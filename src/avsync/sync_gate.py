# src/avsync/sync_gate.py
"""
AV Sync Gate: compares mouth aperture (from video) with audio RMS envelope.

- Robust plotting (no length mismatches)
- Optional quiet mode to suppress TF/Mediapipe/protobuf logs
- JSON + PNG artifacts are written to runs/avsync/
"""

from __future__ import annotations
import argparse, json, math, sys, time, os, warnings
from pathlib import Path

# ---- quiet logs (set before heavy libs import) ----
def _silence_framework_logs():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TF C++ logs
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass
    warnings.filterwarnings("ignore", module="google.protobuf")

import cv2
import numpy as np
import soundfile as sf

# MediaPipe for landmarks (lip tracking)
import mediapipe as mp

RUNS_DIR = Path("runs/avsync")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- low-level helpers ----------------------

def rms_envelope(audio: np.ndarray, sr: int, hop_ms: int = 20) -> tuple[np.ndarray, int]:
    """Return mono RMS envelope sampled every hop_ms."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    hop = max(1, int(sr * hop_ms / 1000.0))
    n = (len(audio) // hop)
    if n == 0:
        return np.zeros(0, dtype=np.float32), hop_ms

    # frame-wise RMS without extra deps
    env = np.empty(n, dtype=np.float32)
    for i in range(n):
        sl = audio[i*hop:(i+1)*hop]
        env[i] = float(np.sqrt(np.mean(sl**2) + 1e-12))

    # normalize to [0,1]
    vmax = float(env.max())
    if vmax > 0:
        env = env / vmax
    return env, hop_ms

def mouth_aperture_series(video_path: Path, hop_ms: int = 20, max_frames: int | None = None) -> tuple[np.ndarray, int]:
    """Track lip opening per frame, then resample to hop_ms grid."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 25.0  # fallback

    mp_face = mp.solutions.face_mesh
    # FaceMesh (468) indices: inner upper/lower lip
    idx_upper = 13
    idx_lower = 14

    aper: list[float] = []
    with mp_face.FaceMesh(static_image_mode=False,
                          max_num_faces=1,
                          refine_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as mesh:
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            if max_frames and count > max_frames:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks:
                aper.append(0.0)
                continue
            lm = res.multi_face_landmarks[0].landmark
            H, W = frame.shape[:2]
            yu = lm[idx_upper].y * H
            yl = lm[idx_lower].y * H
            a = max(0.0, yl - yu)  # vertical gap (pixels)

            # normalize by an approximate face height (nose to chin)
            nose = lm[1].y * H
            chin = lm[152].y * H
            face_h = max(1.0, chin - nose)
            aper.append(float(a / face_h))

    cap.release()
    aper = np.asarray(aper, dtype=np.float32)
    if len(aper) == 0:
        return np.zeros(0, dtype=np.float32), hop_ms

    # light smoothing
    if len(aper) > 5:
        aper = cv2.GaussianBlur(aper.reshape(-1, 1), (5, 1), 0).ravel()

    # resample from per-frame (1/fps) → hop_ms grid
    t_video = np.arange(len(aper)) / float(fps)
    total_t = float(t_video[-1]) if len(t_video) > 0 else 0.0
    if total_t <= 0:
        return np.zeros(0, dtype=np.float32), hop_ms

    hop_s = hop_ms / 1000.0
    t_target = np.arange(0.0, total_t, hop_s)
    vm = np.interp(t_target, t_video, aper)

    # normalize to [0,1]
    vmax = float(vm.max())
    if vmax > 0:
        vm = vm / vmax
    return vm.astype(np.float32), hop_ms

def normxcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of two 1D arrays (same length)."""
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    corr = np.correlate(a, b, mode='full') / len(a)
    return corr

def best_sync_score(v_series: np.ndarray, a_series: np.ndarray, hop_ms: int, max_lag_ms: int = 300) -> tuple[float, float]:
    """Return best correlation score and lag (ms)."""
    n = min(len(v_series), len(a_series))
    if n < 5:
        return 0.0, 0.0
    v = v_series[:n]
    a = a_series[:n]
    corr = normxcorr(v, a)
    center = len(corr) // 2
    max_lag_steps = int(max_lag_ms / hop_ms)
    lo = max(0, center - max_lag_steps)
    hi = min(len(corr), center + max_lag_steps + 1)
    segment = corr[lo:hi]
    k = int(np.argmax(segment))
    best = float(segment[k])
    lag_steps = (lo + k) - center
    lag_ms = float(lag_steps * hop_ms)
    return best, lag_ms

# ---------------------- plotting ----------------------

def _z(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-9)

def _safe_plot(vm: np.ndarray, am: np.ndarray, hop_ms: int, lag_ms: float, out_png: Path):
    """Plot mouth vs audio (shifted by lag) with lengths safely matched."""
    import matplotlib.pyplot as plt
    n = min(len(vm), len(am))
    if n == 0:
        return
    vm = vm[:n]
    am = am[:n]

    # shift audio by integral hop to visualize alignment
    shift_steps = int(round(lag_ms / hop_ms))
    if shift_steps > 0:
        am_viz = np.r_[np.zeros(shift_steps, dtype=np.float32), am[:-shift_steps]]
    elif shift_steps < 0:
        am_viz = np.r_[am[-shift_steps:], np.zeros(-shift_steps, dtype=np.float32)]
    else:
        am_viz = am

    t = np.arange(n, dtype=np.float32) * (hop_ms / 1000.0)
    plt.figure()
    plt.plot(t, _z(vm), label="mouth (z)")
    plt.plot(t, _z(am_viz)[:n], label=f"audio (z, shift {lag_ms:.0f} ms)")
    plt.xlabel("time (s)"); plt.ylabel("z-score")
    plt.title("Lip–Audio Sync (visualization)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()

# ---------------------- CLI task ----------------------

def run(video_path: Path, audio_path: Path, hop_ms: int = 20, max_lag_ms: int = 300,
        plot: bool = False, save_series: bool = False, quiet: bool = False):
    # quiet mode (suppress framework logs)
    if quiet:
        _silence_framework_logs()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    vm, hop_v = mouth_aperture_series(video_path, hop_ms=hop_ms)

    # load audio
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if audio is None or len(audio) == 0:
        raise RuntimeError(f"Could not read audio: {audio_path}")
    am, hop_a = rms_envelope(audio, sr, hop_ms=hop_ms)

    score, lag_ms = best_sync_score(vm, am, hop_ms, max_lag_ms=max_lag_ms)

    result = {
        "video": str(video_path),
        "audio": str(audio_path),
        "hop_ms": hop_ms,
        "max_lag_ms": max_lag_ms,
        "score": round(score, 4),
        "lag_ms": round(lag_ms, 1),
        "sync_ok": bool((score >= 0.45) and (abs(lag_ms) <= 200.0))
    }

    # artifacts
    stamp = int(time.time())
    out_json = RUNS_DIR / f"sync_{stamp}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if save_series:
        np.save(RUNS_DIR / f"vmouth_{stamp}.npy", vm)
        np.save(RUNS_DIR / f"aenv_{stamp}.npy", am)

    if plot:
        try:
            out_png = RUNS_DIR / f"sync_{stamp}.png"
            _safe_plot(vm, am, hop_ms=hop_ms, lag_ms=lag_ms, out_png=out_png)
            result["plot"] = str(out_png)
        except Exception as e:
            if not quiet:
                print(f"[WARN] Plot failed: {e}")

    print(json.dumps(result, indent=2))
    return result

def main():
    ap = argparse.ArgumentParser(description="AV Sync Gate (mouth vs audio envelope)")
    ap.add_argument("--video", required=True, help="Path to video file (used for lip tracking)")
    ap.add_argument("--audio", required=True, help="Path to WAV audio (mono or stereo)")
    ap.add_argument("--hop-ms", type=int, default=20)
    ap.add_argument("--max-lag-ms", type=int, default=300)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save-series", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Suppress TensorFlow/Mediapipe/protobuf logs")
    args = ap.parse_args()
    run(Path(args.video), Path(args.audio),
        hop_ms=args.hop_ms, max_lag_ms=args.max_lag_ms,
        plot=args.plot, save_series=args.save_series, quiet=args.quiet)

if __name__ == "__main__":
    main()
