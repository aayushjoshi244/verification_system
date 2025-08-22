# src/avsync/content_match_strong.py
import argparse, json, math, time
from pathlib import Path
import numpy as np
import cv2
import soundfile as sf
import torch
from pathlib import Path
from datetime import datetime
import json

try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False

# --- Optional deps (handled gracefully) ---
def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

mp = _try_import("mediapipe")
phonemizer = _try_import("phonemizer")
whisperx = _try_import("whisperx")      # preferred (phoneme timing)
whisper = _try_import("whisper")        # fallback (word timing)
g2p_en = _try_import("g2p_en")  # NEW

RUNS = Path("runs/avsync"); RUNS.mkdir(parents=True, exist_ok=True)

# ---------------- Viseme inventory (shared for video & audio) ----------------
# Minimal, discriminative set (expand later if you plug a real VSR model):
VCLASSES = ["C", "F", "R", "O", "N"]  # Closure, Teeth-lip, Rounded, Open, Neutral
V2IDX = {v:i for i,v in enumerate(VCLASSES)}

# ARPABET-ish mapping -> our viseme groups
PH2V = {
    # closures (lips closed: p/b/m)
    "P":"C","B":"C","M":"C",
    # teeth-lip (f/v)
    "F":"F","V":"F",
    # rounded vowels
    "UW":"R","OW":"R","UH":"R","AO":"R","OY":"R","W":"R",
    # open/dropped jaw vowels
    "AA":"O","AE":"O","AH":"O","AW":"O","AY":"O","EH":"O","ER":"O","EY":"O",
    "IH":"O","IY":"O","OH":"O","ʌ":"O",
    # default
}

def map_phone_to_viseme(ph):
    p = ph.upper().strip(":;,.!?")
    # Strip stress digits if present (e.g., AH0, EH1)
    if p and p[-1].isdigit(): p = p[:-1]
    return PH2V.get(p, "N")

def _print_pretty_summary(result: dict) -> None:
    def _yesno(x): return "YES" if bool(x) else "NO"
    cc = result.get("class_counts") or {}
    print("\n=== AVSYNC SUMMARY ===")
    print(f"Video:         {result.get('video')}")
    print(f"Audio:         {result.get('audio')}")
    print(f"Duration (s):  {result.get('duration_s')}")
    print(f"Frames × hop:  {result.get('frames')} × {result.get('hop_ms')} ms")
    print(f"DTW sim:       {result.get('posterior_dtw_sim')}")
    print(f"Top1 agree:    {result.get('top1_agree')}")
    print(f"Avg cost:      {result.get('avg_cost')}")
    print(f"Path len:      {result.get('path_len')}")
    print(f"Threshold:     {result.get('threshold')}")
    print(f"FINAL OK:      {_yesno(result.get('final_ok', False))}")
    p = result.get("params", {}) or {}
    print("\nParams:")
    print(f"  lang={p.get('lang')}  asr_model={p.get('asr_model')}  temp={p.get('temp')}  speech_gate={p.get('speech_gate')}")
    print("\nClass counts (video vs audio):")
    for label, vals in cc.items():
        print(f"  {label}: video={int(vals.get('video_frames',0))}  audio={int(vals.get('audio_frames',0))}")


def _append_excel(result: dict, excel_path: str, meta: dict, transcript_rows=None) -> str:
    """
    Append one run into Excel with sheets:
      - Runs         (one row per run)
      - ClassCounts  (two rows per class: modality=video/audio)
      - Transcript   (one row per word with timestamps and viseme)
    """
    if not _HAVE_PANDAS:
        raise RuntimeError("Excel export requires pandas/openpyxl: pip install pandas openpyxl")

    out = Path(excel_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    p = (result.get("params") or {})
    summary_row = {
        "run_id": run_id,
        "subject_id": meta.get("subject_id", "unknown"),
        "session_id": meta.get("session_id", "s1"),
        "trial": meta.get("trial", "t1"),
        "note": meta.get("note", ""),
        "video": result.get("video"),
        "audio": result.get("audio"),
        "hop_ms": result.get("hop_ms"),
        "frames": result.get("frames"),
        "duration_s": result.get("duration_s"),
        "posterior_dtw_sim": result.get("posterior_dtw_sim"),
        "avg_cost": result.get("avg_cost"),
        "path_len": result.get("path_len"),
        "top1_agree": result.get("top1_agree"),
        "threshold": result.get("threshold"),
        "final_ok": result.get("final_ok"),
        "lang": p.get("lang"),
        "asr_model": p.get("asr_model"),
        "temp": p.get("temp"),
        "speech_gate": p.get("speech_gate"),
    }
    df_runs_new = pd.DataFrame([summary_row])

    cc = result.get("class_counts") or {}
    rows = []
    for label, vals in cc.items():
        rows.append({"run_id": run_id, "class": label, "modality": "video", "frames": int(vals.get("video_frames",0))})
        rows.append({"run_id": run_id, "class": label, "modality": "audio", "frames": int(vals.get("audio_frames",0))})
    df_cc_new = pd.DataFrame(rows)

    df_tx_new = None
    if transcript_rows:
        df_tx_new = pd.DataFrame([dict(run_id=run_id, **r) for r in transcript_rows])

    # Load existing and append
    if out.exists():
        try:
            old = pd.read_excel(out, sheet_name=None, engine="openpyxl")
            df_runs = pd.concat([old.get("Runs", pd.DataFrame()), df_runs_new], ignore_index=True)
            df_cc   = pd.concat([old.get("ClassCounts", pd.DataFrame()), df_cc_new], ignore_index=True)
            if df_tx_new is not None:
                df_tx = pd.concat([old.get("Transcript", pd.DataFrame()), df_tx_new], ignore_index=True)
            else:
                df_tx = old.get("Transcript", None)
        except Exception:
            df_runs, df_cc, df_tx = df_runs_new, df_cc_new, df_tx_new
    else:
        df_runs, df_cc, df_tx = df_runs_new, df_cc_new, df_tx_new

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df_runs.to_excel(xw, sheet_name="Runs", index=False)
        df_cc.to_excel(xw, sheet_name="ClassCounts", index=False)
        if df_tx is not None:
            df_tx.to_excel(xw, sheet_name="Transcript", index=False)

    return out.resolve().as_posix()


# ---------------- Basic DTW (cosine distance on posteriors) ------------------
def dtw_cosine(A, B):
    """
    A: [T1, K] posteriors, B: [T2, K] posteriors
    returns (similarity_in_[0,1], avg_cost, path_len)
    """
    if len(A)==0 or len(B)==0: return 0.0, 1.0, 0
    # normalize rows
    def _norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return X / n
    A = _norm(A); B = _norm(B)

    T1, T2 = len(A), len(B)
    D = np.zeros((T1, T2), dtype=np.float32)
    # cosine distance = 1 - cos
    for i in range(T1):
        D[i,:] = 1.0 - (A[i] @ B.T)

    # DTW
    C = np.full((T1+1, T2+1), np.inf, dtype=np.float32)
    C[0,0] = 0.0
    for i in range(1, T1+1):
        ai = i-1
        for j in range(1, T2+1):
            bj = j-1
            c = D[ai, bj]
            C[i,j] = c + min(C[i-1,j], C[i,j-1], C[i-1,j-1])

    # backtrack to compute path length
    i, j = T1, T2
    path_len = 0
    while i>0 and j>0:
        path_len += 1
        step = np.argmin([C[i-1,j], C[i,j-1], C[i-1,j-1]])
        if step == 0: i -= 1
        elif step == 1: j -= 1
        else: i -= 1; j -= 1
    avg_cost = float(C[T1, T2] / max(path_len, 1))
    sim = float(max(0.0, 1.0 - avg_cost))  # map cost→similarity
    return sim, avg_cost, path_len

# ----------------- Video → viseme posteriorgram (MediaPipe) ------------------
def video_viseme_posterior(video_path, hop_ms=20, temp=0.2):
    if mp is None:
        raise RuntimeError("mediapipe not installed: pip install mediapipe")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hop_s = hop_ms / 1000.0

    feats = []
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks:
                feats.append([0., 0., 0.]); continue
            lm = res.multi_face_landmarks[0].landmark
            H, W = frame.shape[:2]
            yu = lm[13].y * H; yl = lm[14].y * H
            xl = lm[61].x * W; xr = lm[291].x * W
            nose = lm[1].y * H; chin = lm[152].y * H
            face_h = max(1.0, chin - nose)
            h = max(0.0, yl - yu) / face_h
            w = max(1e-3, (xr - xl) / W)
            r = h / w
            feats.append([h, w, r])
    cap.release()
    if not feats:
        return np.zeros((0, len(VCLASSES)), np.float32), hop_ms

    F = np.asarray(feats, dtype=np.float32)
    if len(F) > 5:
        for k in range(F.shape[1]):
            F[:, k] = cv2.GaussianBlur(F[:, k].reshape(-1, 1), (5, 1), 0).ravel()

    proto = {
        "C": np.array([0.02, 0.40, 0.05], dtype=np.float32),
        "F": np.array([0.05, 0.55, 0.10], dtype=np.float32),
        "R": np.array([0.15, 0.18, 0.90], dtype=np.float32),
        "O": np.array([0.35, 0.40, 0.70], dtype=np.float32),
        "N": np.array([0.10, 0.40, 0.25], dtype=np.float32),
    }
    S = np.zeros((len(F), len(VCLASSES)), dtype=np.float32)
    temp = max(1e-3, float(temp))   # soften with larger temp (e.g., 0.2)
    for t, f in enumerate(F):
        scores = []
        for v in VCLASSES:
            d2 = np.sum((f - proto[v])**2)
            scores.append(-d2 / temp)
        e = np.exp(scores - np.max(scores))
        S[t, :] = e / (e.sum() + 1e-9)

    t_frame = np.arange(len(S)) / float(fps)
    t_target = np.arange(0.0, t_frame[-1] if len(t_frame) else 0.0, hop_s)
    if len(t_target) == 0:
        return np.zeros((0, len(VCLASSES)), np.float32), hop_ms

    out = np.zeros((len(t_target), len(VCLASSES)), dtype=np.float32)
    for k in range(len(VCLASSES)):
        out[:, k] = np.interp(t_target, t_frame, S[:, k])
    out = out / (out.sum(axis=1, keepdims=True) + 1e-9)
    return out, hop_ms


# ---------------- Audio → viseme posteriorgram (WhisperX preferred) ----------
def _gaussian_smooth_rows(M, sigma_frames=1.0):
    if len(M)==0: return M
    if sigma_frames <= 0: return M
    from math import exp
    rad = int(max(1, round(3*sigma_frames)))
    w = np.array([exp(-(i*i)/(2*sigma_frames*sigma_frames)) for i in range(-rad,rad+1)], dtype=np.float32)
    w /= w.sum()
    K = len(w); T,Kv = M.shape
    out = np.zeros_like(M)
    for t in range(T):
        acc = np.zeros(Kv, dtype=np.float32)
        for j in range(-rad,rad+1):
            u = t+j
            if 0 <= u < T:
                acc += w[j+rad]*M[u]
        out[t] = acc
    return out

def _rms_envelope(audio, sr, hop_ms=20):
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    hop = max(1, int(sr * (hop_ms / 1000.0)))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, np.float32)
    env = np.empty(n, dtype=np.float32)
    for i in range(n):
        sl = audio[i*hop:(i+1)*hop]
        env[i] = float(np.sqrt(np.mean(sl**2) + 1e-12))
    m = env.max()
    if m > 0: env /= m
    return env

def _words_to_arpabet(word_spans, lang="en"):
    """
    word_spans: list of (word, t0, t1)
    returns list of (ARPABET_phone, start, end)
    """
    phones = []
    # Prefer g2p_en for English (ARPABET with stress digits)
    if lang.startswith("en") and g2p_en is not None:
        g2p = g2p_en.G2p()
        for text, t0, t1 in word_spans:
            seq = [p for p in g2p(text) if p and p != " " and p != "SIL"]
            if not seq:
                continue
            dur = (t1 - t0) / len(seq)
            for i, p in enumerate(seq):
                # ARPABET tokens like 'AH0', 'IY1' → keep; map strips stress later
                s = t0 + i * dur
                e = s + dur
                phones.append((p, s, e))
        return phones
    return None  # signal to fall back to phonemizer

# -------- Word-level transcript helpers (faster-whisper + g2p_en) ----------
def _extract_word_spans(audio_path: Path, lang="en", asr_model="small"):
    """
    Returns a list of tuples: (word, start_s, end_s).
    Uses faster-whisper with word timestamps.
    """
    try:
        from faster_whisper import WhisperModel as FWModel
    except Exception as e:
        raise RuntimeError("faster-whisper is required for transcript export. pip install faster-whisper") from e

    model = FWModel(asr_model, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(audio_path),
        language=(lang if lang else None),
        word_timestamps=True,
        vad_filter=False,
    )
    spans = []
    for seg in segments:
        if getattr(seg, "words", None):
            for w in seg.words:
                if w is None: 
                    continue
                token = (w.word or "").strip()
                if not token or w.start is None or w.end is None:
                    continue
                spans.append((token, float(w.start), float(w.end)))
        else:
            token = (seg.text or "").strip()
            if token and seg.start is not None and seg.end is not None:
                spans.append((token, float(seg.start), float(seg.end)))
    return spans


def _dominant_viseme_for_word(word: str):
    """
    Uses g2p_en to phonemize the word (ARPABET w/ stress) and maps phones -> viseme.
    Returns a single representative viseme for the word (majority vote),
    or 'N' if unavailable.
    """
    if g2p_en is None:
        return "N"
    try:
        g2p = g2p_en.G2p()
        seq = [p for p in g2p(word) if p and p not in (" ", "SIL")]
        if not seq:
            return "N"
        counts = {}
        for ph in seq:
            v = map_phone_to_viseme(ph)
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get)
    except Exception:
        return "N"


def audio_viseme_posterior(audio_path, hop_ms=20, lang="en", asr_model="small"):
    import math, numpy as np, soundfile as sf
    audio, sr = sf.read(str(audio_path), always_2d=False)
    _ = max(1, int(sr * (hop_ms / 1000.0)))

    phones = []

    # ---------- ASR (faster-whisper) to get word spans ----------
    from faster_whisper import WhisperModel as FWModel
    fw = FWModel(asr_model, device="cpu", compute_type="int8")
    segments, _ = fw.transcribe(
        str(audio_path),
        language=(lang if lang else None),
        word_timestamps=True,
        vad_filter=False,
    )
    word_spans = []
    for seg in segments:
        if getattr(seg, "words", None):
            for w in seg.words:
                if w.start is None or w.end is None: continue
                token = (w.word or "").strip()
                if token:
                    word_spans.append((token, float(w.start), float(w.end)))
        elif seg.start is not None and seg.end is not None:
            token = (seg.text or "").strip()
            if token:
                word_spans.append((token, float(seg.start), float(seg.end)))

    # ---------- Prefer ARPABET via g2p_en ----------
    arpa = _words_to_arpabet(word_spans, lang=lang)
    if arpa is not None:
        phones = arpa
    else:
        # ---------- Fallback: phonemizer (IPA) ----------
        try:
            from phonemizer import phonemize
            for text, t0, t1 in word_spans:
                phs = phonemize(text, language=lang, backend="espeak",
                                strip=True, njobs=1, with_stress=True)
                toks = [p for p in phs.replace("|", " ").split() if p.strip()]
                if not toks: continue
                dur = (t1 - t0) / len(toks)
                for i, p in enumerate(toks):
                    s = t0 + i * dur
                    e = s + dur
                    phones.append((p, s, e))
        except Exception:
            raise RuntimeError(
                "Install a phonemizer: pip install 'g2p_en' or 'phonemizer[espeak]'"
            )

    if not phones:
        return np.zeros((0, len(VCLASSES)), np.float32), hop_ms

    # phones -> viseme posterior grid
    t_end = max(e for _, _, e in phones)
    T = int(math.ceil(max(1e-6, t_end) / (hop_ms / 1000.0)))
    P = np.zeros((T, len(VCLASSES)), dtype=np.float32)
    for ph, s, e in phones:
        v = map_phone_to_viseme(ph)  # handles ARPABET (with stress)
        k = V2IDX[v]
        i0 = int(max(0, math.floor(s / (hop_ms / 1000.0))))
        i1 = int(max(i0 + 1, math.ceil(e / (hop_ms / 1000.0))))
        i1 = min(i1, T)
        P[i0:i1, k] += 1.0

    P = P / (P.sum(axis=1, keepdims=True) + 1e-9)
    P = _gaussian_smooth_rows(P, sigma_frames=1.0)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-9)
    return P.astype(np.float32), hop_ms


# --------------------------- Runner / CLI ------------------------------------
def run(video, audio, hop_ms=20, plot=False, quiet=False,
        save_series=False, outdir=RUNS, lang="en", asr_model="small",
        temp=0.2, speech_gate=0.02, threshold=0.55):
    import json, time, matplotlib.pyplot as plt

    video = Path(video); audio = Path(audio)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    def _norm_rows(P):
        if P.size == 0: return P
        return (P / (P.sum(axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    def _resample_post(P, src_hop_ms, dst_hop_ms):
        if P.size == 0 or src_hop_ms == dst_hop_ms: return P
        T_src, C = P.shape
        t_src = np.arange(T_src) * (src_hop_ms / 1000.0)
        total_t = (T_src - 1) * (src_hop_ms / 1000.0) if T_src > 1 else 0.0
        if total_t <= 0: return P
        T_dst = int(np.floor(total_t / (dst_hop_ms / 1000.0))) + 1
        t_dst = np.arange(T_dst) * (dst_hop_ms / 1000.0)
        P_out = np.empty((T_dst, C), dtype=np.float32)
        for c in range(C):
            P_out[:, c] = np.interp(t_dst, t_src, P[:, c])
        return _norm_rows(P_out)

    # --- get posteriors ---
    Pv, hop_v = video_viseme_posterior(video, hop_ms=hop_ms, temp=temp)
    Pa, hop_a = audio_viseme_posterior(audio, hop_ms=hop_ms, lang=lang, asr_model=asr_model)

    if hop_v != hop_a:
        Pa = _resample_post(Pa, hop_a, hop_v)
        hop_ms = hop_v

    Pv = _norm_rows(Pv); Pa = _norm_rows(Pa)

    T = min(len(Pv), len(Pa))
    if T == 0:
        out = {"video": str(video), "audio": str(audio), "error": "empty posterior series"}
        if not quiet: print(json.dumps(out, indent=2))
        return out
    Pv = Pv[:T]; Pa = Pa[:T]

    # --- speech gating (ignore low-energy frames) ---
    try:
        audio_wav, sr = sf.read(str(audio), always_2d=False)
        env = _rms_envelope(audio_wav, sr, hop_ms=hop_ms)
        env = env[:T] if len(env) >= T else np.pad(env, (0, T - len(env)))
        mask = env > float(speech_gate)
        if mask.sum() >= 5:   # keep at least a handful of frames
            Pv = Pv[mask]; Pa = Pa[mask]; T = len(Pv)
    except Exception:
        pass

    # --- DTW ---
    sim, avg_cost, path_len = dtw_cosine(Pv, Pa)

    v_top = Pv.argmax(axis=1); a_top = Pa.argmax(axis=1)
    top1_agree = float(np.mean(v_top == a_top)) if T > 0 else 0.0
    cls_stats = {c: {
        "video_frames": int((v_top == idx).sum()),
        "audio_frames": int((a_top == idx).sum())
    } for c, idx in V2IDX.items()}

    thr = float(threshold)
    out = {
        "video": str(video), "audio": str(audio),
        "hop_ms": int(hop_ms), "frames": int(T),
        "duration_s": round(T * (hop_ms / 1000.0), 3),
        "posterior_dtw_sim": round(float(sim), 4),
        "avg_cost": round(float(avg_cost), 4),
        "path_len": int(path_len),
        "top1_agree": round(top1_agree, 3),
        "threshold": thr, "final_ok": bool(sim >= thr),
        "class_counts": cls_stats,
        "params": {"lang": lang, "asr_model": asr_model, "temp": float(temp), "speech_gate": float(speech_gate)}
    }

    stamp = int(time.time())
    (outdir / f"content_{stamp}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    if save_series:
        np.save(outdir / f"Pv_{stamp}.npy", Pv)
        np.save(outdir / f"Pa_{stamp}.npy", Pa)

    if plot and T > 0:
        try:
            t = np.arange(T) * (hop_ms / 1000.0)
            show_idx = [V2IDX["C"], V2IDX["O"], V2IDX["R"]]
            plt.figure(figsize=(10, 5))
            for idx in show_idx:
                plt.plot(t, Pv[:, idx], label=f"video {VCLASSES[idx]}")
                plt.plot(t, Pa[:, idx], linestyle='--', label=f"audio {VCLASSES[idx]}")
            plt.xlabel("time (s)"); plt.ylabel("posterior")
            plt.title(f"Posterior DTW sim = {sim:.2f}  |  top1 agree = {top1_agree:.2f}")
            plt.legend()
            out_png = outdir / f"content_{stamp}.png"
            plt.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close()
            out["plot"] = str(out_png)
        except Exception as e:
            if not quiet: print(f"[WARN] Plot failed: {e}")

    if not quiet:
        print(json.dumps(out, indent=2))
    return out




def main():
    ap = argparse.ArgumentParser(
        description="Strong content match: viseme posterior DTW (video vs audio)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--hop-ms", type=int, default=20)
    ap.add_argument("--lang", default="en", help="ASR/phonemizer language code (e.g., en)")
    ap.add_argument("--asr-model", default="small", choices=["tiny","base","small","medium","large-v3"],
                    help="faster-whisper model size (CPU INT8)")
    ap.add_argument("--temp", type=float, default=0.2, help="video viseme softmax temperature (higher=softer)")
    ap.add_argument("--speech-gate", type=float, default=0.02, help="RMS gate to ignore non-speech frames [0-1]")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--save-series", action="store_true")
        # ---- pretty print + excel export (new) ----
    ap.add_argument("--pretty", action="store_true",
                    help="Print a human-readable summary to the terminal.")
    ap.add_argument("--export-excel", default=None,
                    help="Append results to an Excel file (e.g., results/avsync_results.xlsx).")
    ap.add_argument("--export-transcript", action="store_true",
                help="Also append a per-word transcript with timestamps/viseme to Excel.")
    ap.add_argument("--subject-id", default="unknown")
    ap.add_argument("--session-id", default="s1")
    ap.add_argument("--trial", default="t1")
    ap.add_argument("--note", default="")
    ap.add_argument("--threshold", type=float, default=0.55,
                help="Similarity threshold for deciding final_ok")
    args = ap.parse_args()

    result = run(Path(args.video), Path(args.audio),
             hop_ms=args.hop_ms, plot=args.plot, quiet=args.quiet,
             save_series=args.save_series, lang=args.lang,
             asr_model=args.asr_model, temp=args.temp, speech_gate=args.speech_gate,
             threshold=args.threshold)

    # Pretty terminal summary (does NOT replace the JSON printed by run)
    if args.pretty and isinstance(result, dict):
        _print_pretty_summary(result)

    spans = _extract_word_spans(Path(args.audio), lang=args.lang, asr_model=args.asr_model)
    hop_s = (result.get("hop_ms", 20)) / 1000.0
    frames_total = int(result.get("frames", 0))
    rows = []
    for i, (word, s, e) in enumerate(spans):
        v = _dominant_viseme_for_word(word)
        sf_idx = int(max(0, round(s / hop_s))) if hop_s > 0 else 0
        ef_idx = int(max(0, round(e / hop_s))) if hop_s > 0 else 0
        if frames_total > 0:
            sf_idx = min(sf_idx, frames_total - 1)
            ef_idx = min(ef_idx, frames_total - 1)
        rows.append({
            "idx": i,
            "word": word,
            "start_s": round(float(s), 3),
            "end_s": round(float(e), 3),
            "duration_s": round(float(e - s), 3),
            "viseme": v,
            "start_frame": sf_idx,
            "end_frame": ef_idx,
        })
    transcript_rows = rows

    # Excel export (appends)
    if args.export_excel and isinstance(result, dict):
        meta = {
            "subject_id": args.subject_id,
            "session_id": args.session_id,
            "trial": args.trial,
            "note": args.note,
        }
        try:
            xlsx_path = _append_excel(result, args.export_excel, meta, transcript_rows=transcript_rows)
            print(f"\nExcel appended: {xlsx_path}")
        except Exception as e:
            print(f"\n[WARN] Excel export failed: {e}")


if __name__ == "__main__":
    main()