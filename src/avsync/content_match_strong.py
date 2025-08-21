# src/avsync/content_match_strong.py
import argparse, json, math, time
from pathlib import Path
import numpy as np
import cv2
import soundfile as sf

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
    "IH":"O","IY":"O","OH":"O","OY":"O","ʌ":"O",
    # default
}

def map_phone_to_viseme(ph):
    p = ph.upper().strip(":;,.!?")
    # Strip stress digits if present (e.g., AH0, EH1)
    if p and p[-1].isdigit(): p = p[:-1]
    return PH2V.get(p, "N")

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
def video_viseme_posterior(video_path, hop_ms=20):
    if mp is None:
        raise RuntimeError("mediapipe not installed: pip install mediapipe")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hop_s = hop_ms/1000.0

    # collect per-frame features
    feats = []   # [ [h_norm, w_norm, r] ... ]
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
                feats.append([0.,0.,0.])
                continue
            lm = res.multi_face_landmarks[0].landmark
            H,W = frame.shape[:2]
            # Points
            yu = lm[13].y*H; yl = lm[14].y*H
            xl = lm[61].x*W; xr = lm[291].x*W
            nose = lm[1].y*H; chin = lm[152].y*H
            face_h = max(1.0, chin - nose)
            h = max(0.0, yl - yu)/face_h
            w = max(1e-3, (xr - xl)/W)              # relative width
            r = h / w                                # "rounding": tall vs wide
            feats.append([h, w, r])

    cap.release()
    if not feats: return np.zeros((0, len(VCLASSES)), np.float32), hop_ms

    F = np.asarray(feats, dtype=np.float32)
    # temporal smoothing
    if len(F) > 5:
        for k in range(F.shape[1]):
            F[:,k] = cv2.GaussianBlur(F[:,k].reshape(-1,1), (5,1), 0).ravel()

    # Heuristic “proto means” per viseme in feature space [h,w,r]
    # C: closed (h~0), normal width; F: small h, normal/big w; R: rounded (small w, moderate h); O: big h
    proto = {
        "C": np.array([0.02, 0.40, 0.05], dtype=np.float32),
        "F": np.array([0.05, 0.55, 0.10], dtype=np.float32),
        "R": np.array([0.15, 0.18, 0.90], dtype=np.float32),
        "O": np.array([0.35, 0.40, 0.70], dtype=np.float32),
        "N": np.array([0.10, 0.40, 0.25], dtype=np.float32),
    }
    S = np.zeros((len(F), len(VCLASSES)), dtype=np.float32)
    for t, f in enumerate(F):
        # negative squared distance → softmax
        scores = []
        for v in VCLASSES:
            d2 = np.sum((f - proto[v])**2)
            scores.append(-d2 / 0.02)  # temperature
        e = np.exp(scores - np.max(scores))
        p = e / (e.sum() + 1e-9)
        S[t,:] = p

    # resample from frame grid to hop grid
    t_frame = np.arange(len(S))/float(fps)
    t_target = np.arange(0.0, t_frame[-1] if len(t_frame) else 0.0, hop_s)
    if len(t_target)==0: return np.zeros((0,len(VCLASSES)), np.float32), hop_ms

    # interp each class independently
    out = np.zeros((len(t_target), len(VCLASSES)), dtype=np.float32)
    for k in range(len(VCLASSES)):
        out[:,k] = np.interp(t_target, t_frame, S[:,k])
    # renormalize rows
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

def audio_viseme_posterior(audio_path, hop_ms=20, lang="en"):
    audio, sr = sf.read(str(audio_path), always_2d=False)
    hop = int(sr * (hop_ms/1000.0))
    if hop < 1: hop = 1

    # Try WhisperX (word+phoneme alignment)
    if whisperx is not None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisperx.load_model("small", device)
        trans = model.transcribe(audio, sr)
        align_model, metadata = whisperx.load_align_model(language_code=trans["language"], device=device)
        aligned = whisperx.align(trans["segments"], align_model, metadata, audio, sr, device)
        phones = []
        for seg in aligned["segments"]:
            for ph in seg.get("characters", []):  # whisperx stores per-character/phoneme depending on lang/model
                label = ph.get("text","").strip()
                start = ph.get("start", seg["start"])
                end   = ph.get("end",   seg["end"])
                if not label: continue
                phones.append((label, float(start), float(end)))
    else:
        # Fallback: Whisper words + phonemizer -> distribute over word span
        if whisper is None or phonemizer is None:
            raise RuntimeError("Need whisperx OR (whisper + phonemizer). Try: pip install whisperx phonemizer")
        mdl = whisper.load_model("small")
        res = mdl.transcribe(str(audio_path), word_timestamps=True)
        from phonemizer import phonemize
        phones = []
        for seg in res.get("segments", []):
            words = seg.get("words", [])
            for w in words:
                text = w["word"]
                t0, t1 = float(w["start"]), float(w["end"])
                phs = phonemize(text, language=lang, backend="espeak", strip=True, njobs=1, with_stress=True)
                tokens = [p for p in phs.replace("|"," ").split() if p.strip()]
                if not tokens:
                    continue
                dur = (t1 - t0) / len(tokens)
                for i, p in enumerate(tokens):
                    start = t0 + i*dur
                    end   = start + dur
                    phones.append((p, start, end))

    # Convert phones → viseme posteriors on the hop grid
    if not phones:
        return np.zeros((0, len(VCLASSES)), np.float32), hop_ms

    t_end = max(e for _,_,e in phones)
    T = int(math.ceil(t_end / (hop_ms/1000.0)))
    P = np.zeros((T, len(VCLASSES)), dtype=np.float32)

    for ph, s, e in phones:
        v = map_phone_to_viseme(ph)
        k = V2IDX[v]
        i0 = int(max(0, math.floor(s / (hop_ms/1000.0))))
        i1 = int(max(i0+1, math.ceil(e / (hop_ms/1000.0))))
        P[i0:i1, k] += 1.0

    # normalize rows; smooth a bit to reflect coarticulation
    P = P / (P.sum(axis=1, keepdims=True) + 1e-9)
    P = _gaussian_smooth_rows(P, sigma_frames=1.0)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-9)
    return P, hop_ms

# --------------------------- Runner / CLI ------------------------------------
def run(video, audio, hop_ms=20, plot=False, quiet=False):
    Pv, _ = video_viseme_posterior(video, hop_ms=hop_ms)
    Pa, _ = audio_viseme_posterior(audio, hop_ms=hop_ms)

    # Trim to common duration
    T = min(len(Pv), len(Pa))
    Pv = Pv[:T]; Pa = Pa[:T]

    sim, avg_cost, path_len = dtw_cosine(Pv, Pa)

    # Simple per-class summary
    v_top = Pv.argmax(axis=1) if len(Pv) else np.array([])
    a_top = Pa.argmax(axis=1) if len(Pa) else np.array([])
    cls_stats = {}
    for c, idx in V2IDX.items():
        cls_stats[c] = {
            "video_frames": int((v_top==idx).sum()),
            "audio_frames": int((a_top==idx).sum())
        }

    out = {
        "video": str(video),
        "audio": str(audio),
        "hop_ms": hop_ms,
        "posterior_dtw_sim": round(float(sim), 4),
        "avg_cost": round(float(avg_cost), 4),
        "path_len": int(path_len),
        "class_counts": cls_stats,
        "final_ok": bool(sim >= 0.55)
    }

    stamp = int(time.time())
    out_json = RUNS / f"content_{stamp}.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if plot and len(Pv) and len(Pa):
        import matplotlib.pyplot as plt
        t = np.arange(len(Pv)) * (hop_ms/1000.0)
        # Show most informative classes
        show_idx = [V2IDX["C"], V2IDX["O"], V2IDX["R"]]
        plt.figure(figsize=(10,5))
        for idx in show_idx:
            plt.plot(t, Pv[:,idx], linestyle='-', label=f"video {VCLASSES[idx]}")
            plt.plot(t, Pa[:,idx], linestyle='--', label=f"audio {VCLASSES[idx]}")
        plt.xlabel("time (s)"); plt.ylabel("posterior")
        plt.title(f"Posterior DTW sim = {sim:.2f}")
        plt.legend()
        out_png = RUNS / f"content_{stamp}.png"
        plt.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close()
        out["plot"] = str(out_png)

    if not quiet:
        print(json.dumps(out, indent=2))
    return out

def main():
    ap = argparse.ArgumentParser(description="Strong content match: video viseme posterior vs audio phoneme→viseme posterior (DTW-cosine)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--hop-ms", type=int, default=20)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(Path(args.video), Path(args.audio), hop_ms=args.hop_ms, plot=args.plot, quiet=args.quiet)

if __name__ == "__main__":
    main()