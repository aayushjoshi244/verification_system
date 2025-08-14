#!/usr/bin/env python3
"""
Calibrate a good cosine similarity threshold for speaker verification.

It:
- loads enrol WAVs from data/raw/voice/<person>/*.wav
- builds L2-normalized ECAPA embeddings per file
- forms positive (same person) and negative (different person) pairs
- computes score distributions, EER threshold, and F1-optimal threshold
- writes runs/voice/calibration.json (and an optional histogram PNG)

Dependencies:
  numpy, soundfile, torch, speechbrain, (optional) matplotlib
"""
from src.voicerec.winlink_shim import patch_links
patch_links()
import os
import sys
import json
import argparse
import itertools
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from speechbrain.pretrained import EncoderClassifier
from src.voicerec.hf_loader import load_ecapa_encoder

# -------- Paths --------
RAW_ROOT   = Path("data/raw/voice")
RUNS_ROOT  = Path("runs/voice")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_JSON = RUNS_ROOT / "calibration.json"
HIST_PNG    = RUNS_ROOT / "calib_hist.png"

# Put HF cache in project + force copy, not links (Windows-friendly)
os.environ.setdefault("HF_HOME", str((RUNS_ROOT / "hf_cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SR = 16000  # expected sample rate

# -------- Utils --------
def _err(m: str):
    print(f"[ERROR] {m}", file=sys.stderr)

def load_audio_ok(path: Path):
    """Return mono float32 wav at 16k if valid; else None."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        _err(f"{path.name} read fail: {e}")
        return None
    if wav is None or wav.size == 0:
        return None
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != SR:
        _err(f"{path.name}: sr={sr} != {SR}, re-record at 16 kHz")
        return None
    if len(wav) < int(0.8 * SR):  # min ~0.8s
        return None
    if not np.isfinite(wav).all():
        return None
    return wav

def l2norm(v: np.ndarray, eps=1e-9) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def encode_files(per_person_files, model_name, device):
    """Return dict: person -> list of L2-normalized embeddings (1-D np arrays)."""
    try:
        from src.voicerec.hf_loader import load_ecapa_encoder
        encoder = load_ecapa_encoder(model_name, device)
    except Exception as e:
        _err(f"Failed to load SpeechBrain model: {e}"); sys.exit(2)

    def as1d(x):
        x = np.asarray(x)
        x = x.squeeze()
        if x.ndim != 1:
            x = x.reshape(-1)
        return x.astype(np.float32)

    out = {}
    for person, files in per_person_files.items():
        embs = []
        for f in files:
            a = load_audio_ok(f)
            if a is None:
                continue
            sig = torch.from_numpy(a).float().unsqueeze(0).to(device)  # [1, T]
            with torch.inference_mode():
                e = encoder.encode_batch(sig)
            emb = as1d(e.detach().cpu().numpy())
            if emb.size == 0 or not np.isfinite(emb).all():
                continue
            embs.append(l2norm(emb))
        if embs:
            out[person] = embs
    return out


def make_pairs(embs_by_person, max_neg_per_person=2000):
    """Return arrays: scores_pos, scores_neg (cosine)."""
    rng = np.random.default_rng(1234)

    def as1d(x):
        x = np.asarray(x).squeeze()
        return x.reshape(-1).astype(np.float32)

    # Positive pairs (within person)
    pos_scores = []
    for _, embs in embs_by_person.items():
        if len(embs) < 2:
            continue
        for a, b in itertools.combinations(embs, 2):
            a1 = as1d(a); b1 = as1d(b)
            if a1.shape != b1.shape:
                # skip inconsistent pair
                continue
            pos_scores.append(float(np.dot(a1, b1)))  # L2-normed → cosine

    # Negative pairs (across persons)
    persons = list(embs_by_person.keys())
    neg_scores = []
    for p in persons:
        anchors = [as1d(e) for e in embs_by_person[p]]
        others_people = [q for q in persons if q != p]
        if not others_people:
            continue
        other_embs = [as1d(e) for q in others_people for e in embs_by_person[q]]
        if not other_embs:
            continue

        # Ensure all same dim
        D = anchors[0].shape[0]
        anchors = [e for e in anchors if e.shape[0] == D]
        other_embs = [e for e in other_embs if e.shape[0] == D]
        if not anchors or not other_embs:
            continue

        num_targets = min(max_neg_per_person, len(other_embs))
        idxs = rng.choice(len(other_embs), size=num_targets,
                          replace=(num_targets > len(other_embs)))
        sel = np.stack([other_embs[i] for i in idxs], axis=0)  # [M, D]
        for a in anchors:
            # cosines: dot since all L2-normalized
            dots = sel @ a  # [M]
            neg_scores.extend(dots.tolist())

    return np.array(pos_scores, dtype=np.float32), np.array(neg_scores, dtype=np.float32)


def metrics_at_threshold(scores_pos, scores_neg, thr):
    # Predict positive if score >= thr
    tp = int((scores_pos >= thr).sum())
    fn = int((scores_pos <  thr).sum())
    fp = int((scores_neg >= thr).sum())
    tn = int((scores_neg <  thr).sum())
    tpr = tp / max(1, (tp + fn))
    fpr = fp / max(1, (fp + tn))
    prec = tp / max(1, (tp + fp))
    rec  = tpr
    f1 = (2 * prec * rec) / max(1e-9, (prec + rec))
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                tpr=float(tpr), fpr=float(fpr), prec=float(prec), rec=float(rec), f1=float(f1))

def find_eer_threshold(scores_pos, scores_neg):
    """Return (thr_at_eer, eer_value, metrics_dict_at_thr)."""
    all_scores = np.unique(np.concatenate([scores_pos, scores_neg]))
    if all_scores.size == 0:
        return 0.5, 1.0, metrics_at_threshold(scores_pos, scores_neg, 0.5)
    mids = (all_scores[:-1] + all_scores[1:]) / 2.0
    cand = np.concatenate([all_scores[:1] - 1e-6, mids, all_scores[-1:] + 1e-6])

    best = None
    best_gap = 1e9
    for thr in cand:
        m = metrics_at_threshold(scores_pos, scores_neg, thr)
        far = m["fpr"]
        frr = 1.0 - m["tpr"]
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best = (thr, far, frr, m)
    thr, far, frr, m = best
    eer = 0.5 * (far + frr)
    return float(thr), float(eer), m

def find_best_f1_threshold(scores_pos, scores_neg):
    """Return (thr_max_f1, f1_value, metrics_dict_at_thr)."""
    all_scores = np.unique(np.concatenate([scores_pos, scores_neg]))
    if all_scores.size == 0:
        return 0.5, 0.0, metrics_at_threshold(scores_pos, scores_neg, 0.5)
    mids = (all_scores[:-1] + all_scores[1:]) / 2.0
    cand = np.concatenate([all_scores[:1] - 1e-6, mids, all_scores[-1:] + 1e-6])

    best_thr, best_f1, best_m = 0.0, -1.0, None
    for thr in cand:
        m = metrics_at_threshold(scores_pos, scores_neg, thr)
        if m["f1"] > best_f1:
            best_f1, best_thr, best_m = m["f1"], thr, m
    return float(best_thr), float(best_f1), best_m

def maybe_save_hist(scores_pos, scores_neg, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available; skipping histogram.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe
    except Exception:
        pass

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.5, 4.5))
    bins = np.linspace(-1.0, 1.0, 80)
    plt.hist(scores_neg, bins=bins, alpha=0.55, label="Different speakers", density=True)
    plt.hist(scores_pos, bins=bins, alpha=0.55, label="Same speaker",     density=True)
    plt.xlabel("Cosine similarity"); plt.ylabel("Density"); plt.title("Speaker verification score distributions")
    plt.legend(); plt.tight_layout()
    try:
        plt.savefig(out_png, dpi=140)
        print(f"[OK] Saved histogram → {out_png}")
    except Exception as e:
        _err(f"Failed to save histogram: {e}")
    finally:
        plt.close()

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb",
                    help="SpeechBrain encoder")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="Force model device (default: auto)")
    ap.add_argument("--max-neg", type=int, default=2000,
                    help="Max negative comparisons per person")
    ap.add_argument("--save-hist", action="store_true",
                    help="Save histogram PNG")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Gather files per person
    persons = [p for p in RAW_ROOT.iterdir() if p.is_dir()]
    if not persons:
        _err(f"No persons found in {RAW_ROOT}. Run 01_record_enroll first.")
        sys.exit(1)

    per_person_files = {}
    for d in persons:
        wavs = sorted(d.glob("*.wav"))
        if len(wavs) >= 1:
            per_person_files[d.name] = wavs

    # Basic feasibility checks
    if len(per_person_files) < 2:
        _err("Need enrol data for at least 2 different people to calibrate negatives.")
        sys.exit(1)
    if all(len(v) < 2 for v in per_person_files.values()):
        _err("Need at least one person with ≥2 utterances to form positive pairs.")
        sys.exit(1)

    # Encode to embeddings
    embs_by_person = encode_files(per_person_files, model_name=args.model, device=device)
    if len(embs_by_person) < 2:
        _err("Embedding stage produced <2 persons; check your enrol WAVs.")
        sys.exit(1)

    # Build score sets
    pos, neg = make_pairs(embs_by_person, max_neg_per_person=args.max_neg)
    if pos.size == 0 or neg.size == 0:
        _err("Insufficient pairs (pos or neg empty). Need more/better enrol data.")
        sys.exit(1)

    # Stats
    stats = {
        "pos_count": int(pos.size),
        "neg_count": int(neg.size),
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std(ddof=1)),
        "neg_mean": float(neg.mean()),
        "neg_std": float(neg.std(ddof=1)),
    }

    # Thresholds
    thr_eer, eer, m_eer = find_eer_threshold(pos, neg)
    thr_f1,  f1,  m_f1 = find_best_f1_threshold(pos, neg)

    report = {
        "model": args.model,
        "device": device,
        "stats": stats,
        "eer": {
            "threshold": float(thr_eer),
            "eer": float(eer),
            "metrics": m_eer
        },
        "f1_opt": {
            "threshold": float(thr_f1),
            "f1": float(f1),
            "metrics": m_f1
        }
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    print(f"[DONE] Wrote calibration report → {REPORT_JSON}")
    print(f"  » EER threshold ≈ {thr_eer:.3f} (EER={eer:.3f})")
    print(f"  » F1-optimal threshold ≈ {thr_f1:.3f} (F1={f1:.3f})")

    if args.save_hist:
        maybe_save_hist(pos, neg, HIST_PNG)

if __name__ == "__main__":
    main()
