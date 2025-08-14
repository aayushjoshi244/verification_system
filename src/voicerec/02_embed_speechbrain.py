# src/voice/02_embed_speechbrain.py
from src.voicerec.winlink_shim import patch_links
patch_links()
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from src.voicerec.hf_loader import load_ecapa_encoder

RAW_ROOT   = Path("data/raw/voice")
RUNS_ROOT  = Path("runs/voice")
PROTO_DIR  = RUNS_ROOT / "prototypes"
INDEX_JSON = RUNS_ROOT / "index.json"

os.environ.setdefault("HF_HOME", str((RUNS_ROOT / "hf_cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SR_TARGET = 16000

def _err(msg): print(f"[ERROR] {msg}", file=sys.stderr)

def load_audio_ok(path: Path):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        _err(f"{path.name} read fail: {e}")
        return None
    if wav is None or wav.size == 0: return None
    if wav.ndim > 1: wav = wav[:, 0]  # mono
    if sr != SR_TARGET:
        _err(f"{path.name}: sr={sr} != {SR_TARGET}. Please re-record at 16 kHz.")
        return None
    if len(wav) < int(0.8 * SR_TARGET):  # min ~0.8s voiced
        return None
    if not np.isfinite(wav).all(): return None
    return wav

def l2norm(v: np.ndarray, eps=1e-9) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def robust_mean(embs: np.ndarray, drop_frac=0.2):
    """Drop far-out outliers w.r.t. median vector (cosine distance), then mean.
       embs: [N, D] (already L2-normalized)."""
    if embs.shape[0] <= 2:
        return embs.mean(axis=0)
    med = l2norm(np.median(embs, axis=0))
    sims = embs @ med
    dists = 1.0 - sims
    k = int(round(drop_frac * embs.shape[0]))
    if k <= 0:
        return embs.mean(axis=0)
    keep_idx = np.argsort(dists)[:-k]  # drop worst k
    return embs[keep_idx].mean(axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb",
                    help="SpeechBrain encoder")
    ap.add_argument("--device", default=None, choices=["cpu","cuda"],
                    help="Force device (default: auto)")
    ap.add_argument("--drop-outliers", type=float, default=0.0,
                    help="Fraction of enrol embeddings to drop as outliers (e.g., 0.2)")
    args = ap.parse_args()

    # device (auto if not provided)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Windows-safe: force a writable savedir for the model files
    savedir = str((RUNS_ROOT / "sb_ecapa").resolve())
    (RUNS_ROOT).mkdir(parents=True, exist_ok=True)
    (PROTO_DIR).mkdir(parents=True, exist_ok=True)

    try:
        encoder = load_ecapa_encoder(args.model, device)
    except Exception as e:
        _err(f"Failed to load SpeechBrain model: {e}")
        sys.exit(2)

    persons = sorted([p for p in RAW_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if not persons:
        _err(f"No persons found in {RAW_ROOT}. Run 01_record_enroll first.")
        sys.exit(1)

    index = {}
    for person in persons:
        wavs = sorted(person.glob("*.wav"))
        if not wavs:
            print(f"[WARN] No WAVs for {person.name}, skipping.")
            continue

        embs = []
        for w in wavs:
            audio = load_audio_ok(w)
            if audio is None:
                print(f"[WARN] Skipping {w.name} (bad or short).")
                continue
            try:
                sig = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # [1, T]
                with torch.inference_mode():
                    emb_t = encoder.encode_batch(sig)  # don't trust shapes; fix below

                # → numpy
                emb = emb_t.detach().cpu().numpy()

                # → force 1-D vector
                emb = np.asarray(emb).squeeze()
                if emb.ndim != 1:
                    emb = emb.reshape(-1)

                # sanity & normalize
                if emb.size == 0 or not np.isfinite(emb).all():
                    continue
                embs.append(l2norm(emb.astype(np.float32)))

            except Exception as e:
                print(f"[WARN] Embedding failed for {w.name}: {e}")

        if not embs:
            print(f"[WARN] No valid embeddings for {person.name}; skipping.")
            continue

        # Make sure it's a clean [N, D] array even if some items were (1,D)
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 3 and embs.shape[1] == 1:
            embs = embs[:, 0, :]
        elif embs.ndim != 2:
            embs = embs.reshape(embs.shape[0], -1)

        proto = robust_mean(embs, drop_frac=args.drop_outliers) if args.drop_outliers > 0.0 else embs.mean(axis=0)
        proto = l2norm(proto)                        # normalize prototype

        outp = PROTO_DIR / f"{person.name}.npy"
        np.save(outp, proto)
        index[person.name] = str(outp)
        print(f"[OK] Prototype saved: {outp.name} (from {embs.shape[0]} utts)")

    if not index:
        _err("No prototypes created. Exiting.")
        sys.exit(1)

    INDEX_JSON.write_text(json.dumps(index, indent=2))
    print(f"[DONE] Wrote index: {INDEX_JSON}  (device={device})")

if __name__ == "__main__":
    main()
