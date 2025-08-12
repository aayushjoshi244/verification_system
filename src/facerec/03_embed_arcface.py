#!/usr/bin/env python3
"""
03_embed_arcface.py — Compute ArcFace embeddings from aligned faces.
Inputs:  runs/face/manifest_align.csv  (from 02_align.py)
Outputs: runs/face/embeddings/5ppl.npz (X,y,label_map,paths)
         runs/face/embeddings/stats.csv
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# paths
from src.common.paths import ensure_dirs, RUNS, EMB_FACE_DIR

# -----------------------------
# InsightFace init
# -----------------------------
def init_face_app():
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("ERROR: insightface not installed. Try: pip install insightface onnxruntime opencv-python")
        sys.exit(1)

    providers = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers() or ["CPUExecutionProvider"]
    except Exception:
        pass

    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    # det_size can be small because faces are already aligned crops
    app.prepare(ctx_id=ctx_id, det_size=(224, 224))
    return app

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()
    ap = argparse.ArgumentParser(description="Embed aligned face crops with ArcFace")
    ap.add_argument("--manifest-in", type=str, default=str(RUNS / "face" / "manifest_align.csv"),
                    help="CSV from 02_align.py containing aligned_path and person")
    ap.add_argument("--out-npz", type=str, default=str(EMB_FACE_DIR / "5ppl.npz"),
                    help="Output NPZ for embeddings (X,y,label_map,paths)")
    ap.add_argument("--out-stats", type=str, default=str(EMB_FACE_DIR / "stats.csv"),
                    help="Per-class stats CSV")
    ap.add_argument("--min-per-person", type=int, default=6,
                    help="Minimum usable aligned images per person to include")
    args = ap.parse_args()

    manifest_in = Path(args.manifest_in)
    if not manifest_in.exists():
        print(f"ERROR: {manifest_in} not found. Run 02_align.py first.")
        sys.exit(2)

    df = pd.read_csv(manifest_in)
    needed = {"aligned_path", "person"}
    if not needed.issubset(df.columns):
        print(f"ERROR: manifest missing required columns {needed}. Found: {list(df.columns)}")
        sys.exit(3)

    # group by person and filter by min-per-person
    grouped = df.groupby("person")
    keep_people = [p for p, g in grouped if (g["aligned_path"].notna().sum() >= args.min_per_person)]
    if len(keep_people) < 2:
        print("ERROR: Need at least 2 people with sufficient aligned images. "
              f"Found: {len(keep_people)} meeting min={args.min_per_person}.")
        sys.exit(4)

    df = df[df["person"].isin(keep_people)].copy().reset_index(drop=True)

    # label map
    names = sorted(df["person"].unique().tolist())
    label_map = {name: i for i, name in enumerate(names)}

    app = init_face_app()

    X = []
    y = []
    paths = []
    skipped = 0

    for i, row in df.iterrows():
        apath = str(row["aligned_path"])
        person = row["person"]
        if not apath or apath.lower() == "nan":
            continue
        p = Path(apath)
        if not p.exists():
            skipped += 1
            continue

        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue

        # even though it's aligned, let the app find the face & embedding
        faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not faces:
            skipped += 1
            continue

        # take the largest (should be one)
        f = max(faces, key=lambda F: (F.bbox[2]-F.bbox[0])*(F.bbox[3]-F.bbox[1]))

        # Handle embedding attribute variants
        if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
        else:
            e = getattr(f, "embedding", None)
            if e is None:
                skipped += 1
                continue
            v = np.asarray(e, dtype=np.float32)
            n = np.linalg.norm(v) + 1e-12
            emb = v / n

        X.append(emb)
        y.append(label_map[person])
        paths.append(apath)

    if len(X) < 10:
        print(f"ERROR: Too few total usable embeddings: {len(X)}. Add more images or check alignment.")
        sys.exit(5)

    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)

    # Save NPZ
    EMB_FACE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, X=X, y=y, label_map=label_map, paths=paths)
    # --- prototypes per class (cosine centroids) ---
    # Each prototype = L2-normalized mean embedding for that person
    protos = np.zeros((len(label_map), X.shape[1]), dtype=np.float32)
    for name, cls_id in label_map.items():
        m = (y == cls_id)
        v = X[m].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        protos[cls_id] = v

    # Save alongside embeddings
    np.save(EMB_FACE_DIR / "prototypes.npy", protos)
    print(f"[OK] Prototypes saved: {EMB_FACE_DIR / 'prototypes.npy'}")
    print(f"[OK] Saved embeddings: {args.out_npz}")
    print(f"     Classes: {len(label_map)}  Samples: {len(y)}  Skipped during embed: {skipped}")

    # Stats per class
    stats_rows = []
    for name, cls_id in label_map.items():
        mask = (y == cls_id)
        cnt = int(mask.sum())
        stats_rows.append({"person": name, "count": cnt})
    pd.DataFrame(stats_rows).to_csv(args.out_stats, index=False)
    print(f"[OK] Stats written: {args.out_stats}")

if __name__ == "__main__":
    main()
