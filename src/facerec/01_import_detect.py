#!/usr/bin/env python3
"""
01_import_detect.py — Detect faces from raw images, save clean crops, log results.
"""

import os
import cv2
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Use your repo's paths.py
try:
    from src.common import paths
except ImportError:
    paths = None  # fallback, will check args for directories

# -----------------------------
# Quality check helpers
# -----------------------------
def calc_blur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calc_illumination(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

# -----------------------------
# Init InsightFace
# -----------------------------
def init_face_app(det_size=(640, 640)):
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError("Please install: pip install insightface onnxruntime opencv-python")

    # Try to get providers; fall back to CPU if onnxruntime missing
    providers = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers() or ["CPUExecutionProvider"]
    except Exception:
        pass

    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

# -----------------------------
# Main detection
# -----------------------------
def detect_and_crop(app, img_path, out_dir, blur_min, illum_min, illum_max):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, "unreadable"

    h, w = img_bgr.shape[:2]
    blur_val = calc_blur(img_bgr)
    illum_val = calc_illumination(img_bgr)

    faces = app.get(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not faces:
        return None, "no_face"

    # pick largest
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)

    # clip bbox to image bounds
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None, "invalid_bbox"

    crop = img_bgr[y1:y2, x1:x2]

    # quality filters
    if blur_val < blur_min:
        return None, f"blurry({blur_val:.1f})"
    if illum_val < illum_min or illum_val > illum_max:
        return None, f"illum({illum_val:.1f})"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), crop)

    det_conf = float(getattr(face, "det_score", 0.0))

    return {
        "raw_path": str(img_path),
        "crop_path": str(out_path),
        "person": out_dir.name,
        "blur": float(blur_val),
        "illum": float(illum_val),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "det_conf": det_conf
    }, None

# -----------------------------
# CLI entry
# -----------------------------
def main():
    # Ensure dirs if we have paths module
    if paths is not None:
        paths.ensure_dirs()

    ap = argparse.ArgumentParser(description="Detect faces from raw images and save crops.")
    ap.add_argument("--in", dest="in_dir", type=str,
                    default=(str(paths.FACE_RAW) if paths else "data/raw/face"),
                    help="Input raw face images root (per-person folders)")
    ap.add_argument("--out-crops", type=str,
                    default=(str(paths.FACE_CROPS) if paths else "data/processed/face/crops"),
                    help="Output directory for cropped faces")
    ap.add_argument("--manifest", type=str,
                    default=(str((paths.RUNS / "face" / "manifest_detect.csv")) if paths else "runs/face/manifest_detect.csv"),
                    help="CSV to log all kept crops")
    ap.add_argument("--skipped", type=str,
                    default=(str((paths.RUNS / "face" / "detect_skipped.csv")) if paths else "runs/face/detect_skipped.csv"),
                    help="CSV to log skipped images and reasons")
    ap.add_argument("--min-per-person", type=int, default=8)
    ap.add_argument("--blur-min", type=float, default=20.0)
    ap.add_argument("--illum-min", type=float, default=30.0)
    ap.add_argument("--illum-max", type=float, default=220.0)
    ap.add_argument("--det-size", type=int, nargs=2, default=[640, 640], help="Detector size W H")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_crops_dir = Path(args.out_crops)
    manifest_path = Path(args.manifest)
    skipped_path = Path(args.skipped)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    app = init_face_app(det_size=tuple(args.det_size))

    manifest_rows = []
    skipped_rows = []

    persons = sorted([p for p in in_dir.iterdir() if p.is_dir()])
    if not persons:
        raise RuntimeError(f"No person folders found in {in_dir}")

    for person_dir in persons:
        kept_count = 0
        for img_path in tqdm(sorted(person_dir.iterdir()), desc=f"Processing {person_dir.name}"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            result, skip_reason = detect_and_crop(
                app,
                img_path,
                out_crops_dir / person_dir.name,
                args.blur_min,
                args.illum_min,
                args.illum_max
            )
            if result:
                manifest_rows.append(result)
                kept_count += 1
            else:
                skipped_rows.append({"raw_path": str(img_path), "person": person_dir.name, "reason": skip_reason})

        if kept_count < args.min_per_person:
            print(f"WARNING: {person_dir.name} has only {kept_count} usable crops (< {args.min_per_person})")

    # Save CSVs
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_path", "crop_path", "person", "blur", "illum", "bbox", "det_conf"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(skipped_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_path", "person", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"\nDetection complete.")
    print(f"Kept: {len(manifest_rows)} images, Skipped: {len(skipped_rows)} images.")
    print(f"Manifest saved to {manifest_path}")
    print(f"Skipped log saved to {skipped_path}")

if __name__ == "__main__":
    main()
