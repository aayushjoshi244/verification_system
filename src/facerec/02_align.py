#!/usr/bin/env python3
"""
02_align.py — Align face crops using MediaPipe FaceMesh and save 112x112 images.
Reads manifest_detect.csv from 01_import_detect and writes manifest_align.csv
"""

import cv2, mediapipe as mp, numpy as np, pandas as pd
from pathlib import Path
from src.common.paths import ensure_dirs, FACE_ALIGNED, RUNS

# Target triad (L eye, R eye, nose) for 112x112; you can tweak if needed
TGT = np.float32([[38, 52], [74, 52], [56, 75]])
SIZE = (112, 112)

mp_mesh = mp.solutions.face_mesh

def get_keypoints(img_bgr):
    # returns np.float32 [[xL,yL],[xR,yR],[xN,yN]] or None
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        res = mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    # MediaPipe indices: 33 (L eye), 263 (R eye), 1 (nose tip)
    return np.float32([[lm[33].x * w, lm[33].y * h],
                       [lm[263].x * w, lm[263].y * h],
                       [lm[1].x * w, lm[1].y * h]])

def align_face(img_bgr):
    pts = get_keypoints(img_bgr)
    if pts is None:
        return None, None
    M, _ = cv2.estimateAffinePartial2D(pts, TGT, method=cv2.LMEDS)
    if M is None:
        return None, None
    aligned = cv2.warpAffine(img_bgr, M, SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned, pts

def main():
    ensure_dirs()
    manifest_in = RUNS / "face" / "manifest_detect.csv"
    manifest_out = RUNS / "face" / "manifest_align.csv"

    if not manifest_in.exists():
        print(f"[ERROR] {manifest_in} not found. Run 01_import_detect first.")
        return

    df = pd.read_csv(manifest_in)
    # sanity check for expected cols from 01_import_detect
    needed = {"crop_path", "person"}
    if not needed.issubset(df.columns):
        print(f"[ERROR] manifest missing required columns {needed}. Found: {list(df.columns)}")
        return

    out_rows = []
    aligned_count = 0
    for _, row in df.iterrows():
        crop_path = str(row["crop_path"])
        person = str(row["person"])
        if not crop_path or crop_path.lower() == "nan":
            continue

        p = Path(crop_path)
        if not p.exists():
            print(f"[WARN] missing crop {p}")
            continue

        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] unreadable crop {p}")
            continue

        aligned, pts = align_face(img)
        if aligned is None:
            print(f"[INFO] align fail {p}")
            continue

        outp = FACE_ALIGNED / person / (p.stem + "_a.jpg")
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), aligned)

        out_rows.append({
            "crop_path": crop_path,
            "person": person,
            "aligned_path": str(outp),
            "landmarks": (pts.tolist() if pts is not None else None)
        })
        aligned_count += 1

    out_df = pd.DataFrame(out_rows, columns=["crop_path", "person", "aligned_path", "landmarks"])
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(manifest_out, index=False)

    print(f"[OK] aligned {aligned_count} faces -> {FACE_ALIGNED}")
    print(f"[INFO] manifest written: {manifest_out}")

if __name__ == "__main__":
    main()
