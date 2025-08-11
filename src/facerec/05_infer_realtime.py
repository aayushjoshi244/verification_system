"""
05_infer_realtime.py — Real-time (or folder) face recognition using:
- InsightFace (detector + ArcFace embeddings)
- Trained Linear SVM classifier (from 04_eval_basic)

Features:
- Unknown-thresholding (defaults to 0.40 or your report value)
- FPS overlay
- Batch mode: annotate images from a folder and save to runs/face/pred_images
- CSV log of predictions
"""

import argparse, json, sys, time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from src.common.paths import ensure_dirs, RUNS, PRED_FACE_DIR, MODELS_FACE_DIR

#--------quality helpers----------
def calc_BLUR(image_bgr):
    return cv2.Laplacian(image_bgr, cv2.CV_64F).var()

def calc_illum(image_bgr):
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())

# -----------------------------
# InsightFace init
# -----------------------------
def init_face_app(det_size=(512, 512)):
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
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

# -----------------------------
# Drawing helpers
# -----------------------------
def draw_label(frame, text, x, y):
    # simple black box with white text
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - h - 8), (x + w + 8, y + 4), (225, 0, 225), -1)
    cv2.putText(frame, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

# -----------------------------
# Embedding helper
# -----------------------------
def embed_faces(app, rgb):
    faces = app.get(rgb)
    out = []
    for f in faces:
        if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
        else:
            e = getattr(f, "embedding", None)
            if e is None:
                continue
            v = np.asarray(e, dtype=np.float32)
            n = np.linalg.norm(v) + 1e-12
            emb = v / n
        out.append((f, emb))
    return out

# -----------------------------
# Prediction with unknown threshold
# -----------------------------
def predict_names(clf, embeds, id2name, unknown_th):
    names, confs = [], []
    svm = clf[-1]
    if hasattr(svm, "predict_proba") and svm.probability:
        P = clf.predict_proba(embeds)
        maxp = P.max(axis=1)
        idx = P.argmax(axis=1)
        for i in range(len(embeds)):
            if maxp[i] < unknown_th:
                names.append("unknown")
            else:
                names.append(id2name[int(idx[i])])
            confs.append(float(maxp[i]))
    else:
        dec = clf.decision_function(embeds)
        if dec.ndim == 1:
            dec = np.stack([-dec, dec], axis=1)
        max_margin = dec.max(axis=1)
        pseudo = 1.0 / (1.0 + np.exp(-max_margin))
        idx = dec.argmax(axis=1)
        for i in range(len(embeds)):
            if pseudo[i] < unknown_th:
                names.append("unknown")
            else:
                names.append(id2name[int(idx[i])])
            confs.append(float(pseudo[i]))
    return names, confs

# -----------------------------
# Load model + labels + default threshold
# -----------------------------
def load_runtime(model_path, labels_path, report_path=None, fallback_th=0.40):
    if not Path(model_path).exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(2)
    if not Path(labels_path).exists():
        print(f"ERROR: labels not found: {labels_path}")
        sys.exit(3)

    clf = joblib.load(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f).items()}

    th = fallback_th
    if report_path and Path(report_path).exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
            th = float(rep.get("threshold_recommended", th))
        except Exception:
            pass
    return clf, id2name, th

# -----------------------------
# Folder mode
# -----------------------------
def run_folder(app, clf, id2name, unknown_th, images_dir, out_dir, csv_log):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    images = [p for p in Path(images_dir).iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
    if not images:
        print(f"[WARN] No images found in {images_dir}")
        return

    for p in images:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"[WARN] unreadable {p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = embed_faces(app, rgb)

        if not faces:
            rows.append({"image": str(p), "name": "", "conf": "", "bbox": "", "note": "no_face"})
        else:
            embeds = np.stack([e for (_, e) in faces], axis=0)
            names, confs = predict_names(clf, embeds, id2name, unknown_th)
            for (f, _), name, conf in zip(faces, names, confs):
                x1,y1,x2,y2 = map(int, f.bbox)
                cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), 2)
                draw_label(bgr, f"{name} ({conf:.2f})", x1, y1)
                rows.append({
                    "image": str(p),
                    "name": name,
                    "conf": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "note": ""
                })

        outp = out_dir / f"pred_{p.name}"
        cv2.imwrite(str(outp), bgr)

    if rows:
        pd.DataFrame(rows).to_csv(csv_log, index=False, encoding="utf-8")
        print(f"[OK] Predictions CSV: {csv_log}")
    print(f"[OK] Annotated images: {out_dir}")

# -----------------------------
# Webcam mode
# -----------------------------
def run_webcam(app, clf, id2name, unknown_th, cam_index=0, window="FaceRec (q to quit)"):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {cam_index}")
        sys.exit(4)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    t0, frames = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = embed_faces(app, rgb)
        if faces:
            embeds = np.stack([e for (_, e) in faces], axis=0)
            names, confs = predict_names(clf, embeds, id2name, unknown_th)
            for (f, _), name, conf in zip(faces, names, confs):
                x1,y1,x2,y2 = map(int, f.bbox)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                draw_label(frame, f"{name} ({conf:.2f})", x1, y1)

        frames += 1
        if frames % 10 == 0:
            fps = frames / (time.time() - t0 + 1e-9)
            draw_label(frame, f"FPS: {fps:.1f}", 10, 30)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Entry
# -----------------------------
def main():
    ensure_dirs()

    ap = argparse.ArgumentParser(description="Real-time/folder face recognition")
    ap.add_argument("--model", type=str, default=str(MODELS_FACE_DIR / "svm_5ppl.pkl"))
    ap.add_argument("--labels", type=str, default=str(MODELS_FACE_DIR / "labels_5ppl.json"))
    ap.add_argument("--report", type=str, default=str(RUNS / "face" / "eval" / "report.json"))
    ap.add_argument("--unknown-threshold", type=float, default=None, help="Override unknown threshold (else uses report)")
    ap.add_argument("--det-size", type=int, nargs=2, default=[512, 512], help="Detector size W H")
    ap.add_argument("--source", type=int, default=None, help="Webcam index (e.g., 0)")
    ap.add_argument("--images", type=str, default=None, help="Folder of images to annotate")
    ap.add_argument("--csv-log", type=str, default=str(RUNS / "face" / "preds.csv"))
    args = ap.parse_args()

    if args.source is None and args.images is None:
        print("ERROR: Provide either --source (webcam index) or --images (folder).")
        sys.exit(5)

    clf, id2name, th_rep = load_runtime(args.model, args.labels, args.report, fallback_th=0.40)
    unknown_th = float(args.unknown_threshold) if args.unknown_threshold is not None else th_rep

    app = init_face_app(det_size=tuple(args.det_size))

    if args.images:
        run_folder(app, clf, id2name, unknown_th, args.images, PRED_FACE_DIR, Path(args.csv_log))
    else:
        run_webcam(app, clf, id2name, unknown_th, cam_index=int(args.source))

if __name__ == "__main__":
    main()
