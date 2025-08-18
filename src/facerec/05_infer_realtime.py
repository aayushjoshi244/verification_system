from pathlib import Path
import argparse, json, sys, time
from pathlib import Path
import cv2, joblib, numpy as np, pandas as pd

from src.common.paths import CLEAN_EMB_DIR, CLEAN_FACE_DIR, ensure_dirs, RUNS, PRED_FACE_DIR, MODELS_FACE_DIR, EMB_FACE_DIR

PROTOS = None  # will be loaded in main()
OVERLAY_FILE = (Path(__file__).resolve().parents[2] / "runs" / "face" / "match_overlay.txt")
OVERLAY_TTL  = 5.0 
# ---------- quality helpers ----------
def calc_blur(image_bgr):
    return cv2.Laplacian(image_bgr, cv2.CV_64F).var()

def calc_illum(image_bgr):
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())

# ---------- insightface ----------
def init_face_app(det_size=(512, 512)):
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("ERROR: pip install insightface onnxruntime opencv-python"); sys.exit(1)
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

def draw_label(frame, text, x, y):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - h - 8), (x + w + 8, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def embed_faces(app, rgb):
    faces = app.get(rgb)
    out = []
    for f in faces:
        if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
        else:
            e = getattr(f, "embedding", None)
            if e is None: continue
            v = np.asarray(e, dtype=np.float32); n = np.linalg.norm(v) + 1e-12
            emb = v / n
        out.append((f, emb))
    return out

def predict_names(clf, embeds, id2name, unknown_th):
    names, confs = [], []
    svm = clf[-1]
    if hasattr(svm, "predict_proba") and svm.probability:
        P = clf.predict_proba(embeds); maxp = P.max(axis=1); idx = P.argmax(axis=1)
        for i in range(len(embeds)):
            names.append(id2name[int(idx[i])] if maxp[i] >= unknown_th else "unknown")
            confs.append(float(maxp[i]))
    else:
        dec = clf.decision_function(embeds)
        if dec.ndim == 1: dec = np.stack([-dec, dec], axis=1)
        max_margin = dec.max(axis=1); pseudo = 1.0 / (1.0 + np.exp(-max_margin)); idx = dec.argmax(axis=1)
        for i in range(len(embeds)):
            names.append(id2name[int(idx[i])] if pseudo[i] >= unknown_th else "unknown")
            confs.append(float(pseudo[i]))
    return names, confs

def load_runtime(model_path, labels_path, report_path=None, fallback_th=0.40):
    if not Path(model_path).exists(): print(f"ERROR: model not found: {model_path}"); sys.exit(2)
    if not Path(labels_path).exists(): print(f"ERROR: labels not found: {labels_path}"); sys.exit(3)
    clf = joblib.load(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f).items()}
    th = fallback_th
    if report_path and Path(report_path).exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                rep = json.load(f); th = float(rep.get("threshold_recommended", th))
        except Exception: pass
    return clf, id2name, th

# ---------- alignment for saving clean (3-point affine like 02) ----------
TGT = np.float32([[38, 52], [74, 52], [56, 75]]); SIZE = (112, 112)
def align_by_mesh(crop_bgr):
    import mediapipe as mp
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        res = mesh.process(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks: return None
    h, w = crop_bgr.shape[:2]; lm = res.multi_face_landmarks[0].landmark
    pts = np.float32([[lm[33].x*w, lm[33].y*h],[lm[263].x*w, lm[263].y*h],[lm[1].x*w, lm[1].y*h]])
    M, _ = cv2.estimateAffinePartial2D(pts, TGT, method=cv2.LMEDS)
    if M is None: return None
    return cv2.warpAffine(crop_bgr, M, SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ---------- clean saver ----------
def maybe_save_clean(person, conf, frame_bgr, face_obj, emb_vec, args):
    if person == "unknown": return None
    x1, y1, x2, y2 = map(int, face_obj.bbox)
    w = max(0, x2 - x1); h = max(0, y2 - y1)
    if min(w, h) < args.min_box: return None
    if conf < args.min_conf: return None

    crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop.size == 0: return None
    blur = calc_blur(crop); illum = calc_illum(crop)
    if blur < args.blur_min or not (args.illum_min <= illum <= args.illum_max): return None

    # aligned (best effort; if fails, we still keep crop)
    aligned = align_by_mesh(crop)

    ts = int(time.time()*1000)
    base = f"{ts}_{x1}_{y1}"
    person_dir = CLEAN_FACE_DIR / person
    crop_dir = person_dir / "crops"; ali_dir = person_dir / "aligned"
    crop_dir.mkdir(parents=True, exist_ok=True); ali_dir.mkdir(parents=True, exist_ok=True)
    CLEAN_EMB_DIR.mkdir(parents=True, exist_ok=True)

    crop_path = crop_dir / f"{base}.jpg"
    cv2.imwrite(str(crop_path), crop)
    aligned_path = None
    if aligned is not None:
        aligned_path = ali_dir / f"{base}_a.jpg"
        cv2.imwrite(str(aligned_path), aligned)

    emb_path = CLEAN_EMB_DIR / f"emb_{base}.npy"
    np.save(str(emb_path), emb_vec.astype(np.float32))

    # append to rolling CSV
    row = {
        "person": person, "conf": float(conf),
        "crop_path": str(crop_path),
        "aligned_path": (str(aligned_path) if aligned_path else ""),
        "emb_path": str(emb_path),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "blur": float(blur), "illum": float(illum),
        "det_size": list(args.det_size)
    }
    log_csv = CLEAN_EMB_DIR / "clean_log.csv"
    pd.DataFrame([row]).to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False, encoding="utf-8")
    return row

def gate_by_similarity(embeds, protos):
    """
    embeds: [N,512] L2-normalized; protos: [C,512] L2-normalized
    returns max_sim [N], best_cls [N]
    """
    sims = embeds @ protos.T
    max_sim = sims.max(axis=1)
    best_cls = sims.argmax(axis=1)
    return max_sim, best_cls


# ---------- folder mode ----------
def run_folder(app, clf, id2name, unknown_th, images_dir, out_dir, csv_log, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    images = [p for p in Path(images_dir).iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
    if not images: print(f"[WARN] No images in {images_dir}"); return
    for p in images:
        bgr = cv2.imread(str(p)); 
        if bgr is None: print(f"[WARN] unreadable {p}"); continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); faces = embed_faces(app, rgb)
        if not faces:
            rows.append({"image": str(p), "name": "", "conf": "", "bbox": "", "note": "no_face"})
        else:
            embeds = np.stack([e for (_, e) in faces], axis=0)

            # 1) cosine gate
            max_sim, best_cls = gate_by_similarity(embeds, PROTOS)
            pass_mask = (max_sim >= args.sim_th)

            names = ["unknown"] * len(embeds)
            confs = [0.0] * len(embeds)

            if args.gate_only:
                for i in range(len(embeds)):
                    if pass_mask[i]:
                        names[i] = id2name[int(best_cls[i])]
                        confs[i] = float(max_sim[i])
            else:
                if pass_mask.any():
                    idx = np.where(pass_mask)[0]
                    sv_names, sv_confs = predict_names(clf, embeds[idx], id2name, unknown_th)
                    for j, k in enumerate(idx):
                        names[k] = sv_names[j]
                        confs[k] = sv_confs[j]

            for i, (f, emb) in enumerate(faces):
                name = names[i]
                conf  = confs[i]
                sim = float(max_sim[i])  # cosine gate similarity
                if (sim >= args.sim_th) and (name != "unknown"):
                    print(f"[FACE] name={name} sim={sim:.3f}", flush=True)
                x1, y1, x2, y2 = map(int, f.bbox)
                cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), 2)
                draw_label(bgr, f"{name} ({conf:.2f})", x1, y1)
                rows.append({"image": str(p), "name": name, "conf": conf, "bbox": [x1,y1,x2,y2], "note": ""})
                if args.save_clean and name != "unknown":
                    maybe_save_clean(name, conf, bgr, f, emb, args)


        outp = out_dir / f"pred_{p.name}"; cv2.imwrite(str(outp), bgr)
    if rows: pd.DataFrame(rows).to_csv(csv_log, index=False, encoding="utf-8"); print(f"[OK] Predictions CSV: {csv_log}")
    print(f"[OK] Annotated images: {out_dir}")

# ---------- webcam mode ----------
def run_webcam(app, clf, id2name, unknown_th, args):
    cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)  # Windows-friendly
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {args.source}")
        sys.exit(4)

    # Full HD request
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    window = "Face Recognition (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

            # 1) cosine gate
            max_sim, best_cls = gate_by_similarity(embeds, PROTOS)
            pass_mask = (max_sim >= args.sim_th)

            names = ["unknown"] * len(embeds)
            confs = [0.0] * len(embeds)

            if args.gate_only:
                for i in range(len(embeds)):
                    if pass_mask[i]:
                        names[i] = id2name[int(best_cls[i])]
                        confs[i] = float(max_sim[i])
            else:
                if pass_mask.any():
                    idx = np.where(pass_mask)[0]
                    sv_names, sv_confs = predict_names(clf, embeds[idx], id2name, unknown_th)
                    for j, k in enumerate(idx):
                        names[k] = sv_names[j]
                        confs[k] = sv_confs[j]

            for i, (f, emb) in enumerate(faces):
                name = names[i]
                conf = confs[i]
                sim = float(max_sim[i])  # cosine gate similarity
                if (sim >= args.sim_th) and (name != "unknown"):
                    print(f"[FACE] name={name} sim={sim:.3f}", flush=True)
                x1, y1, x2, y2 = map(int, f.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_label(frame, f"{name} ({conf:.2f})", x1, y1)
                if args.save_clean and (frames % args.every_n == 0) and name != "unknown":
                    maybe_save_clean(name, conf, frame, f, emb, args)

        frames += 1
        if frames % 10 == 0:
            fps = frames / (time.time() - t0 + 1e-9)
            draw_label(frame, f"FPS: {fps:.1f}", 10, 30)

        # --- overlay banner if a recent match occurred (written by main_detect.py) ---
        try:
            if OVERLAY_FILE.exists():
                age = time.time() - OVERLAY_FILE.stat().st_mtime
                if age <= OVERLAY_TTL:
                    msg = OVERLAY_FILE.read_text(encoding="utf-8").strip()
                    H, W = frame.shape[:2]
                    # green bar at the top
                    cv2.rectangle(frame, (0, 0), (W, 40), (0, 200, 0), -1)
                    cv2.putText(frame, msg, (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        except Exception:
            pass
        display_frame = cv2.resize(frame, (1920, 720))
        cv2.imshow(window, display_frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- entry ----------
def main():
    ensure_dirs()
    ap = argparse.ArgumentParser(description="Real-time/folder face recognition (+ clean capture)")
    ap.add_argument("--model",  type=str, default=str(MODELS_FACE_DIR / "svm_5ppl.pkl"))
    ap.add_argument("--labels", type=str, default=str(MODELS_FACE_DIR / "labels_5ppl.json"))
    ap.add_argument("--report", type=str, default=str(RUNS / "face" / "eval" / "report.json"))
    ap.add_argument("--unknown-threshold", type=float, default=None)
    ap.add_argument("--det-size", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--source", type=int, default=None, help="Webcam index")
    ap.add_argument("--images", type=str, default=None, help="Folder of images to annotate")
    ap.add_argument("--csv-log", type=str, default=str(RUNS / "face" / "preds.csv"))
    ap.add_argument("--save-embeddings", action="store_true", help="Save face embeddings to disk")
    
    # Load cosine prototypes
    proto_path = EMB_FACE_DIR / "prototypes.npy"
    if not proto_path.exists():
        print(f"ERROR: {proto_path} not found. Re-run 03_embed_arcface to create prototypes.")
        sys.exit(6)
    global PROTOS
    PROTOS = np.load(str(proto_path)).astype(np.float32)  # shape [C,512]

    # clean capture options
    ap.add_argument("--save-clean", action="store_true", help="Save confident recognitions to clean dataset")
    ap.add_argument("--min-conf", type=float, default=None, help="Min confidence for saving (default = unknown-threshold)")
    ap.add_argument("--min-box", type=int, default=120, help="Min face box size (pixels)")
    ap.add_argument("--every-n", type=int, default=10, help="Save every Nth frame per camera loop")
    ap.add_argument("--blur-min", type=float, default=20.0)
    ap.add_argument("--illum-min", type=float, default=30.0)
    ap.add_argument("--illum-max", type=float, default=220.0)
    ap.add_argument("--sim-th", type=float, default=0.45,
                help="Cosine similarity threshold for open-set gate")
    ap.add_argument("--gate-only", action="store_true",
                help="Use only cosine gate (nearest prototype) and skip SVM")

    args = ap.parse_args()

    if args.source is None and args.images is None:
        print("ERROR: Provide either --source (webcam index) or --images (folder)."); sys.exit(5)

    clf, id2name, th_rep = load_runtime(args.model, args.labels, args.report, fallback_th=0.40)
    unknown_th = float(args.unknown_threshold) if args.unknown_threshold is not None else th_rep
    if args.min_conf is None: args.min_conf = unknown_th

    app = init_face_app(det_size=tuple(args.det_size))

    if args.images:
        run_folder(app, clf, id2name, unknown_th, args.images, PRED_FACE_DIR, Path(args.csv_log), args)
    else:
        run_webcam(app, clf, id2name, unknown_th, args)

if __name__ == "__main__":
    main()
