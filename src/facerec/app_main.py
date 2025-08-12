#!/usr/bin/env python3
"""
app_main.py — one-file runner for the whole face-rec pipeline.

Modes:
1) Register: guided capture (front/left/right) with FaceMesh overlay → saves to data/raw/face/<Name>/
   then runs 01→02→03→04 automatically, and finally launches 05 (realtime) with open-set gate.

2) Just Detect: directly runs 05_infer_realtime with your preferred thresholds.

Dependencies: opencv-python, mediapipe, insightface, onnxruntime, numpy, pandas, tqdm, scikit-learn
"""

import sys, time, subprocess, json, math
import numpy as np
from pathlib import Path
import cv2

# -------- import your repo paths --------
try:
    from src.common.paths import ensure_dirs, FACE_RAW
except Exception:
    print("[ERROR] cannot import src.common.paths. Make sure your project structure is intact")
    sys.exit(1)

# --------- FaceMesh wrapper (for overlay) ----------
class MeshHelper:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.style = mp.solutions.drawing_styles

    def draw(self, frame_bgr, results):
        if not results or not results.multi_face_landmarks:
            return frame_bgr
        for face_landmarks in results.multi_face_landmarks:
            self.drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self.mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.style.get_default_face_mesh_tesselation_style())
            self.drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self.mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.style.get_default_face_mesh_contours_style()
            )
        return frame_bgr

    def process(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.mesh.process(rgb)

    def __del__(self):
        try:
            # mediapipe >=0.10 has close(); guard for older versions
            self.mesh.close()
        except Exception:
            pass

# ---------- quality gates ----------
def calc_blur(image_bgr):
    return cv2.Laplacian(image_bgr, cv2.CV_64F).var()

def calc_illum(image_bgr):
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())

# ---------- head pose (yaw) using FaceMesh + solvePnP ----------
# MediaPipe landmark indices
LM_NOSE_TIP = 1
LM_CHIN     = 152
LM_EYE_L    = 33    # left eye outer corner
LM_EYE_R    = 263   # right eye outer corner
LM_MOUTH_L  = 61
LM_MOUTH_R  = 291

# Generic 3D face model (mm)
MODEL_POINTS_3D = np.array([
    [0.0,   0.0,   0.0],     # Nose tip
    [0.0, -63.6, -12.5],     # Chin
    [-43.3, 32.7, -26.0],    # Left eye outer corner
    [ 43.3, 32.7, -26.0],    # Right eye outer corner
    [-28.9,-28.9, -24.1],    # Left mouth corner
    [ 28.9,-28.9, -24.1],    # Right mouth corner
], dtype=np.float64)
def estimate_yaw_deg(face_landmarks, w, h):
    """Return yaw in degrees. Right turn = positive; Left = negative."""
    pts2d = []
    for idx in [LM_NOSE_TIP, LM_CHIN, LM_EYE_L, LM_EYE_R, LM_MOUTH_L, LM_MOUTH_R]:
        lm = face_landmarks.landmark[idx]
        pts2d.append([lm.x * w, lm.y * h])
    image_points = np.array(pts2d, dtype=np.float64)

    # Approx camera intrinsics
    f = w  # focal length ~ width (pixels)
    cx, cy = w / 2.0, h / 2.0
    cam_mtx = np.array([[f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    # ZYX euler (yaw around Y)
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    yaw = math.degrees(math.atan2(R[2,0], sy))  # right positive
    return float(yaw)

def yaw_range_for_phase(phase_key):
    """Return (min_yaw, max_yaw) in degrees for FRONT/LEFT/RIGHT."""
    if phase_key == "FRONT":
        return (-10.0, 10.0)
    if phase_key == "LEFT":
        return (-40.0, -15.0)
    if phase_key == "RIGHT":
        return (15.0, 40.0)
    return (-999.0, 999.0)  # fallback

def draw_text(img, text, org, scale=0.8, color=(0,255,0), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_progress(img, x, y, w, h, frac, ok):
    frac = max(0.0, min(1.0, float(frac)))
    cv2.rectangle(img, (x, y), (x+w, y+h), (80, 80, 80), 2)
    fill_w = int(w * frac)
    color = (0, 200, 0) if ok else (0, 165, 255)
    cv2.rectangle(img, (x+2, y+2), (x+2+fill_w, y+h-2), color, -1)

# ---------- helpers ----------
def run_step(module: str, extra_args=None) -> int:
    """Run a module like python -m src.facerec.01_import_detect with optional args list."""
    cmd = [sys.executable, "-m", module]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n[RUN] {' '.join(cmd)}")
    return subprocess.call(cmd)

def big_window(name="Capture (q=quit, space=manual capture)"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # Try to make it fullscreen; fall back if not supported
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass
    return name

def ensure_cam(cap_idx=0, width=1280, height=720):
    cap = cv2.VideoCapture(cap_idx, cv2.CAP_DSHOW)  # Windows-friendly backend
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {cap_idx}.")
        sys.exit(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# ---------- guided capture ----------
def guided_register(name: str,
                    per_pose: int = 10,
                    blur_min: float = 20.0,
                    illum_min: float = 30.0,
                    illum_max: float = 220.0,
                    cam_index: int = 0,
                    stable_needed: int = 6,
                    progress_w: int = 320):
    """
    Automatically capture FRONT, LEFT, and RIGHT shots with FaceMesh overlay.
    Person can start in any angle — system auto-detects yaw and categorizes.
    Limited extremes so we ignore overly turned faces (|yaw| > 40°).
    """
    person_dir = FACE_RAW / name
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = ensure_cam(cam_index, 1280, 720)
    mh = MeshHelper()
    window = big_window(f"Face Registration — {name} (q=quit, space=manual capture)")

    # Consistent classification AND display bounds (no gaps)
    PHASE_BOUNDS = {
        "FRONT": (-12.0,  12.0),
        "LEFT":  (-40.0,  -12.0),
        "RIGHT": ( 12.0,   40.0),
    }

    def classify_phase(yaw: float):
        if yaw is None:
            return None
        if PHASE_BOUNDS["FRONT"][0] <= yaw <= PHASE_BOUNDS["FRONT"][1]:
            return "FRONT"
        if PHASE_BOUNDS["LEFT"][0] <= yaw <  PHASE_BOUNDS["LEFT"][1]:
            return "LEFT"
        if PHASE_BOUNDS["RIGHT"][0] <  yaw <= PHASE_BOUNDS["RIGHT"][1]:
            return "RIGHT"
        return None  # too extreme (|yaw| > 40°) → ignore

    counts = {"FRONT": 0, "LEFT": 0, "RIGHT": 0}
    total_needed = per_pose * len(counts)

    cooldown = 0
    stable = 0
    saved_total = 0

    print(f"[INFO] Need {per_pose} good captures per angle (FRONT/LEFT/RIGHT).")

    while sum(counts.values()) < total_needed:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        overlay = frame.copy()
        cv2.putText(overlay,
                    f"Captures: F={counts['FRONT']} L={counts['LEFT']} R={counts['RIGHT']}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        res = mh.process(frame)
        overlay = mh.draw(overlay, res)

        ok_quality = False
        ok_angle = False
        phase_key = None
        yaw = None

        if res and res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            xs = [int(l.x * w) for l in fl.landmark]
            ys = [int(l.y * h) for l in fl.landmark]
            x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
            y1, y2 = max(min(ys), 0), min(max(ys), h - 1)
            ww, hh = x2 - x1, y2 - y1

            if ww > 160 and hh > 160:
                crop = frame[y1:y2, x1:x2]
                blur = calc_blur(crop)
                illum = calc_illum(crop)
                ok_quality = (blur >= blur_min) and (illum_min <= illum <= illum_max)
                draw_text(overlay, f"blur:{blur:.0f} illum:{illum:.0f}",
                          (20, 80), 0.7, (255,255,255), 2)

            yaw = estimate_yaw_deg(fl, w, h)
            if yaw is not None:
                draw_text(overlay, f"yaw:{yaw:5.1f}°", (20, 110), 0.7, (255,255,255), 2)
                phase_key = classify_phase(yaw)

                if phase_key:
                    ymin, ymax = PHASE_BOUNDS[phase_key]
                    ok_angle = (ymin <= yaw <= ymax)
                    draw_text(overlay,
                              f"{phase_key} target:{ymin:.0f}..{ymax:.0f}",
                              (20, 140), 0.7,
                              (0,255,0) if ok_angle else (0,165,255), 2)
                else:
                    draw_text(overlay, "Angle too extreme — turn slightly toward camera",
                              (20, 140), 0.7, (0,165,255), 2)
        else:
            draw_text(overlay, "No face detected", (20, 80), 0.8, (0,165,255), 2)

        # Stability logic
        if phase_key and ok_angle:
            stable = min(stable + 1, stable_needed)
        else:
            stable = 0

        # Progress bar for angle stability
        draw_progress(overlay, 20, 170, progress_w, 16, stable / stable_needed, ok_angle)

        # Show frame
        cv2.imshow(window, overlay)
        key = cv2.waitKey(1) & 0xFF

        manual = (key == 32)  # spacebar
        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows()
            print("[INFO] Registration aborted by user.")
            sys.exit(0)

        if cooldown > 0:
            cooldown -= 1
            continue

        # Save only when classified, under quota, and quality + stability OK (or manual)
        if phase_key and counts[phase_key] < per_pose and ((stable >= stable_needed and ok_quality) or manual):
            ts = int(time.time() * 1000)
            outp = person_dir / f"{phase_key.lower()}_{ts}.jpg"
            cv2.imwrite(str(outp), frame)
            counts[phase_key] += 1
            saved_total += 1
            cooldown = 10
            stable = 0

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] Saved {saved_total} images for '{name}' in: {person_dir}")
    return person_dir


# ---------- Orchestration ----------
def run_full_pipeline_and_infer(sim_th=0.45, unk_th=None, cam_index=0):
    """
    Runs 01→02→03→04 and then launches 05 with given thresholds.
    """
    # 01
    rc = run_step("src.facerec.01_import_detect")
    if rc != 0:
        print("[ERROR] 01_import_detect failed."); sys.exit(rc)

    # 02
    rc = run_step("src.facerec.02_align")
    if rc != 0:
        print("[ERROR] 02_align failed."); sys.exit(rc)

    # 03
    rc = run_step("src.facerec.03_embed_arcface")
    if rc != 0:
        print("[ERROR] 03_embed_arcface failed."); sys.exit(rc)

    # 04 (train + recommended threshold)
    rc = run_step("src.facerec.04_eval_basic", ["--prob"])
    if rc != 0:
        print("[ERROR] 04_eval_basic failed."); sys.exit(rc)

    # read recommended threshold (optional)
    report = Path("runs/face/eval/report.json")
    rep_th = None
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            rep_th = float(data.get("threshold_recommended", None))
        except Exception:
            pass

    unk = str(unk_th if unk_th is not None else (rep_th if rep_th is not None else 0.60))

    # 05
    args_05 = [
        "--source", str(cam_index),
        "--sim-th", str(sim_th),
        "--unknown-threshold", unk
    ]
    print(f"[INFO] Launching realtime with: sim-th={sim_th} unknown-th={unk}")
    rc = run_step("src.facerec.05_infer_realtime", args_05)
    if rc != 0:
        print("[ERROR] 05_infer_realtime failed."); sys.exit(rc)

def run_realtime_only(sim_th=0.45, unk_th=None, cam_index=0):
    """
    Launch only 05_infer_realtime using the current model & prototypes.
    """
    # try to read recommended threshold if not provided
    rep_th = None
    report = Path("runs/face/eval/report.json")
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            rep_th = float(data.get("threshold_recommended", None))
        except Exception:
            pass

    unk = str(unk_th if unk_th is not None else (rep_th if rep_th is not None else 0.60))
    args_05 = ["--source", str(cam_index), "--sim-th", str(sim_th), "--unknown-threshold", unk]
    print(f"[INFO] Launching realtime (no rebuild) with: sim-th={sim_th} unknown-th={unk}")
    rc = run_step("src.facerec.05_infer_realtime", args_05)
    if rc != 0:
        print("[ERROR] 05_infer_realtime failed."); sys.exit(rc)

def main():
    ensure_dirs()
    print("\n=== FaceRec Main ===")
    print("1) Register a new person (guided capture) and run the full pipeline")
    print("2) Just try detecting (realtime) with the current model")
    choice = input("Choose [1/2]: ").strip()

    cam_index = input("Camera index [default 0]: ").strip()
    cam_index = int(cam_index) if cam_index.isdigit() else 0

    if choice == "1":
        name = input("Enter the person's name (folder-safe, e.g., Aryan_K): ").strip()
        if not name:
            print("[ERROR] Name cannot be empty.")
            sys.exit(1)
        print("\n[INFO] Starting guided capture…")
        guided_register(name=name, per_pose=10, cam_index=cam_index)

        # After registration, run the whole pipeline and launch realtime
        print("\n[INFO] Running full pipeline 01→02→03→04, then realtime 05…")
        # You can tweak default thresholds here
        run_full_pipeline_and_infer(sim_th=0.45, unk_th=None, cam_index=cam_index)

    elif choice == "2":
        # Ask thresholds quickly
        sim = input("Cosine gate threshold (sim-th) [0.45]: ").strip()
        unk = input("Unknown probability threshold [press enter to use recommended or 0.65]: ").strip()
        sim_th = float(sim) if sim else 0.45
        unk_th = float(unk) if unk else None
        print("\n[INFO] Launching realtime only…")
        run_realtime_only(sim_th=sim_th, unk_th=unk_th, cam_index=cam_index)

    else:
        print("[ERROR] Invalid choice.")
        sys.exit(1)

if __name__ == "__main__":
    main()

