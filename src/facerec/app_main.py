#!/usr/bin/env python3
"""
app_main.py — one-file runner for the whole face-rec pipeline.

Modes:
1) Register: guided capture (front/left/right) with FaceMesh overlay → saves to data/raw/face/<Name>/
   then runs 01→02→03→04 automatically, and finally launches 05 (realtime) with open-set gate.

2) Just Detect: directly runs 05_infer_realtime with your preferred thresholds.

Dependencies: opencv-python, mediapipe, insightface, onnxruntime, numpy, pandas, tqdm, scikit-learn
"""

import sys, time, subprocess, json
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
                    cam_index: int = 0):
    """
    Capture front/left/right shots with FaceMesh overlay, store to data/raw/face/<name>/.
    We don't do strict head-pose math here; we guide user + enforce quality gates.
    """
    person_dir = FACE_RAW / name
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = ensure_cam(cam_index, 1280, 720)
    mh = MeshHelper()

    window = big_window(f"Face Registration — {name} (q=quit, space=manual capture)")

    phases = [
        ("FRONT", "Look straight at the camera"),
        ("LEFT",  "Turn your face slightly to the LEFT"),
        ("RIGHT", "Turn your face slightly to the RIGHT"),
    ]

    saved_total = 0

    for phase_idx, (phase_key, phase_msg) in enumerate(phases, 1):
        saved_phase = 0
        hint = f"[{phase_idx}/{len(phases)}] {phase_msg}. Capturing {per_pose} good frames…"
        print(hint)

        cooldown = 0  # cool-down frames between captures
        while saved_phase < per_pose:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # overlay text
            overlay = frame.copy()
            cv2.putText(overlay, hint, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"Saved in phase: {saved_phase}/{per_pose}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

            # FaceMesh (for cool visuals)
            res = mh.process(frame)
            overlay = mh.draw(overlay, res)

            # basic quality check on the estimated face bbox from landmarks
            ok_to_capture = False
            if res and res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                xs = [int(l.x * frame.shape[1]) for l in lm]
                ys = [int(l.y * frame.shape[0]) for l in lm]
                x1, x2 = max(min(xs), 0), min(max(xs), frame.shape[1]-1)
                y1, y2 = max(min(ys), 0), min(max(ys), frame.shape[0]-1)
                w, h = x2 - x1, y2 - y1

                # consider only reasonable face boxes
                if w > 160 and h > 160:
                    crop = frame[y1:y2, x1:x2]
                    blur = calc_blur(crop)
                    illum = calc_illum(crop)
                    if blur >= blur_min and illum_min <= illum <= illum_max:
                        ok_to_capture = True

                    # show quick metrics
                    cv2.putText(overlay, f"blur:{blur:.0f} illum:{illum:.0f}", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # show frame
            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF

            # Manual capture (space)
            manual = (key == 32)  # spacebar
            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows()
                print("[INFO] Registration aborted by user.")
                sys.exit(0)

            if cooldown > 0:
                cooldown -= 1
                continue

            # auto capture when ok_to_capture or manual capture pressed
            if ok_to_capture or manual:
                ts = int(time.time() * 1000)
                outp = person_dir / f"{phase_key.lower()}_{ts}.jpg"
                cv2.imwrite(str(outp), frame)
                saved_phase += 1
                saved_total += 1
                cooldown = 10  # a short cool-down to avoid bursts

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] Saved {saved_total} images for '{name}' to: {person_dir}")
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
