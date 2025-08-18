# src/common/main_register.py
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Optional

from src.common.registry_log import append_registry_row

# Where data lands
FACE_RAW  = Path("data/raw/face")
FACE_ALI  = Path("data/processed/face/aligned")
VOICE_RAW = Path("data/raw/voice")

def run(cmd: List[str], env: Optional[dict] = None) -> int:
    print("[RUN]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env)

def count_files(p: Path, patterns: List[str]) -> int:
    n = 0
    for pat in patterns:
        n += len(list(p.glob(pat)))
    return n

def ensure_name_safe(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name.strip())
    return safe.strip("_-")

def main():
    ap = argparse.ArgumentParser(
        description="Register a person: guided face capture + face pipeline + voice enrol + prototypes + registry log"
    )
    ap.add_argument("--name", type=str, default=None, help="Person name (folder-safe). If omitted, you'll be prompted.")
    ap.add_argument("--per-pose", type=int, default=10, help="Images per angle in guided face capture.")
    ap.add_argument("--min-aligned", type=int, default=12, help="Minimum aligned face images required for this person.")
    ap.add_argument("--max-retries", type=int, default=2, help="Max extra guided-capture attempts if aligned images are too few.")
    ap.add_argument("--skip-face", action="store_true", help="Skip face capture/pipeline (voice only).")
    ap.add_argument("--skip-voice", action="store_true", help="Skip voice enrol/prototypes (face only).")
    ap.add_argument("--utterances", type=int, default=5, help="Number of enrol utterances.")
    ap.add_argument("--seconds", type=int, default=10, help="Seconds per utterance.")
    ap.add_argument("--mic-index", type=int, default=None, help="Audio device index for enrol. If omitted, tool default is used.")
    ap.add_argument("--drop-outliers", type=float, default=0.2, help="Outlier drop fraction for voice prototype embedding.")
    ap.add_argument("--quiet", action="store_true", help="Suppress noisy Python warnings in child processes.")
    args = ap.parse_args()

    # Name
    name = args.name or input("Enter person's name (folder-safe, e.g. Aryan_K): ").strip()
    name = ensure_name_safe(name)
    if not name:
        print("[ERROR] Name cannot be empty.")
        sys.exit(1)

    # Quiet mode env (propagates to children)
    child_env = os.environ.copy()
    if args.quiet:
        child_env["PYTHONWARNINGS"] = "ignore::FutureWarning"

    print("\n=== Biometric Registration (Face + Voice) ===")

    # ---------- FACE ----------
    face_ok = args.skip_face
    if not args.skip_face:
        attempts = 0
        while True:
            attempts += 1
            print("\n[FACE] Starting guided capture… (attempt", attempts, ")")
            # Call your guided register with configurable per-pose
            rc = run(
                [
                    sys.executable, "-c",
                    "import sys; from src.facerec.app_main import guided_register; "
                    "guided_register(sys.argv[1], per_pose=int(sys.argv[2]))",
                    name, str(args.per_pose),
                ],
                env=child_env,
            )
            if rc != 0:
                print("[ERROR] Face guided capture failed.")
                sys.exit(rc)

            # Run the face pipeline 01→02→03→04
            print("\n[FACE] Running face pipeline 01→02→03→04…")
            pipeline = [
                ["src.facerec.01_import_detect"],
                ["src.facerec.02_align"],
                ["src.facerec.03_embed_arcface"],
                ["src.facerec.04_eval_basic", "--prob"],
            ]
            pipe_ok = True
            for mod in pipeline:
                rc = run([sys.executable, "-m", *mod], env=child_env)
                if rc != 0:
                    pipe_ok = False
                    # Common case: 03_embed_arcface wants ≥2 people. We warn & continue to voice.
                    if "03_embed_arcface" in mod[0]:
                        print("[WARN] Face embedding/training step failed (often happens if only 1 person is enrolled).")
                        print("       You can register at least one more person later and rerun 03_embed_arcface.")
                    else:
                        print(f"[ERROR] {mod[0]} failed.")
                    break

            # Check aligned count for THIS person
            ali_dir = FACE_ALI / name
            n_aligned = count_files(ali_dir, ["*.jpg", "*.png", "*.jpeg"])
            print(f"[FACE] Aligned images for {name}: {n_aligned}")

            if pipe_ok and n_aligned >= args.min_aligned:
                face_ok = True
                break

            if n_aligned < args.min_aligned:
                print(f"[WARN] Too few aligned images for {name} (need ≥ {args.min_aligned}).")
            if attempts >= (1 + args.max_retries):
                print("[ERROR] Face data quality insufficient after retries. "
                      "Try better lighting, look front/left/right, remove glasses, move closer, then re-run.")
                break
            print("[INFO] Repeating guided capture to collect more/better images…")

    # ---------- VOICE ----------
    voice_ok = args.skip_voice
    if not args.skip_voice:
        print("\n[VOICE] Recording enrol utterances…")
        cmd = [
            sys.executable, "-m", "src.voicerec.01_record_enroll",
            "--name", name,
            "--utterances", str(args.utterances),
            "--seconds", str(args.seconds),
        ]
        if args.mic_index is not None:
            cmd.extend(["--device", str(args.mic_index)])
        rc = run(cmd, env=child_env)
        if rc != 0:
            print("[ERROR] Voice enrol failed.")
            sys.exit(rc)

        print("\n[VOICE] Computing speaker prototypes…")
        rc = run(
            [
                sys.executable, "-m", "src.voicerec.02_embed_speechbrain",
                "--drop-outliers", str(args.drop_outliers),
            ],
            env=child_env,
        )
        if rc != 0:
            print("[ERROR] Voice prototype embedding failed.")
            sys.exit(rc)
        voice_ok = True

    # ---------- Registry log ----------
    face_dir  = FACE_RAW / name
    voice_dir = VOICE_RAW / name
    n_imgs  = count_files(face_dir,  ["*.jpg", "*.png", "*.jpeg"])
    n_audio = count_files(voice_dir, ["*.wav", "*.flac"])

    append_registry_row(
        name=name,
        face_image_dir=face_dir,
        voice_audio_dir=voice_dir,
        num_images=n_imgs,
        num_audios=n_audio,
    )

    # ---------- Summary ----------
    print("\n[OK] Registration finished.")
    print("   Name:                ", name)
    if not args.skip_face:
        print("   Face raw dir:        ", face_dir.resolve())
        print("   Face aligned dir:    ", (FACE_ALI / name).resolve())
        print("   #face images (raw):  ", n_imgs)
    if not args.skip_voice:
        print("   Voice raw dir:       ", voice_dir.resolve())
        print("   #voice clips:        ", n_audio)
    print("   Registry Excel:       runs/registry/registry.xlsx")

    if not face_ok:
        print("\n[NOTE] Face pipeline did not fully pass. If the failure was 03_embed_arcface with one person,")
        print("      register one more person (or add more images) and rerun:")
        print("      python -m src.facerec.03_embed_arcface && python -m src.facerec.04_eval_basic --prob")

if __name__ == "__main__":
    main()
