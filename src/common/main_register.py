# src/common/main_register.py
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from src.common.registry_log import append_registry_row

# Where data is saved by your existing code
FACE_RAW = Path("data/raw/face")
VOICE_RAW = Path("data/raw/voice")

def run(cmd: list[str]) -> int:
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)

def count_files(p: Path, pattern: str) -> int:
    return len(list(p.glob(pattern)))

def main():
    print("\n=== Biometric Registration (Face + Voice) ===")
    name = input("Enter person's name (folder-safe, e.g. Aryan_K): ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        sys.exit(1)

    # ---------- Face guided capture (auto angles) ----------
    print("\n[FACE] Starting guided capture…")
    # This uses your improved guided_register() inside src.facerec.app_main module
    # Easiest: call the module with choice=1 path OR call a dedicated helper.
    # We call guided_register directly by exposing a light cli via -m.
    rc = run([sys.executable, "-c",
              "import sys; from src.facerec.app_main import guided_register; guided_register(sys.argv[1], per_pose=10)",
              name])
    if rc != 0:
        print("[ERROR] Face guided capture failed.")
        sys.exit(rc)

    # After face capture, run face pipeline 01→02→03→04
    print("\n[FACE] Running face pipeline 01→02→03→04…")
    for mod in ["src.facerec.01_import_detect",
                "src.facerec.02_align",
                "src.facerec.03_embed_arcface",
                "src.facerec.04_eval_basic"]:
        rc = run([sys.executable, "-m", mod, *(["--prob"] if mod.endswith("04_eval_basic") else [])])
        if rc != 0:
            print(f"[ERROR] {mod} failed.")
            sys.exit(rc)

    # ---------- Voice enrol ----------
    print("\n[VOICE] Recording enrol utterances…")
    # Your 01_record_enroll supports: --name, --utterances, --seconds, --device
    rc = run([sys.executable, "-m", "src.voicerec.01_record_enroll",
              "--name", name, "--utterances", "5", "--seconds", "10"])
    if rc != 0:
        print("[ERROR] Voice enrol failed.")
        sys.exit(rc)

    # Build/update prototypes (02_embed_speechbrain)
    print("\n[VOICE] Computing speaker prototypes…")
    rc = run([sys.executable, "-m", "src.voicerec.02_embed_speechbrain", "--drop-outliers", "0.2"])
    if rc != 0:
        print("[ERROR] Voice prototype embedding failed.")
        sys.exit(rc)

    # ---------- Log registry to Excel ----------
    face_dir = (FACE_RAW / name)
    voice_dir = (VOICE_RAW / name)
    n_imgs  = count_files(face_dir, "*.jpg") + count_files(face_dir, "*.png")
    n_audio = count_files(voice_dir, "*.wav")

    append_registry_row(name=name,
                        face_image_dir=face_dir,
                        voice_audio_dir=voice_dir,
                        num_images=n_imgs,
                        num_audios=n_audio)

    print("\n[OK] Registration complete and logged.")
    print("   Name:              ", name)
    print("   Face images dir:   ", face_dir.resolve())
    print("   Voice audio dir:   ", voice_dir.resolve())
    print("   #images / #audio:  ", n_imgs, "/", n_audio)
    print("   Registry Excel:     runs/registry/registry.xlsx")

if __name__ == "__main__":
    main()
