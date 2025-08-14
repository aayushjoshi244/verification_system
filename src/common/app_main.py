# src/common/app_main.py
"""
Unified launcher for the project.

- Register  -> runs src.common.main_register (your face+voice enrollment flow)
- Detect    -> runs src.common.main_detect   (your face+voice verification flow)

You can also skip the menu with:
    python -m src.common.app_main --mode register
    python -m src.common.app_main --mode detect

Any arguments after a lone `--` are forwarded to the target script:
    python -m src.common.app_main --mode detect -- --camera 0 --mic 1
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Optional: make sure we’re running from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REGISTER_MODULE = "src.common.main_register"
DETECT_MODULE   = "src.common.main_detect"

def run_module(module: str, forwarded_args=None) -> int:
    """Run a module as `python -m <module> [forwarded_args...]`."""
    cmd = [sys.executable, "-m", module]
    if forwarded_args:
        cmd.extend(forwarded_args)
    print(f"\n[RUN] {' '.join(cmd)}\n")
    return subprocess.call(cmd)

def menu_choice() -> str:
    print("\n=== Verification System ===")
    print("1) Register a new person (face + voice)")
    print("2) Detect / Verify (face + voice)")
    choice = input("Choose [1/2]: ").strip()
    if choice == "1":
        return "register"
    if choice == "2":
        return "detect"
    print("[ERROR] Invalid choice.")
    sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Main launcher for register/detect flows.")
    ap.add_argument("--mode", choices=["register", "detect"], default=None,
                    help="Skip menu and run the chosen mode directly.")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="Use `--` then any args to forward to the target script.")
    args = ap.parse_args()

    # Split off a leading '--' if present, so everything after is forwarded.
    fwd = args.rest
    if fwd and fwd[0] == "--":
        fwd = fwd[1:]

    mode = args.mode or menu_choice()

    if mode == "register":
        rc = run_module(REGISTER_MODULE, fwd)
    else:
        rc = run_module(DETECT_MODULE, fwd)

    if rc != 0:
        print(f"[ERROR] {mode} flow exited with code {rc}.")
        sys.exit(rc)

if __name__ == "__main__":
    main()
