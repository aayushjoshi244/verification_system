#!/usr/bin/env python3
"""
src/voicerec/app_main.py — one-file runner for the whole VOICE pipeline.

Menu:
1) Enroll a new person (record utterances with VAD) → embed → (optionally) calibrate → verify realtime
2) Calibrate threshold only (from existing enrol WAVs)
3) Realtime verify (use prototypes + best threshold found / manual override)

Requires you already have:
- src.voicerec.01_record_enroll
- src.voicerec.02_embed_speechbrain
- src.voicerec.03_verify_realtime
- src.voicerec.04_calibrate_threshold

Data/Outputs:
- Enrol WAVs → data/raw/voice/<Name>/*.wav
- Prototypes  → runs/voice/prototypes/*.npy + runs/voice/index.json
- Calibration → runs/voice/calibration.json  (optional but recommended)
"""

import sys, json, subprocess
from pathlib import Path

RUNS_ROOT  = Path("runs/voice")
RAW_ROOT   = Path("data/raw/voice")
INDEX_JSON = RUNS_ROOT / "index.json"
CALIB_JSON = RUNS_ROOT / "calibration.json"

def _err(msg): print(f"[ERROR] {msg}", file=sys.stderr)

def run_step(module: str, args=None) -> int:
    """Run a module like: python -m src.voicerec.02_embed_speechbrain --foo bar"""
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend([str(a) for a in args])
    print(f"\n[RUN] {' '.join(cmd)}")
    return subprocess.call(cmd)

def pick_device_index() -> int | None:
    """
    Convenience: call the realtime script in list-only mode to print devices,
    then ask user. We don’t run the realtime loop; we only re-use the same printer.
    """
    try:
        # Reuse the device lister in verify module by running it with a quick exit
        # If you don’t want to depend on it, comment this call; devices will still
        # be printed when realtime starts.
        pass
    except Exception:
        pass
    val = input("Input device index (blank = auto): ").strip()
    return int(val) if val.isdigit() else None

def read_recommended_sim_th() -> float | None:
    """
    Prefer calibration F1-opt threshold from CALIB_JSON.
    Fallback to None so caller can decide a default (e.g., 0.60).
    """
    if not CALIB_JSON.exists():
        return None
    try:
        data = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
        thr = data.get("f1_opt", {}).get("threshold", None)
        if thr is None:
            return None
        return float(thr)
    except Exception:
        return None

def ensure_any_enrol_data() -> bool:
    return RAW_ROOT.exists() and any(p.is_dir() and list(p.glob("*.wav")) for p in RAW_ROOT.iterdir())

def ensure_index_exists() -> bool:
    return INDEX_JSON.exists()

def main():
    print("\n=== VoiceRec Main ===")
    print("1) Enroll a new person  → (01 record) → (02 embed) → [optional 04 calibrate] → 03 realtime")
    print("2) Calibrate threshold only (04) using all existing enrol WAVs")
    print("3) Realtime verify (03) using current prototypes/index")
    print("q) Quit")
    choice = input("Choose [1/2/3/q]: ").strip().lower()

    if choice == "q":
        print("Bye!")
        sys.exit(0)

    # common: ask device (can be None → auto)
    dev_idx = pick_device_index()

    if choice == "1":
        # --- 01: record/enroll ---
        name = input("Enter person name (folder-safe, e.g. Aryan_K): ").strip()
        if not name:
            _err("Name cannot be empty."); sys.exit(1)
        try:
            utt = input("How many utterances to record [5]: ").strip()
            utt = int(utt) if utt.isdigit() else 5
            sec = input("Max seconds per utterance window [10]: ").strip()
            sec = int(sec) if sec.isdigit() else 10
        except Exception:
            utt, sec = 5, 10

        args01 = ["--name", name, "--utterances", utt, "--seconds", sec]
        if dev_idx is not None:
            args01.extend(["--device", dev_idx])

        rc = run_step("src.voicerec.01_record_enroll", args01)
        if rc != 0:
            _err("01_record_enroll failed."); sys.exit(rc)

        # --- 02: embed ---
        drop = input("Drop fraction for outlier enrol embeddings [0.2]: ").strip()
        try:
            drop_f = float(drop) if drop else 0.2
        except Exception:
            drop_f = 0.2

        rc = run_step("src.voicerec.02_embed_speechbrain",
                      ["--drop-outliers", drop_f])
        if rc != 0:
            _err("02_embed_speechbrain failed."); sys.exit(rc)

        # Ask about calibration
        do_calib = input("Run threshold calibration now? [Y/n]: ").strip().lower()
        if do_calib in ("", "y", "yes"):
            rc = run_step("src.voicerec.04_calibrate_threshold", ["--save-hist"])
            if rc != 0:
                _err("04_calibrate_threshold failed."); sys.exit(rc)

        # --- 03: realtime ---
        # Pick threshold: prefer F1-opt from calibration; else default
        sim_th = read_recommended_sim_th() or 0.60
        val = input(f"Cosine threshold for realtime [{sim_th:.2f}]: ").strip()
        try:
            sim_th = float(val) if val else sim_th
        except Exception:
            pass

        args03 = ["--sim-th", sim_th]
        if dev_idx is not None:
            args03.extend(["--device-index", dev_idx])
        rc = run_step("src.voicerec.03_verify_realtime", args03)
        if rc != 0:
            _err("03_verify_realtime failed."); sys.exit(rc)

    elif choice == "2":
        # --- 04 only: calibrate ---
        if not ensure_any_enrol_data():
            _err("No enrol WAVs found in data/raw/voice/. Run option 1 first.")
            sys.exit(1)
        rc = run_step("src.voicerec.04_calibrate_threshold", ["--save-hist"])
        if rc != 0:
            _err("04_calibrate_threshold failed."); sys.exit(rc)
        print("[OK] Calibration complete.")

    elif choice == "3":
        # --- 03 only: realtime ---
        if not ensure_index_exists():
            _err("No prototypes/index found. Run option 1 (enroll + embed) first.")
            sys.exit(1)

        sim_th = read_recommended_sim_th() or 0.60
        val = input(f"Cosine threshold for realtime [{sim_th:.2f}]: ").strip()
        try:
            sim_th = float(val) if val else sim_th
        except Exception:
            pass

        args03 = ["--sim-th", sim_th]
        if dev_idx is not None:
            args03.extend(["--device-index", dev_idx])
        rc = run_step("src.voicerec.03_verify_realtime", args03)
        if rc != 0:
            _err("03_verify_realtime failed."); sys.exit(rc)

    else:
        _err("Invalid choice.")
        sys.exit(1)

if __name__ == "__main__":
    main()
