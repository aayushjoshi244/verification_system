# src/common/main_detect.py — continuous parallel voice+face, Excel/CSV log, no timeouts

import sys, subprocess, re, time, threading, queue, signal, argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import os
from typing import Optional, Tuple, Deque, Dict
from collections import deque

# ---------- parsing ----------
VOICE_RE = re.compile(r"\[VOICE\].*==>\s*([A-Za-z0-9_\- ]+)", re.I)

# Capture only the name token (no trailing " sim=0.69...")
FACE_RE_STRICT   = re.compile(r"\[FACE\].*name\s*=\s*([^\s,]+)", re.I)
FACE_RE_FALLBACK = re.compile(r"(?:pred\s*[:=]\s*|id\s*[:=]\s*)([^\s,]+)", re.I)


print(f"[DEBUG] parent Python: {sys.executable}")
# this file is .../src/common/main_detect.py → project root = parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Registry/logs
REG_DIR      = PROJECT_ROOT / "runs/registry"
ATTEND_XLSX  = REG_DIR / "attendance.xlsx"
ATTEND_CSV   = REG_DIR / "attendance.csv"   # fallback if xlsx locked
OVERLAY_FILE = PROJECT_ROOT / "runs/face/match_overlay.txt"  # optional: for on-cam success banner

# ---------- helpers ----------
def _now_row(name: str, ok: bool, reason: str = "") -> dict:
    now = datetime.now()
    return {
        "name": name if ok else "",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "status": "OK" if ok else "FAILED",
        "reason": reason,
    }

def append_attendance(name: str, ok: bool, reason: str = ""):
    """Append one row into Excel; fall back to CSV if Excel is locked."""
    row = _now_row(name, ok, reason)
    ATTEND_XLSX.parent.mkdir(parents=True, exist_ok=True)
    try:
        if ATTEND_XLSX.exists():
            df = pd.read_excel(ATTEND_XLSX)
            for c in ["name","date","time","status","reason"]:
                if c not in df.columns: df[c] = None
            df = pd.concat([df[["name","date","time","status","reason"]],
                            pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row], columns=["name","date","time","status","reason"])

        tmp = ATTEND_XLSX.with_suffix(".tmp.xlsx")
        with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
            df.to_excel(xw, index=False)
        ATTEND_XLSX.unlink(missing_ok=True)
        tmp.replace(ATTEND_XLSX)
        return
    except PermissionError:
        print("[WARN] attendance.xlsx is open/locked; appending to attendance.csv instead.")
    except Exception as e:
        print(f"[WARN] Excel write failed ({e}). Falling back to CSV.")

    try:
        hdr = not ATTEND_CSV.exists()
        pd.DataFrame([row]).to_csv(ATTEND_CSV, index=False, mode="a", header=hdr)
    except Exception as e:
        print(f"[ERROR] CSV fallback also failed: {e}")

def extract_voice_name(line: str) -> Optional[str]:
    m = VOICE_RE.search(line)
    if not m: return None
    lbl = m.group(1).strip()
    return None if lbl.upper() == "UNKNOWN" else lbl

def extract_face_name(line: str) -> Optional[str]:
    m = FACE_RE_STRICT.search(line)
    if m: return m.group(1).strip()
    m = FACE_RE_FALLBACK.search(line)
    if m: return m.group(1).strip()
    return None

def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def reader_thread(proc: subprocess.Popen, outq: "queue.Queue[Tuple[str, Optional[str]]]", tag: str):
    try:
        for line in proc.stdout:
            outq.put((tag, line.rstrip("\r\n")))
    finally:
        outq.put((tag, None))

def start_proc(cmd, *, cwd: Optional[Path] = PROJECT_ROOT):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,          # line-buffered
        env=env,
        cwd=str(cwd) if cwd else None,
    )

def stop_proc(proc: Optional[subprocess.Popen]):
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform.startswith("win"):
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                proc.wait(timeout=1.0)
            except Exception:
                # fallback if CTRL_BREAK doesn’t stop it
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    proc.kill()
        else:
            proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except Exception:
                proc.kill()
    except Exception:
        pass


def build_face_cmd(cam_index:int, face_sim:float) -> list[str]:
    return [sys.executable, "-u", "-m", "src.facerec.05_infer_realtime",
            "--source", str(cam_index), "--sim-th", f"{face_sim}"]

def build_voice_cmd(mic_index:int, voice_sim:float, vad:int, min_dur:float) -> list[str]:
    return [sys.executable, "-u", "-m", "src.voicerec.03_verify_realtime",
            "--device-index", str(mic_index), "--sim-th", f"{voice_sim}",
            "--vad", str(vad), "--min-dur", f"{min_dur}"]

def should_echo(tag: str, line: str, echo_mode: str) -> bool:
    if echo_mode == "none":
        return False
    if "FutureWarning" in line:
        return False
    # Hide the specific InsightFace transform spam too (the code line)
    if "insightface\\utils\\transform.py" in line or "insightface/utils/transform.py" in line:
        return False
    if echo_mode == "all":
        return True
    # "matched": only show lines that actually produced a name
    if tag == "VOICE" and extract_voice_name(line):
        return True
    if tag == "FACE" and extract_face_name(line):
        return True
    return False


def notify_success(name: str, beep: bool, write_overlay: bool):
    print(f"\n[SUCCESS] Matched: {name}\n")
    if beep and sys.platform.startswith("win"):
        try:
            import winsound
            winsound.Beep(1000, 200); winsound.Beep(1400, 180)
        except Exception:
            pass
    if write_overlay:
        try:
            OVERLAY_FILE.parent.mkdir(parents=True, exist_ok=True)
            OVERLAY_FILE.write_text(f"OK: {name} @ {datetime.now().strftime('%H:%M:%S')}", encoding="utf-8")
        except Exception:
            pass

# ---------------- continuous parallel loop ----------------

def main():
    ap = argparse.ArgumentParser(description="Continuous detection: voice + face (no timeouts)")
    ap.add_argument("--voice-mic", type=int, default=1, help="Microphone device index for voice")
    ap.add_argument("--voice-sim", type=float, default=0.60, help="Voice cosine threshold (03)")
    ap.add_argument("--voice-vad", type=int, default=2, choices=[0,1,2,3], help="VAD aggressiveness")
    ap.add_argument("--voice-min-dur", type=float, default=0.8, help="Minimum voiced duration (s)")
    ap.add_argument("--face-cam", type=int, default=0, help="Camera index for face")
    ap.add_argument("--face-sim", type=float, default=0.45, help="Face cosine gate (05)")
    ap.add_argument("--echo", choices=["all","matched","none"], default="matched",
                help="What to echo from child processes")
    ap.add_argument("--suppress-warnings", action="store_true",
                help="Hide Python warnings (e.g., FutureWarning spam)")


    # new: streaming matching controls
    ap.add_argument("--match-window", type=float, default=3.0, help="Seconds in which voice & face names must coincide")
    ap.add_argument("--cooldown", type=float, default=4.0, help="Seconds to suppress repeated logs for same person")
    ap.add_argument("--beep-on-match", action="store_true", help="Play a short beep when matched")
    ap.add_argument("--overlay-on-match", action="store_true",
                    help="Write runs/face/match_overlay.txt with a message (hook for camera overlay)")

    args = ap.parse_args()
    if args.suppress_warnings:
        os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"



    print("\n=== Continuous Parallel Detection (Voice + Face) ===")
    print("Start voice & face once → parse outputs forever → match within time window → log → continue.\n")
    print("[TIP] Press Ctrl+C to stop.\n")

    vcmd = build_voice_cmd(args.voice_mic, args.voice_sim, args.voice_vad, args.voice_min_dur)
    fcmd = build_face_cmd(args.face_cam, args.face_sim)

    print(f"[DEBUG] Running VOICE: {' '.join(vcmd)}  (cwd={PROJECT_ROOT})")
    print(f"[DEBUG] Running FACE : {' '.join(fcmd)}  (cwd={PROJECT_ROOT})")

    vproc = start_proc(vcmd)
    fproc = start_proc(fcmd)

    q: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
    vt = threading.Thread(target=reader_thread, args=(vproc, q, "VOICE"), daemon=True)
    ft = threading.Thread(target=reader_thread, args=(fproc, q, "FACE"),  daemon=True)
    vt.start(); ft.start()

    # rolling last seen names
    last_voice_name: Optional[str] = None
    last_voice_t: float = 0.0
    last_face_name: Optional[str] = None
    last_face_t: float = 0.0

    # suppress duplicate logs per name
    last_logged: Dict[str, float] = {}

    try:
        while True:
            try:
                tag, line = q.get(timeout=0.2)
            except queue.Empty:
                # check if any child died
                if vproc.poll() is not None:
                    print(f"[ERROR] Voice process exited rc={vproc.returncode}. Restarting…")
                    vproc = start_proc(vcmd)
                    vt = threading.Thread(target=reader_thread, args=(vproc, q, "VOICE"), daemon=True)
                    vt.start()
                    # reset voice side to avoid stale match
                    last_voice_name = None
                    last_voice_t = 0.0

                if fproc.poll() is not None:
                    print(f"[ERROR] Face process exited rc={fproc.returncode}. Restarting…")
                    fproc = start_proc(fcmd)
                    ft = threading.Thread(target=reader_thread, args=(fproc, q, "FACE"), daemon=True)
                    ft.start()
                    # reset face side to avoid stale match
                    last_face_name = None
                    last_face_t = 0.0
                continue

            # show child outputs according to --echo policy
            if should_echo(tag, line, args.echo):
                print(f"[{tag}] {line}")

            now = time.time()
            updated = False

            if tag == "VOICE":
                n = extract_voice_name(line)
                if n:
                    last_voice_name = n
                    last_voice_t = now
                    updated = True
            else:  # FACE
                n = extract_face_name(line)
                if n:
                    last_face_name = n
                    last_face_t = now
                    updated = True

            if not updated:
                continue

            # attempt match when either side updates
            if last_voice_name and last_face_name:
                # temporal proximity
                if abs(last_voice_t - last_face_t) <= args.match_window:
                    vnorm = norm_name(last_voice_name)
                    fnorm = norm_name(last_face_name)
                    if vnorm == fnorm and vnorm:   # names match (case/space-insensitive)
                        # cooldown per person
                        prev = last_logged.get(vnorm, 0.0)
                        if now - prev >= args.cooldown:
                            notify_success(last_face_name, beep=args.beep_on_match, write_overlay=args.overlay_on_match)
                            append_attendance(last_face_name, ok=True, reason="voice+face agree")
                            last_logged[vnorm] = now
                        else:
                            # still in cooldown; skip duplicate log
                            pass
                    else:
                        # both names present but don't match → optional: log failed once per window
                        # To avoid spamming, only log if not recently logged mismatch for this pair
                        pair_key = f"{vnorm}|{fnorm}"
                        prev = last_logged.get(pair_key, 0.0)
                        if now - prev >= args.cooldown:
                            print(f"[FAIL] Voice='{last_voice_name}' vs Face='{last_face_name}' (within {args.match_window:.1f}s)")
                            append_attendance("", ok=False, reason=f"mismatch: voice={last_voice_name}; face={last_face_name}")
                            last_logged[pair_key] = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopping…")
    finally:
        stop_proc(vproc)
        stop_proc(fproc)
        print("[INFO] Stopped.")

if __name__ == "__main__":
    print("[DEBUG] entering main()", flush=True)
    try:
        main()
    except Exception as e:
        import traceback
        print("[FATAL] Uncaught exception in main_detect.py")
        traceback.print_exc()
        raise
