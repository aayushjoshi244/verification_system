# src/common/main_detect.py — sequential voice→face verification + Excel log

import sys, subprocess, re, time, threading, queue, signal, argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import os

# ---------- parsing ----------
VOICE_RE = re.compile(r"\[VOICE\].*==>\s*([A-Za-z0-9_\- ]+)", re.I)
FACE_RE_STRICT   = re.compile(r"\[FACE\].*name\s*=\s*([A-Za-z0-9_\- ]+)", re.I)
FACE_RE_FALLBACK = re.compile(r"(?:pred\s*[:=]\s*|id\s*[:=]\s*)([A-Za-z0-9_\- ]+)", re.I)

print(f"[DEBUG] parent Python: {sys.executable}")
# main_detect.py path: .../src/common/main_detect.py
# project root should be two levels up from this file (../..)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ATTEND_FILE = Path("runs/registry/attendance.xlsx")

def append_attendance(name: str, ok: bool, reason: str = ""):
    ATTEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    row = {"name": name if ok else "", "date": now.strftime("%Y-%m-%d"),
           "time": now.strftime("%H:%M:%S"), "status": "OK" if ok else "FAILED",
           "reason": reason}
    if ATTEND_FILE.exists():
        df = pd.read_excel(ATTEND_FILE)
        for c in ["name","date","time","status","reason"]:
            if c not in df.columns: df[c] = None
        df = pd.concat([df[["name","date","time","status","reason"]],
                        pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=["name","date","time","status","reason"])
    ATTEND_FILE.unlink(missing_ok=True)
    with pd.ExcelWriter(ATTEND_FILE, engine="openpyxl") as xw:
        df.to_excel(xw, index=False)

def extract_voice_name(line: str):
    m = VOICE_RE.search(line)
    if not m: return None
    lbl = m.group(1).strip()
    return None if lbl.upper() == "UNKNOWN" else lbl

def extract_face_name(line: str):
    m = FACE_RE_STRICT.search(line)
    if m: return m.group(1).strip()
    m = FACE_RE_FALLBACK.search(line)
    if m: return m.group(1).strip()
    return None

def reader_thread(proc, outq, tag):
    try:
        for line in proc.stdout:
            outq.put((tag, line.rstrip("\r\n")))
    finally:
        outq.put((tag, None))

def start_proc(cmd, *, cwd: Path | None = PROJECT_ROOT):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,          # decode to str
        bufsize=1,          # line-buffered
        env=env,
        cwd=str(cwd) if cwd else None,
    )


def stop_proc(proc):
    if proc is None or proc.poll() is not None: return
    try:
        if sys.platform.startswith("win"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=1.5)
        except Exception:
            proc.kill()
    except Exception:
        pass

def wait_for_name(cmd, extractor, prompt: str, timeout: float | None) -> str | None:
    print(prompt)
    print(f"[DEBUG] Running: {' '.join(cmd)}  (cwd={PROJECT_ROOT})")

    proc = start_proc(cmd)
    q = queue.Queue()
    t = threading.Thread(target=reader_thread, args=(proc, q, "PROC"), daemon=True)
    t.start()

    t0 = time.time()
    name = None
    tail = []
    TAIL_MAX = 50

    try:
        while True:
            if timeout is not None and (time.time() - t0) >= timeout:
                print("[INFO] Timed out waiting for a name.")
                break
            try:
                _, line = q.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None:
                    # child exited without printing more
                    break
                continue
            if line is None:
                break  # child ended

            print(line)
            tail.append(line)
            if len(tail) > TAIL_MAX:
                tail.pop(0)

            n = extractor(line)
            if n:
                name = n
                break
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        stop_proc(proc)

    if name is None:
        rc = proc.returncode
        print(f"[DEBUG] Child exit code: {rc}")
        if tail:
            print("[DEBUG] Last lines from child:")
            for tl in tail[-10:]:
                print("   ", tl)
    return name



def build_face_cmd(cam_index:int, face_sim:float) -> list[str]:
    return [sys.executable, "-u", "-m", "src.facerec.05_infer_realtime",
            "--source", str(cam_index), "--sim-th", f"{face_sim}"]

def build_voice_cmd(mic_index:int, voice_sim:float, vad:int, min_dur:float) -> list[str]:
    return [sys.executable, "-u", "-m", "src.voicerec.03_verify_realtime",
            "--device-index", str(mic_index), "--sim-th", f"{voice_sim}",
            "--vad", str(vad), "--min-dur", f"{min_dur}"]


def main():
    ap = argparse.ArgumentParser(description="Sequential detection: voice → face")
    ap.add_argument("--voice-mic", type=int, default=1, help="Microphone device index for voice")
    ap.add_argument("--voice-sim", type=float, default=0.60, help="Voice cosine threshold (03)")
    ap.add_argument("--voice-timeout", type=float, default=8.0, help="Seconds to wait for voice")
    ap.add_argument("--voice-vad", type=int, default=2, choices=[0,1,2,3], help="VAD aggressiveness")
    ap.add_argument("--voice-min-dur", type=float, default=0.8, help="Minimum voiced duration (s)")
    ap.add_argument("--face-cam", type=int, default=0, help="Camera index for face")
    ap.add_argument("--face-sim", type=float, default=0.45, help="Face cosine gate (05)")
    ap.add_argument("--face-timeout", type=float, default=10.0, help="Seconds to wait for face after voice")
    args = ap.parse_args()

    print("\n=== Sequential Detection (Voice → Face) ===")
    print("Flow: Detect voice → stop → detect face → stop → match → log → repeat.\n")

    try:
        while True:
            # VOICE first
            vcmd = build_voice_cmd(args.voice_mic, args.voice_sim, args.voice_vad, args.voice_min_dur)
            voice_name = wait_for_name(vcmd, extract_voice_name,
                                       "[LISTENING] Please say your passphrase…", timeout=args.voice_timeout)
            if voice_name is None:
                print("\n[FAIL] No voice recognized in time. Failed in detection.\n")
                append_attendance("", ok=False, reason="voice timeout")
                print("[READY] Next person — please speak…\n")
                continue
            print(f"[VOICE] captured: {voice_name}")

            # FACE next
            fcmd = build_face_cmd(args.face_cam, args.face_sim)
            face_name = wait_for_name(fcmd, extract_face_name,
                                      "[STEP] Look at the camera…", timeout=args.face_timeout)
            if face_name is None:
                print("\n[FAIL] No face recognized in time. Failed in detection.\n")
                append_attendance("", ok=False, reason=f"face timeout after voice={voice_name}")
                print("[READY] Next person — please speak…\n")
                continue
            print(f"[FACE] captured: {face_name}")

            # Decision
            if voice_name == face_name:
                print(f"\n[OK] Matched: {face_name} (voice & face). Marking attendance.\n")
                append_attendance(face_name, ok=True, reason="voice+face agree")
            else:
                print(f"\n[FAIL] Voice='{voice_name}' vs Face='{face_name}' → failed in detection.\n")
                append_attendance("", ok=False, reason=f"mismatch: voice={voice_name}; face={face_name}")

            print("[READY] Next person — please speak…\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

if __name__ == "__main__":
    main()
