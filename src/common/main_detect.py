# src/common/main_detect.py
import sys, subprocess, re, time, threading, queue
from pathlib import Path
from datetime import datetime

# ---- tune these if needed ----
VOICE_CMD = [sys.executable, "-m", "src.voicerec.03_verify_realtime",
             "--device-index", "1", "--sim-th", "0.60"]
FACE_CMD  = [sys.executable, "-m", "src.facerec.05_infer_realtime",
             "--source", "0", "--sim-th", "0.45"]  # add --unknown-threshold if you need

# Unified attendance log (Excel)
ATTEND_FILE = Path("runs/registry/attendance.xlsx")

# Accept only if both sides agree within this many seconds
AGREE_WINDOW_SEC = 5.0

# ---- Face line parsing ----
# Preferred: add in 05_infer_realtime a line like:
#   print(f"[FACE] name={pred_name} sim={sim:.3f}")
FACE_RE_STRICT = re.compile(r"\[FACE\].*name\s*=\s*([A-Za-z0-9_\- ]+)", re.I)

# Fallback: try to parse typical outputs like "pred=Name", "ID: Name", etc.
FACE_RE_FALLBACK = re.compile(
    r"(?:pred\s*[:=]\s*|id\s*[:=]\s*)([A-Za-z0-9_\- ]+)", re.I
)

# ---- Voice parsing (from your 03) ----
VOICE_RE = re.compile(r"\[VOICE\].*==>\s*([A-Za-z0-9_\- ]+)", re.I)


import pandas as pd

def append_attendance(name: str, ok: bool, reason: str = ""):
    ATTEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    row = {
        "name": name if ok else "",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "status": "OK" if ok else "FAILED",
        "reason": reason,
    }
    if ATTEND_FILE.exists():
        df = pd.read_excel(ATTEND_FILE)
        for c in ["name","date","time","status","reason"]:
            if c not in df.columns:
                df[c] = None
        df = pd.concat([df[["name","date","time","status","reason"]],
                        pd.DataFrame([row])[["name","date","time","status","reason"]]], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=["name","date","time","status","reason"])
    ATTEND_FILE.unlink(missing_ok=True)
    with pd.ExcelWriter(ATTEND_FILE, engine="openpyxl") as xw:
        df.to_excel(xw, index=False)

def reader_thread(proc, outq, tag):
    # push each stdout line with its tag ("FACE"/"VOICE")
    for line in proc.stdout:
        try:
            s = line.decode(errors="ignore").strip()
        except Exception:
            s = str(line).strip()
        outq.put((tag, s))
    outq.put((tag, None))  # signals end

def start_proc(cmd):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)

def extract_face_name(s: str):
    m = FACE_RE_STRICT.search(s)
    if m:
        return m.group(1).strip()
    m = FACE_RE_FALLBACK.search(s)
    if m:
        return m.group(1).strip()
    return None

def extract_voice_name(s: str):
    m = VOICE_RE.search(s)
    if not m:
        return None
    lbl = m.group(1).strip()
    if lbl.upper() == "UNKNOWN":
        return None
    return lbl

def main():
    print("\n=== Combined Detection (Face + Voice) ===")
    print("[INFO] Launching realtime processes …")
    print("       On-screen: 'marking you, please wait' until both match.")
    print("       If either side fails → 'failed in detection'.")

    voice = start_proc(VOICE_CMD)
    face  = start_proc(FACE_CMD)

    q = queue.Queue()
    tv = threading.Thread(target=reader_thread, args=(voice, q, "VOICE"), daemon=True)
    tf = threading.Thread(target=reader_thread, args=(face,  q, "FACE"),  daemon=True)
    tv.start(); tf.start()

    last_face_name, t_face = None, 0.0
    last_voice_name, t_voice = None, 0.0

    print("\n[INFO] marking you, please wait…")
    try:
        while True:
            try:
                tag, s = q.get(timeout=0.25)
            except queue.Empty:
                # check time window match periodically
                pass
            else:
                if s is None:
                    # one of the processes ended; keep going until both end
                    if (voice.poll() is not None) and (face.poll() is not None):
                        print("[INFO] Detectors stopped.")
                        break
                    continue

                # parse new line
                if tag == "VOICE":
                    vname = extract_voice_name(s)
                    if vname:
                        last_voice_name, t_voice = vname, time.time()
                        print(f"[VOICE~] {vname}")
                elif tag == "FACE":
                    fname = extract_face_name(s)
                    if fname:
                        last_face_name, t_face = fname, time.time()
                        print(f"[FACE~]  {fname}")

            # Decision check: both present and close in time?
            if last_face_name and last_voice_name:
                if abs(t_face - t_voice) <= AGREE_WINDOW_SEC:
                    if last_face_name == last_voice_name:
                        print(f"\n[OK] Matched: {last_face_name} (face & voice)\n")
                        append_attendance(last_face_name, ok=True, reason="face+voice agree")
                        # reset so we can accept next person
                        last_face_name, last_voice_name = None, None
                        print("[INFO] marking you, please wait…")
                    else:
                        # mismatch → failed attempt
                        print(f"\n[FAIL] Face='{last_face_name}' vs Voice='{last_voice_name}' → failed in detection\n")
                        append_attendance("", ok=False, reason=f"mismatch: face={last_face_name}; voice={last_voice_name}")
                        last_face_name, last_voice_name = None, None
                        print("[INFO] marking you, please wait…")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        for p in (voice, face):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2.0)
                except Exception:
                    p.kill()

if __name__ == "__main__":
    main()
