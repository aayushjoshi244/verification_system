# src/voice/01_record_enroll.py
import sys, time, argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from pathlib import Path
import os
from src.voicerec.hf_loader import load_ecapa_encoder

# --- constants ---
SR = 16000
CHAN = 1
SAMPLE_WIDTH_BYTES = 2
FRAME_MS = 30                      # 10/20/30ms supported by VAD
FRAME_SAMPLES = SR * FRAME_MS // 1000

RUNS_ROOT = Path("runs/voice")  # (already in your files)

# Put HF cache in project + force copy, not links (Windows-friendly)
os.environ.setdefault("HF_HOME", str((RUNS_ROOT / "hf_cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


OUT_ROOT = Path("data/raw/voice")

def _err(msg): print(f"[ERROR] {msg}", file=sys.stderr)

def list_devices_print():
    try:
        devs = sd.query_devices()
        print("\n[INFO] Available audio devices:")
        for i, d in enumerate(devs):
            print(f"  [{i:02d}] in={d.get('max_input_channels',0)} out={d.get('max_output_channels',0)}  {d.get('name','')}")
        print()
        return devs
    except Exception as e:
        _err(f"Could not query audio devices: {e}")
        return []

def ensure_path(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def vad_chunker(vad: webrtcvad.Vad, pcm_int16: np.ndarray):
    """Yield voiced chunks (bytes) using simple hangover logic."""
    bytes_per_frame = FRAME_SAMPLES * SAMPLE_WIDTH_BYTES
    raw = pcm_int16.tobytes()
    voiced = bytearray()
    hang = 0
    HANG_FRAMES = 8               # ~240 ms tail with 30 ms frames
    active = False

    for i in range(0, len(raw), bytes_per_frame):
        frame = raw[i:i+bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        is_speech = vad.is_speech(frame, SR)
        if is_speech:
            active = True
            hang = HANG_FRAMES
            voiced.extend(frame)
        else:
            if active:
                if hang > 0:
                    hang -= 1
                    voiced.extend(frame)
                else:
                    yield bytes(voiced)
                    voiced = bytearray()
                    active = False

    if voiced:
        yield bytes(voiced)

def record_once(seconds=10, device=None):
    try:
        audio = sd.rec(int(seconds * SR), samplerate=SR, channels=CHAN, dtype='float32', device=device)
        sd.wait()
        a = audio[:, 0] if audio.ndim == 2 else audio
        if a is None or a.size == 0:
            return None
        if not np.isfinite(a).all():
            _err("Non-finite samples (NaN/Inf) in capture; discarding.")
            return None
        # Quick feedback on level
        peak = float(np.max(np.abs(a)))
        if peak > 0.98:
            print("[WARN] Audio looks clipped (peak > 0.98). Back off the mic or lower input gain.")
        elif peak < 0.02:
            print("[WARN] Audio very quiet (peak < 0.02). Move closer to the mic or raise input gain.")
        return a
    except Exception as e:
        _err(f"Recording failed: {e}")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Person name (folder-safe)")
    ap.add_argument("--utterances", type=int, default=5, help="How many enrol utterances to save")
    ap.add_argument("--seconds", type=int, default=10, help="Max seconds per recording window")
    ap.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices)")
    ap.add_argument("--vad", type=int, default=2, choices=[0,1,2,3], help="WebRTC VAD aggressiveness (0=loose, 3=strict)")
    ap.add_argument("--min-dur", type=float, default=0.8, help="Minimum voiced duration (seconds) after VAD")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = ap.parse_args()

    # Devices
    devs = list_devices_print() if args.list_devices else list_devices_print()
    if args.list_devices:
        return
    if not devs:
        _err("No audio devices found."); sys.exit(2)
    if args.device is not None:
        if args.device < 0 or args.device >= len(devs):
            _err(f"Invalid device index {args.device}."); sys.exit(2)
        if devs[args.device].get("max_input_channels", 0) < 1:
            _err(f"Selected device [{args.device}] has no input channels."); sys.exit(2)

    out_dir = ensure_path(OUT_ROOT / args.name)
    print(f"[INFO] Saving enrol WAVs to: {out_dir}")
    print("[INFO] Speak naturally when prompted. We auto-trim silence. Press Ctrl+C to stop early.")

    vad = webrtcvad.Vad(args.vad)
    saved = 0

    try:
        while saved < args.utterances:
            print(f"\n[STEP] Utterance {saved+1}/{args.utterances} — recording up to {args.seconds}s…")
            audio = record_once(seconds=args.seconds, device=args.device)
            if audio is None:
                _err("Capture returned None; retrying.")
                continue

            pcm = float_to_int16(audio)
            voiced_segments = list(vad_chunker(vad, pcm))

            if not voiced_segments:
                print("[WARN] No voiced speech detected. Try speaking closer to the mic.")
                continue

            concat = b"".join(voiced_segments)
            pcm_concat = np.frombuffer(concat, dtype=np.int16)
            wav = pcm_concat.astype(np.float32) / 32767.0

            dur = len(wav) / SR
            if dur < args.min_dur:
                print(f"[WARN] Too short after VAD ({dur:.2f}s < {args.min_dur:.2f}s). Please try again.")
                continue

            ts = int(time.time() * 1000)
            outp = out_dir / f"enroll_{ts}.wav"
            try:
                sf.write(str(outp), wav, SR, subtype="PCM_16")
                print(f"[OK] Saved: {outp}  ({dur:.2f}s)")
                saved += 1
            except Exception as e:
                _err(f"Failed to save WAV: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    if saved == 0:
        _err("No utterances saved. Exiting with error."); sys.exit(1)

    print(f"[DONE] Saved {saved} enrol utterances for '{args.name}' in {out_dir}")

if __name__ == "__main__":
    main()
