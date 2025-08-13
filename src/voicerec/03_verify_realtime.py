# src/voice/03_verify_realtime.py
import sys, time, json, argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
import webrtcvad
import torch
from speechbrain.pretrained import EncoderClassifier

RUNS_ROOT = Path("runs/voice")
PROTO_DIR = RUNS_ROOT / "prototypes"
INDEX_JSON = RUNS_ROOT / "index.json"

SR = 16000
CHAN = 1
FRAME_MS = 30
FRAME_SAMPLES = SR * FRAME_MS // 1000
SAMPLE_WIDTH_BYTES = 2

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

def float_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def l2norm(v: np.ndarray, eps=1e-9) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def record_window(seconds=6, device=None):
    try:
        audio = sd.rec(int(seconds * SR), samplerate=SR, channels=CHAN, dtype='float32', device=device)
        sd.wait()
        a = audio[:,0] if audio.ndim == 2 else audio
        if a is None or a.size == 0: return None
        if not np.isfinite(a).all():
            _err("Non-finite samples in capture; discarding.")
            return None
        # quick level feedback
        peak = float(np.max(np.abs(a)))
        if peak > 0.98:
            print("[WARN] Audio clipped (peak > 0.98). Lower input gain / move back.")
        elif peak < 0.02:
            print("[WARN] Audio very quiet (peak < 0.02). Raise input gain / move closer.")
        return a
    except Exception as e:
        _err(f"Recording failed: {e}")
        return None

def vad_collect(audio_f32, aggressiveness=2):
    vad = webrtcvad.Vad(aggressiveness)
    pcm = float_to_int16(audio_f32)
    raw = pcm.tobytes()
    bytes_per_frame = FRAME_SAMPLES * SAMPLE_WIDTH_BYTES
    voiced = bytearray()
    active = False
    hang, HANG_FRAMES = 0, 8

    for i in range(0, len(raw), bytes_per_frame):
        frame = raw[i:i+bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        if vad.is_speech(frame, SR):
            active = True
            hang = HANG_FRAMES
            voiced.extend(frame)
        else:
            if active:
                if hang > 0:
                    hang -= 1
                    voiced.extend(frame)
                else:
                    seg = np.frombuffer(bytes(voiced), dtype=np.int16).astype(np.float32)/32767.0
                    yield seg
                    voiced = bytearray()
                    active = False

    if len(voiced) > 0:
        seg = np.frombuffer(bytes(voiced), dtype=np.int16).astype(np.float32)/32767.0
        yield seg

def cosine(a, b):
    # expects L2-normalized inputs, but safe either way
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

def load_prototypes():
    if not INDEX_JSON.exists():
        _err(f"Prototype index not found: {INDEX_JSON}. Run 02_embed_speechbrain.py first.")
        sys.exit(1)
    idx = json.loads(INDEX_JSON.read_text())
    protos = {}
    for name, path in idx.items():
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Missing proto: {p}")
            continue
        v = np.load(p)
        protos[name] = l2norm(v)  # ensure normalized
    if not protos:
        _err("No prototypes loaded.")
        sys.exit(1)
    return protos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-index", type=int, default=None, help="Input device index (see printed list)")
    ap.add_argument("--sim-th", type=float, default=0.50, help="Cosine similarity accept threshold")
    ap.add_argument("--vad", type=int, default=2, choices=[0,1,2,3], help="VAD aggressiveness (0=loose, 3=strict)")
    ap.add_argument("--min-dur", type=float, default=0.8, help="Minimum voiced duration (s) to consider a segment")
    ap.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--device", choices=["cpu","cuda"], default=None, help="Force model device (default: auto)")
    args = ap.parse_args()

    devs = list_devices_print()
    if len(devs) == 0:
        _err("No audio devices available.")
        sys.exit(2)
    if args.device_index is not None:
        if args.device_index < 0 or args.device_index >= len(devs):
            _err(f"Invalid device index {args.device_index}."); sys.exit(2)
        if devs[args.device_index].get("max_input_channels",0) < 1:
            _err(f"Selected device [{args.device_index}] has no input channels."); sys.exit(2)

    mdl_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        encoder = EncoderClassifier.from_hparams(source=args.model, run_opts={"device": mdl_device})
    except Exception as e:
        _err(f"Failed to load SpeechBrain model: {e}")
        sys.exit(2)

    protos = load_prototypes()
    names = list(protos.keys())
    mats = np.stack([protos[n] for n in names], axis=0)

    print(f"[INFO] Loaded {len(names)} speaker prototypes. (model on {mdl_device})")
    print("[INFO] Speak; segments will be auto-detected. Ctrl+C to stop.")

    try:
        while True:
            audio = record_window(seconds=6, device=args.device_index)
            if audio is None:
                continue
            for seg in vad_collect(audio, aggressiveness=args.vad):
                if len(seg) < int(args.min_dur * SR):
                    continue
                try:
                    sig = torch.from_numpy(seg).float().unsqueeze(0).to(mdl_device)  # [1, T]
                    with torch.inference_mode():
                        emb_t = encoder.encode_batch(sig).squeeze(0)               # [D]
                    emb = emb_t.detach().cpu().numpy()
                    emb = l2norm(emb)
                except Exception as e:
                    print(f"[WARN] Embedding failed for a segment: {e}")
                    continue

                sims = mats @ emb  # since both L2-normalized, dot = cosine
                best_i = int(np.argmax(sims))
                best_name, best_sim = names[best_i], float(sims[best_i])

                label = best_name if best_sim >= args.sim_th else "UNKNOWN"
                print(f"[VOICE] best={best_name:15s}  sim={best_sim:.3f}  ==> {label}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    except Exception as e:
        _err(f"Runtime error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
