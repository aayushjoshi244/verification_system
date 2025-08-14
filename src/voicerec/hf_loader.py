# src/voicerec/hf_loader.py
import os, shutil
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier

RUNS_ROOT   = Path("runs/voice").resolve()
RUNTIME_DIR = (RUNS_ROOT / "sb_runtime").resolve()   # SpeechBrain will write here

# Keep HF cache local & disable link tricks (Windows-safe)
os.environ.setdefault("HF_HOME", str((RUNS_ROOT / "hf_cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_HARD_LINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- Replace hardlink/symlink with *file copy* (avoid WinError 1314) ---
def _copy2(src, dst):
    src = os.fspath(src); dst = os.fspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        shutil.copy(src, dst)

def _safe_link(src, dst, *a, **k): _copy2(src, dst)
def _safe_symlink(src, dst, *a, **k): _copy2(src, dst)

os.link = _safe_link
os.symlink = _safe_symlink
# -----------------------------------------------------------------------

def load_ecapa_encoder(model_id: str, device: str):
    """
    Let SpeechBrain fetch from Hugging Face directly into our RUNTIME_DIR.
    Our monkeypatch ensures it copies files instead of linking, so no admin needed.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    encoder = EncoderClassifier.from_hparams(
        source=str(model_id),
        savedir=os.fspath(RUNTIME_DIR),
        run_opts={"device": device},
    )
    return encoder
