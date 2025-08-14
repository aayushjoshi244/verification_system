# src/common/registry_log.py
from pathlib import Path
from datetime import datetime
import pandas as pd

REG_FILE = Path("runs/registry/registry.xlsx")
REG_FILE.parent.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "name",
    "date",
    "time",
    "face_image_dir",
    "voice_audio_dir",
    "num_images",
    "num_audios",
]

def append_registry_row(name: str,
                        face_image_dir: Path,
                        voice_audio_dir: Path,
                        num_images: int,
                        num_audios: int):
    REG_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    row = {
        "name": name,
        "date": date_str,
        "time": time_str,
        "face_image_dir": str(face_image_dir.resolve()),
        "voice_audio_dir": str(voice_audio_dir.resolve()),
        "num_images": int(num_images),
        "num_audios": int(num_audios),
    }

    if REG_FILE.exists():
        df = pd.read_excel(REG_FILE)
        # Ensure expected columns exist (in case file was edited)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = pd.concat([df[COLUMNS], pd.DataFrame([row])[COLUMNS]], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=COLUMNS)

    REG_FILE.unlink(missing_ok=True)
    with pd.ExcelWriter(REG_FILE, engine="openpyxl") as xw:
        df.to_excel(xw, index=False)
