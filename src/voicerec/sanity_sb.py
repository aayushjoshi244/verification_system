from src.voicerec.winlink_shim import patch_links
patch_links()

import torch
from src.voicerec.hf_loader import load_ecapa_encoder

dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = load_ecapa_encoder("speechbrain/spkrec-ecapa-voxceleb", dev)
print("OK:", type(enc))
