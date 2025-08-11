from pathlib import Path

# Resolve project root from this file's location
ROOT = Path(__file__).resolve().parents[2]

# -------------------------
# Base directories
# -------------------------
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
ANNO = DATA / "annotations"
RUNS = ROOT / "runs"
MODELS = ROOT / "models"

# -------------------------
# Face-specific directories
# -------------------------
FACE_RAW = RAW / "face"                    # Raw per-person images
FACE_CROPS = PROC / "face" / "crops"       # Cropped faces after detection
FACE_ALIGNED = PROC / "face" / "aligned"   # Aligned faces after landmarking
FACE_XLSX = ANNO / "faces.xlsx"            # Optional Excel annotations

# -------------------------
# Outputs / experiments
# -------------------------
EMB_FACE_DIR = RUNS / "face" / "embeddings"  # Saved embeddings .npz
EVAL_FACE_DIR = RUNS / "face" / "eval"       # Evaluation results (reports, confusion matrix)
PRED_FACE_DIR = RUNS / "face" / "pred_images" # Annotated predictions for folder tests

# -------------------------
# Models
# -------------------------
MODELS_FACE_DIR = MODELS / "face"           # Saved SVM model + label map

# -------------------------
# Ensure directories exist
# -------------------------
def ensure_dirs():
    for p in [
        DATA, RAW, PROC, ANNO, RUNS, MODELS,
        FACE_RAW, FACE_CROPS, FACE_ALIGNED,
        EMB_FACE_DIR, EVAL_FACE_DIR, PRED_FACE_DIR,
        MODELS_FACE_DIR
    ]:
        p.mkdir(parents=True, exist_ok=True)
