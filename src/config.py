"""
Shared configuration for the Deep-DMS project.
All stages (preprocessing, dataset, model, training) import from here so that
class order, paths and hyper-parameters stay consistent across the pipeline.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEOS_DIR = ROOT / "data" / "raw_videos"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
LABELS_FILE = PROCESSED_DIR / "labels.json"

# ---------------------------------------------------------------------------
# Classes - the index in this list IS the label used for training.
# Do not reorder once you have trained a model.
# ---------------------------------------------------------------------------
CLASSES = ["Alert", "Drowsy", "Sleeping", "Singing", "Distracted", "Yawning", "Dazzled"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# Sequence / feature settings
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30   # frames per sample (~1 second at 30 fps)
NUM_FEATURES = 2       # [EAR, MAR]

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices (same ones used in the original main.py)
# ---------------------------------------------------------------------------
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 82, 312, 308, 317, 87]
