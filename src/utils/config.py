"""
config.py
----------
Central configuration file for paths, constants, and global settings.
"""

from pathlib import Path

# =========================
# Project root
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================
# Data paths
# =========================
DATA_DIR = PROJECT_ROOT / "project_data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# =========================
# Output paths
# =========================
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
VIDEO_DIR = OUTPUT_DIR / "videos"

# =========================
# Model paths
# =========================
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# =========================
# Runtime settings
# =========================
RANDOM_SEED = 42
DEVICE = "cuda"  # change to "cpu" if needed

# =========================
# Graph parameters
# =========================
KNN_K = 5
MAX_NODES = 6000

# =========================
# Visualization defaults
# =========================
VIDEO_FPS = 24
VIDEO_RESOLUTION = (1920, 1080)

# =========================
# Directory creation
# =========================
def create_dirs():
    """Create all required directories if they don't exist."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURE_DIR,
        METRIC_DIR,
        VIDEO_DIR,
        MODEL_DIR,
        CHECKPOINT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
