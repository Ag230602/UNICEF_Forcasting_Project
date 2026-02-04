"""
utils package
-------------
Common utilities for configuration, IO, and shared helpers.
"""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    FIGURE_DIR,
    METRIC_DIR,
    VIDEO_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    RANDOM_SEED,
    DEVICE,
    KNN_K,
    MAX_NODES,
    VIDEO_FPS,
    VIDEO_RESOLUTION,
    create_dirs,
)

from .io import (
    save_csv,
    load_csv,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_torch,
    load_torch,
)
