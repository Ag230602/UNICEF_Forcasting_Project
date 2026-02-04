"""
io.py
-----
Utility functions for saving and loading data artifacts.
"""

import json
import pickle
from pathlib import Path
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


def ensure_parent(path: Path):
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


# =========================
# CSV
# =========================
def save_csv(df: pd.DataFrame, path: Path):
    ensure_parent(path)
    df.to_csv(path, index=False)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# =========================
# JSON
# =========================
def save_json(obj, path: Path):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Pickle
# =========================
def save_pickle(obj, path: Path):
    ensure_parent(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================
# PyTorch
# =========================
def save_torch(obj, path: Path):
    if torch is None:
        raise ImportError("PyTorch is not installed.")
    ensure_parent(path)
    torch.save(obj, path)


def load_torch(path: Path, map_location="cpu"):
    if torch is None:
        raise ImportError("PyTorch is not installed.")
    return torch.load(path, map_location=map_location)
