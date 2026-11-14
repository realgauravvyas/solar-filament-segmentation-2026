"""Paths and constants shared by every stage.

Two environment variables relocate the data and the outputs, so the same code
runs on a laptop and inside a Kaggle session:

    FILASEG_DATA   folder that contains ``train/`` and ``test/``
    FILASEG_OUT    folder for checkpoints and predictions (things worth keeping)
    FILASEG_WORK   folder for caches and other large scratch files
                   (default: ``<FILASEG_OUT>/work``)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every GONG frame in the competition is 2048 x 2048 pixels.
IMAGE_SIZE = 2048

#: Number of cross-validation folds (observations are the unit that is split).
N_FOLDS = 4

#: Seed for the fold assignment. Changing it changes every validation number.
FOLD_SEED = 2026

#: A prediction counts as a match when its IoU with a drawing exceeds this.
IOU_MATCH = 0.5


def data_root() -> Path:
    """Folder holding ``train/train_images``, the annotation JSON and ``test/test_images``."""
    env = os.environ.get("FILASEG_DATA")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"


def out_root() -> Path:
    """Folder for everything the pipeline writes."""
    env = os.environ.get("FILASEG_OUT")
    root = Path(env) if env else REPO_ROOT / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def train_json() -> Path:
    return data_root() / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def train_image_dir() -> Path:
    return data_root() / "train" / "train_images"


def test_image_dir() -> Path:
    return data_root() / "test" / "test_images"


def work_root() -> Path:
    """Folder for scratch files that can be rebuilt: caches, detector datasets, training runs."""
    env = os.environ.get("FILASEG_WORK")
    root = Path(env) if env else out_root() / "work"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_dir() -> Path:
    path = work_root() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
