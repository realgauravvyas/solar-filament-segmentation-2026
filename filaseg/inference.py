"""Run the two networks over whole subsets and store what they produce.

    probability maps   ``<folder>/<stem>.png``     8-bit, one per frame
    detections         ``<file>.json``             {stem: [[{"score", "counts"}, ...] per view]}

Both are plain files, so the later stages (candidate generation, scoring,
selection) run on a CPU in minutes and can be repeated without a GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import detector, preprocess, rle, unet


def predict_probabilities(
    checkpoints: list[Path],
    stems: list[str],
    subset: str,
    out_dir: Path,
    device: str = "cuda",
    tta: int = 8,
    tile: int = 1024,
) -> None:
    """Average the given U-Net checkpoints and write one probability PNG per frame."""
    models = [unet.load_model(path, device) for path in checkpoints]
    disks = preprocess.load_disks(subset)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for index, stem in enumerate(stems, start=1):
        target = out_dir / f"{stem}.png"
        if target.exists():
            continue
        channels = preprocess.load_channels(stem, subset)
        probability = unet.predict_probability(
            models, channels, disks[stem], device=device, tta=tta, tile=tile
        )
        unet.save_probability(probability, target)
        if index % 25 == 0:
            print(f"  U-Net {subset}: {index}/{len(stems)}  {time.time() - started:.0f} s", flush=True)


def predict_detections(
    weights: Path,
    stems: list[str],
    subset: str,
    out_path: Path,
    imgsz: int = 1536,
    device: str | int = 0,
    flips: int = 4,
) -> None:
    """Run the detector over a subset and write its detections to JSON.

    The file maps each stem to a list of test-time views, each view being a
    list of ``{"score", "counts"}`` detections.
    """
    model = detector.load(weights)
    results: dict[str, list[list[dict]]] = {}
    started = time.time()
    for index, stem in enumerate(stems, start=1):
        channels = preprocess.load_channels(stem, subset)
        views = detector.predict(weights, channels, imgsz=imgsz, device=device, model=model, flips=flips)
        results[stem] = [
            [{"score": round(d["score"], 5), "counts": rle.rle_to_counts(d["rle"])} for d in view]
            for view in views
        ]
        if index % 25 == 0:
            print(f"  detector {subset}: {index}/{len(stems)}  {time.time() - started:.0f} s", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)


def load_detections(path: Path) -> dict[str, list[list[dict]]]:
    """Read a detections file back as {stem: [view, ...]}, each view a list of {"score", "rle"}."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    loaded = {}
    for stem, views in raw.items():
        if views and isinstance(views[0], dict):
            views = [views]          # files written before test-time flips hold a single view
        loaded[stem] = [
            [{"score": d["score"], "rle": rle.counts_to_rle(d["counts"])} for d in view]
            for view in views
        ]
    return loaded


def mean_probability(folders: list[Path], stem: str) -> np.ndarray:
    """Average the probability maps of one frame over several models."""
    total = None
    for folder in folders:
        probability = unet.load_probability(folder / f"{stem}.png")
        total = probability if total is None else total + probability
    return total / len(folders)
