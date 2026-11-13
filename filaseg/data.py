"""Annotations, observations and cross-validation folds.

Vocabulary used throughout the code base:

    observation   one H-alpha frame, identified by its file stem, e.g. ``20140609195854Bh``
    view          one annotator's drawing of an observation (1 to 3 per observation)
    filament      one polygon inside a view

The organisers score a prediction against every view of an observation
separately and pool the counts, so views are kept apart everywhere and never
merged into a single "consensus" label.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, rle


@dataclass
class View:
    """One annotator's drawing of one observation."""

    view_id: str
    polygons: list[list[float]] = field(default_factory=list)

    def rles(self) -> list[dict]:
        return [rle.polygon_to_rle(p) for p in self.polygons]


@dataclass
class Observation:
    """One H-alpha frame together with all of its views."""

    stem: str
    file_name: str
    views: list[View] = field(default_factory=list)


def load_observations(json_path: Path | None = None) -> list[Observation]:
    """Read the training annotations and group the views by observation.

    Returns the observations sorted by stem, so the order is deterministic.
    """
    json_path = json_path or config.train_json()
    with open(json_path, encoding="utf-8") as handle:
        coco = json.load(handle)

    polygons_by_view: dict[str, list[list[float]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        # Every filament is a single closed polygon (see the data description).
        polygons_by_view[annotation["image_id"]].append(annotation["segmentation"][0])

    by_stem: dict[str, Observation] = {}
    for image in coco["images"]:
        stem = Path(image["file_name"]).stem
        observation = by_stem.setdefault(stem, Observation(stem, image["file_name"]))
        observation.views.append(View(image["id"], polygons_by_view.get(image["id"], [])))

    observations = sorted(by_stem.values(), key=lambda o: o.stem)
    for observation in observations:
        observation.views.sort(key=lambda v: v.view_id)
    return observations


def assign_folds(
    observations: list[Observation],
    n_folds: int = config.N_FOLDS,
    seed: int = config.FOLD_SEED,
) -> dict[str, int]:
    """Assign every observation to one fold.

    The split is by observation, so all views of a frame stay together, and it
    is stratified by the number of views: frames drawn by several annotators
    are scored several times, so each fold gets its share of them.
    """
    rng = np.random.default_rng(seed)
    folds: dict[str, int] = {}
    by_view_count: dict[int, list[str]] = defaultdict(list)
    for observation in observations:
        by_view_count[len(observation.views)].append(observation.stem)
    for stems in by_view_count.values():
        order = rng.permutation(len(stems))
        for position, index in enumerate(order):
            folds[stems[index]] = position % n_folds
    return folds


def list_test_stems() -> list[str]:
    return sorted(p.stem for p in config.test_image_dir().glob("*.jpeg"))


def image_path(stem: str, subset: str) -> Path:
    """Path of a frame; ``subset`` is ``"train"`` or ``"test"``."""
    folder = config.train_image_dir() if subset == "train" else config.test_image_dir()
    return folder / f"{stem}.jpeg"
