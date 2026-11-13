"""Glue between the stored network outputs and the candidate / selection stages.

A *recipe* names the networks that take part:

    Recipe(unets=["r34", "rx50"], detectors=["m1536"])

For a training frame in fold k the networks of fold k are used; they have not
seen that frame, so everything measured on it is out-of-fold. For a test frame
the networks of all folds are used and their outputs are averaged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import candidates as cand
from . import config, inference, metric, preprocess, rle
from .data import Observation


@dataclass
class Recipe:
    unets: list[str]
    detectors: list[str]
    folds: list[int] = field(default_factory=lambda: list(range(config.N_FOLDS)))
    flips: int | None = None      # use only the first N test-time views of each detector (None = all)
    det_suffix: str = ""          # detections are read from det/<name>_f<k><suffix>/
    params: cand.CandidateParams = field(default_factory=cand.CandidateParams)


@lru_cache(maxsize=64)
def _detections(path: str) -> dict[str, list[list[dict]]]:
    return inference.load_detections(Path(path))


@lru_cache(maxsize=2)
def _disks(subset: str) -> dict[str, preprocess.Disk]:
    return preprocess.load_disks(subset)


def merge_detections(per_detector: list[list[dict]], iou_same: float = 0.5) -> list[dict]:
    """Fuse the outputs of several detectors run on the same frame.

    Detections whose masks overlap with IoU above ``iou_same`` are taken to be
    the same object. The most confident mask represents the object. Each fused
    detection carries

        score   mean confidence over all detectors, counting 0 for a detector
                that did not report the object
        votes   fraction of the detectors that reported it
        top     the highest single confidence
    """
    n_detectors = len(per_detector)
    pool = sorted(
        ((d, index) for index, dets in enumerate(per_detector) for d in dets),
        key=lambda item: -item[0]["score"],
    )
    merged: list[dict] = []
    seen_by: list[dict[int, float]] = []
    for detection, source in pool:
        if merged:
            ious = rle.iou_matrix([detection["rle"]], [m["rle"] for m in merged])[0]
            best = int(np.argmax(ious))
            if ious[best] > iou_same:
                seen_by[best].setdefault(source, detection["score"])
                continue
        merged.append({"rle": detection["rle"], "top": detection["score"]})
        seen_by.append({source: detection["score"]})
    for detection, sources in zip(merged, seen_by):
        detection["score"] = sum(sources.values()) / n_detectors
        detection["votes"] = len(sources) / n_detectors
    return sorted(merged, key=lambda d: -d["score"])


def frame_inputs(recipe: Recipe, stem: str, fold: int | None):
    """Probability map and detections of one frame.

    ``fold`` is the frame's own fold for a training frame (out-of-fold
    networks are used) and ``None`` for a test frame (all folds are averaged).
    """
    out = config.out_root()
    if fold is None:
        prob_dirs = [out / "prob" / f"{name}_f{k}" / "test" for name in recipe.unets for k in recipe.folds]
        det_files = [out / "det" / f"{name}_f{k}{recipe.det_suffix}" / "test.json" for name in recipe.detectors for k in recipe.folds]
    else:
        prob_dirs = [out / "prob" / f"{name}_f{fold}" / "val" for name in recipe.unets]
        det_files = [out / "det" / f"{name}_f{fold}{recipe.det_suffix}" / "val.json" for name in recipe.detectors]
    probability = inference.mean_probability(prob_dirs, stem)
    # Every (detector, test-time view) pair is one voter in the fusion.
    voters = [
        view for path in det_files
        for view in _detections(str(path)).get(stem, [[]])[:recipe.flips]
    ]
    detections = merge_detections(voters)
    return probability, detections


def frame_candidates(recipe: Recipe, stem: str, fold: int | None) -> list[cand.Candidate]:
    """All candidates of one frame under a recipe."""
    subset = "test" if fold is None else "train"
    probability, detections = frame_inputs(recipe, stem, fold)
    channels = preprocess.load_channels(stem, subset)
    disk = _disks(subset)[stem]
    return cand.generate(probability, detections, channels, disk, recipe.params)


def candidate_values(candidates: list[cand.Candidate], observation: Observation) -> np.ndarray:
    """Training target of each candidate: its matched IoU averaged over the frame's views."""
    if not candidates:
        return np.zeros(0)
    view_rles = [view.rles() for view in observation.views]
    return metric.match_value([c.to_rle() for c in candidates], view_rles).mean(axis=1)
