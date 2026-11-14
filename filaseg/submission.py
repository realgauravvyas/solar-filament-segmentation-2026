"""Submission file: writer and format checks.

The file has one row per predicted filament:

    filament_id,segmentation_rle
    20150125172714Mh_1,<COCO RLE counts>

``filament_id`` is the frame's stem plus a running number. The competition
requires every row to be unique; this pipeline additionally guarantees that
the masks of one frame do not overlap.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from . import rle


def write(predictions: dict[str, list[np.ndarray]], path: Path) -> int:
    """Write ``{stem: [boolean masks]}`` as a submission CSV; returns the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for stem in sorted(predictions):
            for number, mask in enumerate(predictions[stem], start=1):
                writer.writerow([f"{stem}_{number}", rle.rle_to_counts(rle.encode(mask))])
                rows += 1
    return rows


def read(path: Path) -> dict[str, list[dict]]:
    """Read a submission CSV back as ``{stem: [RLE dicts]}``."""
    predictions: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = row["filament_id"].rsplit("_", 1)[0]
            predictions.setdefault(stem, []).append(rle.counts_to_rle(row["segmentation_rle"]))
    return predictions


def check(path: Path, expected_stems: list[str]) -> dict[str, float]:
    """Validate a submission file and return summary statistics.

    Raises ``ValueError`` if an id is duplicated, a stem is unknown, a mask is
    empty, or two masks of the same frame share a pixel.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        ids = [row["filament_id"] for row in csv.DictReader(handle)]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate filament_id")
    predictions = read(path)
    unknown = set(predictions) - set(expected_stems)
    if unknown:
        raise ValueError(f"unknown stems: {sorted(unknown)[:5]}")
    areas = []
    for stem, rles in predictions.items():
        frame_areas = [int(a) for a in mask_util.area(rles)]
        if min(frame_areas) == 0:
            raise ValueError(f"empty mask in {stem}")
        union_area = int(mask_util.area(mask_util.merge(rles)))
        if union_area != sum(frame_areas):
            raise ValueError(f"overlapping masks in {stem}")
        areas.extend(frame_areas)
    return {
        "rows": len(ids),
        "frames_with_predictions": len(predictions),
        "frames_total": len(expected_stems),
        "median_area": float(np.median(areas)) if areas else 0.0,
        "mean_per_frame": len(ids) / max(1, len(expected_stems)),
    }
