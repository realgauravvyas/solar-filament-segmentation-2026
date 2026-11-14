"""COCO run-length helpers.

The competition stores one filament per row as the ``counts`` string of a
compressed COCO RLE; the size is always 2048 x 2048 and is not stored.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pycocotools import mask as mask_util

from .config import IMAGE_SIZE


def encode(mask: np.ndarray) -> dict:
    """Binary mask (H, W) -> compressed RLE dict with a ``bytes`` counts field."""
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def decode(rle: dict) -> np.ndarray:
    """Compressed RLE dict -> boolean mask (H, W)."""
    return mask_util.decode(rle).astype(bool)


def counts_to_rle(counts: str | bytes, size: int = IMAGE_SIZE) -> dict:
    """Wrap a bare counts string, as found in a submission file, into an RLE dict."""
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": [size, size], "counts": counts}


def rle_to_counts(rle: dict) -> str:
    """RLE dict -> the counts string that goes into the submission file."""
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def polygon_to_rle(polygon: Sequence[float], size: int = IMAGE_SIZE) -> dict:
    """One closed polygon ``[x0, y0, x1, y1, ...]`` -> RLE, rasterised by pycocotools.

    This is the rasterisation the organisers use, so masks built here are
    pixel-identical to the ground truth they score against.
    """
    return mask_util.merge(mask_util.frPyObjects([list(polygon)], size, size))


def area(rle: dict) -> int:
    return int(mask_util.area(rle))


def iou_matrix(pred: list[dict], gt: list[dict]) -> np.ndarray:
    """Pairwise IoU, shape (len(pred), len(gt)). Empty inputs give an empty matrix."""
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    return np.asarray(mask_util.iou(pred, gt, [0] * len(gt)), dtype=np.float64)
