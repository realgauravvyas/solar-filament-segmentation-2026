"""From network outputs to candidate filament instances.

Two networks look at every frame:

* the U-Net gives a per-pixel filament probability at full resolution: it
  knows the *shape* of filament material, but it cannot separate two
  filaments that touch and it sometimes leaves one filament in pieces;
* the detector gives instance proposals with coarse masks and a confidence:
  it knows the *grouping* and how likely an annotator is to draw the object,
  but its outlines are rougher.

A candidate (kind ``claim``) is the U-Net foreground inside one detection's
mask, grown by a few pixels because the detector mask is coarse: the
detector's grouping with the U-Net's outline. Every detection yields one
candidate per U-Net probability cut (a fatter and a thinner version of the
same object), and each candidate carries a feature vector that describes the
candidate itself, the detection behind it, and its surroundings (for example:
how many confident detections share the connected component it sits in?).
``selection.py`` decides which candidates are submitted.

A second kind, ``component`` (one connected component of the U-Net
foreground, i.e. the U-Net's own grouping), is switched on with
``CandidateParams.components``. On their own, components score almost as
well as claims; offering both to the selector gives the same held-out score
within noise. The main released configuration leaves them off, the second
one (``weights/manifest_components.json``) uses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from pycocotools import mask as mask_util

from . import config, preprocess, rle

#: Codes of the candidate kinds, as fed to the scorer.
KIND_CODES = {"claim": 0, "component": 1}


@dataclass
class CandidateParams:
    """Knobs of the candidate generator (lengths in full-resolution pixels)."""

    thresholds: tuple[float, ...] = (0.35, 0.5, 0.65)   # U-Net probability cuts
    grow: int = 6                 # detector masks are grown by this much before claiming
    min_area: int = 60            # candidates smaller than this are discarded
    min_det_score: float = 0.03   # detections below this confidence are ignored
    margin: int = 24              # context kept around a candidate for its features
    components: bool = False      # also emit connected components of the U-Net foreground


@dataclass
class Candidate:
    """One candidate instance: a mask stored as a crop, plus its features."""

    y0: int
    x0: int
    mask: np.ndarray               # boolean crop
    features: dict[str, float] = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def full_mask(self, size: int = config.IMAGE_SIZE) -> np.ndarray:
        full = np.zeros((size, size), dtype=bool)
        height, width = self.mask.shape
        full[self.y0:self.y0 + height, self.x0:self.x0 + width] = self.mask
        return full

    def to_rle(self) -> dict:
        return rle.encode(self.full_mask())


_KERNELS: dict[int, np.ndarray] = {}


def _disk_kernel(radius: int) -> np.ndarray:
    if radius not in _KERNELS:
        _KERNELS[radius] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
    return _KERNELS[radius]


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _context(detections: list[dict]) -> list[dict[str, float]]:
    """Features that describe a detection relative to the other detections.

    Detectors often report a long filament both as one object and as two or
    three pieces. These features tell the scorer which reading it is looking at:

        inside_better   largest share of this mask that lies inside a more
                        confident detection (near 1: a piece of a bigger object)
        covers_worse    how many less confident detections lie mostly inside
                        this one (it is the "whole" reading of those pieces)
        best_rival      confidence of the most confident detection that
                        overlaps this one at all
    """
    n = len(detections)
    if n == 0:
        return []
    rles = [d["rle"] for d in detections]
    # With iscrowd=1, pycocotools returns intersection / area(first argument).
    containment = np.asarray(mask_util.iou(rles, rles, [1] * n), dtype=np.float64)
    np.fill_diagonal(containment, 0.0)
    scores = np.array([d["score"] for d in detections])
    context = []
    for i in range(n):
        better = scores > scores[i]
        worse = scores < scores[i]
        rivals = containment[i] > 0.05
        context.append(dict(
            inside_better=float(containment[i, better].max()) if better.any() else 0.0,
            covers_worse=float((containment[worse, i] > 0.6).sum()),
            best_rival=float(scores[rivals].max()) if rivals.any() else 0.0,
        ))
    return context


def _detection_features(detection: dict | None, rank: int, det_area: float, context: dict | None) -> dict:
    """Features of the detection behind a candidate (zeros if there is none)."""
    if detection is None:
        return dict(det_score=0.0, det_votes=0.0, det_top=0.0, det_rank=float(rank), det_area=0.0,
                    inside_better=0.0, covers_worse=0.0, best_rival=0.0)
    return dict(
        det_score=detection["score"],
        det_votes=float(detection.get("votes", 1.0)),
        det_top=float(detection.get("top", detection["score"])),
        det_rank=float(rank),
        det_area=det_area,
        **context,
    )


def generate(
    probability: np.ndarray,
    detections: list[dict],
    channels: np.ndarray,
    disk: preprocess.Disk,
    params: CandidateParams | None = None,
) -> list[Candidate]:
    """Build every candidate of one frame.

    ``probability`` is the U-Net output (H, W); ``detections`` the fused
    detector output (dicts with ``rle``, ``score`` and optionally ``votes``
    and ``top``, see ``pipeline.merge_detections``); ``channels`` the cached
    (2, H, W) image channels, used for the contrast features.
    """
    params = params or CandidateParams()
    size = probability.shape[0]
    inside = preprocess.disk_mask(disk, size=size)
    flat = channels[1]
    detections = [d for d in detections if d["score"] >= params.min_det_score]
    context = _context(detections)
    frame = dict(
        n_detections=float(len(detections)),
        frame_foreground=float(((probability >= 0.5) & inside).sum()),
    )
    pad = params.grow + params.margin

    # Each detection once: its window, its mask and its grown mask inside that window.
    windows = []
    for detection in detections:
        x, y, w, h = (int(v) for v in mask_util.toBbox(detection["rle"]))
        y0, y1 = max(0, y - pad), min(size, y + h + pad)
        x0, x1 = max(0, x - pad), min(size, x + w + pad)
        det_mask = rle.decode(detection["rle"])[y0:y1, x0:x1]
        grown = cv2.dilate(det_mask.astype(np.uint8), _disk_kernel(params.grow)).astype(bool)
        grown &= inside[y0:y1, x0:x1]
        windows.append((y0, y1, x0, x1, det_mask, grown))

    candidates: list[Candidate] = []
    for threshold in params.thresholds:
        foreground = (probability >= threshold) & inside
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8), connectivity=8
        )
        component_area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        # For each component: the detection that claims most of it, and how many
        # confident detections have their claim mostly inside it.
        best_overlap = np.zeros(n_labels)
        best_detection = np.full(n_labels, -1)
        n_inside = np.zeros(n_labels)
        claims = []
        for index, (y0, y1, x0, x1, det_mask, grown) in enumerate(windows):
            local = labels[y0:y1, x0:x1]
            claim = grown & (local > 0)
            claim_area = int(claim.sum())
            ids, counts = (np.unique(local[claim], return_counts=True) if claim_area
                           else (np.zeros(0, int), np.zeros(0, int)))
            claims.append((claim, claim_area, ids, counts))
            if not claim_area:
                continue
            improved = counts > best_overlap[ids]
            best_overlap[ids[improved]] = counts[improved]
            best_detection[ids[improved]] = index
            main = ids[np.argmax(counts)]
            if detections[index]["score"] >= 0.15 and counts.max() >= 0.7 * claim_area:
                n_inside[main] += 1

        for index, (y0, y1, x0, x1, det_mask, grown) in enumerate(windows):
            claim, claim_area, ids, counts = claims[index]
            if claim_area < params.min_area:
                continue
            main = ids[np.argmax(counts)]
            candidate = _finish(claim, y0, x0, probability, flat, disk, det_mask, params)
            candidate.features.update(
                _detection_features(detections[index], index, float(det_mask.sum()), context[index]),
                kind=float(KIND_CODES["claim"]), threshold=threshold,
                component_share=float(counts.max() / component_area[main]),
                n_components=float(len(ids)), n_inside=float(n_inside[main]), **frame,
            )
            candidates.append(candidate)

        if not params.components:
            continue
        for label in range(1, n_labels):
            if component_area[label] < params.min_area:
                continue
            top, left = stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_LEFT]
            height, width = stats[label, cv2.CC_STAT_HEIGHT], stats[label, cv2.CC_STAT_WIDTH]
            y0, y1 = max(0, top - params.margin), min(size, top + height + params.margin)
            x0, x1 = max(0, left - params.margin), min(size, left + width + params.margin)
            mask = labels[y0:y1, x0:x1] == label
            owner = int(best_detection[label])
            if owner >= 0:
                # The owning detection's mask, cut to this component's window.
                det_full = rle.decode(detections[owner]["rle"])[y0:y1, x0:x1]
                det_features = _detection_features(
                    detections[owner], owner, float(windows[owner][4].sum()), context[owner]
                )
            else:
                det_full = np.zeros(mask.shape, dtype=bool)
                det_features = _detection_features(None, len(detections), 0.0, None)
            candidate = _finish(mask, y0, x0, probability, flat, disk, det_full, params)
            candidate.features.update(
                det_features, kind=float(KIND_CODES["component"]), threshold=threshold,
                component_share=1.0, n_components=1.0, n_inside=float(n_inside[label]), **frame,
            )
            candidates.append(candidate)
    return candidates


def _finish(
    mask: np.ndarray,
    y0: int,
    x0: int,
    probability: np.ndarray,
    flat: np.ndarray,
    disk: preprocess.Disk,
    det_mask: np.ndarray,
    params: CandidateParams,
) -> Candidate:
    """Turn a mask (a window at (y0, x0)) into a candidate with its own features."""
    height, width = mask.shape
    prob = probability[y0:y0 + height, x0:x0 + width]
    brightness = flat[y0:y0 + height, x0:x0 + width].astype(np.float32)
    area = int(mask.sum())
    mask_u8 = mask.astype(np.uint8)
    ring = cv2.dilate(mask_u8, _disk_kernel(6)).astype(bool) & ~mask
    core = cv2.erode(mask_u8, _disk_kernel(1), borderValue=0).astype(bool)

    inside_prob = prob[mask]
    ys, xs = np.nonzero(mask)
    centre_y, centre_x = ys.mean() + y0, xs.mean() + x0
    # Shape: the minimum-area rectangle gives a length and a breadth for an elongated object.
    points = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(points)
    length, breadth = max(side_a, side_b) + 1.0, min(side_a, side_b) + 1.0
    hull_area = cv2.contourArea(cv2.convexHull(points)) + 1.0
    overlap = float((mask & det_mask).sum())
    det_area = float(det_mask.sum())
    flat_inside = float(brightness[mask].mean())
    flat_ring = float(brightness[ring].mean()) if ring.any() else flat_inside

    features = dict(
        area=float(area),
        prob_mean=float(inside_prob.mean()),
        prob_p10=float(np.percentile(inside_prob, 10)),
        prob_max=float(inside_prob.max()),
        prob_ring=float(prob[ring].mean()) if ring.any() else 0.0,
        prob_core=float(prob[core].mean()) if core.any() else float(inside_prob.mean()),
        flat_inside=flat_inside,
        flat_ring=flat_ring,
        flat_min=float(np.percentile(brightness[mask], 5)),
        flat_contrast=flat_ring - flat_inside,
        length=float(length),
        breadth=float(breadth),
        mean_width=float(area / length),
        solidity=float(area / hull_area),
        radius=float(np.hypot(centre_x - disk.cx, centre_y - disk.cy) / disk.radius),
        det_iou=overlap / (area + det_area - overlap + 1e-6),
        det_covered=overlap / (det_area + 1e-6),   # share of the detector mask inside the candidate
        in_det=overlap / area,                     # share of the candidate inside the detector mask
    )
    top, bottom, left, right = _bounding_box(mask)
    return Candidate(y0 + top, x0 + left, mask[top:bottom, left:right].copy(), features)


#: The order in which features are fed to the scorer.
FEATURE_NAMES = [
    "kind", "threshold",
    "det_score", "det_votes", "det_top", "det_rank", "det_area",
    "inside_better", "covers_worse", "best_rival",
    "component_share", "n_components", "n_inside",
    "n_detections", "frame_foreground",
    "area", "prob_mean", "prob_p10", "prob_max", "prob_ring", "prob_core",
    "flat_inside", "flat_ring", "flat_min", "flat_contrast",
    "length", "breadth", "mean_width", "solidity", "radius",
    "det_iou", "det_covered", "in_det",
]


def feature_matrix(candidates: list[Candidate]) -> np.ndarray:
    """Stack the features of many candidates into a float32 matrix, in ``FEATURE_NAMES`` order."""
    return np.array(
        [[c.features[name] for name in FEATURE_NAMES] for c in candidates], dtype=np.float32
    ).reshape(len(candidates), len(FEATURE_NAMES))
