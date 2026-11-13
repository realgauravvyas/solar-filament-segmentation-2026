"""The competition's Panoptic Quality, reimplemented on run-length masks.

The organisers' self-evaluation notebook decodes every mask to a dense
2048 x 2048 layer. This module computes exactly the same counts from RLEs,
which is some hundred times faster and lets us score thousands of settings.

Scoring rules, taken from that notebook:

* each view (annotator drawing) of an observation is scored separately against
  the full set of predictions for that observation;
* a (drawing, prediction) pair with IoU > 0.5 is a true positive;
* a prediction with no such pair in a view is a false positive for that view;
* a drawing with no such pair is a false negative;
* ``PQ = sum(IoU of TP pairs) / (TP + 0.5 FP + 0.5 FN)``, pooled over all views.

Predictions inside one observation never overlap in this pipeline, so at most
one prediction can exceed IoU 0.5 with a given drawing and the pooled count is
the standard one-to-one PQ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, rle
from .data import Observation


@dataclass
class Totals:
    """Pooled counts; add them up over views, then read ``pq``."""

    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Totals") -> "Totals":
        return Totals(
            self.iou_sum + other.iou_sum, self.tp + other.tp, self.fp + other.fp, self.fn + other.fn
        )

    @property
    def denominator(self) -> float:
        return self.tp + 0.5 * self.fp + 0.5 * self.fn

    @property
    def pq(self) -> float:
        return self.iou_sum / self.denominator if self.denominator > 0 else 0.0

    @property
    def dq(self) -> float:
        """Detection quality: an F1 score over filaments."""
        return self.tp / self.denominator if self.denominator > 0 else 0.0

    @property
    def sq(self) -> float:
        """Segmentation quality: mean IoU of the matched pairs."""
        return self.iou_sum / self.tp if self.tp > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"PQ {self.pq:.4f}  DQ {self.dq:.4f}  SQ {self.sq:.4f}  "
            f"TP {self.tp}  FP {self.fp}  FN {self.fn}"
        )


def score_view(pred: list[dict], gt: list[dict]) -> Totals:
    """Score the predictions of one observation against one view."""
    if not gt:
        return Totals(fp=len(pred))
    if not pred:
        return Totals(fn=len(gt))
    ious = rle.iou_matrix(pred, gt)
    hit = ious > config.IOU_MATCH
    return Totals(
        iou_sum=float(ious[hit].sum()),
        tp=int(hit.sum()),
        fp=int((~hit.any(axis=1)).sum()),
        fn=int((~hit.any(axis=0)).sum()),
    )


def score_observation(pred: list[dict], observation: Observation) -> Totals:
    """Score one observation against each of its views and pool the counts."""
    total = Totals()
    for view in observation.views:
        total = total + score_view(pred, view.rles())
    return total


def score_dataset(
    predictions: dict[str, list[dict]], observations: list[Observation]
) -> Totals:
    """Pooled score over observations; a missing stem means "no predictions"."""
    total = Totals()
    for observation in observations:
        total = total + score_observation(predictions.get(observation.stem, []), observation)
    return total


def match_value(pred: list[dict], view_rles: list[list[dict]]) -> np.ndarray:
    """Per-prediction, per-view matched IoU.

    Returns an array of shape (len(pred), n_views). Entry ``[i, v]`` is the IoU
    of prediction ``i`` with the drawing it matches in view ``v`` (IoU > 0.5),
    or 0 when it matches none. Averaged over views this is the quantity the
    candidate scorer learns to predict (see ``selection.py``).
    """
    values = np.zeros((len(pred), len(view_rles)), dtype=np.float64)
    for column, gt in enumerate(view_rles):
        if not pred or not gt:
            continue
        ious = rle.iou_matrix(pred, gt)
        best = ious.max(axis=1)
        values[:, column] = np.where(best > config.IOU_MATCH, best, 0.0)
    return values
