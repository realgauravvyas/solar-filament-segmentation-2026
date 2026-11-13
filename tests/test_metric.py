"""The RLE implementation of the metric must agree with the organisers' dense one."""

import numpy as np

from filaseg import metric, rle
from filaseg.data import Observation, View


def _square(y0, x0, side, size=256):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y0 + side, x0:x0 + side] = True
    return mask


def _dense_pq(pred_masks, views):
    """Direct transcription of ``get_pq_score`` from the self-evaluation notebook."""
    iou_sum, tp, fp, fn = 0.0, 0, 0, 0
    for gt_masks in views:
        if not gt_masks:
            fp += len(pred_masks)
            continue
        if not pred_masks:
            fn += len(gt_masks)
            continue
        gt = np.stack(gt_masks).reshape(len(gt_masks), -1).astype(np.float64)
        pred = np.stack(pred_masks).reshape(len(pred_masks), -1).astype(np.float64)
        intersection = gt @ pred.T
        union = gt.sum(1)[:, None] + pred.sum(1)[None, :] - intersection
        iou = np.where(union == 0, 0.0, intersection / np.maximum(union, 1))
        hit = iou > 0.5
        iou_sum += iou[hit].sum()
        tp += hit.sum()
        fp += (hit.sum(0) == 0).sum()
        fn += (hit.sum(1) == 0).sum()
    return iou_sum / (tp + 0.5 * fp + 0.5 * fn)


def test_matches_dense_reference():
    views = [
        [_square(10, 10, 40), _square(100, 100, 30), _square(180, 20, 25)],
        [_square(12, 14, 40), _square(150, 150, 30)],
    ]
    pred = [_square(11, 11, 40), _square(104, 96, 30), _square(200, 200, 20)]
    expected = _dense_pq(pred, views)

    class _View(View):
        def __init__(self, masks):
            super().__init__("v")
            self._masks = masks

        def rles(self):
            return [rle.encode(m) for m in self._masks]

    observation = Observation("x", "x.jpeg", [_View(v) for v in views])
    total = metric.score_observation([rle.encode(m) for m in pred], observation)
    assert abs(total.pq - expected) < 1e-9


def test_empty_cases():
    gt = [rle.encode(_square(0, 0, 10))]
    assert metric.score_view([], gt).fn == 1
    assert metric.score_view(gt, []).fp == 1


def test_match_value_is_zero_below_threshold():
    gt = [[rle.encode(_square(0, 0, 40))]]
    barely = rle.encode(_square(0, 20, 40))      # IoU = 1/3
    exact = rle.encode(_square(0, 0, 40))
    values = metric.match_value([barely, exact], gt)
    assert values[0, 0] == 0.0 and values[1, 0] == 1.0
