"""Stage 9 (optional): what does each component contribute?

    python scripts/09_ablation.py --unets r34 --detectors m1536 s1280 m2048 --out results/ablation.json

Scores four simpler pipelines on the held-out folds, with the same networks
and the same scoring rules as the full method:

    unet_components   connected components of the U-Net foreground, ranked by
                      mean probability (no detector at all)
    detector_masks    the fused detector masks as they are (no U-Net)
    claim_one_view    U-Net pixels inside each detection of the first detector,
                      without mirror views, ranked by detector confidence
    claim_fused       the same with all detectors and mirror views fused

Each variant gets its best score threshold, which favours the simple
variants; the full method's number comes from ``05_fit_selector.py``.

With ``--predictions`` (the out-of-fold predictions of the full method) it
also measures how sensitive the score is to the outline: every predicted
instance is shrunk or grown by one pixel and the file is scored again.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import candidates as cand  # noqa: E402
from filaseg import config, data, metric, pipeline, preprocess, rle, submission  # noqa: E402

MIN_AREA = 60
MAX_OVERLAP = 0.25
_STATE = {}


def _init(unets, detectors, observations):
    _STATE["unets"], _STATE["detectors"] = unets, detectors
    _STATE["observations"] = {o.stem: o for o in observations}


def _greedy(masks_and_scores, view_rles, size=config.IMAGE_SIZE):
    """Non-overlapping greedy acceptance by decreasing score; returns (score, matched IoUs) records."""
    occupied = np.zeros((size, size), dtype=bool)
    records = []
    for score, mask in sorted(masks_and_scores, key=lambda item: -item[0]):
        area = mask.sum()
        if area < MIN_AREA or (mask & occupied).sum() > MAX_OVERLAP * area:
            continue
        mask = mask & ~occupied
        if mask.sum() < MIN_AREA:
            continue
        occupied |= mask
        records.append((score, metric.match_value([rle.encode(mask)], view_rles)[0]))
    return records


def _frame(task):
    stem, fold = task
    observation = _STATE["observations"][stem]
    view_rles = [view.rles() for view in observation.views]
    disk = preprocess.load_disks("train")[stem]
    inside = preprocess.disk_mask(disk)
    full = pipeline.Recipe(unets=_STATE["unets"], detectors=_STATE["detectors"])
    single = pipeline.Recipe(unets=_STATE["unets"], detectors=_STATE["detectors"][:1], flips=1)
    probability, fused = pipeline.frame_inputs(full, stem, fold)
    _, one_view = pipeline.frame_inputs(single, stem, fold)
    foreground = (probability >= 0.5) & inside

    out = {}
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= MIN_AREA:
            mask = labels == label
            components.append((float(probability[mask].mean()), mask))
    out["unet_components"] = _greedy(components, view_rles)
    out["detector_masks"] = _greedy([(d["score"], rle.decode(d["rle"])) for d in fused], view_rles)
    kernel = cand._disk_kernel(6)
    for name, detections in (("claim_one_view", one_view), ("claim_fused", fused)):
        claims = [
            (d["score"], cv2.dilate(rle.decode(d["rle"]).astype(np.uint8), kernel).astype(bool) & foreground)
            for d in detections
        ]
        out[name] = _greedy(claims, view_rles)
    return out, sum(len(v) for v in view_rles)


def _outline_frame(task):
    """Score one frame's predictions as they are, shrunk by one pixel and grown by one pixel."""
    stem, rles = task
    view_rles = [view.rles() for view in _STATE["observations"][stem].views]
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    masks = [rle.decode(r) for r in rles]
    totals = {}
    for name, operation in (("as_predicted", None), ("minus_1px", cv2.erode), ("plus_1px", cv2.dilate)):
        occupied = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=bool)
        changed = []
        for mask in masks:
            new = mask if operation is None else operation(mask.astype(np.uint8), cross).astype(bool)
            new = new & ~occupied          # instances must stay disjoint after growing
            if not new.any():
                new = mask & ~occupied
            occupied |= new
            changed.append(rle.encode(new))
        total = metric.Totals()
        for gt in view_rles:
            total = total + metric.score_view(changed, gt)
        totals[name] = total
    return totals


def _best(records, n_drawings):
    scores = np.array([r[0] for r in records])
    best = None
    for threshold in np.quantile(scores, np.linspace(0.0, 0.95, 60)):
        iou_sum, tp, fp = 0.0, 0, 0
        for score, matched in records:
            if score >= threshold:
                hit = matched > 0
                iou_sum += matched[hit].sum()
                tp += int(hit.sum())
                fp += int((~hit).sum())
        total = metric.Totals(iou_sum, tp, fp, n_drawings - tp)
        if best is None or total.pq > best[1].pq:
            best = (float(threshold), total)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--unets", nargs="+", required=True)
    parser.add_argument("--detectors", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(config.N_FOLDS)))
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions", default=None, help="out-of-fold predictions for the outline test")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    observations = data.load_observations()
    fold_of = data.assign_folds(observations)
    tasks = [(o.stem, fold_of[o.stem]) for o in observations if fold_of[o.stem] in args.folds]
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.unets, args.detectors, observations)) as pool:
        results = list(pool.map(_frame, tasks, chunksize=4))
        outline = None
        if args.predictions:
            predicted = submission.read(Path(args.predictions))
            outline = list(pool.map(
                _outline_frame, [(stem, predicted.get(stem, [])) for stem, _ in tasks], chunksize=4
            ))
    n_drawings = sum(r[1] for r in results)

    report = {}
    for name in ("unet_components", "detector_masks", "claim_one_view", "claim_fused"):
        records = [record for r in results for record in r[0][name]]
        threshold, total = _best(records, n_drawings)
        report[name] = {"threshold": threshold, "pq": total.pq, "dq": total.dq, "sq": total.sq}
        print(f"{name:16s} {total}")
    if outline is not None:
        for name in ("as_predicted", "minus_1px", "plus_1px"):
            total = sum((frame[name] for frame in outline), metric.Totals())
            report[f"outline_{name}"] = {"pq": total.pq, "dq": total.dq, "sq": total.sq}
            print(f"outline {name:13s} {total}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
