"""Stage 8: evaluate a prediction file against the training annotations.

    python scripts/08_evaluate.py --predictions results/oof_predictions.csv --out results

The predictions must be out-of-fold (stage 5 writes such a file), otherwise
the numbers are meaningless. Reports, with the organisers' scoring rules:

* PQ, DQ, SQ with a 95 % bootstrap interval (frames resampled 2000 times);
* the same split by number of annotators and by GONG site;
* on frames drawn by three annotators, how well the third annotator's drawing
  is predicted by one annotator, by the filaments on which the two other
  annotators agree, and by the method;
* the IoU and Dice distributions of all overlapping (drawing, prediction)
  pairs and the one-to-many / many-to-one relations, i.e. the plots of the
  organisers' self-evaluation notebook.

Writes ``metrics.json`` and three PNG figures into ``--out``.
"""

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import data, metric, rle, submission  # noqa: E402


def bootstrap_interval(per_frame: list[metric.Totals], n_resamples: int = 2000, seed: int = 0):
    """95 % interval of PQ, DQ and SQ when the frames are resampled with replacement."""
    rng = np.random.default_rng(seed)
    table = np.array([[t.iou_sum, t.tp, t.fp, t.fn] for t in per_frame], dtype=np.float64)
    samples = []
    for _ in range(n_resamples):
        iou_sum, tp, fp, fn = table[rng.integers(len(table), size=len(table))].sum(axis=0)
        denominator = tp + 0.5 * fp + 0.5 * fn
        samples.append((iou_sum / denominator, tp / denominator, iou_sum / max(tp, 1.0)))
    low, high = np.percentile(np.array(samples), [2.5, 97.5], axis=0)
    return {name: [float(low[i]), float(high[i])] for i, name in enumerate(("pq", "dq", "sq"))}


def summarise(per_frame: list[metric.Totals]) -> dict:
    total = sum(per_frame, metric.Totals())
    return {
        "frames": len(per_frame), "pq": total.pq, "dq": total.dq, "sq": total.sq,
        "tp": total.tp, "fp": total.fp, "fn": total.fn, "ci95": bootstrap_interval(per_frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="evaluate only these folds")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    predictions = submission.read(Path(args.predictions))
    observations = data.load_observations()
    fold_of = data.assign_folds(observations)

    per_frame, by_views, by_site, by_fold = [], defaultdict(list), defaultdict(list), defaultdict(list)
    ious, dices, per_drawing, per_prediction = [], [], [], []
    human = []
    # Frames drawn by three annotators: how well is the third annotator predicted by one
    # annotator, by the filaments two annotators agree on, and by the method?
    third = {"one_annotator": [], "two_annotators_agree": [], "method": []}
    if args.folds is not None:
        observations = [o for o in observations if fold_of[o.stem] in args.folds]
    for observation in observations:
        pred = predictions.get(observation.stem, [])
        total = metric.score_observation(pred, observation)
        per_frame.append(total)
        by_views[len(observation.views)].append(total)
        by_site[observation.stem[-2:]].append(total)
        by_fold[fold_of[observation.stem]].append(total)
        view_rles = [view.rles() for view in observation.views]
        for gt in view_rles:
            if not gt:
                per_prediction.extend([0] * len(pred))
                continue
            if not pred:
                per_drawing.extend([0] * len(gt))
                continue
            iou = rle.iou_matrix(pred, gt)
            area_p = mask_util.area(pred).astype(np.float64)[:, None]
            area_g = mask_util.area(gt).astype(np.float64)[None, :]
            intersection = iou * (area_p + area_g) / (1.0 + iou)
            dice = 2.0 * intersection / (area_p + area_g)
            overlapping = iou > 0
            ious.extend(iou[overlapping].tolist())
            dices.extend(dice[overlapping].tolist())
            per_drawing.extend(overlapping.sum(axis=0).tolist())
            per_prediction.extend(overlapping.sum(axis=1).tolist())
        # Reference: each annotator scored against the other annotators of the same frame.
        frame_human = metric.Totals()
        for i, as_prediction in enumerate(view_rles):
            for j, as_truth in enumerate(view_rles):
                if i != j:
                    frame_human = frame_human + metric.score_view(as_prediction, as_truth)
        if len(view_rles) > 1:
            human.append(frame_human)
        if len(view_rles) == 3:
            frame_third = {name: metric.Totals() for name in third}
            for a, b, c in itertools.permutations(range(3)):
                first, second, target = view_rles[a], view_rles[b], view_rles[c]
                agreed = []
                if first and second:
                    agreed = [r for r, hit in zip(first, (rle.iou_matrix(first, second) > 0.5).any(axis=1)) if hit]
                frame_third["one_annotator"] = frame_third["one_annotator"] + metric.score_view(first, target)
                frame_third["two_annotators_agree"] = (
                    frame_third["two_annotators_agree"] + metric.score_view(agreed, target)
                )
                frame_third["method"] = frame_third["method"] + metric.score_view(pred, target)
            for name in third:
                third[name].append(frame_third[name])

    report = {
        "overall": summarise(per_frame),
        "by_number_of_annotators": {str(k): summarise(v) for k, v in sorted(by_views.items())},
        "by_site": {k: summarise(v) for k, v in sorted(by_site.items())},
        "by_fold": {str(k): summarise(v) for k, v in sorted(by_fold.items())},
        "annotator_vs_annotator": summarise(human),
        "predicting_the_third_annotator": {name: summarise(frames) for name, frames in third.items()},
        "pairs": {
            "mean_iou": float(np.mean(ious)), "mean_dice": float(np.mean(dices)),
            "drawings_hit_by": dict(Counter(per_drawing)),
            "predictions_hitting": dict(Counter(per_prediction)),
        },
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    overall = report["overall"]
    print(f"PQ {overall['pq']:.4f}  95% CI {overall['ci95']['pq'][0]:.4f}-{overall['ci95']['pq'][1]:.4f}  "
          f"DQ {overall['dq']:.4f}  SQ {overall['sq']:.4f}  ({overall['frames']} frames)")
    for key, value in report["by_number_of_annotators"].items():
        print(f"  {key} annotator(s): PQ {value['pq']:.4f}  ({value['frames']} frames)")
    print(f"  annotator vs annotator: PQ {report['annotator_vs_annotator']['pq']:.4f}")
    for name, value in report["predicting_the_third_annotator"].items():
        print(f"  third annotator predicted by {name}: PQ {value['pq']:.4f}  ({value['frames']} frames)")

    # The figures are printed about 4 cm wide in the report, hence the large fonts.
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    for values, name, label in ((ious, "iou", "IoU"), (dices, "dice", "Dice")):
        plt.figure(figsize=(4.0, 3.0))
        plt.hist(values, bins=np.linspace(0, 1, 41), weights=np.full(len(values), 1 / len(values)),
                 color="#1e3a8a", alpha=0.8, edgecolor="white", linewidth=0.3)
        plt.axvline(np.mean(values), color="crimson", linestyle="--", label=f"mean {np.mean(values):.2f}")
        if name == "iou":
            plt.axvline(0.5, color="black", linestyle=":", label="match threshold")
        plt.xlabel(f"{label} of overlapping pairs")
        plt.ylabel("fraction of pairs")
        plt.legend(fontsize=9, frameon=False, loc="upper left")
        plt.tight_layout()
        plt.savefig(out / f"{name}_distribution.png", dpi=300)
        plt.close()

    plt.figure(figsize=(4.0, 3.0))
    drawing_counts, prediction_counts = Counter(per_drawing), Counter(per_prediction)
    degrees = [d for d in sorted(set(drawing_counts) | set(prediction_counts)) if d <= 3]
    width = 0.4
    plt.bar([d - width / 2 for d in degrees], [drawing_counts.get(d, 0) for d in degrees], width,
            label="drawings", color="#15803d", alpha=0.85)
    plt.bar([d + width / 2 for d in degrees], [prediction_counts.get(d, 0) for d in degrees], width,
            label="predictions", color="#1d4ed8", alpha=0.85)
    plt.xticks(degrees, [str(d) if d < 3 else "3" for d in degrees])
    plt.xlabel("overlapping partners in a view")
    plt.ylabel("count")
    plt.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out / "relations.png", dpi=300)
    plt.close()
    print(f"wrote {out / 'metrics.json'} and figures")


if __name__ == "__main__":
    main()
