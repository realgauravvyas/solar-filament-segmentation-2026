"""Stage 11 (optional): is the selection threshold robust to who annotated the frame?

    python scripts/11_threshold_by_group.py --name main --out results/threshold_by_group.json

The test frames may have been drawn by other annotators than the training
frames. MAGFiLO's annotator codes start with a two-digit group number, so for
every annotation group this script finds the value threshold that would
maximise its out-of-fold PQ, and how much PQ the single pooled threshold
loses against it. The candidate values are cross-fitted as in stage 5.
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config, data, metric, rle, selection  # noqa: E402
from filaseg.candidates import feature_matrix  # noqa: E402

THRESHOLDS = np.round(np.arange(0.12, 0.42, 0.02), 2)


def pq_curve(records, n_drawings):
    """PQ at every threshold from (predicted value, matched IoU in this view) records."""
    values = np.array([r[0] for r in records])
    matched = np.array([r[1] for r in records])
    curve = []
    for threshold in THRESHOLDS:
        kept = matched[values >= threshold]
        tp = int((kept > 0).sum())
        curve.append(metric.Totals(float(kept.sum()), tp, len(kept) - tp, n_drawings - tp).pq)
    return np.array(curve)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    folder = config.out_root() / "candidates" / args.name
    with open(folder / "train.pkl", "rb") as handle:
        frames = pickle.load(handle)["frames"]
    with open(folder / "selection.json", encoding="utf-8") as handle:
        chosen = json.load(handle)["value_threshold"]
    observations = {o.stem: o for o in data.load_observations() if o.stem in frames}
    stems = sorted(frames)
    features = np.concatenate([feature_matrix(frames[s]["candidates"]) for s in stems])
    values = np.concatenate([frames[s]["values"] for s in stems])
    groups = np.concatenate([np.full(len(frames[s]["values"]), i) for i, s in enumerate(stems)])
    predicted = selection.cross_fitted_values(features, values, groups)
    split = np.cumsum([0] + [len(frames[s]["values"]) for s in stems])

    loose = selection.SelectionParams(value_threshold=-np.inf)
    records, n_drawings, n_views = defaultdict(list), Counter(), Counter()
    for index, stem in enumerate(stems):
        frame_values = predicted[split[index]:split[index + 1]]
        accepted = selection.select_indexed(frames[stem]["candidates"], frame_values, loose) if len(frame_values) else []
        masks = [rle.encode(mask) for _, mask in accepted]
        accepted_values = [frame_values[k] for k, _ in accepted]
        for view in observations[stem].views:
            group, gt = view.view_id[:2], view.rles()
            n_drawings[group] += len(gt)
            n_views[group] += 1
            matched = np.zeros(len(masks))
            if masks and gt:
                best = rle.iou_matrix(masks, gt).max(axis=1)
                matched = np.where(best > config.IOU_MATCH, best, 0.0)
            records[group].extend(zip(accepted_values, matched))

    at_chosen = int(np.argmin(np.abs(THRESHOLDS - chosen)))
    report = {"chosen_threshold": chosen, "thresholds": THRESHOLDS.tolist(), "groups": {}}
    for group in sorted(records):
        curve = pq_curve(records[group], n_drawings[group])
        best = int(np.argmax(curve))
        report["groups"][group] = {
            "views": n_views[group], "drawings_per_view": n_drawings[group] / n_views[group],
            "best_threshold": float(THRESHOLDS[best]), "pq_best": float(curve[best]),
            "pq_at_chosen": float(curve[at_chosen]), "loss": float(curve[best] - curve[at_chosen]),
        }
        r = report["groups"][group]
        print(f"group {group}: {r['views']:4d} views, best threshold {r['best_threshold']:.2f}, "
              f"PQ {r['pq_best']:.4f}; at {chosen:.2f}: {r['pq_at_chosen']:.4f} (loss {r['loss']:.4f})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
