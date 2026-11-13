"""Stage 5: fit the candidate scorer and choose the selection threshold.

    python scripts/05_fit_selector.py --name main

Reads ``outputs/candidates/<name>/train.pkl`` (out-of-fold candidates with
their targets) and reports, all on held-out frames:

* a baseline that ranks candidates by detector confidence alone;
* the learned scorer, cross-fitted so that every frame is scored by a model
  that was not trained on it;
* the PQ of each as a function of the selection threshold.

It then writes ``oof_predictions.csv`` (the cross-fitted selection of every
training frame, in submission format), fits the final scorer on all
candidates and writes ``scorer.json`` (the LightGBM models) and ``selection.json``
(threshold and report).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config, data, metric, rle, selection, submission  # noqa: E402
from filaseg.candidates import FEATURE_NAMES, KIND_CODES, feature_matrix  # noqa: E402


def greedy_records(frames, observations, predicted, params):
    """Run the greedy selection once with no value threshold.

    Because candidates are visited by decreasing value, the selection at any
    threshold is a prefix of this one. Returns, per accepted instance, its
    predicted value and its matched IoU in every view (0 = false positive
    there), plus the total number of drawings, so PQ can be read off for every
    threshold without redoing the selection.
    """
    records = []  # (value, matched IoUs per view)
    n_drawings = 0
    loose = selection.SelectionParams(
        value_threshold=-np.inf, max_overlap=params.max_overlap, min_area=params.min_area
    )
    for stem, frame in frames.items():
        observation = observations[stem]
        view_rles = [view.rles() for view in observation.views]
        n_drawings += sum(len(v) for v in view_rles)
        values = predicted[stem]
        if not len(values):
            continue
        accepted = selection.select_indexed(frame["candidates"], values, loose)
        matched = metric.match_value([rle.encode(mask) for _, mask in accepted], view_rles)
        records.extend((values[index], row) for (index, _), row in zip(accepted, matched))
    return records, n_drawings


def curve(records, n_drawings, thresholds):
    """PQ, DQ, SQ at each threshold from the greedy records."""
    rows = []
    values = np.array([r[0] for r in records])
    for threshold in thresholds:
        keep = values >= threshold
        iou_sum, tp, fp = 0.0, 0, 0
        for (value, matched), kept in zip(records, keep):
            if not kept:
                continue
            hits = matched > 0
            iou_sum += matched[hits].sum()
            tp += int(hits.sum())
            fp += int((~hits).sum())
        total = metric.Totals(iou_sum, tp, fp, n_drawings - tp)
        rows.append((threshold, total))
    return rows


def best(rows):
    return max(rows, key=lambda row: row[1].pq)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-overlap", type=float, default=0.25)
    args = parser.parse_args()

    folder = config.out_root() / "candidates" / args.name
    with open(folder / "train.pkl", "rb") as handle:
        frames = pickle.load(handle)["frames"]
    observations = {o.stem: o for o in data.load_observations() if o.stem in frames}
    stems = sorted(frames)
    params = selection.SelectionParams(max_overlap=args.max_overlap)

    features = np.concatenate([feature_matrix(frames[s]["candidates"]) for s in stems])
    values = np.concatenate([frames[s]["values"] for s in stems])
    groups = np.concatenate([np.full(len(frames[s]["values"]), i) for i, s in enumerate(stems)])
    print(f"{len(stems)} frames, {len(values)} candidates, mean value {values.mean():.3f}, "
          f"{(values > 0).mean():.1%} match at least one view")

    report = {}

    # Ceiling of the candidate generator: the share of drawings that some candidate matches.
    matched, drawn = 0, 0
    for stem in stems:
        candidate_rles = [c.to_rle() for c in frames[stem]["candidates"]]
        for view in observations[stem].views:
            gt = view.rles()
            drawn += len(gt)
            if gt and candidate_rles:
                matched += int((rle.iou_matrix(candidate_rles, gt).max(axis=0) > config.IOU_MATCH).sum())
    report["candidate_recall"] = matched / drawn
    print(f"{matched / drawn:.1%} of the {drawn} drawings are matched by at least one candidate")

    # Baseline: U-Net pixels claimed by a detection, ranked by detector confidence.
    threshold = features[:, FEATURE_NAMES.index("threshold")]
    det_score = features[:, FEATURE_NAMES.index("det_score")]
    kind = features[:, FEATURE_NAMES.index("kind")]
    baseline = np.where((kind == KIND_CODES["claim"]) & (np.abs(threshold - 0.5) < 1e-6), det_score, -1.0)
    split = np.cumsum([0] + [len(frames[s]["values"]) for s in stems])
    as_dict = lambda array: {s: array[split[i]:split[i + 1]] for i, s in enumerate(stems)}  # noqa: E731
    records, n_drawings = greedy_records(frames, observations, as_dict(baseline), params)
    rows = curve(records, n_drawings, np.arange(0.05, 0.75, 0.05))
    threshold_b, total_b = best(rows)
    print(f"baseline (detector confidence):  best at {threshold_b:.2f}:  {total_b}")
    report["baseline"] = {"threshold": float(threshold_b), "pq": total_b.pq, "dq": total_b.dq, "sq": total_b.sq}

    # Oracle: rank by the true value; an upper bound for any scorer on these candidates.
    records, n_drawings = greedy_records(frames, observations, as_dict(values), params)
    rows = curve(records, n_drawings, np.arange(0.02, 0.5, 0.02))
    threshold_o, total_o = best(rows)
    print(f"oracle (true value):             best at {threshold_o:.2f}:  {total_o}")
    report["oracle"] = {"threshold": float(threshold_o), "pq": total_o.pq}

    # Learned scorer, cross-fitted over frames.
    predicted = selection.cross_fitted_values(features, values, groups)
    records, n_drawings = greedy_records(frames, observations, as_dict(predicted), params)
    rows = curve(records, n_drawings, np.arange(0.06, 0.46, 0.02))
    for threshold_s, total in rows:
        print(f"  scorer threshold {threshold_s:.2f}:  {total}")
    threshold_s, total_s = best(rows)
    print(f"learned scorer (cross-fitted):   best at {threshold_s:.2f}:  {total_s}")
    report["scorer"] = {"threshold": float(threshold_s), "pq": total_s.pq, "dq": total_s.dq, "sq": total_s.sq,
                        "tp": total_s.tp, "fp": total_s.fp, "fn": total_s.fn}

    # Out-of-fold predictions at the chosen threshold, in submission format: every frame is
    # predicted by networks and by a scorer that never saw it. Stage 8 evaluates this file.
    chosen = selection.SelectionParams(value_threshold=float(threshold_s), max_overlap=args.max_overlap)
    predicted_by_stem = as_dict(predicted)
    oof = {s: selection.select(frames[s]["candidates"], predicted_by_stem[s], chosen) for s in stems}
    submission.write(oof, folder / "oof_predictions.csv")
    print(f"wrote {folder / 'oof_predictions.csv'}")

    scorer = selection.fit_scorer(features, values)
    selection.save_scorer(scorer, folder / "scorer.json")
    gain = np.mean([model.feature_importance("gain") for model in scorer], axis=0)
    ranking = sorted(zip(FEATURE_NAMES, gain / gain.sum()), key=lambda item: -item[1])
    print("feature importance (gain): " + ", ".join(f"{n} {g:.2f}" for n, g in ranking[:10]))
    report["feature_importance"] = {n: float(g) for n, g in ranking}
    with open(folder / "selection.json", "w", encoding="utf-8") as handle:
        json.dump({"value_threshold": float(threshold_s), "max_overlap": args.max_overlap, "report": report},
                  handle, indent=2)
    print(f"wrote {folder / 'scorer.json'} and selection.json")


if __name__ == "__main__":
    main()
