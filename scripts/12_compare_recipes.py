"""Stage 12 (optional): is recipe B better than recipe A, out-of-fold?

    python scripts/12_compare_recipes.py --base main --others main_comp main_2048 --out results/comparison.json

Every recipe has out-of-fold predictions for all training frames
(``outputs/candidates/<name>/oof_predictions.csv``, written by stage 5). For each
other recipe the script reports the PQ difference to the base recipe, a 95 %
interval from a paired bootstrap (the same resampled frames score both
recipes), and the share of resamples in which the other recipe wins.

The rule used in this project: a variant replaces the released configuration
only if the interval excludes zero. Of the several variants that were tried,
the best-looking one is biased upwards by selection, so a point estimate alone
is not enough.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config, data, metric, submission  # noqa: E402


def per_frame_counts(name: str, observations) -> np.ndarray:
    """(iou_sum, tp, fp, fn) of every frame for the out-of-fold predictions of one recipe."""
    predictions = submission.read(config.out_root() / "candidates" / name / "oof_predictions.csv")
    rows = []
    for observation in observations:
        total = metric.score_observation(predictions.get(observation.stem, []), observation)
        rows.append((total.iou_sum, total.tp, total.fp, total.fn))
    return np.array(rows, dtype=np.float64)


def pq(table: np.ndarray, index: np.ndarray) -> float:
    iou_sum, tp, fp, fn = table[index].sum(axis=0)
    return float(iou_sum / (tp + 0.5 * fp + 0.5 * fn))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--others", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    observations = data.load_observations()
    n = len(observations)
    everything = np.arange(n)
    rng = np.random.default_rng(0)
    resamples = [rng.integers(n, size=n) for _ in range(args.resamples)]

    base = per_frame_counts(args.base, observations)
    report = {"base": {"name": args.base, "pq": pq(base, everything)}, "others": {}}
    print(f"{args.base}: PQ {report['base']['pq']:.4f}")
    for name in args.others:
        other = per_frame_counts(name, observations)
        samples = np.array([pq(other, b) - pq(base, b) for b in resamples])
        low, high = np.percentile(samples, [2.5, 97.5])
        entry = {
            "pq": pq(other, everything), "difference": pq(other, everything) - report["base"]["pq"],
            "ci95": [float(low), float(high)], "p_better": float((samples > 0).mean()),
        }
        report["others"][name] = entry
        print(f"{name}: PQ {entry['pq']:.4f}  difference {entry['difference']:+.4f}  "
              f"95% interval [{low:+.4f}, {high:+.4f}]  P(better) {entry['p_better']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
