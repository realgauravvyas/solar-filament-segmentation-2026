"""Stage 6: score the test candidates, select, and write the submission file.

    python scripts/06_make_submission.py --name main

Reads ``outputs/candidates/<name>/test.pkl``, ``scorer.json`` and
``selection.json`` and writes ``outputs/submissions/<name>.csv``.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config, data, selection, submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="override the stored value threshold")
    parser.add_argument("--out", default=None, help="output CSV (default: outputs/submissions/<name>.csv)")
    args = parser.parse_args()

    folder = config.out_root() / "candidates" / args.name
    with open(folder / "test.pkl", "rb") as handle:
        frames = pickle.load(handle)["frames"]
    with open(folder / "selection.json", encoding="utf-8") as handle:
        stored = json.load(handle)
    scorer = selection.load_scorer(folder / "scorer.json")
    params = selection.SelectionParams(
        value_threshold=args.threshold if args.threshold is not None else stored["value_threshold"],
        max_overlap=stored["max_overlap"],
    )

    predictions = {}
    for stem, frame in frames.items():
        values = selection.predict_values(scorer, frame["candidates"])
        predictions[stem] = selection.select(frame["candidates"], values, params)

    out_path = Path(args.out) if args.out else config.out_root() / "submissions" / f"{args.name}.csv"
    submission.write(predictions, out_path)
    stats = submission.check(out_path, data.list_test_stems())
    print(f"wrote {out_path}  (value threshold {params.value_threshold:.2f})")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
