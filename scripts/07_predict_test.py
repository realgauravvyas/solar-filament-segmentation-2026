"""Inference only: rebuild the submission from the weights stored in the repository.

    python scripts/07_predict_test.py

Needs the competition's test images (see README, "Data") and a GPU. Nothing is
trained. Writes ``outputs/submissions/submission.csv``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config, reproduce  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--weights", default=str(config.REPO_ROOT / "weights"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_csv = Path(args.out) if args.out else config.out_root() / "submissions" / "submission.csv"
    stats = reproduce.make_submission(Path(args.weights), out_csv, device=args.device)
    print(f"wrote {out_csv}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
