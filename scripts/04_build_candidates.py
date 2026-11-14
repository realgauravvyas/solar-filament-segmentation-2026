"""Stage 4: turn stored network outputs into candidate instances.

    python scripts/04_build_candidates.py --name main --unets r34 rx50 --detectors m1536

For every training frame the out-of-fold networks are used and each candidate
gets its training target (matched IoU averaged over the frame's views). For
every test frame the networks of all folds are averaged. The result is one
pickle per subset under ``outputs/candidates/<name>/``.
"""

import argparse
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config, data, pipeline  # noqa: E402

_STATE = {}


def _init(recipe, observations):
    _STATE["recipe"] = recipe
    _STATE["observations"] = {o.stem: o for o in observations}


def _train_frame(task):
    stem, fold = task
    candidates = pipeline.frame_candidates(_STATE["recipe"], stem, fold)
    values = pipeline.candidate_values(candidates, _STATE["observations"][stem])
    return stem, candidates, values


def _test_frame(stem):
    return stem, pipeline.frame_candidates(_STATE["recipe"], stem, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--name", required=True, help="name of this recipe (output folder)")
    parser.add_argument("--unets", nargs="+", required=True)
    parser.add_argument("--detectors", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(config.N_FOLDS)))
    parser.add_argument("--subsets", nargs="+", default=["train", "test"])
    parser.add_argument("--flips", type=int, default=None, help="use only the first N detector views")
    parser.add_argument("--det-suffix", default="", help="read detections from det/<name>_f<k><suffix>/")
    parser.add_argument("--components", action="store_true",
                        help="also offer whole connected components of the U-Net foreground as candidates")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    recipe = pipeline.Recipe(
        unets=args.unets, detectors=args.detectors, folds=args.folds,
        flips=args.flips, det_suffix=args.det_suffix,
        params=pipeline.cand.CandidateParams(components=args.components),
    )
    observations = data.load_observations()
    fold_of = data.assign_folds(observations)
    out_dir = config.out_root() / "candidates" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(recipe, observations)) as pool:
        if "train" in args.subsets:
            started = time.time()
            tasks = [(o.stem, fold_of[o.stem]) for o in observations if fold_of[o.stem] in args.folds]
            frames = {}
            for index, (stem, candidates, values) in enumerate(pool.map(_train_frame, tasks, chunksize=4), 1):
                frames[stem] = {"candidates": candidates, "values": values, "fold": fold_of[stem]}
                if index % 50 == 0:
                    print(f"  train {index}/{len(tasks)}  {time.time() - started:.0f} s", flush=True)
            with open(out_dir / "train.pkl", "wb") as handle:
                pickle.dump({"recipe": recipe, "frames": frames}, handle)
            total = sum(len(f["candidates"]) for f in frames.values())
            print(f"train: {len(frames)} frames, {total} candidates, {time.time() - started:.0f} s")
        if "test" in args.subsets:
            started = time.time()
            frames = {}
            for stem, candidates in pool.map(_test_frame, data.list_test_stems(), chunksize=4):
                frames[stem] = {"candidates": candidates}
            with open(out_dir / "test.pkl", "wb") as handle:
                pickle.dump({"recipe": recipe, "frames": frames}, handle)
            total = sum(len(f["candidates"]) for f in frames.values())
            print(f"test: {len(frames)} frames, {total} candidates, {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
