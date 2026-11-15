"""Stage 10: collect what the repository ships.

    python scripts/10_package_release.py --name main --unets r34 \\
        --detectors m1536:1536 s1280:1280 m2048:2048 --folds 0 1 2 3

Copies the trained checkpoints, the selector and its threshold into
``weights/`` and writes ``weights/manifest.json`` (read by
``filaseg/reproduce.py``); copies the out-of-fold predictions, the metrics
and the submission file into ``results/``.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config  # noqa: E402


def release_variant(args, out: Path, weights: Path, results: Path) -> None:
    """Add a second configuration that reuses the released networks with another selector."""
    with open(weights / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    recipe_dir = out / "candidates" / args.variant
    with open(recipe_dir / "selection.json", encoding="utf-8") as handle:
        selection = json.load(handle)
    scorer_name = f"scorer_{args.variant_suffix}.json"
    shutil.copyfile(recipe_dir / "scorer.json", weights / scorer_name)
    manifest.update(
        scorer=scorer_name, components=args.components,
        selection={"value_threshold": selection["value_threshold"], "max_overlap": selection["max_overlap"]},
    )
    with open(weights / f"manifest_{args.variant_suffix}.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    shutil.copyfile(recipe_dir / "selection.json", results / f"selection_{args.variant_suffix}.json")
    shutil.copyfile(recipe_dir / "oof_predictions.csv", results / f"oof_predictions_{args.variant_suffix}.csv")
    shutil.copyfile(out / "submissions" / f"{args.variant}.csv",
                    results / f"final_submission_{args.variant_suffix}.csv")
    print(f"released configuration manifest_{args.variant_suffix}.json")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--name", required=True, help="recipe name used in stages 4-6")
    parser.add_argument("--unets", nargs="+", required=True)
    parser.add_argument("--detectors", nargs="+", required=True, help="name:imgsz, e.g. m1536:1536")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(config.N_FOLDS)))
    parser.add_argument("--flips", type=int, default=4)
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--evaluation", default=None, help="folder with metrics.json and figures (stage 8)")
    parser.add_argument("--ablation", default=None, help="ablation.json (stage 9)")
    parser.add_argument("--variant", default=None,
                        help="release only the selector of this recipe as an extra configuration\n"
                             "(same networks), e.g. --variant main_comp --variant-suffix components")
    parser.add_argument("--variant-suffix", default="variant")
    parser.add_argument("--components", action="store_true", help="the variant uses component candidates")
    args = parser.parse_args()

    out = config.out_root()
    weights = config.REPO_ROOT / "weights"
    if args.variant:
        release_variant(args, out, weights, config.REPO_ROOT / "results")
        return
    results = config.REPO_ROOT / "results"
    weights.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    recipe_dir = out / "candidates" / args.name

    manifest = {"unets": {}, "detectors": {}, "flips": args.flips, "tta": args.tta, "scorer": "scorer.json"}
    for name in args.unets:
        files = [f"unet_{name}_f{k}.pt" for k in args.folds]
        manifest["unets"][name] = files
    for spec in args.detectors:
        name, imgsz = spec.split(":")
        files = [f"det_{name}_f{k}.pt" for k in args.folds]
        manifest["detectors"][name] = {"imgsz": int(imgsz), "files": files}
    for files in list(manifest["unets"].values()) + [d["files"] for d in manifest["detectors"].values()]:
        for file_name in files:
            shutil.copyfile(out / "models" / file_name, weights / file_name)

    with open(recipe_dir / "selection.json", encoding="utf-8") as handle:
        selection = json.load(handle)
    manifest["selection"] = {
        "value_threshold": selection["value_threshold"], "max_overlap": selection["max_overlap"],
    }
    shutil.copyfile(recipe_dir / "scorer.json", weights / "scorer.json")
    with open(weights / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    shutil.copyfile(recipe_dir / "selection.json", results / "selection.json")
    shutil.copyfile(recipe_dir / "oof_predictions.csv", results / "oof_predictions.csv")
    shutil.copyfile(out / "submissions" / f"{args.name}.csv", results / "final_submission.csv")
    if args.evaluation:
        for path in Path(args.evaluation).glob("*"):
            shutil.copyfile(path, results / path.name)
    if args.ablation:
        shutil.copyfile(args.ablation, results / "ablation.json")

    size = sum(p.stat().st_size for p in weights.glob("*")) / 1e6
    print(f"weights/: {len(list(weights.glob('*')))} files, {size:.0f} MB")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
