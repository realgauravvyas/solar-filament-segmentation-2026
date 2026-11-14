"""Stage 2: train one semantic U-Net and store its predictions.

    python scripts/02_train_unet.py --fold 0 --name r34

``--fold k`` trains on every observation outside fold k and then predicts
fold k (out-of-fold) and the test set. ``--fold -1`` trains on everything and
predicts the test set only.

Outputs, under ``outputs/``:
    models/unet_<name>_f<fold>.pt
    prob/<name>_f<fold>/val/<stem>.png      (not for --fold -1)
    prob/<name>_f<fold>/test/<stem>.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config, data, inference, unet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--name", default="r34")
    parser.add_argument("--arch", default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--iterations", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="predict only the first N frames (dry runs)")
    args = parser.parse_args()

    observations = data.load_observations()
    folds = data.assign_folds(observations)
    train_obs = [o for o in observations if folds[o.stem] != args.fold]
    val_stems = [o.stem for o in observations if folds[o.stem] == args.fold]
    tag = f"{args.name}_f{args.fold}"
    checkpoint = config.out_root() / "models" / f"unet_{tag}.pt"

    if not checkpoint.exists():
        cfg = unet.UNetConfig(
            arch=args.arch, encoder=args.encoder, iterations=args.iterations,
            batch_size=args.batch_size, crop=args.crop, lr=args.lr,
            workers=args.workers, seed=args.seed,
        )
        print(f"training U-Net {tag} on {len(train_obs)} observations: {cfg}", flush=True)
        unet.train(train_obs, cfg, checkpoint, device=args.device)

    if args.skip_predict:
        return
    prob_dir = config.out_root() / "prob" / tag
    if val_stems:
        inference.predict_probabilities(
            [checkpoint], val_stems[:args.limit], "train", prob_dir / "val", device=args.device, tta=args.tta
        )
    inference.predict_probabilities(
        [checkpoint], data.list_test_stems()[:args.limit], "test", prob_dir / "test",
        device=args.device, tta=args.tta,
    )
    print(f"U-Net {tag} done", flush=True)


if __name__ == "__main__":
    main()
