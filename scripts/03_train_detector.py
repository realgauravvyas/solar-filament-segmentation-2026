"""Stage 3: train one YOLO11-seg detector and store its detections.

    python scripts/03_train_detector.py --fold 0 --name m1536

``--fold k`` trains on every observation outside fold k and then runs the
detector on fold k (out-of-fold) and on the test set. ``--fold -1`` trains on
everything and predicts the test set only.

Outputs, under ``outputs/``:
    models/det_<name>_f<fold>.pt
    det/<name>_f<fold>/val.json      (not for --fold -1)
    det/<name>_f<fold>/test.json
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import config, data, detector, inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--name", default="m1536")
    parser.add_argument("--weights", default="yolo11m-seg.pt")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", action="store_true", help="keep the resized training images in RAM")
    parser.add_argument("--flips", type=int, default=4, help="test-time views per frame (1 = none)")
    parser.add_argument(
        "--gpu-fraction", type=float, default=None,
        help="cap this process's share of GPU memory, e.g. 0.9 (on Windows the driver otherwise\n"
             "lets PyTorch spill into system RAM, which slows training to a crawl)",
    )
    parser.add_argument("--resume", action="store_true", help="continue an interrupted training run")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="predict only the first N frames (dry runs)")
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device
    if args.gpu_fraction:
        import torch

        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, int(args.device))

    observations = data.load_observations()
    folds = data.assign_folds(observations)
    train_obs = [o for o in observations if folds[o.stem] != args.fold]
    val_obs = [o for o in observations if folds[o.stem] == args.fold]
    tag = f"{args.name}_f{args.fold}"
    checkpoint = config.out_root() / "models" / f"det_{tag}.pt"

    if not checkpoint.exists():
        dataset_dir = config.work_root() / "yolo" / f"f{args.fold}"
        # Ultralytics needs a validation split; with --fold -1 it reuses a slice of the training data.
        yaml_path = detector.export_dataset(train_obs, val_obs or train_obs[:40], dataset_dir)
        print(f"training detector {tag} on {len(train_obs)} observations", flush=True)
        last = detector.train(
            yaml_path, config.work_root() / "yolo_runs" / tag, weights=args.weights,
            imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=device,
            seed=args.seed, workers=args.workers, cache=args.cache, resume=args.resume,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(last, checkpoint)

    if args.skip_predict:
        return
    det_dir = config.out_root() / "det" / tag
    if val_obs:
        inference.predict_detections(
            checkpoint, [o.stem for o in val_obs][:args.limit], "train", det_dir / "val.json",
            imgsz=args.imgsz, device=device, flips=args.flips,
        )
    inference.predict_detections(
        checkpoint, data.list_test_stems()[:args.limit], "test", det_dir / "test.json",
        imgsz=args.imgsz, device=device, flips=args.flips,
    )
    print(f"detector {tag} done", flush=True)


if __name__ == "__main__":
    main()
