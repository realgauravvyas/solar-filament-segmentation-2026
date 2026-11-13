"""Build (and optionally push) the Kaggle training session for one fold.

    python kaggle_sessions/build_kernel.py --fold 0 --push

The session script is ``kernel_template.py`` with the repository's code
embedded as a base64 archive, so a session is fully described by one file
that is saved next to this script under ``kaggle_sessions/sessions/``.

The Kaggle user name is read from the ``KAGGLE_USERNAME`` environment variable
or from ``~/.kaggle/kaggle.json``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: What a standard fold session trains. Each string is appended to the stage script's command line.
DEFAULT_UNET_JOBS = [
    "--name r34 --arch unet --encoder resnet34 --iterations 12000 --batch-size 12 --workers 2",
    "--name rx50 --arch unet --encoder resnext50_32x4d --iterations 10000 --batch-size 10 --workers 2",
]
DEFAULT_GPU0_DETECTOR_JOBS = [
    "--name s1280 --weights yolo11s-seg.pt --imgsz 1280 --epochs 60 --batch 4 --workers 2 --cache",
]
DEFAULT_DETECTOR_JOBS = [
    "--name m1536 --weights yolo11m-seg.pt --imgsz 1536 --epochs 80 --batch 4 --workers 2 --cache",
]


def kaggle_username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    with open(Path.home() / ".kaggle" / "kaggle.json", encoding="utf-8") as handle:
        return json.load(handle)["username"]


def code_archive() -> str:
    """The ``filaseg`` package and the stage scripts as a base64 tar.gz."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for folder in ("filaseg", "scripts"):
            for path in sorted((REPO / folder).rglob("*.py")):
                archive.add(path, arcname=path.relative_to(REPO).as_posix())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--tag", default="", help="suffix for the session name, e.g. 'b' for a second run")
    parser.add_argument("--unet-job", action="append", help="override the U-Net jobs (repeatable)")
    parser.add_argument("--detector-job", action="append", help="override the detector jobs (repeatable)")
    parser.add_argument("--gpu0-detector-job", action="append", help="override the GPU-0 detector jobs")
    parser.add_argument("--no-unet", action="store_true")
    parser.add_argument("--no-gpu0-detector", action="store_true")
    parser.add_argument("--no-detector", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    unet_jobs = [] if args.no_unet else (args.unet_job or DEFAULT_UNET_JOBS)
    detector_jobs = [] if args.no_detector else (args.detector_job or DEFAULT_DETECTOR_JOBS)
    gpu0_detector_jobs = [] if args.no_gpu0_detector else (args.gpu0_detector_job or DEFAULT_GPU0_DETECTOR_JOBS)
    script = (HERE / "kernel_template.py").read_text(encoding="utf-8")
    script = (
        script.replace("__FOLD__", str(args.fold))
        .replace("__UNET_JOBS__", repr(unet_jobs))
        .replace("__DETECTOR_JOBS__", repr(detector_jobs))
        .replace("__GPU0_DETECTOR_JOBS__", repr(gpu0_detector_jobs))
        .replace("__CODE_ARCHIVE__", code_archive())
    )

    fold_name = "all" if args.fold < 0 else str(args.fold)
    slug = f"filaseg-train-f{fold_name}{args.tag}"
    session_dir = HERE / "sessions" / slug
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "train_fold.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{kaggle_username()}/{slug}",
        "title": slug,
        "code_file": "train_fold.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [],
    }
    (session_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {session_dir}")

    if args.push:
        subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(session_dir), "--accelerator", "NvidiaTeslaT4"],
            check=True,
        )


if __name__ == "__main__":
    main()
