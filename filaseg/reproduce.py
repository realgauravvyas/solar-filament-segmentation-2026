"""Inference from the released weights: test images in, submission file out.

``weights/manifest.json`` lists everything the final entry uses:

    {
      "unets":     {"r34":   ["unet_r34_f0.pt", ...]},
      "detectors": {"m1536": {"imgsz": 1536, "files": ["det_m1536_f0.pt", ...]}},
      "flips": 4,
      "tta": 8,
      "scorer": "scorer.json",
      "selection": {"value_threshold": 0.24, "max_overlap": 0.25}
    }

The network outputs are written with the same layout the training stages use
(``outputs/prob/<name>_f<k>/test``, ``outputs/det/<name>_f<k>/test.json``), so
the candidate and selection code is the code that was validated out-of-fold.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from . import config, data, inference, pipeline, preprocess, selection, submission


def load_manifest(weights_dir: Path) -> dict:
    with open(weights_dir / "manifest.json", encoding="utf-8") as handle:
        return json.load(handle)


def _fold_of(file_name: str) -> int:
    """``unet_r34_f2.pt`` -> 2."""
    return int(re.search(r"_f(\d+)\.pt$", file_name).group(1))


def recipe_from_manifest(manifest: dict) -> pipeline.Recipe:
    folds = sorted({_fold_of(name) for names in manifest["unets"].values() for name in names})
    return pipeline.Recipe(
        unets=list(manifest["unets"]), detectors=list(manifest["detectors"]),
        folds=folds, flips=manifest.get("flips"),
    )


def predict_networks(weights_dir: Path, device: str = "cuda:0", verbose: bool = True) -> None:
    """Run every released network over the test set and store its output."""
    manifest = load_manifest(weights_dir)
    stems = data.list_test_stems()
    preprocess.build_cache("test", verbose=verbose)
    out = config.out_root()
    for name, files in manifest["unets"].items():
        for file_name in files:
            started = time.time()
            inference.predict_probabilities(
                [weights_dir / file_name], stems, "test",
                out / "prob" / f"{name}_f{_fold_of(file_name)}" / "test",
                device=device, tta=manifest.get("tta", 8),
            )
            if verbose:
                print(f"U-Net {file_name}: {time.time() - started:.0f} s", flush=True)
    yolo_device = int(device.split(":")[1]) if device.startswith("cuda:") else device
    for name, spec in manifest["detectors"].items():
        for file_name in spec["files"]:
            started = time.time()
            target = out / "det" / f"{name}_f{_fold_of(file_name)}" / "test.json"
            if not target.exists():
                inference.predict_detections(
                    weights_dir / file_name, stems, "test", target,
                    imgsz=spec["imgsz"], device=yolo_device, flips=manifest.get("flips", 4),
                )
            if verbose:
                print(f"detector {file_name}: {time.time() - started:.0f} s", flush=True)


def select_test_instances(weights_dir: Path, verbose: bool = True) -> dict[str, list[np.ndarray]]:
    """Candidates -> predicted values -> non-overlapping instances, for every test frame."""
    manifest = load_manifest(weights_dir)
    recipe = recipe_from_manifest(manifest)
    scorer = selection.load_scorer(weights_dir / manifest["scorer"])
    params = selection.SelectionParams(**manifest["selection"])
    predictions: dict[str, list[np.ndarray]] = {}
    started = time.time()
    for index, stem in enumerate(data.list_test_stems(), start=1):
        candidates = pipeline.frame_candidates(recipe, stem, fold=None)
        values = selection.predict_values(scorer, candidates)
        predictions[stem] = selection.select(candidates, values, params)
        if verbose and index % 30 == 0:
            print(f"  selected {index} frames  {time.time() - started:.0f} s", flush=True)
    return predictions


def make_submission(weights_dir: Path, out_csv: Path, device: str = "cuda:0") -> dict[str, float]:
    """The whole inference pipeline; returns the statistics of the written file."""
    predict_networks(weights_dir, device=device)
    submission.write(select_test_instances(weights_dir), out_csv)
    return submission.check(out_csv, data.list_test_stems())
