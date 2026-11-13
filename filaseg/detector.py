"""Instance detector: which pixels belong to the same filament?

The semantic network knows which pixels are filament but not where one
filament ends and the next begins: a single filament is often several dark
fragments, and two filaments can touch. A YOLO11-seg detector (Ultralytics)
looks at the whole disk at reduced resolution and proposes instances. Its
masks are too coarse to submit (one mask cell covers several pixels of a
ten-pixel-wide filament), but they say which pixels go together and how likely
an annotator is to have drawn the object. ``candidates.py`` combines the two.

The detector sees an 8-bit, three-channel rendering of the frame:

    blue    raw image
    green   flat-fielded image
    red     flat-fielded image with the contrast around the quiet-Sun level tripled

Every view is its own training sample. A filament drawn by one annotator of
three is therefore a positive in one sample and background in two, and the
detector's confidence settles near the fraction of annotators who draw it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config, preprocess, rle
from .data import Observation


def render(channels: np.ndarray) -> np.ndarray:
    """Cached (2, H, W) channels -> uint8 BGR image for the detector."""
    raw, flat = channels[0], channels[1]
    stretched = np.clip(
        (flat.astype(np.float32) - preprocess.FLAT_UNITY) * 3.0 + 128.0, 0, 255
    ).astype(np.uint8)
    return np.dstack([raw, flat, stretched])


def export_dataset(
    train: list[Observation], val: list[Observation], folder: Path
) -> Path:
    """Write images and polygon labels in the Ultralytics segmentation layout.

    Returns the path of the ``data.yaml`` that ``train`` expects. Images are
    stored once per observation; a view that shares its image with another
    view gets a hard copy, because the label file is looked up by image name.
    """
    size = float(config.IMAGE_SIZE)
    for split, observations in (("train", train), ("val", val)):
        image_dir = folder / "images" / split
        label_dir = folder / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for observation in observations:
            encoded = None
            for view in observation.views:
                image_path = image_dir / f"{view.view_id}.jpg"
                if not image_path.exists():
                    if encoded is None:
                        channels = preprocess.load_channels(observation.stem, "train")
                        ok, encoded = cv2.imencode(
                            ".jpg", render(np.asarray(channels)), [cv2.IMWRITE_JPEG_QUALITY, 95]
                        )
                        assert ok
                    image_path.write_bytes(encoded.tobytes())
                lines = []
                for polygon in view.polygons:
                    points = np.clip(np.asarray(polygon, dtype=np.float64) / size, 0.0, 1.0)
                    lines.append("0 " + " ".join(f"{value:.6f}" for value in points))
                (label_dir / f"{view.view_id}.txt").write_text("\n".join(lines), encoding="utf-8")

    yaml_path = folder / "data.yaml"
    yaml_path.write_text(
        f"path: {folder.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
        encoding="utf-8",
    )
    return yaml_path


def train(
    data_yaml: Path,
    out_dir: Path,
    weights: str = "yolo11m-seg.pt",
    imgsz: int = 1536,
    epochs: int = 80,
    batch: int = 4,
    device: str | int = 0,
    seed: int = 0,
    workers: int = 4,
    cache: bool = False,
    resume: bool = False,
    mask_ratio: int = 2,
) -> Path:
    """Fine-tune a COCO-pretrained YOLO11-seg model; returns the last-epoch weights.

    Augmentation is the Ultralytics default (mosaic, scale, HSV, horizontal
    flip) plus vertical flips and free rotation: a filament has no preferred
    orientation in the image plane. The last epoch is used rather than the
    epoch with the best validation mAP, so the validation fold stays untouched
    for the later stages. With ``resume`` an interrupted run in ``out_dir`` is
    continued instead of started again. ``mask_ratio`` is the down-sampling of
    the training masks (2 = half resolution); 4 halves the memory at 2048 px.
    """
    from ultralytics import YOLO

    interrupted = out_dir / "weights" / "last.pt"
    if resume and interrupted.exists():
        # Continue an interrupted run from its last finished epoch, with its original settings.
        YOLO(str(interrupted)).train(resume=True)
        return interrupted

    model = YOLO(weights)
    model.train(
        data=str(data_yaml), imgsz=imgsz, epochs=epochs, batch=batch, device=device,
        project=str(out_dir.parent), name=out_dir.name, exist_ok=True, seed=seed,
        workers=workers, cache=cache, single_cls=True, flipud=0.5, fliplr=0.5, degrees=180.0,
        scale=0.25, mosaic=1.0, close_mosaic=10, mask_ratio=mask_ratio, overlap_mask=True,
        cos_lr=True, patience=1000, plots=False, verbose=False, amp=True,
    )
    return out_dir / "weights" / "last.pt"


#: Test-time views of a frame: (flip left-right, flip up-down).
FLIPS = [(False, False), (True, False), (False, True), (True, True)]


def predict(
    weights: Path,
    channels: np.ndarray,
    imgsz: int = 1536,
    conf: float = 0.02,
    device: str | int = 0,
    model=None,
    flips: int = 1,
) -> list[list[dict]]:
    """Run the detector on one frame, once per test-time view.

    The frame is shown to the detector as is and, if ``flips`` > 1, mirrored
    (left-right, up-down, both). Returns one list of detections per view,
    with the masks mapped back to the original orientation. A detection is a
    dict with keys ``score`` (confidence) and ``rle`` (the mask predicted by
    the detector at full resolution, run-length encoded). The views are fused
    later by ``pipeline.merge_detections``, exactly like the outputs of
    different detectors. The confidence threshold is deliberately low: the
    candidate scorer decides later what is kept.
    """
    if model is None:
        model = load(weights)
    image = render(np.asarray(channels))
    use_half = str(device) != "cpu"
    views = []
    for mirror_x, mirror_y in FLIPS[:flips]:
        view = image
        if mirror_x:
            view = view[:, ::-1]
        if mirror_y:
            view = view[::-1]
        result = model.predict(
            np.ascontiguousarray(view), imgsz=imgsz, conf=conf, iou=0.6, device=device,
            retina_masks=True, max_det=200, verbose=False, half=use_half,
        )[0]
        detections = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy() > 0.5
            scores = result.boxes.conf.cpu().numpy()
            for score, mask in zip(scores, masks):
                if not mask.any():
                    continue
                if mirror_x:
                    mask = mask[:, ::-1]
                if mirror_y:
                    mask = mask[::-1]
                detections.append({"score": float(score), "rle": rle.encode(mask)})
        views.append(sorted(detections, key=lambda d: -d["score"]))
    return views


def load(weights: Path):
    from ultralytics import YOLO

    return YOLO(str(weights))
