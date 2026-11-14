"""Solar-disk fit, flat-fielding and the cached network inputs.

A GONG H-alpha frame shows the solar disk on a dark background. Two things
make the raw pixels awkward for a network:

* limb darkening: the disk is brighter at the centre than near the edge, so
  the same filament has different grey values depending on where it sits;
* a bright ring at the limb and a faint halo outside it.

``build_channels`` therefore produces two 8-bit channels per frame:

    channel 0   the raw image
    channel 1   the flat-fielded image: raw divided by a smooth estimate of the
                local quiet-Sun brightness, so 100 means "as bright as the
                surroundings" everywhere on the disk, and 0 outside the disk

A third input, the distance from the disk centre in units of the solar
radius, is computed on the fly from the fitted disk (see ``radius_channel``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config, data, rle

#: Value of the flat-fielded channel where the pixel equals its surroundings.
FLAT_UNITY = 100.0

#: Width (pixels, at full resolution) of the Gaussian that estimates the background.
BACKGROUND_SIGMA = 48.0

#: Nominal geometry of a GONG frame: disk centre (x = y) and radius, in pixels.
NOMINAL_CENTRE = 1023.5
NOMINAL_RADIUS = 904.0

#: The limb is searched within this many pixels of the nominal radius. The
#: apparent solar radius changes by +-1.7 % (15 pixels) over the year.
LIMB_SEARCH = 45.0


@dataclass
class Disk:
    """Fitted solar disk in pixel coordinates."""

    cx: float
    cy: float
    radius: float


def fit_disk(image: np.ndarray) -> Disk:
    """Fit a circle to the solar limb.

    GONG frames are delivered with the Sun centred and at a nearly constant
    scale, so the search starts from that nominal geometry instead of from a
    brightness threshold (a threshold is fooled by the halo outside the limb,
    which is as bright as the disk in some frames). Along each of 360 rays the
    limb is the radius where the brightness drops fastest; a circle is fitted
    to those limb points by least squares, discarding outliers such as
    prominences. If the fit ends up far from the nominal geometry, the nominal
    geometry is returned.
    """
    size = image.shape[0]
    scale = size / config.IMAGE_SIZE
    nominal = Disk(NOMINAL_CENTRE * scale, NOMINAL_CENTRE * scale, NOMINAL_RADIUS * scale)

    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 2.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    radii = np.arange(
        nominal.radius - LIMB_SEARCH * scale, nominal.radius + LIMB_SEARCH * scale, 1.0,
        dtype=np.float32,
    )
    xs = (nominal.cx + np.outer(np.cos(angles), radii)).astype(np.float32)
    ys = (nominal.cy + np.outer(np.sin(angles), radii)).astype(np.float32)
    profile = cv2.remap(
        smooth, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    # Steepest brightness drop along each ray = the limb.
    drop = profile[:, :-4] - profile[:, 4:]
    edge = np.argmax(drop, axis=1)
    rows = np.arange(len(angles))
    edge_x, edge_y = xs[rows, edge + 2], ys[rows, edge + 2]

    cx, cy, radius = nominal.cx, nominal.cy, nominal.radius
    keep = np.ones(len(angles), dtype=bool)
    for _ in range(4):
        cx, cy, radius = _least_squares_circle(edge_x[keep], edge_y[keep])
        residual = np.abs(np.hypot(edge_x - cx, edge_y - cy) - radius)
        keep = residual < max(2.0, 2.5 * np.median(residual[keep]))
        if keep.sum() < 60:
            return nominal
    off_centre = np.hypot(cx - nominal.cx, cy - nominal.cy)
    if off_centre > 25 * scale or abs(radius - nominal.radius) > LIMB_SEARCH * scale:
        return nominal
    return Disk(float(cx), float(cy), float(radius))


def _least_squares_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) circle fit: solve ``x^2 + y^2 = 2 a x + 2 b y + c``."""
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, x**2 + y**2, rcond=None)
    a, b, c = solution
    return a, b, float(np.sqrt(max(c + a**2 + b**2, 1.0)))


def disk_mask(disk: Disk, shrink: float = 0.0, size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Boolean mask of the pixels inside the disk, optionally shrunk by ``shrink`` pixels."""
    yy, xx = np.ogrid[:size, :size]
    return (xx - disk.cx) ** 2 + (yy - disk.cy) ** 2 <= (disk.radius - shrink) ** 2


def flat_field(image: np.ndarray, disk: Disk) -> np.ndarray:
    """Divide the image by a smooth estimate of the local quiet-Sun brightness.

    The background is a Gaussian average restricted to the disk (a normalised
    convolution), so the dark sky does not drag the estimate down near the
    limb. It is computed at quarter resolution, which is exact enough for a
    48-pixel Gaussian and four times faster in each direction.
    """
    size = image.shape[0]
    inside = disk_mask(disk, shrink=4.0, size=size)
    small_size = (size // 4, size // 4)
    values = cv2.resize(
        np.where(inside, image, 0).astype(np.float32), small_size, interpolation=cv2.INTER_AREA
    )
    weight = cv2.resize(inside.astype(np.float32), small_size, interpolation=cv2.INTER_AREA)
    sigma = BACKGROUND_SIGMA / 4.0
    background = cv2.GaussianBlur(values, (0, 0), sigma) / np.maximum(
        cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3
    )
    background = cv2.resize(background, (size, size), interpolation=cv2.INTER_LINEAR)
    flat = image.astype(np.float32) / np.maximum(background, 1.0)
    flat = np.where(disk_mask(disk, size=size), flat, 0.0)
    return np.clip(flat * FLAT_UNITY, 0, 255).astype(np.uint8)


def build_channels(image: np.ndarray) -> tuple[np.ndarray, Disk]:
    """Raw frame -> (uint8 array of shape (2, H, W), fitted disk)."""
    disk = fit_disk(image)
    return np.stack([image, flat_field(image, disk)]), disk


def radius_channel(disk: Disk, y0: int, x0: int, height: int, width: int) -> np.ndarray:
    """Distance from the disk centre in solar radii for a crop, clipped to [0, 1.5]."""
    yy = np.arange(y0, y0 + height, dtype=np.float32)[:, None]
    xx = np.arange(x0, x0 + width, dtype=np.float32)[None, :]
    return np.clip(np.hypot(xx - disk.cx, yy - disk.cy) / disk.radius, 0.0, 1.5)


# --------------------------------------------------------------------------- cache


def _cache_path(stem: str, subset: str) -> Path:
    folder = config.cache_dir() / subset
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stem}.npy"


def _disk_path(subset: str) -> Path:
    return config.cache_dir() / f"{subset}_disks.json"


def build_cache(subset: str, stems: list[str] | None = None, verbose: bool = True) -> dict[str, Disk]:
    """Preprocess every frame of a subset once and store the result on disk.

    Each frame becomes ``cache/<subset>/<stem>.npy`` (uint8, shape (2, H, W)),
    which later stages memory-map, so a random crop costs a few milliseconds.
    """
    if stems is None:
        if subset == "train":
            stems = [o.stem for o in data.load_observations()]
        else:
            stems = data.list_test_stems()
    disks = load_disks(subset) if _disk_path(subset).exists() else {}
    for index, stem in enumerate(stems):
        target = _cache_path(stem, subset)
        if target.exists() and stem in disks:
            continue
        image = cv2.imread(str(data.image_path(stem, subset)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(data.image_path(stem, subset))
        channels, disk = build_channels(image)
        np.save(target, channels)
        disks[stem] = disk
        if verbose and (index + 1) % 100 == 0:
            print(f"  preprocessed {index + 1}/{len(stems)} {subset} frames", flush=True)
    with open(_disk_path(subset), "w", encoding="utf-8") as handle:
        json.dump({stem: asdict(disk) for stem, disk in disks.items()}, handle)
    return disks


def load_disks(subset: str) -> dict[str, Disk]:
    with open(_disk_path(subset), encoding="utf-8") as handle:
        return {stem: Disk(**values) for stem, values in json.load(handle).items()}


def load_channels(stem: str, subset: str, mmap: bool = True) -> np.ndarray:
    """Cached (2, H, W) uint8 channels of one frame."""
    return np.load(_cache_path(stem, subset), mmap_mode="r" if mmap else None)


def view_mask_path(view_id: str) -> Path:
    folder = config.cache_dir() / "masks"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{view_id}.npy"


def build_mask_cache(observations: list[data.Observation]) -> None:
    """Rasterise each view once: the union of its polygons as a uint8 (H, W) array.

    The polygons are rasterised by pycocotools, exactly as the organisers do
    for scoring, and stored uncompressed so that training crops can be read
    through a memory map instead of decoding a full frame per sample.
    """
    from pycocotools import mask as mask_util

    for observation in observations:
        for view in observation.views:
            target = view_mask_path(view.view_id)
            if target.exists():
                continue
            rles = view.rles()
            if rles:
                union = rle.decode(mask_util.merge(rles)).astype(np.uint8)
            else:
                union = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
            np.save(target, union)


def load_view_mask(view_id: str) -> np.ndarray:
    """Memory-mapped union mask of one view (see ``build_mask_cache``)."""
    return np.load(view_mask_path(view_id), mmap_mode="r")
