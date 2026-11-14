"""Semantic network: which pixels are filament?

A U-Net (``segmentation_models_pytorch``) is trained at full resolution on
random crops. Filaments are about ten pixels wide and are matched at IoU 0.5,
so one or two pixels at the boundary decide whether a prediction counts;
nothing is downsampled.

Inputs, three channels per pixel:

    0   raw image, scaled to [0, 1]
    1   flat-fielded image, scaled so that 1.0 = the local quiet-Sun level
    2   distance from the disk centre in solar radii

Target: the pixels inside the polygons of ONE view. Observations drawn by
several annotators contribute one training sample per view, so where the
annotators disagree the network learns the fraction of them who marked the
pixel, and 0.5 is the majority vote.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import config, preprocess
from .data import Observation


@dataclass
class UNetConfig:
    """Everything needed to rebuild and retrain the semantic network."""

    arch: str = "unet"                 # "unet" or "unetplusplus"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    crop: int = 512                    # training crop, pixels
    batch_size: int = 12
    iterations: int = 12000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup: int = 500
    ema_decay: float = 0.999
    filament_crop_fraction: float = 0.7   # crops centred on a drawn filament
    rotate_probability: float = 0.5
    workers: int = 4
    seed: int = 0


def build_model(cfg: UNetConfig, pretrained: bool = True) -> torch.nn.Module:
    """Three input channels, one output logit per pixel."""
    factory = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}[cfg.arch]
    return factory(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights if pretrained else None,
        in_channels=3,
        classes=1,
    )


# --------------------------------------------------------------------------- inputs


def make_input(channels: np.ndarray, disk: preprocess.Disk, y0: int, x0: int) -> np.ndarray:
    """uint8 crop (2, h, w) taken at (y0, x0) -> float32 network input (3, h, w)."""
    _, height, width = channels.shape
    out = np.empty((3, height, width), dtype=np.float32)
    out[0] = channels[0].astype(np.float32) / 255.0
    out[1] = channels[1].astype(np.float32) / preprocess.FLAT_UNITY
    out[2] = preprocess.radius_channel(disk, y0, x0, height, width)
    return out


# --------------------------------------------------------------------------- training data


class CropDataset(Dataset):
    """Random full-resolution crops with the mask of one randomly chosen view."""

    def __init__(self, observations: list[Observation], cfg: UNetConfig, length: int):
        self.cfg = cfg
        self.length = length
        self.disks = preprocess.load_disks("train")
        preprocess.build_mask_cache(observations)
        # One sample per view: (stem, view id, centres of the drawn filaments).
        self.samples: list[tuple[str, str, np.ndarray]] = []
        for observation in observations:
            for view in observation.views:
                if view.polygons:
                    centres = np.array([
                        [np.mean(p[0::2]), np.mean(p[1::2])] for p in view.polygons
                    ])
                else:
                    centres = np.zeros((0, 2))
                self.samples.append((observation.stem, view.view_id, centres))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        cfg = self.cfg
        rng = np.random.default_rng()
        stem, view_id, centres = self.samples[rng.integers(len(self.samples))]
        disk = self.disks[stem]
        size = config.IMAGE_SIZE
        crop = cfg.crop
        rotate = rng.random() < cfg.rotate_probability
        # A rotated crop is cut from a larger window so its corners are real pixels.
        window = int(math.ceil(crop * 1.42)) if rotate else crop

        if len(centres) and rng.random() < cfg.filament_crop_fraction:
            cx, cy = centres[rng.integers(len(centres))] + rng.uniform(-0.3, 0.3, 2) * crop
        else:
            radius = disk.radius * math.sqrt(rng.random())
            angle = rng.uniform(0, 2 * math.pi)
            cx, cy = disk.cx + radius * math.cos(angle), disk.cy + radius * math.sin(angle)
        x0 = int(np.clip(cx - window / 2, 0, size - window))
        y0 = int(np.clip(cy - window / 2, 0, size - window))

        channels = np.asarray(
            preprocess.load_channels(stem, "train")[:, y0:y0 + window, x0:x0 + window]
        )
        image = make_input(channels, disk, y0, x0)
        mask = np.asarray(
            preprocess.load_view_mask(view_id)[y0:y0 + window, x0:x0 + window], dtype=np.float32
        )

        if rotate:
            matrix = cv2.getRotationMatrix2D((window / 2, window / 2), rng.uniform(0, 360), 1.0)
            matrix[:, 2] -= (window - crop) / 2
            image = np.stack([
                cv2.warpAffine(c, matrix, (crop, crop), flags=cv2.INTER_LINEAR) for c in image
            ])
            mask = cv2.warpAffine(mask, matrix, (crop, crop), flags=cv2.INTER_LINEAR)

        # The eight symmetries of the square.
        k = int(rng.integers(4))
        image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k)
        if rng.random() < 0.5:
            image, mask = image[:, :, ::-1], mask[:, ::-1]

        # Photometric jitter: seeing and exposure differ between the six GONG sites.
        gain, offset = rng.uniform(0.85, 1.15), rng.uniform(-0.06, 0.06)
        image = image.copy()
        image[0] = np.clip(image[0] * gain + offset, 0, 1.5)
        image[1] = np.clip((image[1] - 1.0) * rng.uniform(0.8, 1.25) + 1.0 + offset, 0, 2.55)
        if rng.random() < 0.2:
            sigma = rng.uniform(0.5, 1.5)
            image[:2] = np.stack([cv2.GaussianBlur(c, (0, 0), sigma) for c in image[:2]])

        return torch.from_numpy(image), torch.from_numpy(np.ascontiguousarray(mask))[None]


# --------------------------------------------------------------------------- training


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy plus soft Dice, both averaged over the batch."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1.0) / (prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0)
    return bce + (1.0 - dice).mean()


def train(
    observations: list[Observation],
    cfg: UNetConfig,
    out_path: Path,
    device: str = "cuda",
    log_every: int = 100,
) -> Path:
    """Train on the given observations and save the EMA weights to ``out_path``.

    The schedule is a linear warm-up followed by one cosine decay to zero.
    The checkpoint written at the end holds an exponential moving average of
    the weights; no "best epoch" is picked, so no validation data is consumed
    by the training run itself.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_model(cfg).to(device)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.ema_decay)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_factor(step: int) -> float:
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        progress = (step - cfg.warmup) / max(1, cfg.iterations - cfg.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    loader = DataLoader(
        CropDataset(observations, cfg, cfg.iterations * cfg.batch_size),
        batch_size=cfg.batch_size, num_workers=cfg.workers, drop_last=True,
        persistent_workers=cfg.workers > 0, pin_memory=True,
    )

    model.train()
    started = time.time()
    running = 0.0
    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            loss = dice_bce_loss(model(images).float(), masks)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update_parameters(model)

        running += loss.item()
        if step % log_every == 0:
            elapsed = time.time() - started
            print(
                f"  step {step:6d}/{cfg.iterations}  loss {running / log_every:.4f}  "
                f"lr {scheduler.get_last_lr()[0]:.2e}  {elapsed / step:.2f} s/it",
                flush=True,
            )
            running = 0.0

    # BatchNorm statistics are not averaged by the EMA; copy the live ones.
    ema_state = {k.removeprefix("module."): v for k, v in ema.state_dict().items() if k != "n_averaged"}
    for key, value in model.state_dict().items():
        if "running_" in key or "num_batches_tracked" in key:
            ema_state[key] = value
    save_checkpoint(ema_state, cfg, out_path)
    return out_path


def save_checkpoint(state: dict, cfg: UNetConfig, path: Path) -> None:
    """Store half-precision weights (they halve the file and change no prediction)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    half = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
    torch.save({"config": asdict(cfg), "state_dict": half}, path)


def load_model(path: Path, device: str = "cuda") -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = UNetConfig(**checkpoint["config"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in checkpoint["state_dict"].items()})
    return model.to(device).eval()


# --------------------------------------------------------------------------- inference

#: The eight symmetries of the square as (number of 90-degree turns, mirror).
DIHEDRAL = [(k, flip) for k in range(4) for flip in (False, True)]


def _hann_window(tile: int) -> torch.Tensor:
    """2-D weight that fades towards the tile border, so tile seams blend away."""
    ramp = torch.hann_window(tile + 2, periodic=False)[1:-1]
    return torch.outer(ramp, ramp).clamp_min(1e-3)


@torch.no_grad()
def predict_probability(
    models: list[torch.nn.Module],
    channels: np.ndarray,
    disk: preprocess.Disk,
    device: str = "cuda",
    tile: int = 1024,
    overlap: int = 256,
    tta: int = 8,
) -> np.ndarray:
    """Filament probability of one frame, float32 (H, W).

    The frame is covered by overlapping tiles; every tile is run through every
    model under ``tta`` of the eight square symmetries; logits are averaged
    over models and symmetries and blended across tiles with a Hann window.
    """
    size = channels.shape[-1]
    full = torch.from_numpy(make_input(np.asarray(channels), disk, 0, 0)).to(device)
    stride = tile - overlap
    starts = sorted({min(s, size - tile) for s in range(0, size - overlap, stride)})
    window = _hann_window(tile).to(device)
    logit_sum = torch.zeros((size, size), device=device)
    weight_sum = torch.zeros((size, size), device=device)

    for y0 in starts:
        for x0 in starts:
            patch = full[:, y0:y0 + tile, x0:x0 + tile][None]
            accumulated = torch.zeros((tile, tile), device=device)
            for turns, mirror in DIHEDRAL[:tta]:
                view = torch.rot90(patch, turns, dims=(2, 3))
                if mirror:
                    view = view.flip(3)
                for model in models:
                    with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                        logits = model(view).float()
                    if mirror:
                        logits = logits.flip(3)
                    accumulated += torch.rot90(logits, -turns, dims=(2, 3))[0, 0]
            accumulated /= len(models) * len(DIHEDRAL[:tta])
            logit_sum[y0:y0 + tile, x0:x0 + tile] += accumulated * window
            weight_sum[y0:y0 + tile, x0:x0 + tile] += window

    return torch.sigmoid(logit_sum / weight_sum).cpu().numpy()


def save_probability(probability: np.ndarray, path: Path) -> None:
    """Store a probability map as a lossless 8-bit PNG (steps of 1/255)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.round(probability * 255).astype(np.uint8))


def load_probability(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0
