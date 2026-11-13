"""Draw the figures of the report.

    python report/make_figures.py

Writes into ``report/figures/``:
    pipeline.png    the data flow of the method
    examples.png    a full-disk test frame with the submitted instances, and
                    out-of-fold predictions next to the annotators' drawings

``pipeline.png`` needs nothing. ``examples.png`` needs ``results/`` (the
submitted file and the out-of-fold predictions), the competition images and
the preprocessing cache (stage 1).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FIGURES = REPO / "report" / "figures"

INK = "#1f2933"
BLUE, ORANGE, GREEN, GREY = "#dbeafe", "#ffedd5", "#dcfce7", "#f1f5f9"
ANNOTATOR_COLOURS = ["#ef4444", "#22c55e", "#3b82f6"]


def _box(ax, x, y, w, h, title, body, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=colour, edgecolor=INK, linewidth=0.9))
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=5.9, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=6.2, color=INK, linespacing=1.25)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=INK))


def pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.45))
    ax.set_xlim(0, 20.0)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    _box(ax, 0.1, 1.9, 2.7, 1.7, "H-alpha frame", "2048 x 2048\n8-bit grey", GREY)
    _box(ax, 3.4, 1.9, 3.1, 1.7, "Pre-processing", "limb fit (circle)\nflat-field\n+ radius channel", GREY)
    _box(ax, 7.2, 3.3, 4.2, 1.9, "U-Net, full resolution", "512 px crops, 4 folds\n8 flips/rotations\nfilament probability", BLUE)
    _box(ax, 7.2, 0.3, 4.2, 1.9, "YOLO11-seg detectors", "whole disk, 4 folds\n4 mirror views, fused\ninstances + confidence", ORANGE)
    _box(ax, 12.1, 1.9, 3.3, 1.7, "Candidates", "U-Net pixels inside\neach detection, at\n3 probability cuts", GREEN)
    _box(ax, 16.0, 1.9, 3.9, 1.7, "Expected-PQ selection", "LightGBM predicts\nmatched IoU per view;\ngreedy, no overlaps", GREEN)

    _arrow(ax, (2.8, 2.75), (3.4, 2.75))
    _arrow(ax, (6.5, 3.1), (7.2, 4.0))
    _arrow(ax, (6.5, 2.4), (7.2, 1.5))
    _arrow(ax, (11.4, 4.0), (12.1, 3.1))
    _arrow(ax, (11.4, 1.5), (12.1, 2.4))
    _arrow(ax, (15.4, 2.75), (16.0, 2.75))
    ax.text(17.95, 1.6, "submission: one RLE\nmask per filament", ha="center", va="top", fontsize=6.0, color=INK)
    ax.text(9.3, 2.93, "shape", ha="center", va="center", fontsize=6.2, style="italic", color="#1d4ed8")
    ax.text(9.3, 2.55, "grouping", ha="center", va="center", fontsize=6.2, style="italic", color="#c2410c")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", FIGURES / "pipeline.png")


def _outline(ax, mask, colour, linewidth, offset=(0, 0)):
    """Draw the outer contours of a boolean mask."""
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        points = contour[:, 0, :] - offset
        points = np.vstack([points, points[:1]])
        ax.plot(points[:, 0], points[:, 1], color=colour, linewidth=linewidth)


def examples_figure(submission_csv: Path, oof_csv: Path, n_zooms: int = 6) -> None:
    """Left: a whole test frame with the submitted instances. Right: zoomed out-of-fold
    predictions (yellow) with the outlines drawn by the three annotators."""
    import cv2
    import numpy as np

    from filaseg import data, preprocess, rle, submission

    fig = plt.figure(figsize=(7.1, 3.05))
    grid = fig.add_gridspec(2, 4, width_ratios=[2.06, 1, 1, 1], wspace=0.03, hspace=0.12,
                            left=0.004, right=0.996, top=0.955, bottom=0.004)

    # A test frame: the one whose number of submitted instances is the median, i.e. typical.
    submitted = submission.read(submission_csv)
    counts = sorted((len(v), s) for s, v in submitted.items())
    stem = counts[len(counts) // 2][1]
    image = cv2.imread(str(data.image_path(stem, "test")), cv2.IMREAD_GRAYSCALE)
    channels, disk = preprocess.build_channels(image)
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(channels[1], cmap="gray", vmin=45, vmax=150)
    for k, r in enumerate(submitted[stem]):
        _outline(ax, rle.decode(r), plt.cm.tab10(k % 10), 0.7)
    half = disk.radius * 1.02
    ax.set_xlim(disk.cx - half, disk.cx + half)
    ax.set_ylim(disk.cy + half, disk.cy - half)
    ax.set_title(f"test frame {stem}: {len(submitted[stem])} submitted instances", fontsize=6, pad=2)
    ax.axis("off")

    # Zooms: held-out training frames drawn by three annotators.
    predictions = submission.read(oof_csv)
    observations = {o.stem: o for o in data.load_observations()}
    stems = sorted(s for s in predictions if len(observations[s].views) == 3 and len(predictions[s]) >= 4)
    rng = np.random.default_rng(11)
    chosen = [stems[i] for i in rng.choice(len(stems), size=n_zooms, replace=False)]
    for index, stem in enumerate(chosen):
        ax = fig.add_subplot(grid[index // 3, 1 + index % 3])
        flat = np.asarray(preprocess.load_channels(stem, "train")[1])
        masks = [rle.decode(r) for r in predictions[stem]]
        # Centre the zoom on a mid-sized predicted instance.
        target = sorted(masks, key=lambda m: m.sum())[len(masks) // 2 + (index % 2)]
        ys, xs = np.nonzero(target)
        half = 130
        y0 = int(np.clip(ys.mean() - half, 0, flat.shape[0] - 2 * half))
        x0 = int(np.clip(xs.mean() - half, 0, flat.shape[1] - 2 * half))
        crop = flat[y0:y0 + 2 * half, x0:x0 + 2 * half]
        ax.imshow(crop, cmap="gray", vmin=45, vmax=150)
        overlay = np.zeros(crop.shape + (4,))
        for mask in masks:
            overlay[mask[y0:y0 + 2 * half, x0:x0 + 2 * half]] = (1.0, 0.85, 0.0, 0.40)
        ax.imshow(overlay)
        for view, colour in zip(observations[stem].views, ANNOTATOR_COLOURS):
            for polygon in view.polygons:
                points = np.array(polygon).reshape(-1, 2) - (x0, y0)
                ax.plot(points[:, 0], points[:, 1], color=colour, linewidth=0.5)
        ax.set_xlim(0, crop.shape[1])
        ax.set_ylim(crop.shape[0], 0)
        ax.set_title(stem, fontsize=5, pad=1.5)
        ax.axis("off")

    fig.savefig(FIGURES / "examples.png", dpi=300)
    plt.close(fig)
    print("wrote", FIGURES / "examples.png")


if __name__ == "__main__":
    pipeline_figure()
    results = REPO / "results"
    if (results / "oof_predictions.csv").exists() and (results / "final_submission.csv").exists():
        examples_figure(results / "final_submission.csv", results / "oof_predictions.csv")
    else:
        print("results/ is incomplete: skipped examples.png")
