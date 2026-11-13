# Solar filament instance segmentation in GONG H-alpha images

Solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup 2026, sponsored by the NSF National Solar Observatory).

Each frame is a 2048 x 2048 full-disk H-alpha image; the task is one mask per
filament. Submissions are scored with Panoptic Quality (PQ) against every
annotator's drawing of a frame: a predicted mask counts only if its IoU with a
drawing exceeds 0.5.

* **Report:** [`report/main.pdf`](report/main.pdf) (4 pages)
* **Notebook:** [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb) walks through the whole
  pipeline and rebuilds the submitted file from the weights in [`weights/`](weights) without
  retraining
* **Submitted predictions:** [`results/final_submission.csv`](results/final_submission.csv)

## Method in one paragraph

A U-Net working at full resolution says which pixels are filament. A YOLO11-seg
detector looking at the whole disk says which pixels belong to the same filament
and how likely an annotator is to draw it. Each detection claims the U-Net pixels
inside its mask, which gives candidate instances with accurate outlines. A small
gradient-boosted model then predicts, for every candidate, the quantity that the
metric rewards, and candidates are kept only while they are expected to raise PQ.
No ground truth other than the competition's training annotations is used
anywhere, and the test images are used for inference only.

![pipeline](report/figures/pipeline.png)

## Results

All validation numbers are out-of-fold: the 707 training frames are split into
four folds by frame, and a frame is predicted by networks and by a selector that
were fitted without it. Intervals are 95 % bootstrap intervals over frames.

RESULTS_TABLE

For orientation, scoring one annotator against the other annotators of the same
frame gives PQ 0.340 on the training set: the annotators themselves disagree
about which dark structures are filaments and where one ends, and that bounds
what any method can reach on this metric.

## Repository layout

```
filaseg/             the package
  config.py          paths and constants
  data.py            annotations, frames, views, cross-validation folds
  preprocess.py      limb fit, flat-fielding, cached network inputs
  rle.py             COCO run-length helpers
  metric.py          the competition's PQ on run-length masks
  unet.py            semantic network: model, training, tiled inference
  detector.py        YOLO11-seg wrapper: dataset export, training, prediction
  inference.py       run the networks over a subset and store their outputs
  candidates.py      network outputs -> candidate instances with features
  selection.py       expected-PQ scorer and greedy non-overlapping selection
  pipeline.py        glue: which network outputs belong to which frame
  submission.py      CSV writer and format checks
  reproduce.py       inference from the released weights
scripts/             one numbered script per stage (see below)
notebooks/           pipeline.ipynb and the script that generates it
weights/             released weights and manifest.json
results/             submitted file, out-of-fold predictions, metrics
report/              the 4-page report (LaTeX source and PDF)
kaggle_sessions/     how the training sessions on Kaggle were launched
tests/               unit tests (pytest)
```

## Setup

Python 3.11. Install PyTorch for your CUDA version first, then the rest:

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest -q            # 7 tests, no data needed
```

### Data

Download the competition data (needs a Kaggle account that has accepted the rules)
and unpack it so that the folders look like this:

```
data/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpeg
data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
data/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
```

```bash
kaggle competitions download -c filament-segmentation-2026 -p data
unzip -q data/filament-segmentation-2026.zip -d data
```

A different location can be given with the environment variable `FILASEG_DATA`
(the folder that contains `train/` and `test/`). Outputs go to `outputs/`
(`FILASEG_OUT`), large scratch files to `outputs/work/` (`FILASEG_WORK`).

## Reproducing the submission (no training)

```bash
python scripts/07_predict_test.py          # writes outputs/submissions/submission.csv
```

or run [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb) top to bottom, which does the
same and also shows every intermediate step, compares the rebuilt file with the
submitted one, and recomputes the validation numbers. A GPU is needed.

INFERENCE_TIME

## Training from scratch

```bash
python scripts/01_preprocess.py                       # limb fit + flat-field, cached

# for each fold k in 0..3
python scripts/02_train_unet.py     --fold k --name r34 --encoder resnet34 --iterations 12000 --batch-size 12
python scripts/03_train_detector.py --fold k --name m1536 --weights yolo11m-seg.pt --imgsz 1536 --epochs 80 --batch 2  # add --gpu-fraction 0.97 on a 12 GB card
python scripts/03_train_detector.py --fold k --name s1280 --weights yolo11s-seg.pt --imgsz 1280 --epochs 60 --batch 4

python scripts/04_build_candidates.py --name main --unets r34 --detectors m1536 s1280
python scripts/05_fit_selector.py     --name main     # out-of-fold PQ, scorer, threshold
python scripts/06_make_submission.py  --name main     # outputs/submissions/main.csv
python scripts/08_evaluate.py --predictions outputs/candidates/main/oof_predictions.csv --out outputs/evaluation
```

Each training script trains on every frame outside fold `k`, then predicts fold
`k` (out-of-fold) and the test set, so stages 4 to 8 need no GPU.

TRAINING_COST

## What was tried and did not help

NEGATIVE_RESULTS

## Licence and acknowledgements

Code: MIT (see `LICENSE`). The detector is trained with
[Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) starting from
its COCO-pretrained YOLO11 weights; the U-Net uses
[segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
with an ImageNet-pretrained encoder. Data: MAGFiLO (Ahmadzadeh et al., 2024,
doi:10.1038/s41597-024-03876-y), GONG H-alpha observations from the NSO
Integrated Synoptic Program.
