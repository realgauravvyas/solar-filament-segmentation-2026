# How the networks were trained

Folds 0, 1 and 2 were trained in Kaggle sessions (two Tesla T4 GPUs each), fold 3
on a local RTX 3060 (12 GB) with the same commands and settings.

`build_kernel.py` turns the repository into a single-file Kaggle script
(`kernel_template.py` with the code embedded as an archive) and pushes it:

```bash
python kaggle_sessions/build_kernel.py --fold 0 --push
```

A session attaches the competition data, installs `segmentation-models-pytorch`
and `ultralytics`, runs `scripts/01_preprocess.py`, then trains the U-Net on GPU 0
and the large detector on GPU 1 at the same time; the small detector follows the
U-Net on GPU 0. Each network predicts its held-out fold and the test set before
the session ends, and everything is written to `/kaggle/working/out`.

| Network | Command line (per fold `k`) | Time on one T4 |
|---|---|---|
| U-Net `r34` | `02_train_unet.py --fold k --name r34 --encoder resnet34 --iterations 12000 --batch-size 12` | 1.3 h |
| Detector `m1536` | `03_train_detector.py --fold k --name m1536 --weights yolo11m-seg.pt --imgsz 1536 --epochs 80 --batch 2` | 5.5 h |
| Detector `s1280` | `03_train_detector.py --fold k --name s1280 --weights yolo11s-seg.pt --imgsz 1280 --epochs 60 --batch 4` | 1.6 h |

Notes

* The Kaggle sessions asked for `--batch 4` for `m1536`; Ultralytics ran out of memory
  on the 15 GB T4 and fell back to batch 2 on its own, which is what the table shows
  and what the local run uses.
* On the 12 GB local card `m1536` needs `--gpu-fraction 0.97`: without a cap the
  Windows driver lets PyTorch spill into system RAM and training slows to a crawl.
  A few batches still exceed the card; `--resume` continues from the last epoch.
* Detections were written with one test-time view by the Kaggle sessions and
  recomputed locally with four mirror views (`03_train_detector.py --fold k --name ...`
  skips training when the checkpoint exists and only predicts).
* The sessions also trained a second U-Net (ResNeXt-50 encoder). It improved the
  out-of-fold score by 0.002, which is inside the noise, and is not part of the
  released pipeline.
