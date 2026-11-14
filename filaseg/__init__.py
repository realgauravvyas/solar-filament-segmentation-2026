"""filaseg: instance segmentation of solar filaments in GONG H-alpha images.

The package is organised by pipeline stage:

    config       paths and constants
    data         annotations, observations, cross-validation folds
    preprocess   solar-disk fit, flat-fielding, cached network inputs
    rle          COCO run-length helpers
    metric       the competition's Panoptic Quality, reimplemented on RLEs
    unet         semantic network: model, training, tiled inference
    detector     YOLO11-seg wrapper: dataset export, training, prediction
    candidates   turns network outputs into candidate filament instances
    selection    expected-PQ candidate scoring and non-overlapping selection
    submission   CSV writer and format checks
"""

__version__ = "1.0.0"
