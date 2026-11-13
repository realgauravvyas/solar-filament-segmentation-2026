"""The limb fit must recover a synthetic disk, with and without a bright halo."""

import cv2
import numpy as np

from filaseg import preprocess


def _synthetic(cx, cy, radius, halo):
    yy, xx = np.mgrid[:2048, :2048]
    distance = np.hypot(xx - cx, yy - cy)
    image = np.where(distance <= radius, 120.0 - 40.0 * (distance / radius) ** 2, 0.0)
    if halo:
        image = np.where((distance > radius) & (distance < radius + 80), 70.0, image)
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0).clip(0, 255).astype(np.uint8)


def test_limb_fit_plain_and_with_halo():
    for halo in (False, True):
        disk = preprocess.fit_disk(_synthetic(1020.0, 1027.0, 900.0, halo))
        assert abs(disk.cx - 1020.0) < 1.5 and abs(disk.cy - 1027.0) < 1.5
        assert abs(disk.radius - 900.0) < 2.5


def test_flat_field_is_unity_on_a_smooth_disk():
    image = _synthetic(1023.5, 1023.5, 904.0, False)
    channels, disk = preprocess.build_channels(image)
    inside = preprocess.disk_mask(disk, shrink=60)
    assert abs(float(channels[1][inside].mean()) - preprocess.FLAT_UNITY) < 3.0
    assert channels[1][~preprocess.disk_mask(disk)].max() == 0
