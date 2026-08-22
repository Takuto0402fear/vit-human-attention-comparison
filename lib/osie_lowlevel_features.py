"""
Pure image-processing low-level visual feature computation (Phase 2 of the
"shallow=low-level, deep=semantic" hypothesis test). NumPy + SciPy only --
no new dependency was added (cv2/scikit-image are not installed in this
environment; scipy.ndimage already ships Sobel/Laplacian, and
matplotlib.colors already ships RGB->HSV, both already-installed
dependencies of this repo).

All operators are applied to the FULL image first, and only aggregated
inside a mask afterward -- never computed on a mask-cropped image, which
would introduce a spurious gradient/edge AT the mask boundary itself
(explicitly avoided per the task spec: "マスク境界を人工的なエッジとし
て加えない").
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage

LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)  # ITU-R BT.709 relative luminance


def compute_luminance(img_float_hwc: np.ndarray) -> np.ndarray:
    """img_float_hwc: (H,W,3) in [0,1]. Returns (H,W) luminance in [0,1]."""
    r, g, b = LUMINANCE_WEIGHTS
    return r * img_float_hwc[..., 0] + g * img_float_hwc[..., 1] + b * img_float_hwc[..., 2]


def compute_saturation(img_float_hwc: np.ndarray) -> np.ndarray:
    """img_float_hwc: (H,W,3) in [0,1]. Returns (H,W) HSV saturation in [0,1]."""
    hsv = rgb_to_hsv(img_float_hwc)
    return hsv[..., 1]


def compute_gradient_magnitude(luminance_hw: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, applied to the FULL luminance image."""
    gx = ndimage.sobel(luminance_hw, axis=0)
    gy = ndimage.sobel(luminance_hw, axis=1)
    return np.sqrt(gx ** 2 + gy ** 2)


def compute_laplacian(luminance_hw: np.ndarray) -> np.ndarray:
    """Laplacian, applied to the FULL luminance image."""
    return ndimage.laplace(luminance_hw)


def compute_object_lowlevel_targets(
    img_float_224_hwc: np.ndarray, mask_224: np.ndarray, min_mask_pixels: int = 1,
) -> Optional[dict]:
    """
    img_float_224_hwc : (224, 224, 3) float64 in [0,1] -- the actual CLIP
        input image (Resize+CenterCrop applied, NOT yet CLIP-normalized).
    mask_224 : (224, 224) bool -- the OSIE mask transformed through the
        IDENTICAL Resize+CenterCrop (nearest-neighbor).

    Returns None if the mask is empty after the transform (caller should
    log a skip reason, not substitute 0). Otherwise returns a dict of the
    8 continuous targets + 3 geometry-control values.
    """
    if mask_224.sum() < min_mask_pixels:
        return None

    luminance = compute_luminance(img_float_224_hwc)
    saturation = compute_saturation(img_float_224_hwc)
    gradient_magnitude = compute_gradient_magnitude(luminance)
    laplacian = compute_laplacian(luminance)

    m = mask_224
    n_pixels = int(m.sum())
    ys, xs = np.where(m)

    return {
        "mean_luminance": float(luminance[m].mean()),
        "mean_red": float(img_float_224_hwc[..., 0][m].mean()),
        "mean_green": float(img_float_224_hwc[..., 1][m].mean()),
        "mean_blue": float(img_float_224_hwc[..., 2][m].mean()),
        "mean_saturation": float(saturation[m].mean()),
        "luminance_contrast": float(luminance[m].std()),
        "edge_strength": float(gradient_magnitude[m].mean()),
        "fine_texture": float(laplacian[m].var()),
        "mask_area_fraction": float(n_pixels / m.size),
        "centroid_x": float(xs.mean() / m.shape[1]),
        "centroid_y": float(ys.mean() / m.shape[0]),
        "n_mask_pixels_224": n_pixels,
    }
