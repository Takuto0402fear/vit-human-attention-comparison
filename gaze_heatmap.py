"""
Convert fixation points to Gaussian-blurred heatmaps.

Each fixation deposits a 2-D Gaussian blob; the result is a spatial
density map normalised to sum to 1.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from config import Config
from schema import GazeRecord


def fixations_to_heatmap(
    fixations_xy: np.ndarray,
    image_size: Tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """
    Build a fixation density map.

    Parameters
    ----------
    fixations_xy : ndarray, shape (N, 2)
        Pixel coordinates (x, y), origin at top-left.
    image_size : (width, height)
    sigma : Gaussian blur sigma in pixels.

    Returns
    -------
    heatmap : ndarray, shape (H, W), sums to 1 (or all-zero if no valid fixations).
    """
    W, H = image_size
    hmap = np.zeros((H, W), dtype=np.float64)

    for x, y in fixations_xy:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            hmap[iy, ix] += 1.0

    if hmap.sum() == 0:
        return hmap.astype(np.float32)

    hmap = gaussian_filter(hmap, sigma=sigma)
    hmap /= hmap.sum()
    return hmap.astype(np.float32)


def build_gaze_heatmaps(
    records: List[GazeRecord],
    config: Config,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Build heatmaps for every (subject_id, image_id) pair.

    Returns
    -------
    dict
        Key:   (subject_id, image_id)
        Value: ndarray (H, W) summing to 1
    """
    result: Dict[Tuple[str, str], np.ndarray] = {}
    for rec in records:
        pts = np.array([[f.x, f.y] for f in rec.fixations])
        hmap = fixations_to_heatmap(pts, rec.image_size, config.gaussian_sigma)
        result[(rec.subject_id, rec.image_id)] = hmap
    return result
