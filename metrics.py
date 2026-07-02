"""
Saliency-map evaluation metrics.

All functions take a saliency map (the ViT attention heatmap, H x W,
non-negative) and human fixation points, and return a scalar score.

Metrics
-------
- NSS  (Normalized Scanpath Saliency)  -- primary
- AUC-Judd                             -- secondary
- sAUC (shuffled AUC)                  -- centre-bias control
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ======================================================================
# NSS
# ======================================================================

def nss(saliency: np.ndarray, fixation_points: np.ndarray) -> float:
    """
    Normalized Scanpath Saliency.

    Parameters
    ----------
    saliency : (H, W) non-negative map.
    fixation_points : (N, 2) array of (x, y) pixel coordinates.

    Returns
    -------
    float  (higher = better prediction of human fixations).
    """
    if len(fixation_points) == 0:
        return np.nan
    s = saliency.astype(np.float64)
    std = s.std()
    if std < 1e-12:
        return 0.0
    s_norm = (s - s.mean()) / std
    vals = _sample(s_norm, fixation_points)
    return float(np.nanmean(vals))


# ======================================================================
# AUC-Judd
# ======================================================================

def auc_judd(saliency: np.ndarray, fixation_points: np.ndarray) -> float:
    """
    AUC using the Judd et al. method.

    Positive set: fixation locations.
    Negative set: all other pixels (uniformly sampled for efficiency when
    the image is large; cap at 100 000 negatives).
    """
    if len(fixation_points) == 0:
        return np.nan
    H, W = saliency.shape
    s = saliency.astype(np.float64).ravel()

    # positives
    pos_vals = _sample(saliency.astype(np.float64), fixation_points)
    pos_vals = pos_vals[~np.isnan(pos_vals)]
    if len(pos_vals) == 0:
        return np.nan

    # negatives = all pixels except fixation locations
    fix_mask = np.zeros((H, W), dtype=bool)
    for x, y in fixation_points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            fix_mask[iy, ix] = True
    neg_vals = saliency.astype(np.float64)[~fix_mask]

    # subsample negatives for speed
    max_neg = 100_000
    if len(neg_vals) > max_neg:
        rng = np.random.RandomState(0)
        neg_vals = rng.choice(neg_vals, max_neg, replace=False)

    return _roc_auc(pos_vals, neg_vals)


# ======================================================================
# sAUC (shuffled AUC)
# ======================================================================

def sauc(
    saliency: np.ndarray,
    fixation_points: np.ndarray,
    other_fixations: np.ndarray,
    n_splits: int = 10,
) -> float:
    """
    Shuffled AUC for centre-bias control.

    Negative samples are drawn from *other_fixations* (fixation locations
    from other images), not from uniform pixels.

    Parameters
    ----------
    saliency : (H, W)
    fixation_points : (N, 2)  positive fixations on this image.
    other_fixations : (M, 2)  fixations pooled from other images.
    n_splits : number of bootstrap splits to average.
    """
    if len(fixation_points) == 0 or len(other_fixations) == 0:
        return np.nan

    pos_vals = _sample(saliency.astype(np.float64), fixation_points)
    pos_vals = pos_vals[~np.isnan(pos_vals)]
    if len(pos_vals) == 0:
        return np.nan

    rng = np.random.RandomState(0)
    aucs = []
    n_neg = len(pos_vals)  # matched size
    for _ in range(n_splits):
        idx = rng.choice(len(other_fixations), size=min(n_neg, len(other_fixations)),
                         replace=False)
        neg_pts = other_fixations[idx]
        neg_vals = _sample(saliency.astype(np.float64), neg_pts)
        neg_vals = neg_vals[~np.isnan(neg_vals)]
        if len(neg_vals) == 0:
            continue
        aucs.append(_roc_auc(pos_vals, neg_vals))

    return float(np.mean(aucs)) if aucs else np.nan


# ======================================================================
# Internal helpers
# ======================================================================

def _sample(arr: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample arr at (x, y) locations. Returns 1-D array of values."""
    H, W = arr.shape
    vals = []
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            vals.append(arr[iy, ix])
        else:
            vals.append(np.nan)
    return np.array(vals, dtype=np.float64)


def _roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute AUC from positive and negative score arrays (Wilcoxon)."""
    # Efficient O(n log n) implementation via sorting
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.concatenate([pos, neg])
    # sort descending
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    # trapezoidal AUC
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tpr = tp / n_pos
    fpr = fp / n_neg
    # prepend (0, 0)
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapz(tpr, fpr))
