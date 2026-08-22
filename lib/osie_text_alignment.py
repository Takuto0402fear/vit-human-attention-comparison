"""
Pure, model-free helper functions for the OSIE text-alignment / linear-probe
pilot (outputs/osie_text_alignment_pilot/). No CLIP, no torch tensors here
(numpy only) -- these are the primitives unit-tested directly in
tests/test_osie_text_alignment.py, and imported by the extraction/analysis/
probe scripts.

Mask <-> patch-grid convention mirrors the earlier attribute-grounding
pilot exactly: bottom/right zero-pad to a patch_size multiple (same
geometry as vit_extractor.py::_pad_to_patch, reimplemented here for a 2D
boolean mask rather than an (H,W,3) image array), then each patch's weight
is its fractional area coverage in [0, 1].
"""
from __future__ import annotations

import itertools
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Mask <-> patch grid
# ----------------------------------------------------------------------

def pad_mask_to_patch(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """Zero-pad a 2D boolean/float mask on the bottom/right to a patch_size
    multiple -- same geometry as vit_extractor.py::_pad_to_patch, for a 2D
    (H, W) array instead of (H, W, 3)."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    h, w = mask.shape
    hp = (patch_size - h % patch_size) % patch_size
    wp = (patch_size - w % patch_size) % patch_size
    if hp or wp:
        mask = np.pad(mask, ((0, hp), (0, wp)))
    return mask


def mask_to_patch_weights(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """
    mask : (H, W) bool or float, native (unpadded) resolution.
    Returns (grid_h, grid_w) float64 in [0, 1] -- each patch's fractional
    area coverage by the mask, after zero-padding mask to a patch_size
    multiple (so boundary patches that partially fall outside the native
    image get a correspondingly reduced, not undefined, weight).
    """
    padded = pad_mask_to_patch(mask.astype(np.float64), patch_size)
    ph, pw = padded.shape
    if ph % patch_size or pw % patch_size:
        raise RuntimeError(f"STOP: padded mask shape {padded.shape} not a multiple of {patch_size}")
    gh, gw = ph // patch_size, pw // patch_size
    reshaped = padded.reshape(gh, patch_size, gw, patch_size)
    weights = reshaped.mean(axis=(1, 3))
    return weights


# ----------------------------------------------------------------------
# Weighted pooling
# ----------------------------------------------------------------------

def weighted_pool_tokens(
    tokens: np.ndarray, weights: np.ndarray, min_weight_sum: float = 1e-6,
) -> Optional[np.ndarray]:
    """
    tokens  : (grid_h, grid_w, d) or (n_patches, d) float.
    weights : (grid_h, grid_w) or (n_patches,) float, same leading shape.
    Returns (d,) weighted mean, or None if the total weight is below
    min_weight_sum (caller should log a skip reason, not treat this as 0).
    """
    t = tokens.reshape(-1, tokens.shape[-1])
    w = weights.reshape(-1)
    if t.shape[0] != w.shape[0]:
        raise ValueError(f"STOP: token count {t.shape[0]} != weight count {w.shape[0]}")
    total = float(w.sum())
    if total < min_weight_sum:
        return None
    return (t * w[:, None]).sum(axis=0) / total


def l2_normalize(vec: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    return vec / np.clip(norm, eps, None)


def build_equal_ensemble(vectors: np.ndarray) -> np.ndarray:
    """vectors: (K, d), each already L2-normalized. Returns the L2-renormalized mean."""
    mean = vectors.mean(axis=0)
    return l2_normalize(mean, axis=-1)


# ----------------------------------------------------------------------
# Positive/negative attribute split for one multi-label object
# ----------------------------------------------------------------------

def split_positive_negative_attributes(
    features: Sequence[float], mat_attr_names: Sequence[str], mat_to_task: dict,
) -> Tuple[List[str], List[str]]:
    """
    features      : (n_attrs,) 0/1 (or >0 / <=0) values, in mat_attr_names order.
    mat_attr_names : attrs.mat attribute names, same order as `features`.
    mat_to_task    : {mat_name: task_name} mapping (attribute_mapping.json).

    Returns (positive_task_names, negative_task_names) -- a strict
    partition of all task attribute names (no name in both, none omitted),
    so a multi-label object's OTHER positive attributes can never leak
    into its own negative set.
    """
    if len(features) != len(mat_attr_names):
        raise ValueError(f"STOP: features length {len(features)} != {len(mat_attr_names)}")
    positive, negative = [], []
    for value, mat_name in zip(features, mat_attr_names):
        task_name = mat_to_task[mat_name]
        (positive if value > 0 else negative).append(task_name)
    if set(positive) & set(negative):
        raise RuntimeError("STOP: an attribute was classified as both positive and negative")
    return positive, negative


# ----------------------------------------------------------------------
# Label Alignment Margin
# ----------------------------------------------------------------------

def alignment_margin(positive_similarity: float, negative_similarities: Sequence[float]) -> float:
    """margin = positive_similarity - mean(negative_similarities).
    Raises if negative_similarities is empty (caller must guard -- an
    object positive on every attribute has no valid margin)."""
    if len(negative_similarities) == 0:
        raise ValueError("negative_similarities must be non-empty")
    return float(positive_similarity - float(np.mean(negative_similarities)))


# ----------------------------------------------------------------------
# K-subset sampling for prompt-count stability
# ----------------------------------------------------------------------

def enumerate_or_sample_subsets(
    n: int, k: int, seed: int = 42, max_subsets: int = 100,
) -> List[Tuple[int, ...]]:
    """
    Returns a list of tuples (each a sorted subset of indices in range(n),
    size k). If C(n,k) <= max_subsets, ALL subsets are returned (exhaustive,
    no randomness). Otherwise, up to max_subsets DISTINCT subsets are drawn
    via a seeded RandomState, rejection-sampling for uniqueness.
    """
    if not (1 <= k <= n):
        raise ValueError(f"STOP: k={k} out of range [1,{n}]")
    from math import comb
    total_possible = comb(n, k)
    if total_possible <= max_subsets:
        return [tuple(c) for c in itertools.combinations(range(n), k)]

    rng = np.random.RandomState(seed)
    seen = set()
    subsets: List[Tuple[int, ...]] = []
    # Rejection sampling for uniqueness; total_possible >> max_subsets here,
    # so collisions are rare and this terminates quickly in practice.
    max_attempts = max_subsets * 200
    attempts = 0
    while len(subsets) < max_subsets and attempts < max_attempts:
        attempts += 1
        candidate = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        if candidate not in seen:
            seen.add(candidate)
            subsets.append(candidate)
    if len(subsets) < max_subsets:
        raise RuntimeError(
            f"STOP: could not draw {max_subsets} distinct size-{k} subsets "
            f"of {n} within {max_attempts} attempts (got {len(subsets)})")
    return subsets


# ----------------------------------------------------------------------
# Spatial circular-shift baseline
# ----------------------------------------------------------------------

def circular_shift_2d(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


def random_nonzero_shift(grid_h: int, grid_w: int, rng: np.random.RandomState) -> Tuple[int, int]:
    """Draws a (dy, dx) shift that is NOT the identity (0,0) modulo the grid
    size, so the circular-shift baseline is never accidentally a no-op."""
    while True:
        dy = int(rng.randint(0, grid_h))
        dx = int(rng.randint(0, grid_w))
        if (dy % grid_h, dx % grid_w) != (0, 0):
            return dy, dx


# ----------------------------------------------------------------------
# Object-crop geometry (Phase 1 of the text-alignment crop pilot)
# ----------------------------------------------------------------------

# CLIP's own normalization mean, converted to 0-255 RGB -- used as the
# neutral fill color for square-padding AND for masked-out background
# pixels, so no crop ever contains an arbitrary/out-of-distribution
# black region.
CLIP_MEAN_RGB_255 = (
    round(0.48145466 * 255), round(0.4578275 * 255), round(0.40821073 * 255),
)


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    mask : (H, W) bool.
    Returns (y0, y1, x0, x1), an INCLUSIVE pixel bounding box, or None if
    the mask is empty.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def expand_bbox(
    bbox: Tuple[int, int, int, int], frac: float, img_h: int, img_w: int,
) -> Tuple[int, int, int, int]:
    """
    Expands an inclusive (y0,y1,x0,x1) bbox by `frac` of its own height/width
    in each direction, then clips to [0, img_h-1] / [0, img_w-1] ("safely
    clip anything that falls outside the image").
    """
    y0, y1, x0, x1 = bbox
    h, w = y1 - y0 + 1, x1 - x0 + 1
    dy, dx = int(round(h * frac)), int(round(w * frac))
    return (
        max(0, y0 - dy), min(img_h - 1, y1 + dy),
        max(0, x0 - dx), min(img_w - 1, x1 + dx),
    )


def pad_to_square(image: np.ndarray, fill_color: Sequence[int] = CLIP_MEAN_RGB_255) -> np.ndarray:
    """
    image : (H, W, 3) uint8. Returns a (S, S, 3) uint8 array, S=max(H,W),
    with `image` centered and the surrounding border filled with
    `fill_color` (never plain black, so the padded region never reads as
    an out-of-distribution black square to CLIP's preprocessing).
    """
    h, w = image.shape[:2]
    s = max(h, w)
    canvas = np.empty((s, s, 3), dtype=image.dtype)
    canvas[:, :] = fill_color
    top, left = (s - h) // 2, (s - w) // 2
    canvas[top:top + h, left:left + w] = image
    return canvas


# ----------------------------------------------------------------------
# Probe utilities
# ----------------------------------------------------------------------

def safe_n_splits(n_positive_images: int, n_negative_images: int, max_splits: int = 5) -> Optional[int]:
    """
    Returns a safe n_splits for StratifiedGroupKFold (<= max_splits, and
    <= the smaller class's image count), or None if there is not enough
    data for ANY meaningful split (fewer than 2 positive or 2 negative
    images -- caller must report 'insufficient_data', not a number).
    """
    if n_positive_images < 2 or n_negative_images < 2:
        return None
    return max(2, min(max_splits, n_positive_images, n_negative_images))
