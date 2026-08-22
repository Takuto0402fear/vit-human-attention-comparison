"""
Pure, model-free helper functions for [B-2] "M-shape spatial-distribution
comparison" (L4 vs L8 vs L12 CLS->patch attention, OSIE full700, Phase 2
38x50 patch grid). No CLIP, no torch tensors here (numpy only) -- unit
tested directly in tests/test_clip_attention_b2_metrics.py.

Convention: all "raw" inputs are the un-renormalized [CLS]->patch attention
row (196 or, for this experiment, 1900=38*50 patches), matching
lib/clip_vit.py::cls_to_patch_grid's convention (head-averaged, NOT
renormalized after dropping the CLS column -- so raw.sum() < 1 in general,
the gap being the layer's CLS-self-attention mass).

Naming convention (per task spec): a value derived from p = raw/raw.sum()
is called "conditional" (i.e. conditional on attending to a patch at all),
NEVER "raw mass" -- "raw mass" is reserved for sums of the un-renormalized
`raw` array.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

EPS = 1e-8


class DegenerateAttentionError(ValueError):
    """Raised when an attention map's total mass is at/below EPS."""


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

def normalize_distribution(raw: np.ndarray, eps: float = EPS) -> np.ndarray:
    """raw -> p = raw / raw.sum(). Raises DegenerateAttentionError if the
    total mass is at/below eps (caller must treat this image/layer as
    missing, not silently divide by ~0)."""
    total = float(raw.sum())
    if total <= eps:
        raise DegenerateAttentionError(f"attention map sum={total} <= eps={eps}")
    return raw / total


# ----------------------------------------------------------------------
# Concentration / spread metrics
# ----------------------------------------------------------------------

def normalized_entropy(p: np.ndarray, n_patches: int) -> float:
    """H(p)/log(n_patches), in [0, 1]. p must already sum to 1 (a
    normalized distribution, non-negative). 0*log(0) := 0."""
    flat = p.reshape(-1).astype(np.float64)
    if flat.size != n_patches:
        raise ValueError(f"p has {flat.size} entries, expected n_patches={n_patches}")
    pos = flat[flat > 0]
    h = float(-(pos * np.log(pos)).sum())
    return h / np.log(n_patches)


def hoyer_sparsity(raw: np.ndarray) -> float:
    """(sqrt(n) - ||raw||_1/||raw||_2) / (sqrt(n) - 1), in [0, 1].
    Scale-invariant (||.||_1/||.||_2 is a ratio) -- gives the identical
    result whether called on `raw` or on the normalized `p`, so raw is
    used directly (no normalization needed, and no risk of a
    DegenerateAttentionError from a near-zero total)."""
    flat = raw.reshape(-1).astype(np.float64)
    n = flat.size
    l1 = np.abs(flat).sum()
    l2 = np.sqrt((flat ** 2).sum())
    if l2 <= EPS:
        raise DegenerateAttentionError(f"||raw||_2={l2} <= eps={EPS}")
    sqrt_n = np.sqrt(n)
    return float((sqrt_n - l1 / l2) / (sqrt_n - 1))


def max_patch_attention(p: np.ndarray) -> float:
    """max_k p_k. Lower bound is 1/n_patches (the uniform distribution)."""
    return float(p.max())


def top_k_mass(p: np.ndarray, k: int) -> float:
    """Sum of the k largest entries of p. k=190 for 'top-10% of 1900
    patches' (ceil(0.10*1900) == 190 exactly, no rounding ambiguity)."""
    flat = p.reshape(-1)
    if not (1 <= k <= flat.size):
        raise ValueError(f"k={k} out of range [1, {flat.size}]")
    part = np.partition(flat, flat.size - k)[flat.size - k:]
    return float(part.sum())


def coverage_for_mass(p: np.ndarray, target_mass: float) -> Tuple[int, float]:
    """Smallest number of patches (by descending p) whose cumulative mass
    reaches >= target_mass, and that count's area fraction (count/n).
    target_mass in (0, 1]; raises if p's total mass (should be ~1) can't
    reach target_mass due to numerical underflow (should not happen for a
    valid normalized distribution and target_mass <= 1)."""
    flat = p.reshape(-1)
    n = flat.size
    if not (0.0 < target_mass <= 1.0 + 1e-9):
        raise ValueError(f"target_mass={target_mass} out of (0, 1]")
    sorted_desc = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_desc)
    idx = np.searchsorted(cumsum, target_mass - 1e-12)
    if idx >= n:
        idx = n - 1
    count = idx + 1
    return int(count), float(count) / n


# ----------------------------------------------------------------------
# Centroid / spatial location
# ----------------------------------------------------------------------

def centroid(p: np.ndarray, grid_hw: Tuple[int, int]) -> Tuple[float, float]:
    """Returns (x, y) in [0, 1]^2: the attention-mass-weighted centroid,
    with each patch's center placed at ((col+0.5)/gw, (row+0.5)/gh) --
    i.e. row and column are each independently normalized to [0, 1], so
    the non-square 38x50 grid is treated as a unit square (this discards
    the ~0.75 physical aspect ratio of the 38x50 grid vs. the 600x800
    image; deliberate choice, see report.md)."""
    gh, gw = grid_hw
    grid = p.reshape(gh, gw)
    if grid.shape != (gh, gw):
        raise ValueError(f"p has shape {p.shape}, expected ({gh},{gw}) after reshape")
    rows = (np.arange(gh, dtype=np.float64) + 0.5) / gh
    cols = (np.arange(gw, dtype=np.float64) + 0.5) / gw
    y = float((grid.sum(axis=1) * rows).sum())
    x = float((grid.sum(axis=0) * cols).sum())
    return x, y


def normalized_center_distance(centroid_xy: Tuple[float, float]) -> float:
    """Euclidean distance from centroid_xy to the unit square's center
    (0.5, 0.5), normalized by the half-diagonal sqrt(0.5) so the result
    is in [0, 1] (1 == a corner)."""
    x, y = centroid_xy
    dist = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return float(dist / np.sqrt(0.5))


# ----------------------------------------------------------------------
# Foreground / background (raw mass, conditional mass, area-normalized
# enrichment). `coverage` is a (gh, gw) float in [0, 1] -- each patch's
# fractional area covered by the foreground mask (e.g.
# lib.osie_text_alignment.mask_to_patch_weights output).
# ----------------------------------------------------------------------

def cls_self_mass_from_raw(raw: np.ndarray) -> float:
    """cls_self_mass = 1 - patch_mass_raw. Exact up to float32 softmax-sum
    rounding (~1e-7): the full [CLS] row (self + all patches) is a
    head-averaged softmax output, which sums to 1 by construction -- so
    this does NOT require a fresh forward pass to recover full_row_sum
    (verified bit-exact against an actual full_row_sum recompute for 10
    pilot images: max deviation from 1.0 was 1.19e-7, i.e. float32 eps)."""
    return float(1.0 - raw.reshape(-1).sum())


DEGENERATE_AREA_FRACTION = 1.0 / 1900  # < 1 whole patch of area -- treat as missing


def foreground_background_metrics(
    raw: np.ndarray, coverage: np.ndarray, area_threshold: float = DEGENERATE_AREA_FRACTION,
) -> dict:
    """
    raw      : (n_patches,) or (gh, gw) raw [CLS]->patch attention.
    coverage : same shape, in [0, 1] -- foreground area-coverage fraction
        per patch (0 == pure background, 1 == pure foreground).

    Returns a dict with:
      patch_mass_raw, foreground_area_fraction, background_area_fraction,
      foreground_absolute_mass  (= sum(raw * coverage), in raw-mass units),
      foreground_conditional_mass (= sum(p * coverage), p = raw/raw.sum()),
      background_conditional_mass (= 1 - foreground_conditional_mass;
          the exact complement -- NOT independently tested downstream),
      foreground_enrichment (= foreground_conditional_mass / foreground_area_fraction,
          or None if foreground_area_fraction < area_threshold),
      is_degenerate (True if foreground OR background area_fraction < area_threshold
          -- caller must exclude this image/layer from fg/bg-dependent
          statistics, not treat it as zero).
    """
    raw_flat = raw.reshape(-1).astype(np.float64)
    cov_flat = coverage.reshape(-1).astype(np.float64)
    if raw_flat.shape != cov_flat.shape:
        raise ValueError(f"raw shape {raw.shape} != coverage shape {coverage.shape}")
    if np.any(cov_flat < -1e-9) or np.any(cov_flat > 1.0 + 1e-9):
        raise ValueError("coverage must be in [0, 1]")

    n = raw_flat.size
    fg_area_fraction = float(cov_flat.sum()) / n
    bg_area_fraction = 1.0 - fg_area_fraction
    patch_mass_raw = float(raw_flat.sum())

    p = normalize_distribution(raw_flat)
    foreground_absolute_mass = float((raw_flat * cov_flat).sum())
    foreground_conditional_mass = float((p * cov_flat).sum())
    background_conditional_mass = 1.0 - foreground_conditional_mass

    is_degenerate = (fg_area_fraction < area_threshold) or (bg_area_fraction < area_threshold)
    foreground_enrichment = (
        None if fg_area_fraction < area_threshold
        else foreground_conditional_mass / fg_area_fraction
    )

    return {
        "patch_mass_raw": patch_mass_raw,
        "foreground_area_fraction": fg_area_fraction,
        "background_area_fraction": bg_area_fraction,
        "foreground_absolute_mass": foreground_absolute_mass,
        "foreground_conditional_mass": foreground_conditional_mass,
        "background_conditional_mass": background_conditional_mass,
        "foreground_enrichment": foreground_enrichment,
        "is_degenerate": bool(is_degenerate),
    }


# ----------------------------------------------------------------------
# Layer-pair spatial similarity
# ----------------------------------------------------------------------

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two equal-shape attention maps.
    Scale-invariant (r(cX,Y) == r(X,Y) for c>0), so raw and normalized
    inputs give identical results -- raw is used directly by convention.
    Raises ValueError if either input has zero variance (undefined
    correlation), rather than silently returning NaN."""
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    if x.std() <= EPS or y.std() <= EPS:
        raise ValueError("Pearson correlation undefined: a zero-variance (e.g. uniform) map")
    return float(np.corrcoef(x, y)[0, 1])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a, b) = a.b / (||a|| ||b||). Scale-invariant, same convention
    as pearson_corr (raw or normalized input give the identical result)."""
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx <= EPS or ny <= EPS:
        raise DegenerateAttentionError("cosine similarity undefined: a near-zero-norm map")
    return float(np.dot(x, y) / (nx * ny))


def jensen_shannon_divergence_normalized(p: np.ndarray, q: np.ndarray) -> float:
    """JSD(p,q) / log(2), in [0, 1]. p, q must already be normalized
    distributions (sum to 1, non-negative). 0*log(0/anything) := 0."""
    pf = p.reshape(-1).astype(np.float64)
    qf = q.reshape(-1).astype(np.float64)
    m = 0.5 * (pf + qf)

    def _kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log(x[mask] / y[mask])).sum())

    jsd = 0.5 * _kl(pf, m) + 0.5 * _kl(qf, m)
    return jsd / np.log(2)


# ----------------------------------------------------------------------
# Fisher z (for averaging/comparing Pearson correlations across images)
# ----------------------------------------------------------------------

def fisher_z(r: float, clip: float = 1 - 1e-9) -> float:
    r_clipped = float(np.clip(r, -clip, clip))
    return float(0.5 * np.log((1 + r_clipped) / (1 - r_clipped)))


def fisher_z_inverse(z: float) -> float:
    e2z = np.exp(2 * z)
    return float((e2z - 1) / (e2z + 1))


# ----------------------------------------------------------------------
# Statistics: paired bootstrap CI, sign-flip permutation, effect size, BH-FDR
# ----------------------------------------------------------------------

def cohens_dz(diff: np.ndarray) -> float:
    """Standardized mean difference for paired data: mean(diff)/std(diff, ddof=1).
    Returns NaN if std is ~0 (all diffs identical, e.g. n<2 or a constant
    diff) -- caller must treat NaN as 'undefined', not 0."""
    d = np.asarray(diff, dtype=np.float64)
    if d.size < 2:
        return float("nan")
    s = d.std(ddof=1)
    if s <= EPS:
        return float("nan")
    return float(d.mean() / s)


def bootstrap_ci_mean_diff(
    diff: np.ndarray, seed: int, n_boot: int = 10000, alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for mean(diff), resampling image
    indices with replacement. Deterministic given (diff, seed, n_boot)."""
    d = np.asarray(diff, dtype=np.float64)
    n = d.size
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def sign_flip_test(
    diff: np.ndarray, seed: int, n_reps: int = 10000,
) -> Tuple[float, float]:
    """Paired sign-flip permutation test on mean(diff) under the null of
    no systematic direction (each image's diff sign is independently,
    randomly flipped). Two-sided Monte Carlo p-value with the standard
    +1 correction: p = (1 + #{|perm_stat| >= |obs_stat|}) / (1 + n_reps),
    so p is never exactly 0 and is a valid (slightly conservative) p-value
    for any n_reps. Returns (observed_mean_diff, p_value)."""
    d = np.asarray(diff, dtype=np.float64)
    n = d.size
    obs = d.mean()
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_reps, n))
    perm_means = (d[None, :] * signs).mean(axis=1)
    n_extreme = int((np.abs(perm_means) >= np.abs(obs)).sum())
    p = (1 + n_extreme) / (1 + n_reps)
    return float(obs), float(p)


def refuse_if_exists(paths: Sequence[str]) -> None:
    """Raises RuntimeError listing every path in `paths` that already
    exists as a file. Per task spec: a script must stop only when a file
    it is about to WRITE already exists -- never merely because its
    parent output directory exists."""
    existing = [p for p in paths if os.path.isfile(p)]
    if existing:
        raise RuntimeError(
            "STOP: output file(s) already exist, refusing to overwrite:\n  "
            + "\n  ".join(existing))


def fisher_z_array(r: np.ndarray, clip: float = 1 - 1e-9) -> np.ndarray:
    """Vectorized fisher_z, for transforming a whole array of per-image
    correlations at once (used by the repeated-shuffle robustness check,
    where this runs inside a per-derangement loop)."""
    r_clipped = np.clip(np.asarray(r, dtype=np.float64), -clip, clip)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))


# ----------------------------------------------------------------------
# Repeated-shuffle robustness check: many independent derangements
# (fixed-point-free permutations) of the 700 images, instead of the
# single seed=42 derangement used by the primary Family-2 test.
# ----------------------------------------------------------------------

def fixed_point_free_permutation(n: int, seed: int, max_attempts: int = 1000) -> np.ndarray:
    """A single seeded derangement of range(n) (a permutation with NO
    index i such that perm[i] == i), via rejection sampling on
    np.random.RandomState(seed).permutation(n). Deterministic: the same
    (n, seed) always returns the identical array."""
    rng = np.random.RandomState(seed)
    for _ in range(max_attempts):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    raise RuntimeError(
        f"STOP: could not draw a fixed-point-free permutation of size {n} "
        f"within {max_attempts} attempts (seed={seed})")


def derangement_seed(base_seed: int, k: int) -> int:
    """Deterministic per-index sub-seed for the k-th of many derangements
    drawn from a given base_seed. The SAME (base_seed, k) always yields
    the same seed (hence the same derangement); a DIFFERENT base_seed
    yields an entirely different family of 32-bit seeds."""
    return (base_seed * 1_000_003 + k) % (2 ** 32 - 1)


def iter_derangements(n: int, base_seed: int, n_deran: int):
    """Yields n_deran fixed-point-free permutations of range(n), ONE AT A
    TIME (never all held in memory simultaneously). Fully reproducible:
    the same (n, base_seed, n_deran) always yields an identical sequence
    of derangements; a different base_seed yields a different sequence."""
    for k in range(n_deran):
        yield fixed_point_free_permutation(n, derangement_seed(base_seed, k))


def dataset_mean_pearson_for_permutation(
    centered_a: np.ndarray, norms_a: np.ndarray,
    centered_b: np.ndarray, norms_b: np.ndarray, perm: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """centered_a/centered_b : (n_images, n_patches) row-mean-centered raw
    attention for layers A and B. norms_a/norms_b : (n_images,) L2 norm of
    each centered row. Returns (dataset_mean, per_image_r) where
    dataset_mean is the Fisher-z-averaged, back-transformed mean Pearson
    correlation between image i's layer-A map and image perm[i]'s
    layer-B map, over all i."""
    cb = centered_b[perm]
    nb = norms_b[perm]
    denom = norms_a * nb
    if np.any(denom <= EPS):
        raise DegenerateAttentionError("zero-variance row in dataset_mean_pearson_for_permutation")
    r = np.sum(centered_a * cb, axis=1) / denom
    z_mean = fisher_z_array(r).mean()
    return fisher_z_inverse(float(z_mean)), r


def dataset_mean_cosine_for_permutation(
    a: np.ndarray, norms_a: np.ndarray, b: np.ndarray, norms_b: np.ndarray, perm: np.ndarray,
) -> Tuple[float, np.ndarray]:
    bp = b[perm]
    nbp = norms_b[perm]
    denom = norms_a * nbp
    if np.any(denom <= EPS):
        raise DegenerateAttentionError("zero-norm row in dataset_mean_cosine_for_permutation")
    cos = np.sum(a * bp, axis=1) / denom
    return float(cos.mean()), cos


def dataset_mean_jsd_for_permutation(
    pa: np.ndarray, log_pa: np.ndarray, pb: np.ndarray, log_pb: np.ndarray, perm: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """pa/pb : (n_images, n_patches) normalized distributions (all
    entries strictly > 0 for real softmax attention -- no 0*log(0)
    handling needed here; use jensen_shannon_divergence_normalized for
    the general case with possible zeros). log_pa/log_pb : elementwise
    log(pa)/log(pb), precomputed once per layer and reused across every
    derangement (only log(m) is recomputed per derangement)."""
    pbp = pb[perm]
    log_pbp = log_pb[perm]
    m = 0.5 * (pa + pbp)
    log_m = np.log(m)
    kl_p = np.sum(pa * (log_pa - log_m), axis=1)
    kl_q = np.sum(pbp * (log_pbp - log_m), axis=1)
    jsd = 0.5 * (kl_p + kl_q) / np.log(2)
    return float(jsd.mean()), jsd


def monte_carlo_one_sided_p(observed: float, null_distribution: np.ndarray, direction: str) -> float:
    """+1-corrected one-sided Monte Carlo p-value against an empirical
    null distribution. direction='greater' -> p = P(null >= observed);
    direction='less' -> p = P(null <= observed). Never exactly 0."""
    null = np.asarray(null_distribution, dtype=np.float64)
    n = null.size
    if direction == "greater":
        n_extreme = int((null >= observed).sum())
    elif direction == "less":
        n_extreme = int((null <= observed).sum())
    else:
        raise ValueError(f"direction must be 'greater' or 'less', got {direction!r}")
    return (1 + n_extreme) / (1 + n)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (same order as input).
    NaN p-values pass through as NaN and are excluded from the family
    size used for adjustment."""
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan)
    valid_mask = ~np.isnan(p)
    valid_p = p[valid_mask]
    m = valid_p.size
    if m == 0:
        return q

    order = np.argsort(valid_p)
    ranked = valid_p[order]
    raw_q = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity from the largest p-value downward
    adj_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    adj_q = np.clip(adj_q, 0.0, 1.0)

    result_valid = np.empty(m, dtype=np.float64)
    result_valid[order] = adj_q
    q[valid_mask] = result_valid
    return q
