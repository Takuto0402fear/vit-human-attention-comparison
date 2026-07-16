"""
Center-bias direct verification (Yoshida-sensei's MTG request).

Purpose
-------
Exp B / Exp B-control showed that firstk (first-2 fixations) has higher
NSS / AUC-Judd against ViT attention than 'all' (all fixations), and that
this advantage mostly disappears once sAUC (a centre-bias control metric)
is used -- except for a small residual at layers 8-10. That is an
*indirect* argument (sAUC's shuffled negatives implicitly subtract out
whatever centre-bias is shared across images).

This script is the *direct* check requested by Yoshida-sensei: does the
firstk fixation distribution actually sit closer to the image centre and
have a narrower spread than the 'all' fixation distribution? It does NOT
touch the ViT / attention side at all -- it only re-describes the human
fixation-map ground truth that Exp A/B/B-control already use.

Reuses (no recomputation):
  - The 700 OSIE healthy-group images and subjects (same population as
    Exp A/B/B-control).
  - The firstk / all / randk / lastk point definitions and coordinate
    conversion (0-indexed pixel maps) already stored in
    outputs/fixmaps/healthy/*.npz (points, points_firstk, points_randk,
    points_lastk).

Primary comparison: firstk vs all (paired by image).
Supplementary (context only, still image-paired, not fixation-pooled):
  firstk vs randk, firstk vs lastk.

Outputs -> outputs/center_bias/
"""

import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import glob
import json
import os
import time

import numpy as np
import scipy.ndimage
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ======================= CONFIG =======================
GROUP        = "healthy"
IMG_W        = 800
IMG_H        = 600
SEED         = 42
N_BOOT       = 10000

FIXMAP_DIR   = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
OUT_DIR      = r"C:\Users\user\gaze\outputs\center_bias"
EXPB_CONTROL_SAUC_CSV = (
    r"C:\Users\user\gaze\outputs\expB_control\healthy\stats_firstk_vs_randk.csv"
)

PRIMARY_CONDITIONS = ["all", "firstk"]
SUPP_CONDITIONS    = ["randk", "lastk"]
ALL_CONDITIONS     = PRIMARY_CONDITIONS + SUPP_CONDITIONS

COND_KEY = {
    "all":    "points",
    "firstk": "points_firstk",
    "randk":  "points_randk",
    "lastk":  "points_lastk",
}
COND_LABEL = {
    "all":    "All fixations",
    "firstk": "First-K (K=2)",
    "randk":  "Random-K (K=2)",
    "lastk":  "Last-K (K=2)",
}
COND_COLOR = {
    "all":    "steelblue",
    "firstk": "coral",
    "randk":  "mediumseagreen",
    "lastk":  "mediumpurple",
}

CENTER_BOX_HALF   = 0.10   # central 20% x 20% box: |x-0.5|<=0.10 & |y-0.5|<=0.10
CENTER_RADIUS     = 0.10   # central circle, radius 0.10 in normalized units

HEAT_BINS_X      = 100     # 800/100 = 8 px/bin
HEAT_BINS_Y      = 75      # 600/75  = 8 px/bin  (square physical bins)
HEAT_SIGMA_BINS  = 2.5     # Gaussian blur (visualization only), same for both panels

IMAGE_METRICS = ["mean_center_distance", "pct_center_box20", "pct_center_radius01",
                  "var_x", "var_y", "ellipse_area_1sigma"]
# ======================================================


# ==============================================================
# Step 1: Load firstk/all/randk/lastk points from existing fixmaps,
#         normalized to [0,1] x [0,1] (x/W, y/H). No new fixation
#         definitions, no ViT recomputation.
# ==============================================================

def load_condition_points():
    """
    Returns
    -------
    per_image : {condition: {stem: (N,2) float32 normalized (x,y)}}
    stems     : sorted list of image stems (700)
    audit     : dict of counters for the completeness/undercount log
    """
    per_image = {c: {} for c in ALL_CONDITIONS}
    files = sorted(glob.glob(os.path.join(FIXMAP_DIR, "*.npz")))

    audit = {
        "n_images": 0,
        "missing_key": {c: 0 for c in ALL_CONDITIONS},
        "empty_points": {c: 0 for c in ALL_CONDITIONS},
        "out_of_range": {c: 0 for c in ALL_CONDITIONS},
        "raw_n_fix_all": 0,          # d["n_fix"], pre-collision
        "raw_n_fix_firstk_expected": 0,  # n_subjects * 2 (K=2), pre-collision
        "points_sum_all": 0,          # d["points"].sum(), post-collision (unique px)
        "points_sum_firstk": 0,
    }

    stems = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)
        stems.append(stem)
        audit["n_images"] += 1
        audit["raw_n_fix_all"] += int(d["n_fix"])
        audit["raw_n_fix_firstk_expected"] += int(d["n_subjects"]) * 2

        for c in ALL_CONDITIONS:
            key = COND_KEY[c]
            if key not in d.files:
                audit["missing_key"][c] += 1
                continue
            ys, xs = np.where(d[key])
            if len(xs) == 0:
                audit["empty_points"][c] += 1
                per_image[c][stem] = np.zeros((0, 2), dtype=np.float32)
                continue
            x_norm = (xs.astype(np.float64) + 0.5) / IMG_W
            y_norm = (ys.astype(np.float64) + 0.5) / IMG_H
            if x_norm.min() < 0 or x_norm.max() > 1 or y_norm.min() < 0 or y_norm.max() > 1:
                audit["out_of_range"][c] += 1
            pts = np.column_stack([x_norm, y_norm]).astype(np.float32)
            per_image[c][stem] = pts
            if c == "all":
                audit["points_sum_all"] += len(pts)
            elif c == "firstk":
                audit["points_sum_firstk"] += len(pts)

    stems = sorted(stems)
    return per_image, stems, audit


# ==============================================================
# Step 2: Gaussian fit (pooled, per condition) -- statistics on the
#         RAW fixation points, not on the smoothed visualization map.
# ==============================================================

def gaussian_fit(points):
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    n = len(x)
    mu_x, mu_y = float(x.mean()), float(y.mean())
    cov = np.cov(x, y, ddof=1)
    var_x, var_y, cov_xy = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    lam1, lam2 = float(eigvals[0]), float(eigvals[1])
    angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    ellipse_area = float(np.pi * np.sqrt(max(lam1, 0) * max(lam2, 0)))
    corr_xy = cov_xy / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else np.nan
    return {
        "n_points": n, "mu_x": mu_x, "mu_y": mu_y,
        "sigma_x": np.sqrt(var_x), "sigma_y": np.sqrt(var_y),
        "cov_xy": cov_xy, "corr_xy": corr_xy,
        "lambda1": lam1, "lambda2": lam2, "angle_deg": angle_deg,
        "ellipse_area_1sigma": ellipse_area,
    }


# ==============================================================
# Step 3: Direct, non-Gaussian metrics -- pooled (descriptive) and
#         per-image (for paired inferential stats).
# ==============================================================

def center_distance(points):
    return np.sqrt((points[:, 0] - 0.5) ** 2 + (points[:, 1] - 0.5) ** 2)


def pooled_direct_summary(points):
    d = center_distance(points)
    x, y = points[:, 0], points[:, 1]
    in_box = (np.abs(x - 0.5) <= CENTER_BOX_HALF) & (np.abs(y - 0.5) <= CENTER_BOX_HALF)
    return {
        "n_points": len(points),
        "mean_center_distance": float(d.mean()),
        "median_center_distance": float(np.median(d)),
        "pct_center_box20": float(in_box.mean()),
        "pct_center_radius01": float((d <= CENTER_RADIUS).mean()),
    }


def per_image_metrics(per_image, stems):
    """rows: one per (condition, image) with n_points >= 1."""
    rows = []
    for c in ALL_CONDITIONS:
        for stem in stems:
            pts = per_image[c].get(stem)
            if pts is None or len(pts) == 0:
                continue
            x, y = pts[:, 0], pts[:, 1]
            d = center_distance(pts)
            n = len(pts)
            in_box = (np.abs(x - 0.5) <= CENTER_BOX_HALF) & (np.abs(y - 0.5) <= CENTER_BOX_HALF)
            in_radius = d <= CENTER_RADIUS
            if n >= 2:
                cov = np.cov(x, y, ddof=1)
                var_x, var_y, cov_xy = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.clip(eigvals, 0, None)
                ellipse_area = float(np.pi * np.sqrt(eigvals[0] * eigvals[1]))
            else:
                var_x = var_y = cov_xy = ellipse_area = np.nan
            rows.append({
                "image": stem, "condition": c, "n_points": n,
                "mean_center_distance": float(d.mean()),
                "pct_center_box20": float(in_box.mean()),
                "pct_center_radius01": float(in_radius.mean()),
                "var_x": var_x, "var_y": var_y, "cov_xy": cov_xy,
                "ellipse_area_1sigma": ellipse_area,
            })
    return rows


# ==============================================================
# Statistics (same pattern as run_expB*.py: paired Wilcoxon +
# Holm correction + bootstrap CI on the difference).
# ==============================================================

def wilcoxon_test(a, b, alternative="two-sided"):
    diff = np.asarray(a) - np.asarray(b)
    diff_nz = diff[diff != 0]
    n = len(diff_nz)
    if n < 10:
        return np.nan, np.nan, np.nan
    res = sp_stats.wilcoxon(diff_nz, alternative=alternative)
    stat, p = res.statistic, res.pvalue
    total_rank = n * (n + 1) / 2
    r = abs(1 - (2 * stat) / total_rank)
    return float(stat), float(p), float(r)


def holm_correction(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, pvals[idx] * (n - rank))
    running_max = 0
    for idx in order:
        running_max = max(running_max, corrected[idx])
        corrected[idx] = running_max
    return corrected


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values)
    n = len(arr)
    means = np.array([rng.choice(arr, n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_compare(image_level, cond_a, cond_b, role, stems):
    """
    image_level: {condition: {metric: {stem: value}}}
    Paired (image-matched) comparison, cond_a - cond_b, per metric.
    NaNs (images with < 2 points, i.e. only for var/ellipse metrics) are
    dropped pairwise per metric.
    """
    rows = []
    for m in IMAGE_METRICS:
        a = np.array([image_level[cond_a][m][s] for s in stems])
        b = np.array([image_level[cond_b][m][s] for s in stems])
        mask = ~(np.isnan(a) | np.isnan(b))
        a2, b2 = a[mask], b[mask]
        stat, p, r = wilcoxon_test(a2, b2)
        delta = float(np.mean(a2) - np.mean(b2))
        ci_lo, ci_hi = bootstrap_ci(a2 - b2)
        rows.append({
            "role": role, "comparison": f"{cond_a}_vs_{cond_b}", "metric": m,
            "n_pairs": int(mask.sum()),
            f"mean_{cond_a}": float(np.mean(a2)), f"mean_{cond_b}": float(np.mean(b2)),
            "delta": delta, "delta_ci_lo": ci_lo, "delta_ci_hi": ci_hi,
            "statistic": stat, "p": p, "p_corrected": np.nan, "effect_r": r,
        })
    ps = np.array([r["p"] for r in rows])
    ps_corr = holm_correction(ps)
    for r, pc in zip(rows, ps_corr):
        r["p_corrected"] = pc
    return rows


# ==============================================================
# Plot 1: side-by-side density heatmaps, firstk vs all, same bins /
# sigma / colorscale / coordinate range, with 1-sigma ellipse overlay.
# ==============================================================

def build_density(points):
    H, _, _ = np.histogram2d(points[:, 0], points[:, 1],
                              bins=[HEAT_BINS_X, HEAT_BINS_Y],
                              range=[[0, 1], [0, 1]])
    H_smooth = scipy.ndimage.gaussian_filter(H, sigma=HEAT_SIGMA_BINS)
    s = H_smooth.sum()
    if s > 0:
        H_smooth /= s
    return H_smooth  # (bins_x, bins_y)


def _add_ellipse(ax, fit, color, lw=2.0, ls="--"):
    ell = Ellipse((fit["mu_x"], fit["mu_y"]),
                  width=2 * np.sqrt(max(fit["lambda1"], 0)),
                  height=2 * np.sqrt(max(fit["lambda2"], 0)),
                  angle=fit["angle_deg"],
                  edgecolor=color, facecolor="none", lw=lw, ls=ls)
    ax.add_patch(ell)


def plot_heatmaps(densities, fits, n_img, out_path):
    conds = ["firstk", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    vmax = max(densities[c].max() for c in conds)
    im = None
    for ax, c in zip(axes, conds):
        im = ax.imshow(densities[c].T, origin="lower", extent=[0, 1, 0, 1],
                        aspect=IMG_H / IMG_W, cmap="inferno", vmin=0, vmax=vmax)
        _add_ellipse(ax, fits[c], "white", lw=2.2, ls="--")
        ax.plot(fits[c]["mu_x"], fits[c]["mu_y"], "+", color="white", ms=12, mew=2)
        ax.plot(0.5, 0.5, "x", color="cyan", ms=9, mew=2, label="image center (0.5,0.5)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("x / W")
        ax.set_ylabel("y / H")
        ax.set_title(f"{COND_LABEL[c]}\n1$\\sigma$ ellipse area = "
                      f"{fits[c]['ellipse_area_1sigma']:.4f}")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    fig.colorbar(im, ax=axes, shrink=0.8, label="normalized fixation density")
    fig.suptitle(f"Center-bias direct check: fixation density, First-K vs All "
                 f"({GROUP}, {n_img} images)",
                 fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================
# Plot 2: Gaussian-width comparison (overlaid ellipses + bar chart).
# ==============================================================

def plot_width_comparison(fits, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    plot_conds = ["all", "firstk"] + [c for c in SUPP_CONDITIONS if c in fits]
    for c in plot_conds:
        primary = c in PRIMARY_CONDITIONS
        _add_ellipse(ax, fits[c], COND_COLOR[c],
                     lw=2.5 if primary else 1.3, ls="-" if primary else "--")
        ax.plot(fits[c]["mu_x"], fits[c]["mu_y"], "+", color=COND_COLOR[c], ms=10, mew=2,
                label=COND_LABEL[c])
    ax.plot(0.5, 0.5, "x", color="black", ms=8, mew=2, label="image center")
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)
    ax.set_aspect(IMG_H / IMG_W)
    ax.set_xlabel("x / W")
    ax.set_ylabel("y / H")
    ax.set_title("1$\\sigma$ ellipses (overlaid, zoomed near center)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    bar_metrics = ["sigma_x", "sigma_y", "ellipse_area_1sigma"]
    bar_labels = ["$\\sigma_x$", "$\\sigma_y$", "1$\\sigma$ ellipse area"]
    x = np.arange(len(bar_metrics))
    n_c = len(plot_conds)
    width = 0.8 / n_c
    for i, c in enumerate(plot_conds):
        primary = c in PRIMARY_CONDITIONS
        vals = [fits[c][m] for m in bar_metrics]
        ax2.bar(x + i * width, vals, width=width, color=COND_COLOR[c],
                alpha=1.0 if primary else 0.55, label=COND_LABEL[c])
    ax2.set_xticks(x + width * (n_c - 1) / 2)
    ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel("value (normalized units)")
    ax2.set_title("Gaussian width: narrower vs. broader")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"2D Gaussian width comparison ({GROUP}, 700 images)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================
# Plot 3: center-distance distribution (pooled ECDF + paired
# per-image violin/box).
# ==============================================================

def plot_center_distance(per_image_points, image_level, stems, primary_stats, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    plot_conds = ["all", "firstk"] + [c for c in SUPP_CONDITIONS if c in per_image_points]
    for c in plot_conds:
        pooled = np.concatenate([per_image_points[c][s] for s in stems
                                  if len(per_image_points[c][s])])
        d_sorted = np.sort(center_distance(pooled))
        ecdf = np.arange(1, len(d_sorted) + 1) / len(d_sorted)
        primary = c in PRIMARY_CONDITIONS
        ax.plot(d_sorted, ecdf, color=COND_COLOR[c],
                lw=2.2 if primary else 1.3, ls="-" if primary else "--",
                label=COND_LABEL[c])
    ax.axvline(CENTER_RADIUS, color="gray", ls=":", lw=1,
               label=f"center radius = {CENTER_RADIUS}")
    ax.set_xlabel("normalized center distance d = sqrt((x-0.5)^2+(y-0.5)^2)")
    ax.set_ylabel("ECDF (pooled fixation-level, descriptive)")
    ax.set_title("Pooled center-distance ECDF")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    data = [np.array([image_level[c]["mean_center_distance"][s] for s in stems])
            for c in ["all", "firstk"]]
    parts = ax2.violinplot(data, showmeans=True, showextrema=True)
    for pc, c in zip(parts["bodies"], ["all", "firstk"]):
        pc.set_facecolor(COND_COLOR[c])
        pc.set_alpha(0.5)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([COND_LABEL["all"], COND_LABEL["firstk"]])
    ax2.set_ylabel("per-image mean center distance")
    stat_row = [r for r in primary_stats if r["metric"] == "mean_center_distance"][0]
    ax2.set_title(f"Paired per-image mean (n={stat_row['n_pairs']})\n"
                  f"delta(firstk-all)={stat_row['delta']:+.4f}  "
                  f"p_holm={stat_row['p_corrected']:.2e}  r={stat_row['effect_r']:.3f}")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Center-distance distribution ({GROUP}, 700 images)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================
# Main
# ==============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    # -- Step 1: load points (reuse existing definitions, no recompute) --
    print("Loading firstk/all/randk/lastk points from existing fixmaps ...")
    per_image, stems, audit = load_condition_points()
    n_img = len(stems)
    print(f"  {n_img} images loaded.")

    for c in ALL_CONDITIONS:
        if audit["missing_key"][c] or audit["empty_points"][c] or audit["out_of_range"][c]:
            print(f"  WARNING [{c}]: missing_key={audit['missing_key'][c]}, "
                  f"empty_points={audit['empty_points'][c]}, "
                  f"out_of_range={audit['out_of_range'][c]}")
    collisions_all = audit["raw_n_fix_all"] - audit["points_sum_all"]
    collisions_fk = audit["raw_n_fix_firstk_expected"] - audit["points_sum_firstk"]
    print(f"  [all]    raw fixations={audit['raw_n_fix_all']}, "
          f"unique-pixel points={audit['points_sum_all']} "
          f"(collision undercount={collisions_all}, "
          f"{100*collisions_all/audit['raw_n_fix_all']:.2f}%)")
    print(f"  [firstk] expected (n_subjects*2)={audit['raw_n_fix_firstk_expected']}, "
          f"unique-pixel points={audit['points_sum_firstk']} "
          f"(collision undercount={collisions_fk}, "
          f"{100*collisions_fk/audit['raw_n_fix_firstk_expected']:.2f}%)")
    print("  (collisions = >=2 fixations landing on the same 1-px bin in the "
          "existing binary point maps; same convention as Exp A/B/B-control, "
          "not introduced by this script.)")

    supp_available = [c for c in SUPP_CONDITIONS
                      if audit["missing_key"][c] == 0]
    if supp_available != SUPP_CONDITIONS:
        print(f"  Supplementary conditions available: {supp_available} "
              f"(dropped: {set(SUPP_CONDITIONS) - set(supp_available)})")

    active_conditions = PRIMARY_CONDITIONS + supp_available

    # -- Step 2: pooled Gaussian fits + pooled direct summary --
    print("\nFitting pooled 2D Gaussian per condition ...")
    pooled_points = {c: np.concatenate([per_image[c][s] for s in stems])
                     for c in active_conditions}
    fits = {c: gaussian_fit(pooled_points[c]) for c in active_conditions}
    direct_summary = {c: pooled_direct_summary(pooled_points[c]) for c in active_conditions}

    for c in active_conditions:
        fit = fits[c]
        print(f"  {c:7s}: mu=({fit['mu_x']:.4f},{fit['mu_y']:.4f})  "
              f"sigma=({fit['sigma_x']:.4f},{fit['sigma_y']:.4f})  "
              f"1sigma_area={fit['ellipse_area_1sigma']:.5f}")

    # -- Step 3: per-image direct metrics --
    print("\nComputing per-image direct metrics ...")
    image_rows = per_image_metrics(per_image, stems)
    image_level = {c: {m: {} for m in IMAGE_METRICS} for c in active_conditions}
    for row in image_rows:
        c = row["condition"]
        for m in IMAGE_METRICS:
            image_level[c][m][row["image"]] = row[m]

    for c in active_conditions:
        n_present = len(image_level[c]["mean_center_distance"])
        if n_present != n_img:
            print(f"  WARNING: condition {c} has {n_present}/{n_img} images "
                  f"with >=1 fixation point.")

    # -- Step 4: paired statistics --
    print("\nPaired statistics (image-matched, firstk vs all = primary) ...")
    stats_rows = paired_compare(image_level, "firstk", "all", "primary", stems)
    for r in stats_rows:
        sig = ("***" if r["p_corrected"] < 0.001 else "**" if r["p_corrected"] < 0.01
               else "*" if r["p_corrected"] < 0.05 else "ns")
        print(f"  [primary] {r['metric']:24s} delta(fk-all)={r['delta']:+.5f} "
              f"[{r['delta_ci_lo']:+.5f},{r['delta_ci_hi']:+.5f}]  "
              f"p_holm={r['p_corrected']:.2e} r={r['effect_r']:.3f} {sig}")

    for c in supp_available:
        rows = paired_compare(image_level, "firstk", c, "supplementary", stems)
        stats_rows += rows
        for r in rows:
            sig = ("***" if r["p_corrected"] < 0.001 else "**" if r["p_corrected"] < 0.01
                   else "*" if r["p_corrected"] < 0.05 else "ns")
            print(f"  [supp firstk_vs_{c}] {r['metric']:24s} "
                  f"delta={r['delta']:+.5f}  p_holm={r['p_corrected']:.2e} "
                  f"r={r['effect_r']:.3f} {sig}")

    # -- Step 5: save CSVs --
    csv_gauss = os.path.join(OUT_DIR, "gaussian_fit_summary.csv")
    gauss_fields = ["role", "condition", "n_points", "mu_x", "mu_y", "sigma_x", "sigma_y",
                    "cov_xy", "corr_xy", "lambda1", "lambda2", "angle_deg",
                    "ellipse_area_1sigma"]
    with open(csv_gauss, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gauss_fields)
        w.writeheader()
        for c in active_conditions:
            row = {"role": "primary" if c in PRIMARY_CONDITIONS else "supplementary",
                   "condition": c, **fits[c]}
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"\nSaved: {csv_gauss}")

    csv_summary = os.path.join(OUT_DIR, "center_bias_summary.csv")
    summary_fields = ["role", "condition", "n_images", "n_points",
                       "mean_center_distance", "median_center_distance",
                       "pct_center_box20", "pct_center_radius01"]
    with open(csv_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for c in active_conditions:
            row = {"role": "primary" if c in PRIMARY_CONDITIONS else "supplementary",
                   "condition": c, "n_images": n_img, **direct_summary[c]}
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"Saved: {csv_summary}")

    csv_image = os.path.join(OUT_DIR, "center_bias_image_level.csv")
    image_fields = ["image", "condition", "n_points", "mean_center_distance",
                     "pct_center_box20", "pct_center_radius01",
                     "var_x", "var_y", "cov_xy", "ellipse_area_1sigma"]
    with open(csv_image, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=image_fields)
        w.writeheader()
        for row in image_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"Saved: {csv_image} ({len(image_rows)} rows)")

    csv_stats = os.path.join(OUT_DIR, "center_bias_stats.csv")
    stats_fields = ["role", "comparison", "metric", "n_pairs",
                     "mean_firstk", "mean_all", "mean_randk", "mean_lastk",
                     "delta", "delta_ci_lo", "delta_ci_hi",
                     "statistic", "p", "p_corrected", "effect_r"]
    with open(csv_stats, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=stats_fields)
        w.writeheader()
        for row in stats_rows:
            full = {k: row.get(k, "") for k in stats_fields}
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                        for k, v in full.items()})
    print(f"Saved: {csv_stats}")

    # -- Step 6: plots --
    print("\nBuilding plots ...")
    densities = {c: build_density(pooled_points[c]) for c in ["firstk", "all"]}
    plot_heatmaps(densities, fits, n_img, os.path.join(OUT_DIR, "center_bias_heatmaps.png"))
    plot_width_comparison(fits, os.path.join(OUT_DIR, "gaussian_width_comparison.png"))
    primary_stats = [r for r in stats_rows if r["role"] == "primary"]
    plot_center_distance(per_image, image_level, stems, primary_stats,
                         os.path.join(OUT_DIR, "center_distance_distribution.png"))

    # -- Step 7: data-driven judgment (no fixed conclusion) --
    print("\n" + "=" * 70)
    print("JUDGMENT SUMMARY (data-driven, not pre-decided)")
    print("=" * 70)

    area_row = [r for r in primary_stats if r["metric"] == "ellipse_area_1sigma"][0]
    dist_row = [r for r in primary_stats if r["metric"] == "mean_center_distance"][0]
    box_row = [r for r in primary_stats if r["metric"] == "pct_center_box20"][0]
    rad_row = [r for r in primary_stats if r["metric"] == "pct_center_radius01"][0]

    narrower = area_row["delta"] < 0 and area_row["p_corrected"] < 0.05
    print(f"\n[1] Is the firstk 2D Gaussian narrower & more central than all?")
    print(f"    Pooled 1-sigma ellipse area: firstk={fits['firstk']['ellipse_area_1sigma']:.5f}, "
          f"all={fits['all']['ellipse_area_1sigma']:.5f}")
    print(f"    Paired per-image delta(firstk-all) area = {area_row['delta']:+.5f} "
          f"[{area_row['delta_ci_lo']:+.5f},{area_row['delta_ci_hi']:+.5f}], "
          f"p_holm={area_row['p_corrected']:.2e}, r={area_row['effect_r']:.3f}")
    if narrower:
        print("    -> YES: firstk's 1-sigma ellipse is significantly SMALLER "
              "(narrower spread) than all.")
    elif area_row["p_corrected"] < 0.05:
        print("    -> NO: firstk's 1-sigma ellipse is significantly LARGER "
              "(broader spread) than all.")
    else:
        print("    -> NOT SIGNIFICANT: no reliable width difference detected.")

    print(f"\n[2] Do center-distance / center-area rate corroborate this?")
    print(f"    mean_center_distance: delta(firstk-all)={dist_row['delta']:+.5f}, "
          f"p_holm={dist_row['p_corrected']:.2e}")
    print(f"    pct_center_box20:     delta(firstk-all)={box_row['delta']:+.5f}, "
          f"p_holm={box_row['p_corrected']:.2e}")
    print(f"    pct_center_radius01:  delta(firstk-all)={rad_row['delta']:+.5f}, "
          f"p_holm={rad_row['p_corrected']:.2e}")
    corroborates = (dist_row["delta"] < 0 and dist_row["p_corrected"] < 0.05
                    and box_row["delta"] > 0 and box_row["p_corrected"] < 0.05
                    and rad_row["delta"] > 0 and rad_row["p_corrected"] < 0.05)
    if corroborates and narrower:
        print("    -> CONSISTENT: firstk sits closer to center on all three "
              "direct indices, matching the Gaussian-width result.")
    elif narrower or corroborates:
        print("    -> PARTIALLY CONSISTENT: some but not all direct indices "
              "agree with the Gaussian-width conclusion.")
    else:
        print("    -> NOT CONSISTENT: direct center-bias indices do not "
              "support a narrower/more-central firstk distribution.")

    print(f"\n[3] Connection to sAUC (Exp B-control, firstk vs randk) ---")
    if os.path.isfile(EXPB_CONTROL_SAUC_CSV):
        with open(EXPB_CONTROL_SAUC_CSV, "r", encoding="utf-8") as f:
            sauc_rows = [r for r in csv.DictReader(f) if r["metric"] == "sAUC"]
        n_sig = sum(1 for r in sauc_rows if float(r["p_corrected"]) < 0.05)
        deep_rows = [r for r in sauc_rows if 8 <= int(r["layer"]) <= 10]
        mean_deep_delta = np.mean([float(r["delta"]) for r in deep_rows])
        print(f"    sAUC firstk-vs-randk: {n_sig}/12 layers significant "
              f"(Holm p<0.05); L8-L10 mean delta={mean_deep_delta:+.4f}.")
        if narrower and corroborates:
            print("    -> The center-bias direct check shows firstk IS more "
                  "centrally concentrated than all. This is consistent with "
                  "sAUC's near-complete removal of the raw firstk>all "
                  "advantage: sAUC's shuffled-negative design specifically "
                  "cancels out shared spatial (center) bias, so a real "
                  "center-bias effect in firstk predicts exactly this "
                  "pattern. The small residual at L8-L10 in the sAUC "
                  "firstk-vs-randk comparison is therefore plausibly a "
                  "genuine (non-center-bias) temporal/earliness effect, "
                  "since randk is already center-bias-matched to firstk "
                  "in point count and (per Exp B-control) similar spatial "
                  "spread.")
        else:
            print("    -> The direct center-bias check does not clearly show "
                  "firstk as narrower/more central than all, which would mean "
                  "sAUC's near-removal of the firstk advantage is NOT fully "
                  "explained by center-bias alone -- some of it may reflect "
                  "other shared structure between firstk and the sAUC "
                  "shuffled-negative pool. The L8-L10 residual should be "
                  "interpreted with this caveat.")
    else:
        print(f"    (Exp B-control sAUC csv not found at {EXPB_CONTROL_SAUC_CSV}; "
              "skipping connection.)")

    # -- Step 8: run_config.json --
    elapsed = time.time() - t_start
    cfg = {
        "experiment": "center_bias",
        "purpose": "Direct verification of center bias in firstk vs all fixation "
                    "distributions, as an auxiliary check for interpreting the "
                    "firstk NSS/AUC-Judd advantage (not a research goal in itself)",
        "primary_conditions": PRIMARY_CONDITIONS,
        "supplementary_conditions": supp_available,
        "n_images": n_img,
        "center_box_half_width": CENTER_BOX_HALF,
        "center_radius": CENTER_RADIUS,
        "heat_bins": [HEAT_BINS_X, HEAT_BINS_Y],
        "heat_sigma_bins": HEAT_SIGMA_BINS,
        "n_boot": N_BOOT,
        "seed": SEED,
        "fixmap_dir": FIXMAP_DIR,
        "collision_undercount_all": int(collisions_all),
        "collision_undercount_firstk": int(collisions_fk),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg_path = os.path.join(OUT_DIR, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nSaved: {cfg_path}")

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
