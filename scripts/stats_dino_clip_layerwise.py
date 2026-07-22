"""
DINO ViT-S/16 vs CLIP ViT-B/16 -- paired, image-level statistics and
peak-layer bootstrap stability analysis.

Tone-and-manner / statistical-methodology reference (read, not modified):
  scripts/stats_and_plots_expA.py
    -- wilcoxon_test() (paired signed-rank + rank-biserial effect size),
       holm_correction() (Holm-Bonferroni), bootstrap_ci() (percentile
       bootstrap of the mean), plot_3metrics() (figsize=(15,5), 1x3
       panels, steelblue "o-", fill_between alpha=0.15, grid(alpha=0.3),
       suptitle fontsize=13, dpi=150).
  scripts/plot_dino_clip_layerwise.py
    -- CLIP curve color (darkorange), the DINO-vs-CLIP comparison plot
       this script's paired_difference_by_layer.png extends.

Key methodological choice (per the task spec): every bootstrap CI in
this script resamples IMAGE IDs (with replacement), and the SAME
resampled index array is applied to every layer, every metric, and both
models within a single bootstrap iteration. This preserves the paired
(within-image) correspondence for every quantity derived from that
iteration -- including the joint per-iteration comparison needed in
Analysis 5 (each model's own peak layer/value from the SAME resampled
image set). A single shared (N_BOOT, N_IMAGES) index array is generated
once (seed=42) and reused for all analyses in this script.

Reads (read-only):
  outputs/expA/healthy/metrics_per_image.csv       (DINO)
  outputs/expA_clip/healthy/metrics_per_image.csv  (CLIP)

Writes (new directory only): outputs/expA_clip/statistics/
  paired_model_by_layer_tests.csv
  clip_m_shape_contrasts.csv
  model_shape_difference_contrasts.csv
  peak_layer_bootstrap.csv
  max_performance_bootstrap.csv
  statistics_summary.json
  statistics_summary.md
  peak_layer_stability.png / .pdf
  paired_difference_by_layer.png / .pdf
  stats_config.json

Does not modify any existing DINO/CLIP file, CSV, plot, or attention
figure. Only the representative-image selection from
scripts/plot_dino_clip_layerwise.py used all 700 images for aggregate
statistics here as well (the 3 representative images picked there are
for the attention *figures*, not used to bias or subset this script's
statistics).

Usage (PowerShell):
    python scripts\\stats_dino_clip_layerwise.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import math
import os

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
METRICS = ["NSS", "AUC_Judd", "sAUC"]
METRIC_DISPLAY = {"NSS": "NSS", "AUC_Judd": "AUC-Judd", "sAUC": "sAUC"}
N_EXPECTED_IMAGES = 700
SEED = 42
N_BOOT = 10000
ALPHA = 0.05
ROUND_DECIMALS_FOR_TIEBREAK = 10  # near-tie tolerance for argmax layer selection

DINO_PER_IMAGE = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"
CLIP_PER_IMAGE = r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_per_image.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\expA_clip\statistics"

STYLE = {"DINO": "steelblue", "CLIP": "darkorange"}

CLIP_MSHAPE_CONTRASTS = [
    ("early_rise", 4, 1, "positive"),
    ("middle_drop", 8, 4, "negative"),
    ("late_recovery", 12, 8, "positive"),
]
SHAPE_DIFF_CONTRASTS = [
    ("middle_shape_difference", 8, 4),
    ("late_shape_difference", 12, 8),
    ("overall_development_difference", 12, 1),
]
# ======================================================


# ======================================================================
# Load + validate (mirrors scripts/plot_dino_clip_layerwise.py's checks)
# ======================================================================

def load_matrix(csv_path, model_name):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if len(rows) != N_EXPECTED_IMAGES * 12:
        raise RuntimeError(f"STOP [{model_name}]: expected {N_EXPECTED_IMAGES * 12} rows, got {len(rows)}")

    per_image_layers = {}
    values = {m: {} for m in METRICS}
    seen_pairs = set()

    for row in rows:
        img_id = os.path.splitext(row["image"])[0]
        layer = int(row["layer"])
        key = (img_id, layer)
        if key in seen_pairs:
            raise RuntimeError(f"STOP [{model_name}]: duplicate (image,layer) pair {key}")
        seen_pairs.add(key)
        per_image_layers.setdefault(img_id, set()).add(layer)

        for m in METRICS:
            raw = row.get(m)
            if raw is None or raw == "":
                raise RuntimeError(f"STOP [{model_name}]: missing {m} for {key}")
            v = float(raw)
            if math.isnan(v) or math.isinf(v):
                raise RuntimeError(f"STOP [{model_name}]: NaN/Inf {m} for {key}")
            values[m][key] = v

    image_ids = sorted(per_image_layers)
    if len(image_ids) != N_EXPECTED_IMAGES:
        raise RuntimeError(f"STOP [{model_name}]: {len(image_ids)} unique images, expected {N_EXPECTED_IMAGES}")
    bad = [img for img, layers in per_image_layers.items() if layers != set(range(1, 13))]
    if bad:
        raise RuntimeError(f"STOP [{model_name}]: {len(bad)} images missing layers 1-12, e.g. {bad[:3]}")

    return values, image_ids


def build_matrices(values, image_ids):
    """{metric: (N_IMAGES, 12) ndarray}, rows ordered by image_ids."""
    mats = {}
    for m in METRICS:
        mats[m] = np.array([[values[m][(img, l)] for l in range(1, 13)] for img in image_ids])
    return mats


def validate_and_load():
    print("--- Loading + validating DINO ---")
    dino_values, dino_ids = load_matrix(DINO_PER_IMAGE, "DINO")
    print(f"  {len(dino_ids)} images  OK")

    print("--- Loading + validating CLIP ---")
    clip_values, clip_ids = load_matrix(CLIP_PER_IMAGE, "CLIP")
    print(f"  {len(clip_ids)} images  OK")

    if set(dino_ids) != set(clip_ids):
        raise RuntimeError("STOP: DINO and CLIP image-ID sets differ")
    image_ids = sorted(dino_ids)
    print(f"  DINO and CLIP image-ID sets are identical ({len(image_ids)} images)  OK")

    merged_count = 0
    for img in image_ids:
        for l in range(1, 13):
            key = (img, l)
            if key in dino_values["NSS"] and key in clip_values["NSS"]:
                merged_count += 1
    if merged_count != N_EXPECTED_IMAGES * 12:
        raise RuntimeError(f"STOP: merged (image,layer) count {merged_count} != {N_EXPECTED_IMAGES * 12}")
    print(f"  1:1 merge on (image_id, layer): {merged_count} rows  OK")

    dino_mat = build_matrices(dino_values, image_ids)
    clip_mat = build_matrices(clip_values, image_ids)
    return dino_mat, clip_mat, image_ids


# ======================================================================
# Statistics primitives (duplicated from stats_and_plots_expA.py, same
# parameters -- not imported, matching this repo's per-script convention)
# ======================================================================

def wilcoxon_test(x, y, alternative="two-sided"):
    diff = np.array(x) - np.array(y)
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


def make_boot_indices(n_images, n_boot=N_BOOT, seed=SEED):
    """(n_boot, n_images) int array: each row is one bootstrap resample of
    image indices (with replacement). Shared across every analysis in
    this script so all bootstraps preserve the same image-level pairing."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_images, size=(n_boot, n_images))


def bootstrap_ci_mean(diff_1d, boot_indices):
    boot_means = diff_1d[boot_indices].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


# ======================================================================
# Analysis 1: CLIP M-shape contrasts
# ======================================================================

def analysis1_clip_mshape(clip_mat, boot_indices):
    rows = []
    for m in METRICS:
        pvals, contrast_rows = [], []
        for name, la, lb, expected_dir in CLIP_MSHAPE_CONTRASTS:
            arr_a = clip_mat[m][:, la - 1]
            arr_b = clip_mat[m][:, lb - 1]
            diff = arr_a - arr_b
            stat, p, r = wilcoxon_test(arr_a, arr_b)
            ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
            observed_dir = "positive" if diff.mean() > 0 else "negative"
            contrast_rows.append({
                "metric": m, "contrast": name, "layer_a": la, "layer_b": lb,
                "N": int(len(diff)),
                "layer_a_mean": float(arr_a.mean()), "layer_a_median": float(np.median(arr_a)),
                "layer_b_mean": float(arr_b.mean()), "layer_b_median": float(np.median(arr_b)),
                "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "wilcoxon_stat": stat, "p_raw": p, "effect_r": r,
                "expected_direction": expected_dir, "observed_direction": observed_dir,
                "matches_expected": bool(observed_dir == expected_dir),
            })
            pvals.append(p)
        p_holm = holm_correction(np.array(pvals))
        for cr, ph in zip(contrast_rows, p_holm):
            cr["p_holm"] = float(ph)
        rows.extend(contrast_rows)
    return rows


# ======================================================================
# Analysis 2: DINO vs CLIP shape-difference (difference-in-differences)
# ======================================================================

def analysis2_shape_difference(dino_mat, clip_mat, boot_indices):
    rows = []
    for m in METRICS:
        pvals, contrast_rows = [], []
        for name, l_hi, l_lo in SHAPE_DIFF_CONTRASTS:
            clip_component = clip_mat[m][:, l_hi - 1] - clip_mat[m][:, l_lo - 1]
            dino_component = dino_mat[m][:, l_hi - 1] - dino_mat[m][:, l_lo - 1]
            diff = clip_component - dino_component
            stat, p, r = wilcoxon_test(clip_component, dino_component)
            ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
            contrast_rows.append({
                "metric": m, "contrast": name,
                "component_layers": f"L{l_hi}-L{l_lo}",
                "N": int(len(diff)),
                "clip_component_mean": float(clip_component.mean()),
                "clip_component_median": float(np.median(clip_component)),
                "dino_component_mean": float(dino_component.mean()),
                "dino_component_median": float(np.median(dino_component)),
                "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "wilcoxon_stat": stat, "p_raw": p, "effect_r": r,
            })
            pvals.append(p)
        p_holm = holm_correction(np.array(pvals))
        for cr, ph in zip(contrast_rows, p_holm):
            cr["p_holm"] = float(ph)
        rows.extend(contrast_rows)
    return rows


# ======================================================================
# Analysis 3: same-layer model comparison (DINO vs CLIP), 12 layers
# ======================================================================

def analysis3_same_layer(dino_mat, clip_mat, boot_indices):
    rows = []
    for m in METRICS:
        pvals, layer_rows = [], []
        for l in range(1, 13):
            d = dino_mat[m][:, l - 1]
            c = clip_mat[m][:, l - 1]
            diff = d - c
            stat, p, r = wilcoxon_test(d, c)
            ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
            layer_rows.append({
                "metric": m, "layer": l, "N": int(len(diff)),
                "dino_mean": float(d.mean()), "dino_median": float(np.median(d)),
                "clip_mean": float(c.mean()), "clip_median": float(np.median(c)),
                "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "wilcoxon_stat": stat, "p_raw": p, "effect_r": r,
            })
            pvals.append(p)
        p_holm = holm_correction(np.array(pvals))
        for lr, ph in zip(layer_rows, p_holm):
            lr["p_holm"] = float(ph)
        rows.extend(layer_rows)
    return rows


# ======================================================================
# Analysis 4: peak-layer bootstrap stability
# ======================================================================

def analysis4_peak_stability(dino_mat, clip_mat, boot_indices):
    rows = []
    peak_value_boot_store = {}

    for model_name, mat in [("DINO", dino_mat), ("CLIP", clip_mat)]:
        for m in METRICS:
            data = mat[m]  # (N_IMAGES, 12)
            observed_means = data.mean(axis=0)
            observed_means_r = np.round(observed_means, ROUND_DECIMALS_FOR_TIEBREAK)
            observed_peak_layer = int(np.argmax(observed_means_r)) + 1
            observed_peak_value = float(observed_means[observed_peak_layer - 1])

            resampled = data[boot_indices]              # (N_BOOT, N_IMAGES, 12)
            boot_layer_means = resampled.mean(axis=1)     # (N_BOOT, 12)
            del resampled
            boot_layer_means_r = np.round(boot_layer_means, ROUND_DECIMALS_FOR_TIEBREAK)
            boot_peak_layer = boot_layer_means_r.argmax(axis=1) + 1   # (N_BOOT,)
            boot_peak_value = boot_layer_means.max(axis=1)             # (N_BOOT,)

            peak_value_boot_store[(model_name, m)] = boot_peak_value

            counts = np.bincount(boot_peak_layer, minlength=13)[1:13]
            probs = counts / N_BOOT
            modal_layer = int(np.argmax(counts)) + 1
            modal_prob = float(probs[modal_layer - 1])
            ci_lo, ci_hi = np.percentile(boot_peak_value, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

            for l in range(1, 13):
                rows.append({
                    "model": model_name, "metric": m, "layer": l,
                    "peak_count": int(counts[l - 1]), "peak_probability": float(probs[l - 1]),
                    "modal_peak_layer": modal_layer, "modal_peak_probability": modal_prob,
                    "observed_peak_layer": observed_peak_layer, "observed_peak_value": observed_peak_value,
                    "peak_value_ci_lo": float(ci_lo), "peak_value_ci_hi": float(ci_hi),
                })
    return rows, peak_value_boot_store


# ======================================================================
# Analysis 5: max-performance difference between models (exploratory)
# ======================================================================

def analysis5_max_performance(peak_value_boot_store, analysis4_rows):
    observed = {}
    for r in analysis4_rows:
        key = (r["model"], r["metric"])
        observed.setdefault(key, (r["observed_peak_layer"], r["observed_peak_value"]))

    rows = []
    for m in METRICS:
        dino_boot = peak_value_boot_store[("DINO", m)]
        clip_boot = peak_value_boot_store[("CLIP", m)]
        max_diff_boot = dino_boot - clip_boot

        dino_layer, dino_val = observed[("DINO", m)]
        clip_layer, clip_val = observed[("CLIP", m)]
        observed_max_diff = dino_val - clip_val

        ci_lo, ci_hi = np.percentile(max_diff_boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        rows.append({
            "metric": m,
            "dino_peak_layer": dino_layer, "dino_peak_value": dino_val,
            "clip_peak_layer": clip_layer, "clip_peak_value": clip_val,
            "observed_max_difference": float(observed_max_diff),
            "bootstrap_mean_diff": float(max_diff_boot.mean()),
            "bootstrap_median_diff": float(np.median(max_diff_boot)),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "proportion_dino_greater": float((max_diff_boot > 0).mean()),
            "note": ("EXPLORATORY: the compared layer for each model is its own "
                     "data-driven peak (post-hoc selection from the same 700-image "
                     "sample), not a pre-registered layer -- interpret as a "
                     "descriptive best-case comparison, not a confirmatory test."),
        })
    return rows


# ======================================================================
# CSV writers
# ======================================================================

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})


# ======================================================================
# Plots
# ======================================================================

def plot_peak_layer_stability(peak_rows, out_png, out_pdf):
    layers_arr = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    width = 0.35

    for ax, m in zip(axes, METRICS):
        for offset, model_name in [(-width / 2, "DINO"), (width / 2, "CLIP")]:
            sub = [r for r in peak_rows if r["metric"] == m and r["model"] == model_name]
            sub = sorted(sub, key=lambda r: r["layer"])
            probs = np.array([r["peak_probability"] for r in sub])
            obs_layer = sub[0]["observed_peak_layer"]
            ax.bar(layers_arr + offset, probs, width=width, color=STYLE[model_name],
                   alpha=0.85, label=model_name)
            obs_prob = probs[obs_layer - 1]
            ax.plot(obs_layer + offset, obs_prob + 0.02, marker="v", color=STYLE[model_name],
                    ms=7, mec="black", mew=0.6, linestyle="None")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Peak probability")
        ax.set_title(METRIC_DISPLAY[m])
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Peak-layer bootstrap stability (10,000 resamples, image-level; "
        "▼ = observed peak)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_paired_difference_by_layer(layer_rows, out_png, out_pdf):
    layers_arr = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    color = "black"

    for ax, m in zip(axes, METRICS):
        sub = sorted([r for r in layer_rows if r["metric"] == m], key=lambda r: r["layer"])
        mean_diff = np.array([r["mean_diff"] for r in sub])
        ci_lo = np.array([r["ci_lo"] for r in sub])
        ci_hi = np.array([r["ci_hi"] for r in sub])

        ax.plot(layers_arr, mean_diff, "D-", color=color, lw=2, ms=5, label="DINO - CLIP")
        ax.fill_between(layers_arr, ci_lo, ci_hi, alpha=0.15, color=color)
        ax.axhline(0, color="gray", linestyle=":", lw=1.2)

        ax.set_xlabel("Layer")
        ax.set_ylabel(f"{METRIC_DISPLAY[m]} (DINO - CLIP)")
        ax.set_title(METRIC_DISPLAY[m])
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Paired per-layer difference, DINO - CLIP (mean ± 95% bootstrap CI)",
                 fontsize=13)
    fig.text(0.5, -0.02,
              "Positive: DINO favored at that layer.  Negative: CLIP favored at that layer.",
              ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Verification
# ======================================================================

def verify_outputs(a1, a2, a3, a4, a5):
    for tag, rows in [("clip_mshape", a1), ("shape_diff", a2), ("same_layer", a3)]:
        for r in rows:
            if not (r["ci_lo"] <= r["mean_diff"] <= r["ci_hi"]):
                print(f"  WARNING [{tag}]: mean_diff outside its own 95% CI for {r.get('contrast', r.get('layer'))}")
            for k, v in r.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    raise RuntimeError(f"STOP: NaN/Inf found in {tag} row {r}")

    by_model_metric = {}
    for r in a4:
        by_model_metric.setdefault((r["model"], r["metric"]), []).append(r["peak_probability"])
    for key, probs in by_model_metric.items():
        total = sum(probs)
        if abs(total - 1.0) > 1e-9:
            raise RuntimeError(f"STOP: peak_probability for {key} sums to {total}, expected 1.0")
        est = [r for r in a4 if (r["model"], r["metric"]) == key][0]
        if not (est["peak_value_ci_lo"] <= est["observed_peak_value"] <= est["peak_value_ci_hi"]):
            print(f"  NOTE: observed_peak_value for {key} falls outside its own bootstrap 95% CI "
                  f"(can happen for a peak-of-max statistic; reported as-is, not an error)")

    for r in a5:
        if not (r["ci_lo"] <= r["bootstrap_mean_diff"] <= r["ci_hi"]):
            print(f"  WARNING [max_performance]: bootstrap_mean_diff outside its own CI for {r['metric']}")

    print("  Verification checks complete (see any WARNING/NOTE lines above).")


# ======================================================================
# Main
# ======================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  DINO vs CLIP paired statistics + peak-layer bootstrap stability")
    print("=" * 65)

    dino_mat, clip_mat, image_ids = validate_and_load()
    n_images = len(image_ids)

    print(f"\n--- Building shared bootstrap index array (seed={SEED}, n_boot={N_BOOT}) ---")
    boot_indices = make_boot_indices(n_images)
    print(f"  boot_indices shape: {boot_indices.shape}")

    print("\n--- Analysis 1: CLIP M-shape contrasts ---")
    a1 = analysis1_clip_mshape(clip_mat, boot_indices)
    for r in a1:
        print(f"  {r['metric']:10s} {r['contrast']:16s} mean_diff={r['mean_diff']:+.4f}  "
              f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p_holm={r['p_holm']:.2e}  "
              f"r={r['effect_r']:.3f}  matches_expected={r['matches_expected']}")

    print("\n--- Analysis 2: DINO vs CLIP shape-difference ---")
    a2 = analysis2_shape_difference(dino_mat, clip_mat, boot_indices)
    for r in a2:
        print(f"  {r['metric']:10s} {r['contrast']:32s} mean_diff={r['mean_diff']:+.4f}  "
              f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p_holm={r['p_holm']:.2e}  r={r['effect_r']:.3f}")

    print("\n--- Analysis 3: same-layer model comparison (DINO - CLIP) ---")
    a3 = analysis3_same_layer(dino_mat, clip_mat, boot_indices)
    sig_count = sum(1 for r in a3 if r["p_holm"] < 0.05)
    print(f"  {len(a3)} tests, {sig_count} significant at Holm p<0.05")

    print("\n--- Analysis 4: peak-layer bootstrap stability ---")
    a4, peak_value_boot_store = analysis4_peak_stability(dino_mat, clip_mat, boot_indices)
    for model_name in ["DINO", "CLIP"]:
        for m in METRICS:
            rs = [r for r in a4 if r["model"] == model_name and r["metric"] == m]
            r0 = rs[0]
            print(f"  {model_name:5s} {m:10s} observed_peak=L{r0['observed_peak_layer']} "
                  f"({r0['observed_peak_value']:.4f})  modal_peak=L{r0['modal_peak_layer']} "
                  f"(p={r0['modal_peak_probability']:.3f})")

    print("\n--- Analysis 5: max-performance difference (exploratory) ---")
    a5 = analysis5_max_performance(peak_value_boot_store, a4)
    for r in a5:
        print(f"  {r['metric']:10s} observed_max_diff={r['observed_max_difference']:+.4f}  "
              f"boot_mean={r['bootstrap_mean_diff']:+.4f}  CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  "
              f"P(DINO>CLIP)={r['proportion_dino_greater']:.3f}")

    print("\n--- Verification ---")
    verify_outputs(a1, a2, a3, a4, a5)

    print("\n--- Writing CSVs ---")
    write_csv(os.path.join(OUT_DIR, "clip_m_shape_contrasts.csv"), a1,
              ["metric", "contrast", "layer_a", "layer_b", "N",
               "layer_a_mean", "layer_a_median", "layer_b_mean", "layer_b_median",
               "mean_diff", "median_diff", "ci_lo", "ci_hi",
               "wilcoxon_stat", "p_raw", "p_holm", "effect_r",
               "expected_direction", "observed_direction", "matches_expected"])
    write_csv(os.path.join(OUT_DIR, "model_shape_difference_contrasts.csv"), a2,
              ["metric", "contrast", "component_layers", "N",
               "clip_component_mean", "clip_component_median",
               "dino_component_mean", "dino_component_median",
               "mean_diff", "median_diff", "ci_lo", "ci_hi",
               "wilcoxon_stat", "p_raw", "p_holm", "effect_r"])
    write_csv(os.path.join(OUT_DIR, "paired_model_by_layer_tests.csv"), a3,
              ["metric", "layer", "N", "dino_mean", "dino_median", "clip_mean", "clip_median",
               "mean_diff", "median_diff", "ci_lo", "ci_hi",
               "wilcoxon_stat", "p_raw", "p_holm", "effect_r"])
    write_csv(os.path.join(OUT_DIR, "peak_layer_bootstrap.csv"), a4,
              ["model", "metric", "layer", "peak_count", "peak_probability",
               "modal_peak_layer", "modal_peak_probability",
               "observed_peak_layer", "observed_peak_value",
               "peak_value_ci_lo", "peak_value_ci_hi"])
    write_csv(os.path.join(OUT_DIR, "max_performance_bootstrap.csv"), a5,
              ["metric", "dino_peak_layer", "dino_peak_value", "clip_peak_layer", "clip_peak_value",
               "observed_max_difference", "bootstrap_mean_diff", "bootstrap_median_diff",
               "ci_lo", "ci_hi", "proportion_dino_greater", "note"])
    print(f"  Saved 5 CSVs to {OUT_DIR}")

    print("\n--- Plotting ---")
    plot_peak_layer_stability(a4, os.path.join(OUT_DIR, "peak_layer_stability.png"),
                              os.path.join(OUT_DIR, "peak_layer_stability.pdf"))
    plot_paired_difference_by_layer(a3, os.path.join(OUT_DIR, "paired_difference_by_layer.png"),
                                    os.path.join(OUT_DIR, "paired_difference_by_layer.pdf"))
    print("  Saved peak_layer_stability.{png,pdf}, paired_difference_by_layer.{png,pdf}")

    print("\n--- Writing stats_config.json ---")
    stats_config = {
        "seed": SEED, "n_boot": N_BOOT, "alpha": ALPHA, "n_images": n_images,
        "bootstrap_method": (
            "Image-ID-level resampling with replacement; a single shared "
            "(n_boot, n_images) index array is generated once and reused for "
            "every analysis so all layers/metrics/models stay paired within "
            "each bootstrap iteration."),
        "tie_break_rule": (
            f"argmax over the 12 layer means, rounded to {ROUND_DECIMALS_FOR_TIEBREAK} "
            "decimal places before comparison; numpy's argmax returns the first "
            "(lowest-index) maximum, so exact or near-exact ties resolve to the "
            "smallest layer number."),
        "clip_mshape_contrasts": [{"name": n, "layer_a": a, "layer_b": b, "expected_direction": d}
                                   for n, a, b, d in CLIP_MSHAPE_CONTRASTS],
        "shape_difference_contrasts": [{"name": n, "layer_hi": a, "layer_lo": b}
                                        for n, a, b in SHAPE_DIFF_CONTRASTS],
        "holm_families": {
            "analysis1_clip_mshape": "3 contrasts per metric (3 families, one per metric)",
            "analysis2_shape_difference": "3 contrasts per metric (3 families, one per metric)",
            "analysis3_same_layer": "12 layers per metric (3 families, one per metric)",
        },
        "independent_two_sample_tests_used": False,
        "test_used": "scipy.stats.wilcoxon (paired signed-rank), duplicated from scripts/stats_and_plots_expA.py::wilcoxon_test",
        "effect_size": (
            "paired rank-biserial correlation, r = |1 - 2*T / (n*(n+1)/2)|, where T is "
            "scipy.stats.wilcoxon's returned statistic (min of the summed positive/negative "
            "signed ranks) and n is the number of non-zero paired differences. "
            "SIGN CONVENTION: r is always reported as an absolute value (0 = no consistent "
            "direction, 1 = every non-zero-difference image agrees in sign) -- it does NOT "
            "indicate which direction the effect runs. Direction must be read from the "
            "accompanying mean_diff / median_diff sign (or expected_direction/"
            "observed_direction columns in clip_m_shape_contrasts.csv)."
        ),
        "peak_analyses_are_exploratory": True,
        "representative_images_not_used_for_statistics": (
            "The 3 images selected in scripts/plot_dino_clip_layerwise.py "
            "(representative_image_selection.csv) are for the attention "
            "figures only; all statistics here use all 700 images."),
        "source_files": {"dino_per_image": DINO_PER_IMAGE, "clip_per_image": CLIP_PER_IMAGE},
    }
    with open(os.path.join(OUT_DIR, "stats_config.json"), "w", encoding="utf-8") as f:
        json.dump(stats_config, f, indent=2)

    print("--- Writing statistics_summary.json / .md ---")
    write_summary(a1, a2, a3, a4, a5, n_images)

    print("\n" + "=" * 65)
    print("  Done. Only outputs/expA_clip/statistics/ was written to.")
    print("=" * 65)


def write_summary(a1, a2, a3, a4, a5, n_images):
    summary = {
        "n_images": n_images, "seed": SEED, "n_boot": N_BOOT,
        "analysis1_clip_mshape": a1,
        "analysis2_shape_difference": a2,
        "analysis3_same_layer_highlights": [
            r for r in a3 if r["layer"] in (1, 4, 8, 9, 10, 12)
        ],
        "analysis4_peak_stability_summary": [
            {"model": r["model"], "metric": r["metric"],
             "observed_peak_layer": r["observed_peak_layer"],
             "observed_peak_value": r["observed_peak_value"],
             "modal_peak_layer": r["modal_peak_layer"],
             "modal_peak_probability": r["modal_peak_probability"]}
            for r in a4 if r["layer"] == r["observed_peak_layer"]
        ],
        "analysis5_max_performance": a5,
    }
    with open(os.path.join(OUT_DIR, "statistics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("# DINO vs CLIP -- paired statistics summary\n")
    lines.append(f"N = {n_images} images (all 700, paired by image_id). "
                 f"Bootstrap: seed={SEED}, n_boot={N_BOOT}, image-ID-level resampling "
                 "with a shared index array across every analysis below.\n")
    lines.append("Every p-value below is reported together with its effect size "
                 "(paired rank-biserial r) and its absolute mean/median difference "
                 "with 95% bootstrap CI -- p-values alone are not used to draw "
                 "conclusions.\n")

    lines.append("\n## 1. CLIP M-shape contrasts\n")
    lines.append("| metric | contrast | mean diff | 95% CI | p (Holm) | r | matches expected |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in a1:
        lines.append(f"| {r['metric']} | {r['contrast']} | {r['mean_diff']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_holm']:.2e} | "
                     f"{r['effect_r']:.3f} | {r['matches_expected']} |")

    lines.append("\n## 2. DINO vs CLIP shape-difference (difference-in-differences)\n")
    lines.append("| metric | contrast | mean diff (CLIP-comp - DINO-comp) | 95% CI | p (Holm) | r |")
    lines.append("|---|---|---|---|---|---|")
    for r in a2:
        lines.append(f"| {r['metric']} | {r['contrast']} | {r['mean_diff']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_holm']:.2e} | {r['effect_r']:.3f} |")

    lines.append("\n## 3. Same-layer model comparison (DINO - CLIP), selected layers\n")
    lines.append("| metric | layer | DINO mean | CLIP mean | mean diff | 95% CI | p (Holm) | r |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in a3:
        if r["layer"] in (1, 4, 8, 9, 10, 12):
            lines.append(f"| {r['metric']} | L{r['layer']} | {r['dino_mean']:.4f} | "
                         f"{r['clip_mean']:.4f} | {r['mean_diff']:+.4f} | "
                         f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_holm']:.2e} | {r['effect_r']:.3f} |")
    lines.append(f"\n(full 12-layer table in paired_model_by_layer_tests.csv)\n")

    lines.append("\n## 4. Peak-layer bootstrap stability\n")
    lines.append("| model | metric | observed peak | modal peak (bootstrap) | modal probability |")
    lines.append("|---|---|---|---|---|")
    for r in a4:
        if r["layer"] == r["observed_peak_layer"]:
            lines.append(f"| {r['model']} | {r['metric']} | L{r['observed_peak_layer']} "
                         f"({r['observed_peak_value']:.4f}) | L{r['modal_peak_layer']} | "
                         f"{r['modal_peak_probability']:.3f} |")

    lines.append("\n## 5. Max-performance difference between models (EXPLORATORY)\n")
    lines.append("Peak layers are selected from the same 700-image sample being compared "
                 "-- this is a descriptive, data-driven comparison, not a confirmatory test.\n")
    lines.append(
        "**P(DINO>CLIP) is a bootstrap proportion, not a p-value**: it is the fraction "
        "of the 10,000 image-level bootstrap resamples in which DINO's own peak-layer "
        "value exceeded CLIP's own peak-layer value in that same resample. No null-hypothesis "
        "significance test is performed for this analysis; only the effect (observed/bootstrap "
        "mean difference) and its 95% CI are used to draw conclusions.\n")
    lines.append("| metric | DINO peak (layer, value) | CLIP peak (layer, value) | observed diff | "
                 "bootstrap mean diff | 95% CI | P(DINO>CLIP), bootstrap proportion |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in a5:
        lines.append(f"| {r['metric']} | L{r['dino_peak_layer']} ({r['dino_peak_value']:.4f}) | "
                     f"L{r['clip_peak_layer']} ({r['clip_peak_value']:.4f}) | "
                     f"{r['observed_max_difference']:+.4f} | {r['bootstrap_mean_diff']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['proportion_dino_greater']:.3f} |")

    lines.append("")
    for r in a5:
        ci_includes_zero = r["ci_lo"] <= 0.0 <= r["ci_hi"]
        if ci_includes_zero:
            lines.append(
                f"- **{r['metric']}**: the 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                f"includes 0 -- a clear difference between the models' peak performance "
                f"cannot be established (not \"no difference\"; the data are simply "
                f"insufficient to rule out zero or either sign).")
        else:
            favored = "DINO" if r["bootstrap_mean_diff"] > 0 else "CLIP"
            lines.append(
                f"- **{r['metric']}**: the 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                f"excludes 0 -- peak performance favors {favored}.")
    lines.append("")

    with open(os.path.join(OUT_DIR, "statistics_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
