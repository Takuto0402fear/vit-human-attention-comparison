"""
DINO ViT-S/16 vs DINO ViT-B/16 vs CLIP ViT-B/16 -- paired, image-level
three-model statistics.

Research questions:
  1. Does DINO keep its layer-profile shape from S/16 to B/16?
  2. Do DINO-B and CLIP-B differ in layer profile even at matched
     (Base) scale?
  3. Does CLIP's "early rise -> mid-layer drop -> late recovery" hold up
     statistically against DINO-B specifically?
  4. Are the three models' peak layers / max-reached values stable
     under image resampling?

Methodology reference (duplicated, not imported, matching this repo's
per-script convention): scripts/stats_and_plots_expA.py (paired
Wilcoxon signed-rank + rank-biserial + Holm-Bonferroni + percentile
bootstrap) and scripts/stats_dino_clip_layerwise.py (image-ID-level
shared bootstrap resampling so every layer/metric/model stays paired
within one bootstrap iteration).

New this script (extending the 2-model version):
  - rank_biserial_signed alongside rank_biserial_absolute (sign matches
    the paired-difference direction: model_a - model_b).
  - A global, curve-level sign-flip permutation test (Analysis 4) in
    addition to the per-layer / per-contrast Wilcoxon tests.
  - Three model pairs, always reported as "left model - right model":
      DINO-S - DINO-B   (model-size effect)
      DINO-B - CLIP-B   (training-method effect at matched Base scale)
      DINO-S - CLIP-B   (re-confirmation within the 3-model framework)

Reads (read-only):
  outputs/expA/healthy/metrics_per_image.csv               (DINO-S)
  outputs/expA_dino_vitb16/healthy/metrics_per_image.csv    (DINO-B)
  outputs/expA_clip/healthy/metrics_per_image.csv           (CLIP-B)

Writes (new directory only): outputs/expA_dino_vitb16/statistics_three_model/
  paired_three_model_by_layer_tests.csv       (Analysis 1)
  within_model_shape_contrasts.csv             (Analysis 2)
  between_model_shape_contrasts.csv            (Analysis 3)
  global_profile_permutation_tests.csv         (Analysis 4)
  three_model_peak_bootstrap.csv               (Analysis 5)
  three_model_max_performance_bootstrap.csv    (Analysis 6)
  dino_s_b_similarity.csv                      (Analysis 7, curve-level)
  per_image_dino_s_b_profile_similarity.csv    (Analysis 7, per-image)
  statistics_three_model_summary.json / .md
  stats_three_model_config.json
  three_model_peak_stability.png / .pdf
  three_model_paired_difference_by_layer.png / .pdf
  shape_contrast_forest.png / .pdf
  dino_s_b_profile_similarity.png / .pdf

Does not modify any existing DINO-S/DINO-B/CLIP-B file, CSV, statistics
output, or plot, including scripts/stats_dino_clip_layerwise.py,
scripts/stats_and_plots_expA.py, and scripts/plot_three_model_layerwise.py.

Usage (PowerShell):
    python scripts\\stats_three_model_layerwise.py
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
BOOT_SEED = 42
N_BOOT = 10000
PERM_SEED = 42
N_PERM = 10000
ALPHA = 0.05
ROUND_DECIMALS_FOR_TIEBREAK = 10

DINO_S_CSV = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"
DINO_B_CSV = r"C:\Users\user\gaze\outputs\expA_dino_vitb16\healthy\metrics_per_image.csv"
CLIP_B_CSV = r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_per_image.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\expA_dino_vitb16\statistics_three_model"

MODEL_NAMES = ["DINO-S", "DINO-B", "CLIP-B"]
MODEL_PAIRS = [("DINO-S", "DINO-B"), ("DINO-B", "CLIP-B"), ("DINO-S", "CLIP-B")]
STYLE = {"DINO-S": "steelblue", "DINO-B": "seagreen", "CLIP-B": "darkorange"}
PAIR_STYLE = {
    ("DINO-S", "DINO-B"): {"color": "purple", "marker": "D", "linestyle": "-"},
    ("DINO-B", "CLIP-B"): {"color": "firebrick", "marker": "v", "linestyle": "--"},
    ("DINO-S", "CLIP-B"): {"color": "black", "marker": "o", "linestyle": ":"},
}

SHAPE_CONTRASTS = [
    ("early_rise", 4, 1),
    ("middle_change", 8, 4),
    ("late_change", 12, 8),
    ("overall_change", 12, 1),
]
# ======================================================


# ======================================================================
# Load + validate
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
            if m in ("AUC_Judd", "sAUC") and not (0.0 <= v <= 1.0):
                raise RuntimeError(f"STOP [{model_name}]: {m}={v} out of [0,1] for {key}")
            values[m][key] = v

    image_ids = sorted(per_image_layers)
    if len(image_ids) != N_EXPECTED_IMAGES:
        raise RuntimeError(f"STOP [{model_name}]: {len(image_ids)} unique images, expected {N_EXPECTED_IMAGES}")
    bad = [img for img, layers in per_image_layers.items() if layers != set(range(1, 13))]
    if bad:
        raise RuntimeError(f"STOP [{model_name}]: {len(bad)} images missing layers 1-12, e.g. {bad[:3]}")

    return values, image_ids


def build_matrices(values, image_ids):
    return {m: np.array([[values[m][(img, l)] for l in range(1, 13)] for img in image_ids]) for m in METRICS}


def validate_and_load():
    print("--- Loading + validating (per-model) ---")
    values, ids = {}, {}
    for name, path in zip(MODEL_NAMES, [DINO_S_CSV, DINO_B_CSV, CLIP_B_CSV]):
        values[name], ids[name] = load_matrix(path, name)
        print(f"  {name}: {len(ids[name])} images  OK")

    print("\n--- Cross-model validation ---")
    id_sets = [set(ids[n]) for n in MODEL_NAMES]
    if not (id_sets[0] == id_sets[1] == id_sets[2]):
        raise RuntimeError("STOP: image-ID sets differ across models")
    image_ids = sorted(id_sets[0])
    print(f"  Image-ID sets identical across all 3 models ({len(image_ids)} images)  OK")

    layer_sets = [set(range(1, 13))] * 3  # already enforced per-model above
    print("  Layer sets identical (1..12) across all 3 models  OK")

    merged = 0
    for img in image_ids:
        for l in range(1, 13):
            key = (img, l)
            if all(key in values[n]["NSS"] for n in MODEL_NAMES):
                merged += 1
    if merged != N_EXPECTED_IMAGES * 12:
        raise RuntimeError(f"STOP: 3-way merge on (image,layer) gave {merged} rows, expected {N_EXPECTED_IMAGES * 12}")
    print(f"  3-way merge on (image_id, layer): {merged} rows  OK")

    mats = {name: build_matrices(values[name], image_ids) for name in MODEL_NAMES}
    return mats, image_ids


# ======================================================================
# Statistics primitives
# ======================================================================

def wilcoxon_test_signed(x, y, alternative="two-sided"):
    """
    Paired Wilcoxon signed-rank test on x vs y (both raw arrays, same
    length, paired by image). Returns dict with:
      stat, p                    -- scipy's statistic and p-value
      r_signed, r_absolute       -- rank-biserial correlation;
                                     r_signed = (W_pos - W_neg) / (n*(n+1)/2),
                                     positive when x tends to exceed y
                                     (matches sign of mean(x-y));
                                     r_absolute = abs(r_signed).
      n_pos, n_neg, n_zero, n_nonzero -- counts over the FULL (x-y) array
                                     (n_zero counts exact ties, which are
                                     excluded from the Wilcoxon test itself
                                     but still reported here).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x - y
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    n_zero = int((diff == 0).sum())

    diff_nz = diff[diff != 0]
    n = len(diff_nz)
    if n < 10:
        return {"stat": np.nan, "p": np.nan, "r_signed": np.nan, "r_absolute": np.nan,
                "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "n_nonzero": n}

    abs_diff = np.abs(diff_nz)
    ranks = sp_stats.rankdata(abs_diff)
    signs = np.sign(diff_nz)
    total_rank = n * (n + 1) / 2
    w_pos = ranks[signs > 0].sum()
    w_neg = ranks[signs < 0].sum()

    res = sp_stats.wilcoxon(diff_nz, alternative=alternative)
    r_signed = float((w_pos - w_neg) / total_rank)
    r_absolute = abs(r_signed)

    return {"stat": float(res.statistic), "p": float(res.pvalue),
            "r_signed": r_signed, "r_absolute": r_absolute,
            "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "n_nonzero": n}


def holm_correction(pvals):
    pvals = np.asarray(pvals, dtype=np.float64)
    n = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, pvals[idx] * (n - rank))
    running_max = 0.0
    for idx in order:
        running_max = max(running_max, corrected[idx])
        corrected[idx] = running_max
    return corrected


def make_boot_indices(n_images, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_images, size=(n_boot, n_images))


def make_perm_signs(n_images, n_perm=N_PERM, seed=PERM_SEED):
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=(n_perm, n_images))


def bootstrap_ci_mean(diff_1d, boot_indices):
    boot_means = diff_1d[boot_indices].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def bootstrap_ci_nanmean(vals_1d, boot_indices):
    boot_means = np.nanmean(vals_1d[boot_indices], axis=1)
    lo, hi = np.nanpercentile(boot_means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


# ======================================================================
# Analysis 1: same-layer 3-model comparison
# ======================================================================

def analysis1_same_layer(mats, boot_indices):
    rows = []
    for m in METRICS:
        pvals, metric_rows = [], []
        for model_a, model_b in MODEL_PAIRS:
            for l in range(1, 13):
                a = mats[model_a][m][:, l - 1]
                b = mats[model_b][m][:, l - 1]
                diff = a - b
                wt = wilcoxon_test_signed(a, b)
                ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
                metric_rows.append({
                    "metric": m, "layer": l, "model_a": model_a, "model_b": model_b,
                    "N": int(len(diff)),
                    "model_a_mean": float(a.mean()), "model_a_median": float(np.median(a)),
                    "model_b_mean": float(b.mean()), "model_b_median": float(np.median(b)),
                    "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "wilcoxon_stat": wt["stat"], "p_raw": wt["p"],
                    "rank_biserial_signed": wt["r_signed"], "rank_biserial_absolute": wt["r_absolute"],
                    "positive_count": wt["n_pos"], "negative_count": wt["n_neg"], "zero_count": wt["n_zero"],
                })
                pvals.append(wt["p"])
        p_holm = holm_correction(np.array(pvals))
        for r, ph in zip(metric_rows, p_holm):
            r["p_holm"] = float(ph)
        rows.extend(metric_rows)
    return rows


# ======================================================================
# Analysis 2: within-model shape contrasts
# ======================================================================

def analysis2_within_model(mats, boot_indices):
    rows = []
    for m in METRICS:
        pvals, metric_rows = [], []
        for model_name in MODEL_NAMES:
            for name, l_hi, l_lo in SHAPE_CONTRASTS:
                a = mats[model_name][m][:, l_hi - 1]
                b = mats[model_name][m][:, l_lo - 1]
                diff = a - b
                wt = wilcoxon_test_signed(a, b)
                ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
                metric_rows.append({
                    "metric": m, "model": model_name, "contrast": name,
                    "layer_hi": l_hi, "layer_lo": l_lo, "N": int(len(diff)),
                    "contrast_mean": float(diff.mean()), "contrast_median": float(np.median(diff)),
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "wilcoxon_stat": wt["stat"], "p_raw": wt["p"],
                    "rank_biserial_signed": wt["r_signed"], "rank_biserial_absolute": wt["r_absolute"],
                    "positive_count": wt["n_pos"], "negative_count": wt["n_neg"], "zero_count": wt["n_zero"],
                })
                pvals.append(wt["p"])
        p_holm = holm_correction(np.array(pvals))
        for r, ph in zip(metric_rows, p_holm):
            r["p_holm"] = float(ph)
        rows.extend(metric_rows)
    return rows


# ======================================================================
# Analysis 3: between-model shape difference (difference-in-differences)
# ======================================================================

def analysis3_between_model(mats, boot_indices):
    rows = []
    highlights = {}
    for m in METRICS:
        pvals, metric_rows = [], []
        for model_a, model_b in MODEL_PAIRS:
            for name, l_hi, l_lo in SHAPE_CONTRASTS:
                comp_a = mats[model_a][m][:, l_hi - 1] - mats[model_a][m][:, l_lo - 1]
                comp_b = mats[model_b][m][:, l_hi - 1] - mats[model_b][m][:, l_lo - 1]
                diff = comp_a - comp_b
                wt = wilcoxon_test_signed(comp_a, comp_b)
                ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
                row = {
                    "metric": m, "model_a": model_a, "model_b": model_b, "contrast": name,
                    "layer_hi": l_hi, "layer_lo": l_lo, "N": int(len(diff)),
                    "model_a_component_mean": float(comp_a.mean()),
                    "model_b_component_mean": float(comp_b.mean()),
                    "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "wilcoxon_stat": wt["stat"], "p_raw": wt["p"],
                    "rank_biserial_signed": wt["r_signed"], "rank_biserial_absolute": wt["r_absolute"],
                    "positive_count": wt["n_pos"], "negative_count": wt["n_neg"], "zero_count": wt["n_zero"],
                }
                metric_rows.append(row)
                pvals.append(wt["p"])
                if name == "middle_change" and (model_a, model_b) == ("DINO-S", "DINO-B") and m == "NSS":
                    highlights["A_dino_s_vs_dino_b_middle"] = row
                if name == "middle_change" and (model_a, model_b) == ("DINO-B", "CLIP-B") and m == "NSS":
                    highlights["B_dino_b_vs_clip_b_middle"] = row
                if name == "late_change" and (model_a, model_b) == ("DINO-B", "CLIP-B") and m == "NSS":
                    highlights["C_dino_b_vs_clip_b_late"] = row
        p_holm = holm_correction(np.array(pvals))
        for r, ph in zip(metric_rows, p_holm):
            r["p_holm"] = float(ph)
        rows.extend(metric_rows)
    return rows, highlights


# ======================================================================
# Analysis 4: global curve-shape sign-flip permutation test
# ======================================================================

def analysis4_global_permutation(mats, perm_signs):
    rows = []
    all_p = []
    row_refs = []
    for model_a, model_b in MODEL_PAIRS:
        for m in METRICS:
            mean_a = mats[model_a][m].mean(axis=0)  # (12,)
            mean_b = mats[model_b][m].mean(axis=0)
            D = mats[model_a][m] - mats[model_b][m]  # (700, 12), a-b convention
            observed_layer_means = D.mean(axis=0)
            t_observed = float(np.sum(observed_layer_means ** 2))

            M = perm_signs @ D  # (n_perm, 700) @ (700, 12) -> (n_perm, 12)
            n_images = D.shape[0]
            t_perm = np.sum((M / n_images) ** 2, axis=1)  # (n_perm,)
            p_perm = float((np.sum(t_perm >= t_observed) + 1) / (N_PERM + 1))

            rmse = float(np.sqrt(np.mean((mean_a - mean_b) ** 2)))
            pearson_r = float(sp_stats.pearsonr(mean_a, mean_b)[0])
            spearman_r = float(sp_stats.spearmanr(mean_a, mean_b)[0])

            row = {
                "model_a": model_a, "model_b": model_b, "metric": m,
                "observed_statistic": t_observed, "permutation_p": p_perm,
                "n_permutations": N_PERM,
                "mean_curve_rmse": rmse, "mean_curve_pearson_r": pearson_r,
                "mean_curve_spearman_r": spearman_r,
            }
            rows.append(row)
            all_p.append(p_perm)
            row_refs.append(row)

    p_holm = holm_correction(np.array(all_p))
    for r, ph in zip(row_refs, p_holm):
        r["p_holm"] = float(ph)
    return rows


# ======================================================================
# Analysis 5: peak-layer bootstrap stability (3 models)
# ======================================================================

def analysis5_peak_stability(mats, boot_indices):
    rows = []
    peak_value_boot_store = {}

    for model_name in MODEL_NAMES:
        for m in METRICS:
            data = mats[model_name][m]
            observed_means = data.mean(axis=0)
            observed_means_r = np.round(observed_means, ROUND_DECIMALS_FOR_TIEBREAK)
            observed_peak_layer = int(np.argmax(observed_means_r)) + 1
            observed_peak_value = float(observed_means[observed_peak_layer - 1])

            resampled = data[boot_indices]
            boot_layer_means = resampled.mean(axis=1)
            del resampled
            boot_layer_means_r = np.round(boot_layer_means, ROUND_DECIMALS_FOR_TIEBREAK)
            boot_peak_layer = boot_layer_means_r.argmax(axis=1) + 1
            boot_peak_value = boot_layer_means.max(axis=1)

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
                    "observed_peak_layer": observed_peak_layer, "observed_peak_value": observed_peak_value,
                    "modal_peak_layer": modal_layer, "modal_peak_probability": modal_prob,
                    "bootstrap_peak_value_mean": float(boot_peak_value.mean()),
                    "bootstrap_peak_value_median": float(np.median(boot_peak_value)),
                    "peak_value_ci_lo": float(ci_lo), "peak_value_ci_hi": float(ci_hi),
                })
    return rows, peak_value_boot_store


# ======================================================================
# Analysis 6: max-performance 3-model comparison (exploratory)
# ======================================================================

def analysis6_max_performance(peak_value_boot_store, analysis5_rows):
    observed = {}
    for r in analysis5_rows:
        key = (r["model"], r["metric"])
        observed.setdefault(key, (r["observed_peak_layer"], r["observed_peak_value"]))

    rows = []
    for model_a, model_b in MODEL_PAIRS:
        for m in METRICS:
            boot_a = peak_value_boot_store[(model_a, m)]
            boot_b = peak_value_boot_store[(model_b, m)]
            diff_boot = boot_a - boot_b

            layer_a, val_a = observed[(model_a, m)]
            layer_b, val_b = observed[(model_b, m)]
            observed_diff = val_a - val_b

            ci_lo, ci_hi = np.percentile(diff_boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
            rows.append({
                "model_a": model_a, "model_b": model_b, "metric": m,
                "model_a_peak_layer": layer_a, "model_a_peak_value": val_a,
                "model_b_peak_layer": layer_b, "model_b_peak_value": val_b,
                "observed_max_difference": float(observed_diff),
                "bootstrap_mean_diff": float(diff_boot.mean()),
                "bootstrap_median_diff": float(np.median(diff_boot)),
                "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "proportion_model_a_greater": float((diff_boot > 0).mean()),
                "note": ("EXPLORATORY: peak layer is each model's own data-driven "
                         "peak from the same 700-image sample being compared; "
                         "proportion_model_a_greater is a bootstrap proportion, "
                         "NOT a p-value. No equivalence margin was pre-specified, "
                         "so equivalence is never claimed here."),
            })
    return rows


# ======================================================================
# Analysis 7: DINO-S vs DINO-B similarity
# ======================================================================

def analysis7_dino_s_b_similarity(mats, image_ids, boot_indices):
    curve_rows = []
    per_image_rows = []

    for m in METRICS:
        s_mat = mats["DINO-S"][m]  # (700, 12)
        b_mat = mats["DINO-B"][m]

        mean_s = s_mat.mean(axis=0)
        mean_b = b_mat.mean(axis=0)
        pearson_r = float(sp_stats.pearsonr(mean_s, mean_b)[0])
        spearman_r = float(sp_stats.spearmanr(mean_s, mean_b)[0])
        rmse = float(np.sqrt(np.mean((mean_s - mean_b) ** 2)))
        mae = float(np.mean(np.abs(mean_s - mean_b)))

        peak_s_layer = int(np.argmax(np.round(mean_s, ROUND_DECIMALS_FOR_TIEBREAK))) + 1
        peak_b_layer = int(np.argmax(np.round(mean_b, ROUND_DECIMALS_FOR_TIEBREAK))) + 1
        peak_s_value = float(mean_s[peak_s_layer - 1])
        peak_b_value = float(mean_b[peak_b_layer - 1])

        # per-image profile correlation (Pearson, across the 12 layers)
        n_images = s_mat.shape[0]
        per_image_corr = np.full(n_images, np.nan)
        n_noncomputable = 0
        for i in range(n_images):
            s_vec, b_vec = s_mat[i], b_mat[i]
            if np.std(s_vec) == 0 or np.std(b_vec) == 0:
                n_noncomputable += 1
                continue
            per_image_corr[i] = np.corrcoef(s_vec, b_vec)[0, 1]

        computable_mask = ~np.isnan(per_image_corr)
        computable_vals = per_image_corr[computable_mask]
        n_negative = int((computable_vals < 0).sum())

        corr_mean = float(np.mean(computable_vals)) if len(computable_vals) else float("nan")
        corr_median = float(np.median(computable_vals)) if len(computable_vals) else float("nan")
        corr_sd = float(np.std(computable_vals, ddof=1)) if len(computable_vals) > 1 else float("nan")
        ci_lo, ci_hi = bootstrap_ci_nanmean(per_image_corr, boot_indices)

        curve_rows.append({
            "metric": m,
            "pearson_r": pearson_r, "spearman_r": spearman_r, "rmse": rmse, "mae": mae,
            "peak_layer_dino_s": peak_s_layer, "peak_layer_dino_b": peak_b_layer,
            "peak_layer_diff": peak_s_layer - peak_b_layer,
            "peak_value_dino_s": peak_s_value, "peak_value_dino_b": peak_b_value,
            "peak_value_diff": peak_s_value - peak_b_value,
            "per_image_corr_mean": corr_mean, "per_image_corr_median": corr_median,
            "per_image_corr_sd": corr_sd, "per_image_corr_ci_lo": ci_lo, "per_image_corr_ci_hi": ci_hi,
            "n_negative_corr_images": n_negative, "n_noncomputable_images": n_noncomputable,
        })

        for img_id, corr, computable in zip(image_ids, per_image_corr, computable_mask):
            per_image_rows.append({
                "image_id": img_id, "metric": m,
                "pearson_r": ("" if not computable else f"{corr:.6f}"),
                "computable": bool(computable),
            })

    return curve_rows, per_image_rows


# ======================================================================
# CSV writer
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

def plot_peak_stability(a5_rows, out_png, out_pdf):
    layers_arr = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    width = 0.25
    offsets = {"DINO-S": -width, "DINO-B": 0.0, "CLIP-B": width}

    for ax, m in zip(axes, METRICS):
        for model_name in MODEL_NAMES:
            sub = sorted([r for r in a5_rows if r["metric"] == m and r["model"] == model_name],
                        key=lambda r: r["layer"])
            probs = np.array([r["peak_probability"] for r in sub])
            obs_layer = sub[0]["observed_peak_layer"]
            ax.bar(layers_arr + offsets[model_name], probs, width=width,
                   color=STYLE[model_name], alpha=0.85, label=model_name)
            obs_prob = probs[obs_layer - 1]
            ax.plot(obs_layer + offsets[model_name], obs_prob + 0.02, marker="v",
                    color=STYLE[model_name], ms=7, mec="black", mew=0.6, linestyle="None")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Peak probability")
        ax.set_title(METRIC_DISPLAY[m])
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle("Three-model peak-layer bootstrap stability (10,000 resamples, image-level; "
                 "▼ = observed peak)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_paired_difference_by_layer(a1_rows, out_png, out_pdf):
    layers_arr = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, m in zip(axes, METRICS):
        for model_a, model_b in MODEL_PAIRS:
            sub = sorted([r for r in a1_rows if r["metric"] == m
                         and r["model_a"] == model_a and r["model_b"] == model_b],
                        key=lambda r: r["layer"])
            mean_diff = np.array([r["mean_diff"] for r in sub])
            ci_lo = np.array([r["ci_lo"] for r in sub])
            ci_hi = np.array([r["ci_hi"] for r in sub])
            st = PAIR_STYLE[(model_a, model_b)]
            label = f"{model_a} - {model_b}"
            ax.plot(layers_arr, mean_diff, marker=st["marker"], linestyle=st["linestyle"],
                    color=st["color"], lw=2, ms=5, label=label)
            ax.fill_between(layers_arr, ci_lo, ci_hi, alpha=0.12, color=st["color"])
        ax.axhline(0, color="gray", linestyle=":", lw=1.2)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"{METRIC_DISPLAY[m]} (model_a - model_b)")
        ax.set_title(METRIC_DISPLAY[m])
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Paired per-layer differences, three model pairs (mean ± 95% bootstrap CI)",
                 fontsize=13)
    fig.text(0.5, -0.02,
              "Positive: left model (model_a) favored at that layer.  Negative: right model (model_b) favored.",
              ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_shape_contrast_forest(a2_rows, a3_rows, out_png, out_pdf):
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))

    for ax, m in zip(axes, METRICS):
        categories = []
        means = []
        ci_los = []
        ci_his = []
        colors = []

        for name, _, _ in SHAPE_CONTRASTS:
            for model_a, model_b in MODEL_PAIRS:
                row = next(r for r in a3_rows if r["metric"] == m and r["contrast"] == name
                          and r["model_a"] == model_a and r["model_b"] == model_b)
                categories.append(f"{name}\n{model_a}-{model_b}")
                means.append(row["mean_diff"])
                ci_los.append(row["ci_lo"])
                ci_his.append(row["ci_hi"])
                colors.append(PAIR_STYLE[(model_a, model_b)]["color"])

        y_pos = np.arange(len(categories))
        means_arr = np.array(means)
        err_lo = means_arr - np.array(ci_los)
        err_hi = np.array(ci_his) - means_arr
        for yi, mi, elo, ehi, ci in zip(y_pos, means_arr, err_lo, err_hi, colors):
            ax.errorbar([mi], [yi], xerr=[[elo], [ehi]], fmt="o", color=ci,
                       ms=5, elinewidth=1.5, capsize=3, zorder=3)
        ax.axvline(0, color="gray", linestyle=":", lw=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"{METRIC_DISPLAY[m]} shape-difference (model_a - model_b)")
        ax.set_title(METRIC_DISPLAY[m])
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Between-model shape-contrast forest plot (mean ± 95% bootstrap CI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_dino_s_b_similarity(mats, a7_curve_rows, out_png, out_pdf):
    layers_arr = np.arange(1, 13)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for col, m in enumerate(METRICS):
        mean_s = mats["DINO-S"][m].mean(axis=0)
        mean_b = mats["DINO-B"][m].mean(axis=0)
        row_stat = next(r for r in a7_curve_rows if r["metric"] == m)

        ax_top = axes[0, col]
        ax_top.plot(layers_arr, mean_s, "o-", color=STYLE["DINO-S"], lw=2, ms=5, label="DINO-S")
        ax_top.plot(layers_arr, mean_b, "^-.", color=STYLE["DINO-B"], lw=2, ms=5, label="DINO-B")
        ax_top.set_title(METRIC_DISPLAY[m])
        ax_top.set_xlabel("Layer")
        ax_top.set_ylabel(METRIC_DISPLAY[m])
        ax_top.set_xticks(layers_arr)
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(fontsize=8)
        ax_top.text(0.02, 0.98,
                    f"Pearson r={row_stat['pearson_r']:.3f}\nSpearman r={row_stat['spearman_r']:.3f}\n"
                    f"RMSE={row_stat['rmse']:.4f}",
                    transform=ax_top.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        ax_bot = axes[1, col]
        diff = mean_s - mean_b
        ax_bot.plot(layers_arr, diff, "D-", color="black", lw=2, ms=5, label="DINO-S - DINO-B")
        ax_bot.axhline(0, color="gray", linestyle=":", lw=1.2)
        ax_bot.set_xlabel("Layer")
        ax_bot.set_ylabel(f"{METRIC_DISPLAY[m]} diff")
        ax_bot.set_xticks(layers_arr)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=8)

    fig.suptitle("DINO-S vs DINO-B mean layer profile similarity", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Verification
# ======================================================================

def verify_all(a1, a2, a3, a4, a5, a6, a7_curve):
    def check_nan_inf(tag, rows, skip_keys=()):
        for r in rows:
            for k, v in r.items():
                if k in skip_keys:
                    continue
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    raise RuntimeError(f"STOP: NaN/Inf found in {tag} row {r}")

    check_nan_inf("analysis1", a1)
    check_nan_inf("analysis2", a2)
    check_nan_inf("analysis3", a3)
    check_nan_inf("analysis4", a4)
    check_nan_inf("analysis5", a5)
    check_nan_inf("analysis6", a6)
    check_nan_inf("analysis7_curve", a7_curve,
                  skip_keys=("per_image_corr_sd",))  # sd can be nan only if <2 computable images (not expected)

    n_sign_mismatch = 0
    for tag, rows in [("analysis1", a1), ("analysis2", a2), ("analysis3", a3)]:
        for r in rows:
            if not (r["ci_lo"] <= r["mean_diff" if "mean_diff" in r else "contrast_mean"] <= r["ci_hi"]):
                print(f"  WARNING [{tag}]: estimate outside its own 95% CI: {r}")
            rs = r["rank_biserial_signed"]
            md = r["mean_diff"] if "mean_diff" in r else r["contrast_mean"]
            if not math.isnan(rs) and not math.isnan(md) and rs != 0 and md != 0:
                if (rs > 0) != (md > 0):
                    n_sign_mismatch += 1
                    p_holm = r.get("p_holm")
                    print(f"  NOTE [{tag}]: mean_diff and rank_biserial_signed disagree in sign "
                          f"(mean_diff={md:+.4f}, r_signed={rs:+.4f}, p_holm={p_holm:.3f}) -- expected "
                          f"for a near-null, non-significant effect where the mean (magnitude-weighted) "
                          f"and the rank statistic (count/rank-weighted) can point opposite ways; not a "
                          f"computation error. metric={r.get('metric')} layer={r.get('layer')} "
                          f"model_a={r.get('model_a', r.get('model'))} model_b={r.get('model_b')}")
    if n_sign_mismatch:
        print(f"  Total mean_diff / rank_biserial_signed sign disagreements: {n_sign_mismatch} "
              f"(all should be non-significant at Holm p<0.05 -- verified below)")
        mismatch_all_nonsig = True
        for tag, rows in [("analysis1", a1), ("analysis2", a2), ("analysis3", a3)]:
            for r in rows:
                rs, md = r["rank_biserial_signed"], r["mean_diff"] if "mean_diff" in r else r["contrast_mean"]
                if not math.isnan(rs) and not math.isnan(md) and rs != 0 and md != 0 and (rs > 0) != (md > 0):
                    if r["p_holm"] < 0.05:
                        mismatch_all_nonsig = False
        if not mismatch_all_nonsig:
            raise RuntimeError(
                "STOP: a mean_diff/rank_biserial_signed sign disagreement occurred on a "
                "Holm-significant result -- this would indicate a real computation problem, "
                "not expected near-null noise.")
        print("  Confirmed: all sign disagreements occur only on non-significant (Holm p>=0.05) results  OK")

    by_model_metric = {}
    for r in a5:
        by_model_metric.setdefault((r["model"], r["metric"]), []).append(r["peak_probability"])
    for key, probs in by_model_metric.items():
        total = sum(probs)
        if abs(total - 1.0) > 1e-9:
            raise RuntimeError(f"STOP: peak_probability for {key} sums to {total}, expected 1.0")

    print("  Verification checks complete (see any WARNING lines above).")


# ======================================================================
# Main
# ======================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("  DINO-S vs DINO-B vs CLIP-B -- paired three-model statistics")
    print("=" * 70)

    mats, image_ids = validate_and_load()
    n_images = len(image_ids)

    print(f"\n--- Shared bootstrap index array (seed={BOOT_SEED}, n_boot={N_BOOT}) ---")
    boot_indices = make_boot_indices(n_images)
    print(f"  boot_indices shape: {boot_indices.shape}")

    print(f"--- Shared permutation sign array (seed={PERM_SEED}, n_perm={N_PERM}) ---")
    perm_signs = make_perm_signs(n_images)
    print(f"  perm_signs shape: {perm_signs.shape}")

    print("\n--- Analysis 1: same-layer 3-model comparison (108 tests) ---")
    a1 = analysis1_same_layer(mats, boot_indices)
    sig1 = sum(1 for r in a1 if r["p_holm"] < 0.05)
    print(f"  {len(a1)} tests, {sig1} significant at Holm p<0.05")

    print("\n--- Analysis 2: within-model shape contrasts (36 tests) ---")
    a2 = analysis2_within_model(mats, boot_indices)
    for r in a2:
        if r["metric"] == "NSS":
            print(f"  {r['model']:8s} {r['contrast']:16s} mean={r['contrast_mean']:+.4f}  "
                  f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p_holm={r['p_holm']:.2e}  r={r['rank_biserial_signed']:+.3f}")

    print("\n--- Analysis 3: between-model shape difference (36 tests) ---")
    a3, highlights = analysis3_between_model(mats, boot_indices)
    for key, row in highlights.items():
        print(f"  [{key}] mean_diff={row['mean_diff']:+.4f}  CI=[{row['ci_lo']:+.4f},{row['ci_hi']:+.4f}]  "
              f"p_holm={row['p_holm']:.2e}  r={row['rank_biserial_signed']:+.3f}")

    print("\n--- Analysis 4: global curve-shape permutation test (9 tests) ---")
    a4 = analysis4_global_permutation(mats, perm_signs)
    for r in a4:
        print(f"  {r['model_a']}-{r['model_b']:8s} {r['metric']:10s} T={r['observed_statistic']:.4f}  "
              f"p_perm={r['permutation_p']:.4f}  p_holm={r['p_holm']:.4f}  "
              f"RMSE={r['mean_curve_rmse']:.4f}  Pearson={r['mean_curve_pearson_r']:.3f}  "
              f"Spearman={r['mean_curve_spearman_r']:.3f}")

    print("\n--- Analysis 5: peak-layer bootstrap stability (3 models) ---")
    a5, peak_value_boot_store = analysis5_peak_stability(mats, boot_indices)
    for model_name in MODEL_NAMES:
        for m in METRICS:
            r0 = next(r for r in a5 if r["model"] == model_name and r["metric"] == m
                     and r["layer"] == r["observed_peak_layer"])
            print(f"  {model_name:8s} {m:10s} observed=L{r0['observed_peak_layer']} "
                  f"({r0['observed_peak_value']:.4f})  modal=L{r0['modal_peak_layer']} "
                  f"(p={r0['modal_peak_probability']:.3f})")

    print("\n--- Analysis 6: max-performance 3-model comparison (exploratory) ---")
    a6 = analysis6_max_performance(peak_value_boot_store, a5)
    for r in a6:
        print(f"  {r['model_a']}-{r['model_b']:8s} {r['metric']:10s} obs_diff={r['observed_max_difference']:+.4f}  "
              f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  P(a>b)={r['proportion_model_a_greater']:.3f}")

    print("\n--- Analysis 7: DINO-S vs DINO-B similarity ---")
    a7_curve, a7_per_image = analysis7_dino_s_b_similarity(mats, image_ids, boot_indices)
    for r in a7_curve:
        print(f"  {r['metric']:10s} Pearson={r['pearson_r']:.4f}  Spearman={r['spearman_r']:.4f}  "
              f"RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  "
              f"per_image_corr_mean={r['per_image_corr_mean']:.4f}  "
              f"neg_corr_images={r['n_negative_corr_images']}  noncomputable={r['n_noncomputable_images']}")

    print("\n--- Verification ---")
    verify_all(a1, a2, a3, a4, a5, a6, a7_curve)

    print("\n--- Writing CSVs ---")
    write_csv(os.path.join(OUT_DIR, "paired_three_model_by_layer_tests.csv"), a1,
              ["metric", "layer", "model_a", "model_b", "N",
               "model_a_mean", "model_a_median", "model_b_mean", "model_b_median",
               "mean_diff", "median_diff", "ci_lo", "ci_hi",
               "wilcoxon_stat", "p_raw", "p_holm",
               "rank_biserial_signed", "rank_biserial_absolute",
               "positive_count", "negative_count", "zero_count"])
    write_csv(os.path.join(OUT_DIR, "within_model_shape_contrasts.csv"), a2,
              ["metric", "model", "contrast", "layer_hi", "layer_lo", "N",
               "contrast_mean", "contrast_median", "ci_lo", "ci_hi",
               "wilcoxon_stat", "p_raw", "p_holm",
               "rank_biserial_signed", "rank_biserial_absolute",
               "positive_count", "negative_count", "zero_count"])
    write_csv(os.path.join(OUT_DIR, "between_model_shape_contrasts.csv"), a3,
              ["metric", "model_a", "model_b", "contrast", "layer_hi", "layer_lo", "N",
               "model_a_component_mean", "model_b_component_mean",
               "mean_diff", "median_diff", "ci_lo", "ci_hi",
               "wilcoxon_stat", "p_raw", "p_holm",
               "rank_biserial_signed", "rank_biserial_absolute",
               "positive_count", "negative_count", "zero_count"])
    write_csv(os.path.join(OUT_DIR, "global_profile_permutation_tests.csv"), a4,
              ["model_a", "model_b", "metric", "observed_statistic", "permutation_p", "p_holm",
               "n_permutations", "mean_curve_rmse", "mean_curve_pearson_r", "mean_curve_spearman_r"])
    write_csv(os.path.join(OUT_DIR, "three_model_peak_bootstrap.csv"), a5,
              ["model", "metric", "layer", "peak_count", "peak_probability",
               "observed_peak_layer", "observed_peak_value",
               "modal_peak_layer", "modal_peak_probability",
               "bootstrap_peak_value_mean", "bootstrap_peak_value_median",
               "peak_value_ci_lo", "peak_value_ci_hi"])
    write_csv(os.path.join(OUT_DIR, "three_model_max_performance_bootstrap.csv"), a6,
              ["model_a", "model_b", "metric",
               "model_a_peak_layer", "model_a_peak_value", "model_b_peak_layer", "model_b_peak_value",
               "observed_max_difference", "bootstrap_mean_diff", "bootstrap_median_diff",
               "ci_lo", "ci_hi", "proportion_model_a_greater", "note"])
    write_csv(os.path.join(OUT_DIR, "dino_s_b_similarity.csv"), a7_curve,
              ["metric", "pearson_r", "spearman_r", "rmse", "mae",
               "peak_layer_dino_s", "peak_layer_dino_b", "peak_layer_diff",
               "peak_value_dino_s", "peak_value_dino_b", "peak_value_diff",
               "per_image_corr_mean", "per_image_corr_median", "per_image_corr_sd",
               "per_image_corr_ci_lo", "per_image_corr_ci_hi",
               "n_negative_corr_images", "n_noncomputable_images"])
    write_csv(os.path.join(OUT_DIR, "per_image_dino_s_b_profile_similarity.csv"), a7_per_image,
              ["image_id", "metric", "pearson_r", "computable"])
    print(f"  Saved 8 CSVs to {OUT_DIR}")

    print("\n--- Plotting ---")
    plot_peak_stability(a5, os.path.join(OUT_DIR, "three_model_peak_stability.png"),
                        os.path.join(OUT_DIR, "three_model_peak_stability.pdf"))
    plot_paired_difference_by_layer(a1, os.path.join(OUT_DIR, "three_model_paired_difference_by_layer.png"),
                                    os.path.join(OUT_DIR, "three_model_paired_difference_by_layer.pdf"))
    plot_shape_contrast_forest(a2, a3, os.path.join(OUT_DIR, "shape_contrast_forest.png"),
                              os.path.join(OUT_DIR, "shape_contrast_forest.pdf"))
    plot_dino_s_b_similarity(mats, a7_curve, os.path.join(OUT_DIR, "dino_s_b_profile_similarity.png"),
                             os.path.join(OUT_DIR, "dino_s_b_profile_similarity.pdf"))
    print("  Saved 4 figure pairs (png+pdf)")

    print("\n--- Writing stats_three_model_config.json ---")
    stats_config = {
        "seed_bootstrap": BOOT_SEED, "n_boot": N_BOOT,
        "seed_permutation": PERM_SEED, "n_perm": N_PERM,
        "alpha": ALPHA, "n_images": n_images,
        "model_pairs_convention": "always left model minus right model (model_a - model_b)",
        "model_pairs": [f"{a} - {b}" for a, b in MODEL_PAIRS],
        "bootstrap_method": (
            "Image-ID-level resampling with replacement; a single shared "
            "(n_boot, n_images) index array is generated once and reused for "
            "every analysis (including peak-layer/max-performance bootstraps) "
            "so all layers/metrics/models stay paired within each iteration."),
        "permutation_method": (
            "Sign-flip permutation on the paired per-image, per-layer difference "
            "D[i,l] = model_a[i,l] - model_b[i,l]. Each permutation draws one "
            "+-1 sign per IMAGE (not per layer) from a shared (n_perm, n_images) "
            "array, preserving the within-image correlation across the 12 "
            "layers. T = sum_l (mean_i(sign_i * D[i,l]))^2; "
            "p = (#{T_perm >= T_observed} + 1) / (n_perm + 1)."),
        "tie_break_rule": (
            f"argmax over the 12 layer means, rounded to {ROUND_DECIMALS_FOR_TIEBREAK} "
            "decimal places before comparison; numpy's argmax returns the first "
            "(lowest-index) maximum, so exact or near-exact ties resolve to the "
            "smallest layer number."),
        "effect_size": {
            "definition": "paired rank-biserial correlation, r = (W_pos - W_neg) / (n*(n+1)/2)",
            "rank_biserial_signed": (
                "Signed: positive when model_a tends to exceed model_b (matches the "
                "sign of mean_diff / contrast_mean); negative when model_b tends to "
                "exceed model_a. Computed directly from the signed-rank sums W_pos, "
                "W_neg, NOT derived after the fact from scipy's single 'statistic' "
                "value (which does not carry a sign)."),
            "rank_biserial_absolute": "abs(rank_biserial_signed); the magnitude-only version used in the earlier 2-model analysis.",
        },
        "holm_families": {
            "analysis1_same_layer": "3 model pairs x 12 layers = 36 tests per metric (3 families, one per metric)",
            "analysis2_within_model": "3 models x 4 contrasts = 12 tests per metric (3 families, one per metric)",
            "analysis3_between_model": "3 model pairs x 4 contrasts = 12 tests per metric (3 families, one per metric)",
            "analysis4_global_permutation": "3 model pairs x 3 metrics = 9 tests, ONE single family (not per-metric)",
        },
        "independent_two_sample_tests_used": False,
        "peak_and_max_performance_analyses_are_exploratory": True,
        "equivalence_never_claimed": (
            "No equivalence margin was pre-specified for any comparison in this "
            "script; a CI including 0 is reported as 'no clear advantage can be "
            "established', never as 'no difference' or 'equivalent'."),
        "n_significant_at_holm_0_05_analysis1": int(sig1),
        "source_files": {"dino_s": DINO_S_CSV, "dino_b": DINO_B_CSV, "clip_b": CLIP_B_CSV},
    }
    with open(os.path.join(OUT_DIR, "stats_three_model_config.json"), "w", encoding="utf-8") as f:
        json.dump(stats_config, f, indent=2)

    print("--- Writing statistics_three_model_summary.json / .md ---")
    write_summary(a1, a2, a3, highlights, a4, a5, a6, a7_curve, n_images)

    print("\n" + "=" * 70)
    print("  Done. Only outputs/expA_dino_vitb16/statistics_three_model/ was written to.")
    print("=" * 70)


def write_summary(a1, a2, a3, highlights, a4, a5, a6, a7_curve, n_images):
    summary = {
        "n_images": n_images, "seed_bootstrap": BOOT_SEED, "n_boot": N_BOOT,
        "seed_permutation": PERM_SEED, "n_perm": N_PERM,
        "analysis1_same_layer_count": len(a1),
        "analysis2_within_model": a2,
        "analysis3_highlights": highlights,
        "analysis4_global_permutation": a4,
        "analysis5_peak_summary": [
            {"model": r["model"], "metric": r["metric"], "observed_peak_layer": r["observed_peak_layer"],
             "observed_peak_value": r["observed_peak_value"], "modal_peak_layer": r["modal_peak_layer"],
             "modal_peak_probability": r["modal_peak_probability"]}
            for r in a5 if r["layer"] == r["observed_peak_layer"]
        ],
        "analysis6_max_performance": a6,
        "analysis7_dino_s_b_similarity": a7_curve,
    }
    with open(os.path.join(OUT_DIR, "statistics_three_model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("# DINO-S vs DINO-B vs CLIP-B -- paired three-model statistics summary\n")
    lines.append(f"N = {n_images} images (all 700, paired by image_id across all three models). "
                 f"Bootstrap: seed={BOOT_SEED}, n_boot={N_BOOT}. Permutation: seed={PERM_SEED}, "
                 f"n_perm={N_PERM}. Model-pair convention: left minus right (model_a - model_b).\n")
    lines.append("Every p-value is reported together with its effect size (signed and absolute "
                 "paired rank-biserial r) and its absolute mean/median difference with 95% "
                 "bootstrap CI -- p-values alone are not used to draw conclusions. Peak-layer "
                 "and max-performance comparisons (Analyses 5-6) are exploratory: peaks are "
                 "selected from the same data being compared. No equivalence margin was "
                 "pre-specified anywhere in this script, so equivalence is never claimed.\n")

    lines.append("\n## Analysis 3 highlights (middle/late shape difference, NSS)\n")
    lines.append("| highlight | mean diff | 95% CI | p (Holm) | r (signed) |")
    lines.append("|---|---|---|---|---|")
    for key, row in highlights.items():
        lines.append(f"| {key} | {row['mean_diff']:+.4f} | [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] | "
                     f"{row['p_holm']:.2e} | {row['rank_biserial_signed']:+.3f} |")

    lines.append("\n## Analysis 4: global curve-shape permutation test\n")
    lines.append("| model_a - model_b | metric | T | perm p | Holm p | RMSE | Pearson | Spearman |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in a4:
        lines.append(f"| {r['model_a']} - {r['model_b']} | {r['metric']} | {r['observed_statistic']:.4f} | "
                     f"{r['permutation_p']:.4f} | {r['p_holm']:.4f} | {r['mean_curve_rmse']:.4f} | "
                     f"{r['mean_curve_pearson_r']:.3f} | {r['mean_curve_spearman_r']:.3f} |")
    lines.append("\nNote: with N=700, even DINO-S vs DINO-B can reach statistical significance "
                 "from small per-layer differences. Significance is NOT the same as \"curves "
                 "look different\" -- RMSE / Pearson / Spearman / absolute differences above "
                 "should be read alongside the p-value, not in place of it.\n")

    lines.append("\n## Analysis 5: peak-layer bootstrap stability\n")
    lines.append("| model | metric | observed peak | modal peak (bootstrap) | modal probability |")
    lines.append("|---|---|---|---|---|")
    for r in a5:
        if r["layer"] == r["observed_peak_layer"]:
            lines.append(f"| {r['model']} | {r['metric']} | L{r['observed_peak_layer']} "
                         f"({r['observed_peak_value']:.4f}) | L{r['modal_peak_layer']} | "
                         f"{r['modal_peak_probability']:.3f} |")

    lines.append("\n## Analysis 6: max-performance comparison (EXPLORATORY)\n")
    lines.append("**proportion_model_a_greater is a bootstrap proportion, NOT a p-value.** "
                 "It is the fraction of the 10,000 image-level bootstrap resamples in which "
                 "model_a's own peak-layer value exceeded model_b's own peak-layer value in "
                 "that same resample.\n")
    lines.append("| model_a - model_b | metric | observed diff | boot mean diff | 95% CI | P(a>b) | interpretation |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in a6:
        ci_includes_zero = r["ci_lo"] <= 0.0 <= r["ci_hi"]
        interp = ("no clear advantage can be established (CI includes 0)" if ci_includes_zero
                  else f"favors {'model_a (' + r['model_a'] + ')' if r['bootstrap_mean_diff'] > 0 else 'model_b (' + r['model_b'] + ')'}")
        lines.append(f"| {r['model_a']} - {r['model_b']} | {r['metric']} | {r['observed_max_difference']:+.4f} | "
                     f"{r['bootstrap_mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                     f"{r['proportion_model_a_greater']:.3f} | {interp} |")

    lines.append("\n## Analysis 7: DINO-S vs DINO-B similarity\n")
    lines.append("| metric | Pearson | Spearman | RMSE | MAE | per-image corr mean | neg-corr images | noncomputable |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in a7_curve:
        lines.append(f"| {r['metric']} | {r['pearson_r']:.4f} | {r['spearman_r']:.4f} | {r['rmse']:.4f} | "
                     f"{r['mae']:.4f} | {r['per_image_corr_mean']:.4f} | {r['n_negative_corr_images']} | "
                     f"{r['n_noncomputable_images']} |")
    lines.append("\nNote: statistical significance in Analyses 1/4 does not, by itself, mean "
                 "DINO-S and DINO-B are qualitatively different -- if per-image/curve "
                 "correlations are high and RMSE / difference-in-differences effect sizes are "
                 "small, the appropriate description is \"the numbers differ, but the "
                 "qualitative layer profile is similar\", not \"statistically equivalent\" "
                 "(no equivalence test was run).\n")

    with open(os.path.join(OUT_DIR, "statistics_three_model_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
