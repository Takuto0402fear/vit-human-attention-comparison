"""
M-shape statistical analysis: DeiT/SL vs DINO-S vs DINO-B vs CLIP-B,
layerwise agreement with human gaze (OSIE 700, healthy).

Research questions:
  1. Is CLIP-B's M-shape statistically confirmed?
  2. Is CLIP-B's M-shape stronger than DeiT/SL, DINO-S, DINO-B?
  3. At which layers is the CLIP-B / other-model gap largest?
  4. Do conclusions depend on the metric or on which DeiT trial is used?

Reuses the EXACT statistical machinery already validated in
scripts/stats_three_model_layerwise.py (bootstrap seed=42/N=10000 image-
level resampling, permutation seed=42/N=10000 image-level sign-flip,
rank-biserial r=(W+-W-)/(n(n+1)/2), Holm step-down correction) --
re-implemented here (not imported) per this repo's own per-script
convention, restricted to CLIP-B as the reference model and extended
with the M-shape early/middle/late bin contrast and DeiT's 6-trial
sensitivity analysis (neither of which existed in the 3-model script).

Primary per-image data sources (verified column names, not assumed):
  DeiT/SL (6-trial mean, PRIMARY): outputs/expA_sl/combined/
    metrics_by_image_meanacrosstrials.csv -- cols {m}_mean_across_trials
  DeiT/SL (per trial, for Section 6 ONLY):
    outputs/expA_sl/trial_{01..06}/metrics_per_image.csv -- cols {m}
  DINO-S: outputs/expA/healthy/metrics_per_image.csv -- cols {m}
  DINO-B: outputs/expA_dino_vitb16/healthy/metrics_per_image.csv -- cols {m}
  CLIP-B: outputs/expA_clip/healthy/metrics_per_image.csv -- cols {m}

All four share the SAME 700 OSIE images (verified below), 12 layers,
patch16. The 6 DeiT trials are NOT pooled into 4200 independent samples
anywhere -- the primary analysis uses the 8400-row (700x12) 6-trial-MEAN
file; Section 6 analyzes each trial's own 700-image sample separately.

Output: outputs/model_comparison_statistics/ (new directory; no existing
outputs/expA*/ or outputs/model_comparison_layerwise/ file is modified).

Usage (PowerShell):
    python scripts\\stats_model_comparison_mcontrast.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import math
import os
import time

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
METRICS = ["NSS", "AUC_Judd", "sAUC"]
PRIMARY_METRIC = "NSS"
METRIC_DISPLAY = {"NSS": "NSS", "AUC_Judd": "AUC-Judd", "sAUC": "sAUC"}
N_EXPECTED_IMAGES = 700
BOOT_SEED = 42
N_BOOT = 10000
PERM_SEED = 42
N_PERM = 10000
ALPHA = 0.05

EARLY_LAYERS = [3, 4, 5]
MIDDLE_LAYERS = [7, 8, 9]
LATE_LAYERS = [11, 12]

REFERENCE_MODEL = "CLIP-B"
COMPARE_MODELS = ["DeiT-SL", "DINO-S", "DINO-B"]
MODEL_NAMES = ["DeiT-SL", "DINO-S", "DINO-B", "CLIP-B"]
STYLE = {"DeiT-SL": "firebrick", "DINO-S": "steelblue", "DINO-B": "seagreen", "CLIP-B": "darkorange"}
MARKER = {"DeiT-SL": "o", "DINO-S": "^", "DINO-B": "D", "CLIP-B": "s"}

DEIT_MEAN_CSV = r"C:\Users\user\gaze\outputs\expA_sl\combined\metrics_by_image_meanacrosstrials.csv"
DEIT_TRIAL_CSV = {t: rf"C:\Users\user\gaze\outputs\expA_sl\trial_{t:02d}\metrics_per_image.csv"
                   for t in range(1, 7)}
DINO_S_CSV = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"
DINO_B_CSV = r"C:\Users\user\gaze\outputs\expA_dino_vitb16\healthy\metrics_per_image.csv"
CLIP_B_CSV = r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_per_image.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\model_comparison_statistics"
# ======================================================


# ======================================================================
# Load + validate (reports issue counts rather than raising per-row, then
# raises once with the full report if anything is non-zero)
# ======================================================================

def load_matrix(csv_path, model_name, col_template="{m}"):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    n_rows = len(rows)

    per_image_layers = {}
    values = {m: {} for m in METRICS}
    seen_pairs = set()
    n_duplicates = 0
    n_nan_inf = 0
    n_range_violations = 0

    for row in rows:
        img_id = os.path.splitext(row["image"])[0]
        layer = int(row["layer"])
        key = (img_id, layer)
        if key in seen_pairs:
            n_duplicates += 1
            continue
        seen_pairs.add(key)
        per_image_layers.setdefault(img_id, set()).add(layer)
        for m in METRICS:
            col = col_template.format(m=m)
            v = float(row[col])
            if math.isnan(v) or math.isinf(v):
                n_nan_inf += 1
            if m in ("AUC_Judd", "sAUC") and not (0.0 <= v <= 1.0):
                n_range_violations += 1
            values[m][key] = v

    image_ids = sorted(per_image_layers)
    bad_layer_images = [img for img, layers in per_image_layers.items() if layers != set(range(1, 13))]

    report = {
        "model": model_name, "source_file": csv_path, "n_rows": n_rows,
        "n_unique_images": len(image_ids), "n_duplicates": n_duplicates,
        "n_nan_inf": n_nan_inf, "n_range_violations": n_range_violations,
        "n_images_missing_layers": len(bad_layer_images),
    }
    ok = (n_rows == N_EXPECTED_IMAGES * 12 and len(image_ids) == N_EXPECTED_IMAGES and
          n_duplicates == 0 and n_nan_inf == 0 and n_range_violations == 0 and len(bad_layer_images) == 0)
    report["ok"] = ok
    if not ok:
        raise RuntimeError(f"STOP [{model_name}]: data integrity failed: {report}")
    return values, image_ids, report


def build_matrices(values, image_ids):
    return {m: np.array([[values[m][(img, l)] for l in range(1, 13)] for img in image_ids]) for m in METRICS}


def validate_and_load():
    print("--- Loading + validating (per-model) ---")
    values, ids, reports = {}, {}, {}
    sources = [("DeiT-SL", DEIT_MEAN_CSV, "{m}_mean_across_trials"), ("DINO-S", DINO_S_CSV, "{m}"),
               ("DINO-B", DINO_B_CSV, "{m}"), ("CLIP-B", CLIP_B_CSV, "{m}")]
    for name, path, tmpl in sources:
        values[name], ids[name], reports[name] = load_matrix(path, name, tmpl)
        print(f"  {name}: {reports[name]['n_unique_images']} images, "
              f"{reports[name]['n_duplicates']} dup, {reports[name]['n_nan_inf']} nan/inf  OK")

    print("\n--- Cross-model validation ---")
    id_sets = {n: set(ids[n]) for n in MODEL_NAMES}
    common = set.intersection(*id_sets.values())
    all_equal = all(id_sets[n] == common for n in MODEL_NAMES)
    print(f"  Common images across all 4 models: {len(common)}  (all-4-identical: {all_equal})")
    if not all_equal or len(common) != N_EXPECTED_IMAGES:
        raise RuntimeError(f"STOP: image-ID sets are not identical 700-image sets across all 4 models "
                           f"(common={len(common)})")
    image_ids = sorted(common)

    mats = {name: build_matrices(values[name], image_ids) for name in MODEL_NAMES}

    integrity = {
        "per_model": reports,
        "common_images_across_all_4": len(common),
        "all_4_identical_image_sets": bool(all_equal),
        "n_images_used": len(image_ids),
        "n_layers": 12,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return mats, image_ids, integrity


# ======================================================================
# Statistics primitives (identical formulas to scripts/stats_three_model_layerwise.py)
# ======================================================================

def wilcoxon_test_signed(x, y, alternative="two-sided"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x - y
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    n_zero = int((diff == 0).sum())

    diff_nz = diff[diff != 0]
    n = len(diff_nz)
    if n < 10:
        return {"stat": np.nan, "p": np.nan, "r_signed": np.nan,
                "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "n_nonzero": n}

    abs_diff = np.abs(diff_nz)
    ranks = sp_stats.rankdata(abs_diff)
    signs = np.sign(diff_nz)
    total_rank = n * (n + 1) / 2
    w_pos = ranks[signs > 0].sum()
    w_neg = ranks[signs < 0].sum()

    res = sp_stats.wilcoxon(diff_nz, alternative=alternative)
    r_signed = float((w_pos - w_neg) / total_rank)

    return {"stat": float(res.statistic), "p": float(res.pvalue), "r_signed": r_signed,
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


# ======================================================================
# M-shape bin contrasts
# ======================================================================

def compute_bins(mat):
    """mat: (n_images, 12). Returns early, middle, late, m_contrast,
    early_drop, late_recovery, each (n_images,)."""
    early = mat[:, [l - 1 for l in EARLY_LAYERS]].mean(axis=1)
    middle = mat[:, [l - 1 for l in MIDDLE_LAYERS]].mean(axis=1)
    late = mat[:, [l - 1 for l in LATE_LAYERS]].mean(axis=1)
    m_contrast = (early + late) / 2 - middle
    early_drop = early - middle
    late_recovery = late - middle
    return early, middle, late, m_contrast, early_drop, late_recovery


def section2_m_contrast_per_image_and_summary(mats, image_ids, boot_indices):
    per_image_rows = []
    bins_cache = {}  # (model, metric) -> dict of arrays
    for model in MODEL_NAMES:
        for m in METRICS:
            early, middle, late, mc, ed, lr = compute_bins(mats[model][m])
            bins_cache[(model, m)] = {"early": early, "middle": middle, "late": late,
                                       "m_contrast": mc, "early_drop": ed, "late_recovery": lr}
            for i, img in enumerate(image_ids):
                per_image_rows.append({
                    "model": model, "image": img, "metric": m,
                    "early": float(early[i]), "middle": float(middle[i]), "late": float(late[i]),
                    "m_contrast": float(mc[i]), "early_drop": float(ed[i]), "late_recovery": float(lr[i]),
                })

    summary_rows = []
    measures = ["m_contrast", "early_drop", "late_recovery"]
    for m in METRICS:
        pvals, metric_rows = [], []
        for model in MODEL_NAMES:
            for measure in measures:
                x = bins_cache[(model, m)][measure]
                zeros = np.zeros_like(x)
                wt = wilcoxon_test_signed(x, zeros)
                ci_lo, ci_hi = bootstrap_ci_mean(x, boot_indices)
                row = {
                    "model": model, "metric": m, "measure": measure, "N": int(len(x)),
                    "mean": float(x.mean()), "median": float(np.median(x)), "sd": float(x.std(ddof=1)),
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "pct_images_positive": float((x > 0).mean()),
                    "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
                    "wilcoxon_stat": wt["stat"], "p_raw": wt["p"], "rank_biserial_signed": wt["r_signed"],
                }
                metric_rows.append(row)
                pvals.append(wt["p"])
        p_holm = holm_correction(np.array(pvals))
        for r, ph in zip(metric_rows, p_holm):
            r["p_holm"] = float(ph)
        summary_rows.extend(metric_rows)

    return per_image_rows, summary_rows, bins_cache


# ======================================================================
# Section 3: CLIP-B vs {DeiT-SL, DINO-S, DINO-B} -- M-shape measures
# ======================================================================

def section3_model_comparisons(bins_cache, boot_indices):
    measures = ["m_contrast", "early_drop", "late_recovery"]
    rows = []
    for m in METRICS:
        for measure in measures:
            pvals, sub_rows = [], []
            ref = bins_cache[(REFERENCE_MODEL, m)][measure]
            for model in COMPARE_MODELS:
                x = bins_cache[(model, m)][measure]
                diff = x - ref  # compare_model - CLIP-B
                wt = wilcoxon_test_signed(x, ref)
                ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
                sub_rows.append({
                    "comparison": f"{model} - {REFERENCE_MODEL}", "metric": m, "measure": measure,
                    "N": int(len(diff)),
                    f"{model}_mean": float(x.mean()), f"{REFERENCE_MODEL}_mean": float(ref.mean()),
                    "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "wilcoxon_stat": wt["stat"], "p_raw": wt["p"], "rank_biserial_signed": wt["r_signed"],
                })
                pvals.append(wt["p"])
            p_holm = holm_correction(np.array(pvals))  # family = 3 model comparisons, within this (metric, measure)
            for r, ph in zip(sub_rows, p_holm):
                r["p_holm"] = float(ph)
                r["holm_family"] = f"metric={m},measure={measure} (3 tests: DeiT/DINO-S/DINO-B vs CLIP-B)"
            rows.extend(sub_rows)
    return rows


# ======================================================================
# Section 4: layerwise paired comparisons, CLIP-B vs {DeiT,DINO-S,DINO-B}
# ======================================================================

def section4_layerwise(mats, boot_indices):
    rows = []
    for m in METRICS:
        pvals, metric_rows = [], []
        for model in COMPARE_MODELS:
            for l in range(1, 13):
                a = mats[model][m][:, l - 1]
                b = mats[REFERENCE_MODEL][m][:, l - 1]
                diff = a - b
                wt = wilcoxon_test_signed(a, b)
                ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
                metric_rows.append({
                    "comparison": f"{model} - {REFERENCE_MODEL}", "metric": m, "layer": l, "N": int(len(diff)),
                    f"model_mean": float(a.mean()), "clip_mean": float(b.mean()),
                    "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "wilcoxon_stat": wt["stat"], "p_raw": wt["p"], "rank_biserial_signed": wt["r_signed"],
                })
                pvals.append(wt["p"])
        p_holm = holm_correction(np.array(pvals))  # family = 3 comparisons x 12 layers = 36, per metric
        for r, ph in zip(metric_rows, p_holm):
            r["p_holm"] = float(ph)
        rows.extend(metric_rows)
    return rows


# ======================================================================
# Section 5: curve-level permutation test, CLIP-B vs {DeiT,DINO-S,DINO-B}
# ======================================================================

def section5_curve_permutation(mats, perm_signs):
    rows = []
    all_p = []
    for model in COMPARE_MODELS:
        for m in METRICS:
            D = mats[model][m] - mats[REFERENCE_MODEL][m]  # (700, 12), model - CLIP-B
            observed_layer_means = D.mean(axis=0)
            t_observed = float(np.sum(observed_layer_means ** 2))

            n_images = D.shape[0]
            M = perm_signs @ D  # (n_perm, 700) @ (700, 12) -> (n_perm, 12)
            t_perm = np.sum((M / n_images) ** 2, axis=1)
            p_perm = float((np.sum(t_perm >= t_observed) + 1) / (N_PERM + 1))

            mean_model = mats[model][m].mean(axis=0)
            mean_clip = mats[REFERENCE_MODEL][m].mean(axis=0)
            rmse = float(np.sqrt(np.mean((mean_model - mean_clip) ** 2)))
            pearson_r = float(sp_stats.pearsonr(mean_model, mean_clip)[0])

            row = {
                "comparison": f"{model} - {REFERENCE_MODEL}", "metric": m,
                "observed_statistic": t_observed, "permutation_p": p_perm, "n_permutations": N_PERM,
                "mean_curve_rmse": rmse, "mean_curve_pearson_r": pearson_r,
            }
            rows.append(row)
            all_p.append(p_perm)

    p_holm = holm_correction(np.array(all_p))  # ONE single family: 3 comparisons x 3 metrics = 9 tests
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = float(ph)
    return rows


# ======================================================================
# Section 6: DeiT trial sensitivity
# ======================================================================

def section6_deit_trial_sensitivity(mats, image_ids, boot_indices):
    """
    Loads each of the 6 DeiT trials independently (each its own 700-image
    sample; never pooled into 4200 rows), computes M_contrast per trial,
    compares each trial vs CLIP-B, and re-derives the leave-one-trial-out
    5-trial mean (per image, per layer) to check whether dropping any
    single trial changes the qualitative conclusion reached by the
    primary 6-trial-mean analysis.
    """
    trial_mats = {}
    for t in range(1, 7):
        values, ids, _ = load_matrix(DEIT_TRIAL_CSV[t], f"DeiT-trial{t:02d}", "{m}")
        if sorted(ids) != image_ids:
            raise RuntimeError(f"STOP: DeiT trial {t:02d} image-ID set does not match the shared 700-image set")
        trial_mats[t] = build_matrices(values, image_ids)

    clip_mc_ref = compute_bins(mats[REFERENCE_MODEL][PRIMARY_METRIC])[3]  # m_contrast, for quick reuse below

    rows = []

    # --- reference: 6-trial-mean result (primary analysis anchor) ---
    ref_rows = {}
    for m in METRICS:
        mc_deit = compute_bins(mats["DeiT-SL"][m])[3]
        mc_clip = compute_bins(mats[REFERENCE_MODEL][m])[3]
        diff = mc_deit - mc_clip
        wt = wilcoxon_test_signed(mc_deit, mc_clip)
        ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
        ref_rows[m] = {"mean_diff": float(diff.mean()), "ci_lo": ci_lo, "ci_hi": ci_hi, "p": wt["p"]}
        rows.append({
            "row_type": "six_trial_mean", "trial_or_excluded": "mean_of_6", "metric": m,
            "m_contrast_mean": float(mc_deit.mean()), "m_contrast_median": float(np.median(mc_deit)),
            "vs_clip_mean_diff": float(diff.mean()), "vs_clip_ci_lo": ci_lo, "vs_clip_ci_hi": ci_hi,
            "vs_clip_wilcoxon_p": wt["p"], "vs_clip_rank_biserial_r": wt["r_signed"],
            "conclusion_consistent_with_primary": True,
        })

    # --- single trials ---
    for t in range(1, 7):
        for m in METRICS:
            mc_trial = compute_bins(trial_mats[t][m])[3]
            mc_clip = compute_bins(mats[REFERENCE_MODEL][m])[3]
            diff = mc_trial - mc_clip
            wt = wilcoxon_test_signed(mc_trial, mc_clip)
            ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
            ref = ref_rows[m]
            same_sign = np.sign(diff.mean()) == np.sign(ref["mean_diff"])
            same_sig = (ci_lo > 0 or ci_hi < 0) == (ref["ci_lo"] > 0 or ref["ci_hi"] < 0)
            rows.append({
                "row_type": "single_trial", "trial_or_excluded": f"trial{t:02d}", "metric": m,
                "m_contrast_mean": float(mc_trial.mean()), "m_contrast_median": float(np.median(mc_trial)),
                "vs_clip_mean_diff": float(diff.mean()), "vs_clip_ci_lo": ci_lo, "vs_clip_ci_hi": ci_hi,
                "vs_clip_wilcoxon_p": wt["p"], "vs_clip_rank_biserial_r": wt["r_signed"],
                "conclusion_consistent_with_primary": bool(same_sign and same_sig),
            })

    # --- leave-one-trial-out (5-trial mean per image, per layer) ---
    for excl in range(1, 7):
        remaining = [t for t in range(1, 7) if t != excl]
        for m in METRICS:
            stacked = np.stack([trial_mats[t][m] for t in remaining], axis=0)  # (5, 700, 12)
            loo_mean = stacked.mean(axis=0)  # (700, 12)
            mc_loo = compute_bins(loo_mean)[3]
            mc_clip = compute_bins(mats[REFERENCE_MODEL][m])[3]
            diff = mc_loo - mc_clip
            wt = wilcoxon_test_signed(mc_loo, mc_clip)
            ci_lo, ci_hi = bootstrap_ci_mean(diff, boot_indices)
            ref = ref_rows[m]
            same_sign = np.sign(diff.mean()) == np.sign(ref["mean_diff"])
            same_sig = (ci_lo > 0 or ci_hi < 0) == (ref["ci_lo"] > 0 or ref["ci_hi"] < 0)
            rows.append({
                "row_type": "leave_one_out", "trial_or_excluded": f"excl_trial{excl:02d}", "metric": m,
                "m_contrast_mean": float(mc_loo.mean()), "m_contrast_median": float(np.median(mc_loo)),
                "vs_clip_mean_diff": float(diff.mean()), "vs_clip_ci_lo": ci_lo, "vs_clip_ci_hi": ci_hi,
                "vs_clip_wilcoxon_p": wt["p"], "vs_clip_rank_biserial_r": wt["r_signed"],
                "conclusion_consistent_with_primary": bool(same_sign and same_sig),
            })

    return rows, trial_mats


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
# Plots (auxiliary only -- do not duplicate outputs/model_comparison_layerwise/)
# ======================================================================

def plot_mcontrast_distribution(bins_cache, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m in zip(axes, METRICS):
        data = [bins_cache[(model, m)]["m_contrast"] for model in MODEL_NAMES]
        bp = ax.boxplot(data, labels=MODEL_NAMES, showmeans=True, patch_artist=True)
        for patch, model in zip(bp["boxes"], MODEL_NAMES):
            patch.set_facecolor(STYLE[model])
            patch.set_alpha(0.35)
        ax.axhline(0, color="gray", linestyle=":", lw=1.2)
        ax.set_ylabel("M_contrast = ((early+late)/2) - middle")
        ax.set_title(METRIC_DISPLAY[m])
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("M-shape contrast distribution per model (N=700 images)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_layerwise_diff_vs_clip(a4_rows, out_png):
    layers_arr = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m in zip(axes, METRICS):
        for model in COMPARE_MODELS:
            comp = f"{model} - {REFERENCE_MODEL}"
            sub = sorted([r for r in a4_rows if r["metric"] == m and r["comparison"] == comp],
                        key=lambda r: r["layer"])
            mean_diff = np.array([r["mean_diff"] for r in sub])
            ci_lo = np.array([r["ci_lo"] for r in sub])
            ci_hi = np.array([r["ci_hi"] for r in sub])
            ax.plot(layers_arr, mean_diff, marker=MARKER[model], color=STYLE[model], lw=2, ms=5, label=comp)
            ax.fill_between(layers_arr, ci_lo, ci_hi, alpha=0.15, color=STYLE[model])
        ax.axhline(0, color="gray", linestyle=":", lw=1.2)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"{METRIC_DISPLAY[m]} (model - CLIP-B)")
        ax.set_title(METRIC_DISPLAY[m])
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Paired per-layer difference vs CLIP-B, mean +/- 95% bootstrap CI", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_deit_trial_mcontrast(a6_rows, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m in zip(axes, METRICS):
        trial_rows = sorted([r for r in a6_rows if r["row_type"] == "single_trial" and r["metric"] == m],
                            key=lambda r: r["trial_or_excluded"])
        ref_row = next(r for r in a6_rows if r["row_type"] == "six_trial_mean" and r["metric"] == m)
        xs = np.arange(1, 7)
        means = [r["m_contrast_mean"] for r in trial_rows]
        ax.bar(xs, means, color="firebrick", alpha=0.7)
        ax.axhline(ref_row["m_contrast_mean"], color="black", lw=1.6, ls="--", label="6-trial mean")
        ax.axhline(0, color="gray", lw=1.0, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"t{t:02d}" for t in range(1, 7)])
        ax.set_ylabel("M_contrast (mean over 700 images)")
        ax.set_title(METRIC_DISPLAY[m])
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("DeiT/SL: M_contrast per trial vs the 6-trial mean", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  M-shape statistics: DeiT/SL vs DINO-S vs DINO-B vs CLIP-B")
    print("=" * 65)

    mats, image_ids, integrity = validate_and_load()
    n_images = len(image_ids)
    boot_indices = make_boot_indices(n_images)
    perm_signs = make_perm_signs(n_images)

    with open(os.path.join(OUT_DIR, "data_integrity_report.json"), "w", encoding="utf-8") as f:
        json.dump(integrity, f, indent=2)
    print(f"\nSaved: data_integrity_report.json")

    print("\n--- Section 2: M-shape bin contrasts (per image + per-model summary) ---")
    per_image_rows, summary_rows, bins_cache = section2_m_contrast_per_image_and_summary(
        mats, image_ids, boot_indices)
    write_csv(os.path.join(OUT_DIR, "m_contrast_per_image.csv"), per_image_rows,
              ["model", "image", "metric", "early", "middle", "late", "m_contrast", "early_drop", "late_recovery"])
    write_csv(os.path.join(OUT_DIR, "m_contrast_summary.csv"), summary_rows,
              ["model", "metric", "measure", "N", "mean", "median", "sd", "ci_lo", "ci_hi",
               "pct_images_positive", "ci_excludes_zero", "wilcoxon_stat", "p_raw", "p_holm",
               "rank_biserial_signed"])
    print(f"  m_contrast_per_image.csv: {len(per_image_rows)} rows")
    print(f"  m_contrast_summary.csv: {len(summary_rows)} rows")
    for r in summary_rows:
        if r["metric"] == PRIMARY_METRIC and r["measure"] == "m_contrast":
            print(f"    {r['model']:10s} M_contrast(NSS): mean={r['mean']:+.4f} "
                  f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p_holm={r['p_holm']:.2e}")

    print("\n--- Section 3: model comparisons vs CLIP-B (M-shape measures) ---")
    a3_rows = section3_model_comparisons(bins_cache, boot_indices)
    write_csv(os.path.join(OUT_DIR, "m_contrast_model_comparisons.csv"), a3_rows,
              ["comparison", "metric", "measure", "N"] +
              [f"{m}_mean" for m in COMPARE_MODELS] + [f"{REFERENCE_MODEL}_mean"] +
              ["mean_diff", "median_diff", "ci_lo", "ci_hi", "wilcoxon_stat", "p_raw", "p_holm",
               "rank_biserial_signed", "holm_family"])
    print(f"  m_contrast_model_comparisons.csv: {len(a3_rows)} rows")
    for r in a3_rows:
        if r["metric"] == PRIMARY_METRIC and r["measure"] == "m_contrast":
            print(f"    {r['comparison']:20s} mean_diff={r['mean_diff']:+.4f} "
                  f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p_holm={r['p_holm']:.2e} r={r['rank_biserial_signed']:+.3f}")

    print("\n--- Section 4: layerwise paired comparisons vs CLIP-B ---")
    a4_rows = section4_layerwise(mats, boot_indices)
    write_csv(os.path.join(OUT_DIR, "layerwise_paired_comparisons.csv"), a4_rows,
              ["comparison", "metric", "layer", "N", "model_mean", "clip_mean", "mean_diff", "median_diff",
               "ci_lo", "ci_hi", "wilcoxon_stat", "p_raw", "p_holm", "rank_biserial_signed"])
    print(f"  layerwise_paired_comparisons.csv: {len(a4_rows)} rows")
    n_sig4 = sum(1 for r in a4_rows if r["p_holm"] < ALPHA)
    print(f"  significant at Holm alpha=0.05: {n_sig4}/{len(a4_rows)}")

    print("\n--- Section 5: curve-level permutation test vs CLIP-B ---")
    a5_rows = section5_curve_permutation(mats, perm_signs)
    write_csv(os.path.join(OUT_DIR, "curve_level_permutation_tests.csv"), a5_rows,
              ["comparison", "metric", "observed_statistic", "permutation_p", "p_holm", "n_permutations",
               "mean_curve_rmse", "mean_curve_pearson_r"])
    print(f"  curve_level_permutation_tests.csv: {len(a5_rows)} rows")
    for r in a5_rows:
        print(f"    {r['comparison']:20s} {r['metric']:9s} T={r['observed_statistic']:.4f} "
              f"p_perm={r['permutation_p']:.4f} p_holm={r['p_holm']:.4f}")

    print("\n--- Section 6: DeiT trial sensitivity ---")
    a6_rows, trial_mats = section6_deit_trial_sensitivity(mats, image_ids, boot_indices)
    write_csv(os.path.join(OUT_DIR, "deit_trial_sensitivity.csv"), a6_rows,
              ["row_type", "trial_or_excluded", "metric", "m_contrast_mean", "m_contrast_median",
               "vs_clip_mean_diff", "vs_clip_ci_lo", "vs_clip_ci_hi", "vs_clip_wilcoxon_p",
               "vs_clip_rank_biserial_r", "conclusion_consistent_with_primary"])
    n_inconsistent = sum(1 for r in a6_rows if not r["conclusion_consistent_with_primary"])
    print(f"  deit_trial_sensitivity.csv: {len(a6_rows)} rows, {n_inconsistent} inconsistent-with-primary")

    # ------------------------------------------------------------------
    # Auxiliary plots
    # ------------------------------------------------------------------
    print("\n--- Auxiliary plots ---")
    p1 = os.path.join(OUT_DIR, "m_contrast_by_model_distribution.png")
    plot_mcontrast_distribution(bins_cache, p1)
    print(f"  Saved: {p1}")
    p2 = os.path.join(OUT_DIR, "layerwise_diff_vs_clip_with_ci.png")
    plot_layerwise_diff_vs_clip(a4_rows, p2)
    print(f"  Saved: {p2}")
    p3 = os.path.join(OUT_DIR, "deit_trial_mcontrast.png")
    plot_deit_trial_mcontrast(a6_rows, p3)
    print(f"  Saved: {p3}")

    # ------------------------------------------------------------------
    # analysis_config.json
    # ------------------------------------------------------------------
    config = {
        "seed_bootstrap": BOOT_SEED, "n_boot": N_BOOT, "seed_permutation": PERM_SEED, "n_perm": N_PERM,
        "alpha": ALPHA, "n_images": n_images,
        "layer_bins": {"early": EARLY_LAYERS, "middle": MIDDLE_LAYERS, "late": LATE_LAYERS},
        "m_contrast_definition": "((mean(early_layers) + mean(late_layers)) / 2) - mean(middle_layers)",
        "early_drop_definition": "mean(early_layers) - mean(middle_layers)",
        "late_recovery_definition": "mean(late_layers) - mean(middle_layers)",
        "reference_model": REFERENCE_MODEL, "compare_models": COMPARE_MODELS,
        "primary_metric": PRIMARY_METRIC, "secondary_metrics": [m for m in METRICS if m != PRIMARY_METRIC],
        "bootstrap_method": "Image-ID-level resampling with replacement; a single shared (n_boot, n_images) "
                            "index array reused across all analyses.",
        "permutation_method": "Sign-flip permutation on the paired per-image, per-layer difference "
                              "D[i,l] = model[i,l] - CLIP_B[i,l]. One +-1 sign per IMAGE (shared array), "
                              "preserving within-image correlation across 12 layers. "
                              "T = sum_l (mean_i(sign_i * D[i,l]))^2; p = (#{T_perm>=T_obs}+1)/(n_perm+1).",
        "effect_size": "matched-pairs rank-biserial correlation, r = (W_pos - W_neg) / (n*(n+1)/2), "
                       "signed to match the sign of mean_diff.",
        "holm_families": {
            "section2_per_model_summary": "4 models x 3 measures = 12 tests per metric (3 families, one per metric)",
            "section3_model_comparisons": "3 model comparisons per (metric, measure) = 9 families of 3 tests "
                                          "(one family per metric x measure combination)",
            "section4_layerwise": "3 comparisons x 12 layers = 36 tests per metric (3 families, one per metric)",
            "section5_curve_permutation": "3 comparisons x 3 metrics = 9 tests, ONE single family "
                                          "(matches scripts/stats_three_model_layerwise.py's analysis4 convention)",
            "section6_trial_sensitivity": "not Holm-corrected -- each trial/LOO row is an independent "
                                          "sensitivity check against the SAME primary-analysis reference, "
                                          "not a family of tests answering one shared question",
        },
        "deit_trial_handling": "6 trials analyzed independently (Section 6) and as their 6-trial MEAN "
                               "(primary analysis, Sections 2-5) -- never pooled into 4200 (=6x700) "
                               "independent samples for any hypothesis test.",
        "source_files": {
            "deit_sl_primary": DEIT_MEAN_CSV, "deit_sl_trials": DEIT_TRIAL_CSV,
            "dino_s": DINO_S_CSV, "dino_b": DINO_B_CSV, "clip_b": CLIP_B_CSV,
        },
        "n_significant_layerwise_holm_0_05": int(n_sig4),
        "n_deit_trial_sensitivity_inconsistent": int(n_inconsistent),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT_DIR, "analysis_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved: analysis_config.json")

    # ------------------------------------------------------------------
    # statistics_summary.md
    # ------------------------------------------------------------------
    write_summary(summary_rows, a3_rows, a4_rows, a5_rows, a6_rows, n_images, n_sig4, n_inconsistent)
    print(f"Saved: statistics_summary.md")

    print("\n" + "=" * 65)
    print("  M-shape statistics: DONE")
    print("  No existing outputs/expA*/ or outputs/model_comparison_layerwise/ file was modified.")
    print("=" * 65)


def write_summary(summary_rows, a3_rows, a4_rows, a5_rows, a6_rows, n_images, n_sig4, n_inconsistent):
    lines = []
    lines.append("# M-shape statistical analysis -- DeiT/SL vs DINO-S vs DINO-B vs CLIP-B\n")
    lines.append(f"N = {n_images} images (all 4 models paired by image_id). "
                 f"Bootstrap: seed={BOOT_SEED}, n_boot={N_BOOT}. Permutation: seed={PERM_SEED}, n_perm={N_PERM}. "
                 f"Reference model: {REFERENCE_MODEL}. DeiT/SL primary analysis uses its 6-trial MEAN "
                 f"(never the pooled 6x700 rows). Layer bins: early={EARLY_LAYERS}, middle={MIDDLE_LAYERS}, "
                 f"late={LATE_LAYERS}.\n")
    lines.append("Every p-value is reported together with its effect size (signed rank-biserial r) and its "
                 "mean/median difference with 95% bootstrap CI -- p-values alone are not used to draw "
                 "conclusions.\n")

    lines.append("## Section 2: M_contrast per-model summary (primary metric: NSS)\n")
    lines.append("| model | measure | mean | 95% CI | p (Holm) | r (signed) |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary_rows:
        if r["metric"] == PRIMARY_METRIC:
            lines.append(f"| {r['model']} | {r['measure']} | {r['mean']:+.4f} | "
                         f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_holm']:.2e} | "
                         f"{r['rank_biserial_signed']:+.3f} |")
    lines.append("")

    lines.append("## Section 3: M_contrast, model vs CLIP-B (all 3 metrics)\n")
    lines.append("| comparison | metric | measure | mean diff | 95% CI | p (Holm) | r (signed) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in a3_rows:
        if r["measure"] == "m_contrast":
            lines.append(f"| {r['comparison']} | {r['metric']} | {r['measure']} | {r['mean_diff']:+.4f} | "
                         f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_holm']:.2e} | "
                         f"{r['rank_biserial_signed']:+.3f} |")
    lines.append("")

    lines.append("## Section 4: layerwise paired comparisons vs CLIP-B\n")
    lines.append(f"{n_sig4}/{len(a4_rows)} tests significant at Holm alpha=0.05 (see "
                 "layerwise_paired_comparisons.csv for the full table). Largest |mean_diff| per comparison "
                 "(primary metric NSS):\n")
    lines.append("| comparison | layer | mean diff | 95% CI | p (Holm) |")
    lines.append("|---|---|---|---|---|")
    for comp in [f"{m} - {REFERENCE_MODEL}" for m in COMPARE_MODELS]:
        sub = [r for r in a4_rows if r["comparison"] == comp and r["metric"] == PRIMARY_METRIC]
        best = max(sub, key=lambda r: abs(r["mean_diff"]))
        lines.append(f"| {comp} | L{best['layer']} | {best['mean_diff']:+.4f} | "
                     f"[{best['ci_lo']:+.4f}, {best['ci_hi']:+.4f}] | {best['p_holm']:.2e} |")
    lines.append("")

    lines.append("## Section 5: curve-level permutation test (single Holm family, 9 tests)\n")
    lines.append("| comparison | metric | T | perm p | Holm p | RMSE | Pearson r |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in a5_rows:
        lines.append(f"| {r['comparison']} | {r['metric']} | {r['observed_statistic']:.4f} | "
                     f"{r['permutation_p']:.4f} | {r['p_holm']:.4f} | {r['mean_curve_rmse']:.4f} | "
                     f"{r['mean_curve_pearson_r']:.3f} |")
    lines.append("")

    lines.append("## Section 6: DeiT trial sensitivity\n")
    lines.append(f"{n_inconsistent}/{len(a6_rows)} rows disagree in sign or significance with the primary "
                 "6-trial-mean result (see deit_trial_sensitivity.csv).\n")
    lines.append("| row_type | trial/excluded | metric | M_contrast mean | vs CLIP-B diff | 95% CI | consistent |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in a6_rows:
        if r["metric"] == PRIMARY_METRIC:
            lines.append(f"| {r['row_type']} | {r['trial_or_excluded']} | {r['metric']} | "
                         f"{r['m_contrast_mean']:+.4f} | {r['vs_clip_mean_diff']:+.4f} | "
                         f"[{r['vs_clip_ci_lo']:+.4f}, {r['vs_clip_ci_hi']:+.4f}] | "
                         f"{r['conclusion_consistent_with_primary']} |")
    lines.append("")

    with open(os.path.join(OUT_DIR, "statistics_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
