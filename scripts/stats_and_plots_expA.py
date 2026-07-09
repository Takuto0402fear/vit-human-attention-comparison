"""
Experiment A -- statistical analysis and publication-ready plots.

Reads metrics_per_image.csv (700 images x 12 layers) and produces:
  (A) stats_summary.csv     -- mean / SD / 95% CI per layer x metric
      stats_tests.csv       -- Wilcoxon tests (adjacent, peak, L1, baseline)
      metrics_by_layer.csv  -- updated with SD + 95% CI columns
  (B) layerwise_3metrics_full.png     -- +/- SD band  (subset style)
      layerwise_3metrics_full_ci.png  -- 95% CI band  (same layout)
"""

import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import os
import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── CONFIG ───
OUT_DIR  = r"C:\Users\user\gaze\outputs\expA\healthy"
CSV_IN   = os.path.join(OUT_DIR, "metrics_per_image.csv")
SEED     = 42
N_BOOT   = 10000
METRICS  = ["NSS", "AUC_Judd", "sAUC"]
BASELINE_NSS = 0.7304   # center-Gaussian mean NSS (pre-computed on 700 images)
# ──────────────


# ==============================================================
# Load data
# ==============================================================

def load_per_image(csv_path):
    """Return {metric: {layer: [values]}} from metrics_per_image.csv."""
    data = {m: {l: [] for l in range(1, 13)} for m in METRICS}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer = int(row["layer"])
            for m in METRICS:
                data[m][layer].append(float(row[m]))
    return data


# ==============================================================
# (A) Statistics
# ==============================================================

def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Return (ci_lo, ci_hi) for the mean via percentile bootstrap."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = rng.choice(arr, size=n, replace=True).mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def wilcoxon_test(x, y, alternative="two-sided"):
    """Wilcoxon signed-rank test. Returns (stat, p, r_effect).
    Effect size: matched-pairs rank-biserial correlation."""
    diff = np.array(x) - np.array(y)
    diff_nz = diff[diff != 0]
    n = len(diff_nz)
    if n < 10:
        return np.nan, np.nan, np.nan
    res = sp_stats.wilcoxon(diff_nz, alternative=alternative)
    stat, p = res.statistic, res.pvalue
    # rank-biserial r = 1 - (2*T) / (n*(n+1)/2)
    total_rank = n * (n + 1) / 2
    r = abs(1 - (2 * stat) / total_rank)
    return float(stat), float(p), float(r)


def holm_correction(pvals):
    """Holm-Bonferroni correction. Returns corrected p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, pvals[idx] * (n - rank))
    # enforce monotonicity
    running_max = 0
    for idx in order:
        running_max = max(running_max, corrected[idx])
        corrected[idx] = running_max
    return corrected


def run_stats(data):
    """Run all statistical analyses. Returns (summary_rows, test_rows)."""

    # ── Summary table (mean / SD / 95% CI per layer per metric) ──
    summary_rows = []
    ci_data = {m: {} for m in METRICS}   # for plotting later
    for layer in range(1, 13):
        row = {"layer": layer}
        for m in METRICS:
            vals = data[m][layer]
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1)
            ci_lo, ci_hi = bootstrap_ci(vals)
            row[f"{m}_mean"] = mean_v
            row[f"{m}_std"] = std_v
            row[f"{m}_ci_lo"] = ci_lo
            row[f"{m}_ci_hi"] = ci_hi
            row[f"{m}_N"] = len(vals)
            ci_data[m][layer] = (ci_lo, ci_hi)
        summary_rows.append(row)

    # ── Tests ──
    test_rows = []

    # 1) Adjacent layer comparison (L vs L+1), NSS primary
    print("\n=== Adjacent layer Wilcoxon (NSS) ===")
    for l in range(1, 12):
        stat, p, r = wilcoxon_test(data["NSS"][l], data["NSS"][l + 1])
        test_rows.append({
            "test": "adjacent",
            "metric": "NSS",
            "comparison": f"L{l} vs L{l+1}",
            "statistic": stat,
            "p": p,
            "p_corrected": np.nan,   # fill below
            "effect_r": r,
            "direction": "L<L+1" if np.mean(data["NSS"][l]) < np.mean(data["NSS"][l+1]) else "L>L+1",
        })
    # Holm correct the adjacent tests
    adj_rows = [r for r in test_rows if r["test"] == "adjacent"]
    adj_p = np.array([r["p"] for r in adj_rows])
    adj_p_corr = holm_correction(adj_p)
    for r, pc in zip(adj_rows, adj_p_corr):
        r["p_corrected"] = pc
        sig = "***" if pc < 0.001 else "**" if pc < 0.01 else "*" if pc < 0.05 else "ns"
        print(f"  {r['comparison']:12s}  p={r['p']:.2e}  p_holm={pc:.2e}  r={r['effect_r']:.3f}  {r['direction']}  {sig}")

    # 2) Peak layer (L10) vs L8, L9, L11, L12
    print("\n=== Peak L10 vs neighbours (NSS, Holm) ===")
    peak_layer = 10
    peak_tests = []
    for other in [8, 9, 11, 12]:
        stat, p, r = wilcoxon_test(data["NSS"][peak_layer], data["NSS"][other])
        row = {
            "test": "peak_vs",
            "metric": "NSS",
            "comparison": f"L{peak_layer} vs L{other}",
            "statistic": stat,
            "p": p,
            "p_corrected": np.nan,
            "effect_r": r,
            "direction": "L10>" if np.mean(data["NSS"][peak_layer]) > np.mean(data["NSS"][other]) else "L10<",
        }
        peak_tests.append(row)
        test_rows.append(row)
    peak_p = np.array([r["p"] for r in peak_tests])
    peak_p_corr = holm_correction(peak_p)
    for r, pc in zip(peak_tests, peak_p_corr):
        r["p_corrected"] = pc
        sig = "***" if pc < 0.001 else "**" if pc < 0.01 else "*" if pc < 0.05 else "ns"
        print(f"  {r['comparison']:12s}  p={r['p']:.2e}  p_holm={pc:.2e}  r={r['effect_r']:.3f}  {r['direction']}  {sig}")

    # 3) L1 below chance
    print("\n=== L1 below chance ===")
    # NSS < 0 (one-sided)
    l1_nss = np.array(data["NSS"][1])
    stat, p, r = wilcoxon_test(l1_nss, np.zeros_like(l1_nss), alternative="less")
    row_l1_nss = {
        "test": "L1_below_zero",
        "metric": "NSS",
        "comparison": "L1 NSS < 0",
        "statistic": stat,
        "p": p,
        "p_corrected": p,  # single test
        "effect_r": r,
        "direction": f"mean={np.mean(l1_nss):.4f}",
    }
    test_rows.append(row_l1_nss)
    print(f"  NSS < 0:  p={p:.2e}  r={r:.3f}  mean={np.mean(l1_nss):.4f}")

    # AUC-Judd < 0.5 (one-sided)
    l1_auc = np.array(data["AUC_Judd"][1])
    stat, p, r = wilcoxon_test(l1_auc, np.full_like(l1_auc, 0.5), alternative="less")
    row_l1_auc = {
        "test": "L1_below_chance",
        "metric": "AUC_Judd",
        "comparison": "L1 AUC < 0.5",
        "statistic": stat,
        "p": p,
        "p_corrected": p,
        "effect_r": r,
        "direction": f"mean={np.mean(l1_auc):.4f}",
    }
    test_rows.append(row_l1_auc)
    print(f"  AUC < 0.5: p={p:.2e}  r={r:.3f}  mean={np.mean(l1_auc):.4f}")

    # 4) Each layer NSS > baseline (center Gaussian = 0.7304)
    print(f"\n=== Each layer NSS > baseline ({BASELINE_NSS}) ===")
    bl_tests = []
    for l in range(1, 13):
        vals = np.array(data["NSS"][l])
        stat, p, r = wilcoxon_test(vals, np.full_like(vals, BASELINE_NSS), alternative="greater")
        row = {
            "test": "vs_baseline",
            "metric": "NSS",
            "comparison": f"L{l} > CG_baseline",
            "statistic": stat,
            "p": p,
            "p_corrected": np.nan,
            "effect_r": r,
            "direction": f"mean={np.mean(vals):.4f}",
        }
        bl_tests.append(row)
        test_rows.append(row)
    bl_p = np.array([r["p"] for r in bl_tests])
    bl_p_corr = holm_correction(bl_p)
    for r, pc in zip(bl_tests, bl_p_corr):
        r["p_corrected"] = pc
        sig = "***" if pc < 0.001 else "**" if pc < 0.01 else "*" if pc < 0.05 else "ns"
        print(f"  {r['comparison']:20s}  p_holm={pc:.2e}  {sig}")

    return summary_rows, test_rows, ci_data


# ==============================================================
# (B) Plots -- exact subset_20_test.png style
# ==============================================================

def plot_3metrics(layers_arr, means, bands_lo, bands_hi, suptitle, out_path,
                  band_label="SD"):
    """
    Replicate the subset_20_test.png style exactly:
      figsize=(15, 5), dpi=150
      1x3 subplots, 'o-' steelblue lw=2 ms=5
      fill_between alpha=0.15 steelblue
      xlabel='Layer', ylabel=metric, title=metric
      xticks=layers, grid(True, alpha=0.3)
      suptitle fontsize=13, tight_layout
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_names = ["NSS", "AUC-Judd", "sAUC"]
    metric_keys = ["NSS", "AUC_Judd", "sAUC"]

    for ax, mname_display, mkey in zip(axes, metric_names, metric_keys):
        m = means[mkey]
        lo = bands_lo[mkey]
        hi = bands_hi[mkey]
        ax.plot(layers_arr, m, "o-", color="steelblue", lw=2, ms=5,
                label="DINO ViT-S/16")
        ax.fill_between(layers_arr, lo, hi, alpha=0.15, color="steelblue")
        ax.set_xlabel("Layer")
        ax.set_ylabel(mname_display)
        ax.set_title(mname_display)
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================
# Main
# ==============================================================

def main():
    data = load_per_image(CSV_IN)
    n_images = len(data["NSS"][1])
    print(f"Loaded {n_images} images x 12 layers from {CSV_IN}")

    # ── (A) Statistics ──
    summary_rows, test_rows, ci_data = run_stats(data)

    # Save stats_summary.csv
    summary_fields = ["layer"]
    for m in METRICS:
        summary_fields += [f"{m}_mean", f"{m}_std", f"{m}_ci_lo", f"{m}_ci_hi", f"{m}_N"]
    summary_path = os.path.join(OUT_DIR, "stats_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                        for k, v in row.items()})
    print(f"\nSaved: {summary_path}")

    # Save stats_tests.csv
    test_fields = ["test", "metric", "comparison", "statistic", "p",
                   "p_corrected", "effect_r", "direction"]
    test_path = os.path.join(OUT_DIR, "stats_tests.csv")
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=test_fields)
        w.writeheader()
        for row in test_rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) and k in ("statistic", "p", "p_corrected", "effect_r") else v)
                        for k, v in row.items()})
    print(f"Saved: {test_path}")

    # Update metrics_by_layer.csv with CI columns
    updated_fields = ["layer"]
    for m in METRICS + ["CC", "SIM"]:
        updated_fields += [f"{m}_mean", f"{m}_std"]
    for m in METRICS:
        updated_fields += [f"{m}_ci_lo", f"{m}_ci_hi"]

    # Read existing metrics_by_layer to get CC/SIM
    existing = {}
    existing_path = os.path.join(OUT_DIR, "metrics_by_layer.csv")
    with open(existing_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing[int(row["layer"])] = row

    updated_rows = []
    for sr in summary_rows:
        l = sr["layer"]
        row = {"layer": l}
        for m in METRICS:
            row[f"{m}_mean"] = f"{sr[f'{m}_mean']:.6f}"
            row[f"{m}_std"] = f"{sr[f'{m}_std']:.6f}"
            row[f"{m}_ci_lo"] = f"{sr[f'{m}_ci_lo']:.6f}"
            row[f"{m}_ci_hi"] = f"{sr[f'{m}_ci_hi']:.6f}"
        for m in ["CC", "SIM"]:
            row[f"{m}_mean"] = existing[l][f"{m}_mean"]
            row[f"{m}_std"] = existing[l][f"{m}_std"]
        updated_rows.append(row)

    with open(existing_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=updated_fields)
        w.writeheader()
        w.writerows(updated_rows)
    print(f"Updated: {existing_path}")

    # ── (B) Plots ──
    layers_arr = np.arange(1, 13)
    means = {}
    sd_lo, sd_hi = {}, {}
    ci_lo_d, ci_hi_d = {}, {}

    for m in METRICS:
        m_means = np.array([np.mean(data[m][l]) for l in range(1, 13)])
        m_stds = np.array([np.std(data[m][l], ddof=1) for l in range(1, 13)])
        means[m] = m_means
        sd_lo[m] = m_means - m_stds
        sd_hi[m] = m_means + m_stds
        ci_lo_d[m] = np.array([ci_data[m][l][0] for l in range(1, 13)])
        ci_hi_d[m] = np.array([ci_data[m][l][1] for l in range(1, 13)])

    # SD version
    plot_3metrics(layers_arr, means, sd_lo, sd_hi,
                  "Full run (700 images)",
                  os.path.join(OUT_DIR, "layerwise_3metrics_full.png"),
                  band_label="SD")

    # CI version
    plot_3metrics(layers_arr, means, ci_lo_d, ci_hi_d,
                  "Full run (700 images)",
                  os.path.join(OUT_DIR, "layerwise_3metrics_full_ci.png"),
                  band_label="95% CI")

    # Print summary table
    print("\n=== metrics_by_layer (mean +/- SD [95% CI]) ===")
    print(f"{'Layer':>5}  {'NSS':>22}  {'AUC_Judd':>22}  {'sAUC':>22}")
    for sr in summary_rows:
        l = sr["layer"]
        parts = []
        for m in METRICS:
            parts.append(f"{sr[f'{m}_mean']:6.3f}+/-{sr[f'{m}_std']:.3f} [{sr[f'{m}_ci_lo']:.3f},{sr[f'{m}_ci_hi']:.3f}]")
        print(f"{l:5d}  {'  '.join(parts)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
