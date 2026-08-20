"""
Full700 OSIE attribute-grounding: paired, image-level statistics for
L4 vs L8, L8 vs L12, and L4 vs L12, per attribute (foreground / background /
the 12 attrs.mat attributes).

Direction convention (never ambiguous): each comparison is named
"L{B}-L{A}" and every diff/mean_diff/median_diff/ci is computed as
value(L{B}) - value(L{A}). E.g. "L12-L4": positive means L12's enrichment is
higher than L4's for that attribute.

Statistical methodology mirrors the existing repo convention (read, not
imported -- see scripts/stats_dino_clip_layerwise.py / stats_and_plots_expA.py):
  - wilcoxon_test(): paired signed-rank test + matched-pairs rank-biserial
    effect size r = |1 - 2*W/(n(n+1)/2)|, n<10 -> NaN (not enough non-zero
    diffs for a meaningful signed-rank test).
  - holm_correction(): Holm-Bonferroni step-down, applied ONCE across ALL
    attribute x comparison p-values in this run (14 attributes x 3
    comparisons = 42 tests), not per-attribute or per-comparison in
    isolation.
  - Paired bootstrap 95% CI: resamples IMAGE indices (with replacement);
    the SAME resampled index array is applied to both layers in a
    comparison within one bootstrap iteration, preserving the paired
    (within-image) correspondence. A fresh (N_BOOT, n_valid_images) index
    array is built per attribute (seed=42) since n_valid_images differs by
    attribute (6 to ~500+).

Per attribute, the set of images with a valid (non-missing) mask is
IDENTICAL across L4/L8/L12 (the mask doesn't depend on layer) -- this
script asserts that explicitly and stops if it's ever violated.

Reads (read-only): outputs/osie_attribute_grounding_full700/per_image_long.csv
Requires: scripts/check_pilot_reproduction.py must have PASSed first
(this script checks for pilot_reproduction_check.json with passed=true).
Writes only: outputs/osie_attribute_grounding_full700/paired_layer_tests.csv
and stats_config.json.

Usage (PowerShell):
    python scripts\\stats_osie_attribute_grounding_full700.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os

import numpy as np
from scipy import stats as sp_stats

# ======================= CONFIG =======================
OUT_DIR = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_full700"
LONG_CSV = os.path.join(OUT_DIR, "per_image_long.csv")
REPRO_CHECK_JSON = os.path.join(OUT_DIR, "pilot_reproduction_check.json")
OUT_CSV = os.path.join(OUT_DIR, "paired_layer_tests.csv")
STATS_CONFIG_JSON = os.path.join(OUT_DIR, "stats_config.json")

ATTR_NAMES = [
    "text", "face", "emotion", "sound", "smell", "taste", "touch",
    "motion", "operability", "watchability", "touched", "gazed",
]
LOW_SAMPLE_ATTRS = {"sound", "smell"}
CATEGORIES = ["foreground", "background"] + ATTR_NAMES

LAYERS_DISPLAY = [4, 8, 12]
# (comparison_name, layer_B, layer_A) -- diff := value(layer_B) - value(layer_A)
COMPARISONS = [
    ("L8-L4", 8, 4),
    ("L12-L8", 12, 8),
    ("L12-L4", 12, 4),
]

SEED = 42
N_BOOT = 10000
ALPHA = 0.05
# ======================================================


def wilcoxon_test(x, y):
    """
    x, y: 1D arrays, paired. diff := x - y. Mirrors
    scripts/stats_dino_clip_layerwise.py::wilcoxon_test exactly (same
    formula, same n<10 -> NaN guard, not imported per repo convention).
    """
    diff = np.array(x) - np.array(y)
    diff_nz = diff[diff != 0]
    n = len(diff_nz)
    if n < 10:
        return np.nan, np.nan, np.nan
    res = sp_stats.wilcoxon(diff_nz, alternative="two-sided")
    stat, p = res.statistic, res.pvalue
    total_rank = n * (n + 1) / 2
    r = abs(1 - (2 * stat) / total_rank)
    return float(stat), float(p), float(r)


def holm_correction(pvals):
    """Holm-Bonferroni step-down. NaN p-values pass through as NaN."""
    pvals = np.asarray(pvals, dtype=float)
    valid_mask = ~np.isnan(pvals)
    corrected = np.full_like(pvals, np.nan)
    valid_p = pvals[valid_mask]
    n = len(valid_p)
    if n == 0:
        return corrected
    order = np.argsort(valid_p)
    tmp = np.empty(n)
    for rank, idx in enumerate(order):
        tmp[idx] = min(1.0, valid_p[idx] * (n - rank))
    running_max = 0.0
    for idx in order:
        running_max = max(running_max, tmp[idx])
        tmp[idx] = running_max
    corrected[valid_mask] = tmp
    return corrected


def make_boot_indices(n_images, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_images, size=(n_boot, n_images))


def bootstrap_ci_mean_diff(diff_1d, boot_indices):
    boot_means = diff_1d[boot_indices].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def load_long_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_per_attribute_matrices(rows):
    """
    Returns {attribute: (image_ids_sorted list, {layer: np.array aligned to image_ids})}.
    Asserts the valid-image set is identical across L4/L8/L12 for every attribute.
    """
    by_attr_layer = {}
    for r in rows:
        attr = r["attribute"]
        layer = int(r["layer"])
        stem = os.path.splitext(r["image_name"])[0]
        by_attr_layer.setdefault(attr, {}).setdefault(layer, {})[stem] = float(r["area_normalized_enrichment"])

    result = {}
    for attr in CATEGORIES:
        layer_dicts = by_attr_layer.get(attr, {})
        image_sets = [set(layer_dicts.get(l, {}).keys()) for l in LAYERS_DISPLAY]
        if any(s != image_sets[0] for s in image_sets[1:]):
            raise RuntimeError(
                f"STOP: attribute '{attr}' has a different valid-image set across "
                f"L4/L8/L12 (mask presence must be layer-independent): "
                f"sizes={[len(s) for s in image_sets]}")
        image_ids = sorted(image_sets[0])
        layer_arrays = {
            l: np.array([layer_dicts[l][img] for img in image_ids], dtype=float)
            for l in LAYERS_DISPLAY
        }
        result[attr] = (image_ids, layer_arrays)
    return result


def main():
    print("=" * 65)
    print("  Full700 paired layer statistics: L4 vs L8 vs L12")
    print("=" * 65)

    if not os.path.isfile(REPRO_CHECK_JSON):
        raise RuntimeError(
            f"STOP: {REPRO_CHECK_JSON} not found -- run "
            "scripts/check_pilot_reproduction.py first and confirm it PASSes")
    with open(REPRO_CHECK_JSON, encoding="utf-8") as f:
        repro = json.load(f)
    if not repro.get("passed", False):
        raise RuntimeError(
            "STOP: pilot reproduction check did not PASS -- refusing to compute "
            f"statistics. See {REPRO_CHECK_JSON}")
    print(f"  Pilot reproduction check: PASS (max abs diff: {repro['max_abs_diff']})")

    if not os.path.isfile(LONG_CSV):
        raise RuntimeError(f"STOP: {LONG_CSV} not found")
    rows = load_long_csv(LONG_CSV)
    print(f"  Loaded {len(rows)} rows from {LONG_CSV}")

    print("\n--- Building per-attribute paired matrices ---")
    per_attr = build_per_attribute_matrices(rows)
    for attr in CATEGORIES:
        image_ids, _ = per_attr[attr]
        flag = "  (LOW SAMPLE)" if attr in LOW_SAMPLE_ATTRS else ""
        print(f"  {attr:15s}: {len(image_ids):4d} valid images{flag}")

    print("\n--- Running paired tests (Wilcoxon + bootstrap CI), Holm across all 42 tests ---")
    prelim_rows = []
    raw_pvals = []
    for attr in CATEGORIES:
        image_ids, layer_arrays = per_attr[attr]
        n_valid = len(image_ids)
        boot_idx = make_boot_indices(n_valid, seed=SEED) if n_valid >= 2 else None

        for comp_name, layer_b, layer_a in COMPARISONS:
            arr_b = layer_arrays[layer_b]
            arr_a = layer_arrays[layer_a]
            diff = arr_b - arr_a

            stat, p_raw, eff_r = wilcoxon_test(arr_b, arr_a)
            if boot_idx is not None:
                ci_lo, ci_hi = bootstrap_ci_mean_diff(diff, boot_idx)
            else:
                ci_lo = ci_hi = float("nan")

            row = {
                "attribute": attr,
                "comparison": comp_name,
                "layer_minuend": layer_b,
                "layer_subtrahend": layer_a,
                "valid_image_count": n_valid,
                f"mean_L{layer_b}": float(arr_b.mean()) if n_valid else float("nan"),
                f"median_L{layer_b}": float(np.median(arr_b)) if n_valid else float("nan"),
                f"sd_L{layer_b}": float(np.std(arr_b, ddof=1)) if n_valid > 1 else float("nan"),
                f"mean_L{layer_a}": float(arr_a.mean()) if n_valid else float("nan"),
                f"median_L{layer_a}": float(np.median(arr_a)) if n_valid else float("nan"),
                f"sd_L{layer_a}": float(np.std(arr_a, ddof=1)) if n_valid > 1 else float("nan"),
                "mean_diff": float(diff.mean()) if n_valid else float("nan"),
                "median_diff": float(np.median(diff)) if n_valid else float("nan"),
                "ci95_lo_diff": ci_lo,
                "ci95_hi_diff": ci_hi,
                "wilcoxon_stat": stat,
                "p_raw": p_raw,
                "rank_biserial_r": eff_r,
                "low_sample_flag": int(attr in LOW_SAMPLE_ATTRS),
            }
            prelim_rows.append(row)
            raw_pvals.append(p_raw)

    p_holm = holm_correction(np.array(raw_pvals, dtype=float))
    for row, ph in zip(prelim_rows, p_holm):
        row["p_holm"] = float(ph) if not np.isnan(ph) else float("nan")

    # ------------------------------------------------------------------
    # Normalize columns: every row gets the SAME fixed column set so the
    # CSV is rectangular even though mean_L{layer} column names differ by
    # comparison (mean_L4/mean_L8/mean_L12 as applicable, NaN elsewhere).
    # ------------------------------------------------------------------
    all_layer_cols = []
    for l in LAYERS_DISPLAY:
        all_layer_cols += [f"mean_L{l}", f"median_L{l}", f"sd_L{l}"]
    fieldnames = (["attribute", "comparison", "layer_minuend", "layer_subtrahend",
                    "valid_image_count"] + all_layer_cols +
                  ["mean_diff", "median_diff", "ci95_lo_diff", "ci95_hi_diff",
                    "wilcoxon_stat", "p_raw", "p_holm", "rank_biserial_r", "low_sample_flag"])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in prelim_rows:
            full_row = {k: row.get(k, "") for k in fieldnames}
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in full_row.items()})
    print(f"\n  Saved: {OUT_CSV}  ({len(prelim_rows)} rows = {len(CATEGORIES)} attrs x {len(COMPARISONS)} comparisons)")

    print("\n=== L12-L4 (positive => L12 higher than L4): mean_diff [95% CI]  p_holm  effect_r ===")
    for row in prelim_rows:
        if row["comparison"] != "L12-L4":
            continue
        flag = " (LOW SAMPLE)" if row["low_sample_flag"] else ""
        print(f"  {row['attribute']:15s} n={row['valid_image_count']:4d}  "
              f"diff={row['mean_diff']:+.3f} [{row['ci95_lo_diff']:+.3f},{row['ci95_hi_diff']:+.3f}]  "
              f"p_holm={row['p_holm']:.4f}  r={row['rank_biserial_r']:.3f}{flag}")

    stats_config = {
        "seed": SEED, "n_boot": N_BOOT, "alpha": ALPHA,
        "comparisons": [{"name": n, "layer_minuend": b, "layer_subtrahend": a} for n, b, a in COMPARISONS],
        "direction_convention": "diff := value(layer_minuend) - value(layer_subtrahend); "
                                "e.g. comparison 'L12-L4' -> diff = L12 - L4, positive means L12 > L4",
        "holm_scope": f"Holm-Bonferroni applied ONCE across all {len(prelim_rows)} "
                      f"(attribute x comparison) p-values in this run",
        "wilcoxon_min_nonzero_diffs": 10,
        "low_sample_attrs": sorted(LOW_SAMPLE_ATTRS),
        "n_valid_images_per_attribute": {attr: len(per_attr[attr][0]) for attr in CATEGORIES},
        "source_csv": LONG_CSV,
        "output_csv": OUT_CSV,
    }
    with open(STATS_CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(stats_config, fh, indent=2)
    print(f"  Saved: {STATS_CONFIG_JSON}")

    print("\n" + "=" * 65)
    print("  Full700 paired layer statistics: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
