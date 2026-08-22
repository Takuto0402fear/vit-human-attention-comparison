"""
[B-2] Full700 main analysis, step 2/3 (stats): paired, image-level
statistics on the outputs of scripts/compute_clip_attention_b2_full700.py.

Statistical unit: images (700, or the non-degenerate subset for
foreground/background-dependent metrics). Direction convention: every
comparison "L{B}-L{A}" is diff = value(L{B}) - value(L{A}).

TWO INDEPENDENT BH-FDR FAMILIES (per task spec -- "spatial similarity
tests, if run, form a separate family"):

  Family 1 "layer_scalar_metrics" (27 tests = 9 metrics x 3 comparisons):
    normalized_entropy, hoyer_sparsity, max_patch_attention, top10pct_mass,
    mass50_area_fraction, mass80_area_fraction, normalized_center_distance,
    foreground_conditional_mass, foreground_enrichment
    -- background_conditional_mass is NOT tested here (it is the exact
       complement of foreground_conditional_mass: 1 - fg_conditional_mass,
       so testing it would duplicate the foreground test under a sign
       flip; reported descriptively in summary.json only).
    -- patch_mass_raw / cls_self_mass are reported descriptively only
       (they are about total un-normalized attention budget, not spatial
       distribution, which is out of B-2's scope).

  Family 2 "spatial_similarity_same_vs_shuffled" (9 tests = 3 metrics x
    3 layer pairs): for each layer pair (L4-L8, L8-L12, L4-L12) and each
    similarity metric (pearson, cosine, normalized_jsd), diff_i =
    same_image_sim_i - shuffled_baseline_sim_i (paired per image via the
    SAME fixed derangement used in the compute step -- NOT a test of
    "correlation > 0", but of whether same-image layer-pair similarity
    exceeds the shuffled-image (center-bias) baseline). Pearson diffs use
    a Fisher-z transform before differencing (task spec); cosine/JSD use
    raw differences. This is the single-derangement version; a
    1000-derangement robustness check is a separate script/commit, not
    performed here.

Per test: mean_diff (raw scale), median_diff, 95% percentile bootstrap CI
(image-level resample, 10000 reps, seed=42 by default), Cohen's dz,
two-sided sign-flip permutation p-value (10000 reps, seed=42 by default,
+1 Monte-Carlo correction), then BH-FDR q-value within its family.

All paths are resolved relative to the repository root or overridable
via CLI flags -- no machine-specific absolute path is hardcoded. Writes
only under --output-dir (refuses to overwrite only the specific files
below; no statistical figures are produced by this script):
  paired_layer_comparisons.csv, summary.json

Usage (PowerShell):
    python scripts\\stats_clip_attention_b2_full700.py
    python scripts\\stats_clip_attention_b2_full700.py --input-dir D:\\tmp\\b2_smoke\\full700 --output-dir D:\\tmp\\b2_smoke\\full700
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import argparse
import csv
import json

import numpy as np

from lib.clip_attention_b2 import (
    bh_fdr, bootstrap_ci_mean_diff, cohens_dz, fisher_z, fisher_z_inverse,
    refuse_if_exists, sign_flip_test,
)

SEED_DEFAULT = 42
N_BOOT_DEFAULT = 10000
N_SIGNFLIP_DEFAULT = 10000
ALPHA = 0.05

DEFAULT_DATA_DIR = REPO_ROOT / "outputs" / "clip_attention_b2_distribution"

LAYERS_DISPLAY = [4, 8, 12]
COMPARISONS = [("L8-L4", 8, 4), ("L12-L8", 12, 8), ("L12-L4", 12, 4)]
LAYER_PAIRS = [(4, 8), (8, 12), (4, 12)]

# metric_name -> requires_nondegenerate_subset
FAMILY1_METRICS = {
    "normalized_entropy": False,
    "hoyer_sparsity": False,
    "max_patch_attention": False,
    "top10pct_mass": False,
    "mass50_area_fraction": False,
    "mass80_area_fraction": False,
    "normalized_center_distance": False,
    "foreground_conditional_mass": True,
    "foreground_enrichment": True,
}
DESCRIPTIVE_ONLY_METRICS = [
    "patch_mass_raw", "cls_self_mass", "background_conditional_mass",
    "foreground_area_fraction", "background_area_fraction",
    "mass50_n_patches", "mass80_n_patches", "centroid_x", "centroid_y",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="[B-2] Full700 stats: paired bootstrap + sign-flip + BH-FDR over "
                    "scripts/compute_clip_attention_b2_full700.py's outputs (single "
                    "seed=42 derangement Family 2; the 1000-derangement robustness "
                    "check is a separate script).")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Directory containing per_image_metrics.csv and "
                        f"layer_pair_similarity.csv, read-only (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Directory to write paired_layer_comparisons.csv and "
                        "summary.json into (created if missing; refuses to overwrite "
                        f"either file if it already exists) (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help=f"Seed for the bootstrap CI and sign-flip test (default: {SEED_DEFAULT}).")
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT,
                    help=f"Number of bootstrap resamples per test (default: {N_BOOT_DEFAULT}).")
    p.add_argument("--n-signflip", type=int, default=N_SIGNFLIP_DEFAULT,
                    help=f"Number of sign-flip permutations per test (default: {N_SIGNFLIP_DEFAULT}).")
    return p.parse_args(argv)


def read_csv_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float_or_none(s):
    if s in ("", "None", None):
        return None
    return float(s)


def build_layer_matrix(rows, metric, require_nondegenerate):
    """Returns (image_ids sorted list, {layer: np.array aligned to image_ids})."""
    by_layer = {l: {} for l in LAYERS_DISPLAY}
    for r in rows:
        layer = int(r["layer"])
        val = to_float_or_none(r[metric])
        if require_nondegenerate and (r["is_degenerate_fgbg"] == "True" or val is None):
            continue
        if val is None:
            continue
        by_layer[layer][r["image_id"]] = val

    image_sets = [set(by_layer[l].keys()) for l in LAYERS_DISPLAY]
    common = set.intersection(*image_sets)
    if any(len(s) != len(common) for s in image_sets):
        dropped = [f"L{l}:{len(image_sets[i])}" for i, l in enumerate(LAYERS_DISPLAY)]
        print(f"    WARNING: {metric} valid-image sets differ across layers "
              f"({dropped}); using intersection n={len(common)}")
    image_ids = sorted(common)
    arrays = {l: np.array([by_layer[l][img] for img in image_ids]) for l in LAYERS_DISPLAY}
    return image_ids, arrays


def run_paired_test(diff, seed, n_boot, n_signflip):
    mean_diff = float(diff.mean())
    median_diff = float(np.median(diff))
    ci_lo, ci_hi = bootstrap_ci_mean_diff(diff, seed=seed, n_boot=n_boot, alpha=ALPHA)
    dz = cohens_dz(diff)
    obs, p = sign_flip_test(diff, seed=seed, n_reps=n_signflip)
    return {
        "n_images": int(diff.size), "mean_diff": mean_diff, "median_diff": median_diff,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "cohens_dz": dz,
        "sign_flip_observed_mean": obs, "sign_flip_p": p,
    }


def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    seed = args.seed
    n_boot = args.n_boot
    n_signflip = args.n_signflip

    metrics_csv = input_dir / "per_image_metrics.csv"
    similarity_csv = input_dir / "layer_pair_similarity.csv"
    paired_csv = output_dir / "paired_layer_comparisons.csv"
    summary_json = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_if_exists([str(paired_csv), str(summary_json)])

    print("=" * 70)
    print("  [B-2] Full700 stats: paired bootstrap + sign-flip + BH-FDR")
    print("=" * 70)
    print(f"  input_dir  = {input_dir}")
    print(f"  output_dir = {output_dir}")
    print(f"  seed={seed}  n_boot={n_boot}  n_signflip={n_signflip}")

    if not (metrics_csv.is_file() and similarity_csv.is_file()):
        raise RuntimeError(
            f"STOP: {metrics_csv} and/or {similarity_csv} not found -- run "
            f"scripts/compute_clip_attention_b2_full700.py first")

    metric_rows = read_csv_rows(metrics_csv)
    sim_rows = read_csv_rows(similarity_csv)

    # ------------------------------------------------------------------
    # Family 1: layer-scalar metrics x 3 comparisons
    # ------------------------------------------------------------------
    print("\n--- Family 1: layer_scalar_metrics ---")
    family1_records = []
    for metric, require_nondeg in FAMILY1_METRICS.items():
        image_ids, arrays = build_layer_matrix(metric_rows, metric, require_nondeg)
        for comp_name, lb, la in COMPARISONS:
            diff = arrays[lb] - arrays[la]
            result = run_paired_test(diff, seed, n_boot, n_signflip)
            family1_records.append({
                "family": "layer_scalar_metrics", "metric": metric, "comparison": comp_name,
                **result,
            })
            print(f"    {metric:28s} {comp_name:8s} n={result['n_images']:4d} "
                  f"mean_diff={result['mean_diff']:+.5f} dz={result['cohens_dz']:+.3f} "
                  f"p={result['sign_flip_p']:.5f}")

    pvals1 = np.array([r["sign_flip_p"] for r in family1_records])
    q1 = bh_fdr(pvals1)
    for r, q in zip(family1_records, q1):
        r["bh_fdr_q"] = float(q)

    # ------------------------------------------------------------------
    # Family 2: spatial similarity, same-image vs. shuffled baseline
    # (single seed=42 derangement -- see verify_clip_attention_b2_shuffle_robustness.py
    # for the 1000-derangement version)
    # ------------------------------------------------------------------
    print("\n--- Family 2: spatial_similarity_same_vs_shuffled (single derangement) ---")
    by_pair_type = {}
    for r in sim_rows:
        key = (int(r["layer_a"]), int(r["layer_b"]), r["pair_type"])
        by_pair_type.setdefault(key, {})[r["image_id"]] = r

    family2_records = []
    similarity_descriptive = {}
    for la, lb in LAYER_PAIRS:
        same = by_pair_type[(la, lb, "same_image")]
        shuf = by_pair_type[(la, lb, "shuffled_baseline")]
        common_ids = sorted(set(same.keys()) & set(shuf.keys()))
        if len(common_ids) != 700:
            raise RuntimeError(f"STOP: L{la}-L{lb} similarity n={len(common_ids)} != 700")

        for metric in ("pearson", "cosine", "normalized_jsd"):
            same_vals = np.array([float(same[i][metric]) for i in common_ids])
            shuf_vals = np.array([float(shuf[i][metric]) for i in common_ids])

            if metric == "pearson":
                same_z = np.array([fisher_z(v) for v in same_vals])
                shuf_z = np.array([fisher_z(v) for v in shuf_vals])
                diff = same_z - shuf_z
                same_mean_report = fisher_z_inverse(float(same_z.mean()))
                shuf_mean_report = fisher_z_inverse(float(shuf_z.mean()))
                scale_note = "fisher_z_diff"
            else:
                diff = same_vals - shuf_vals
                same_mean_report = float(same_vals.mean())
                shuf_mean_report = float(shuf_vals.mean())
                scale_note = "raw_diff"

            result = run_paired_test(diff, seed, n_boot, n_signflip)
            family2_records.append({
                "family": "spatial_similarity_same_vs_shuffled",
                "metric": metric, "comparison": f"L{la}-L{lb}_same_minus_shuffled",
                "scale": scale_note, **result,
            })
            similarity_descriptive[f"L{la}-L{lb}_{metric}"] = {
                "same_image_mean": same_mean_report,
                "shuffled_baseline_mean": shuf_mean_report,
                "same_image_median": float(np.median(same_vals)),
                "shuffled_baseline_median": float(np.median(shuf_vals)),
            }
            print(f"    L{la}-L{lb} {metric:14s} same={same_mean_report:+.4f} "
                  f"shuffled={shuf_mean_report:+.4f} diff_dz={result['cohens_dz']:+.3f} "
                  f"p={result['sign_flip_p']:.5f}")

    pvals2 = np.array([r["sign_flip_p"] for r in family2_records])
    q2 = bh_fdr(pvals2)
    for r, q in zip(family2_records, q2):
        r["bh_fdr_q"] = float(q)

    # ------------------------------------------------------------------
    # Write paired_layer_comparisons.csv
    # ------------------------------------------------------------------
    all_records = family1_records + family2_records
    fieldnames = [
        "family", "metric", "comparison", "scale", "n_images", "mean_diff", "median_diff",
        "ci_lo", "ci_hi", "cohens_dz", "sign_flip_observed_mean", "sign_flip_p", "bh_fdr_q",
    ]
    for r in all_records:
        r.setdefault("scale", "raw_diff")
    with open(paired_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_records)
    print(f"\n  Saved: {paired_csv}  ({len(all_records)} rows: "
          f"{len(family1_records)} family1 + {len(family2_records)} family2)")

    # ------------------------------------------------------------------
    # Descriptive per-layer summary (mean/median/95% bootstrap CI) for
    # every metric (family1 + descriptive-only), plus similarity summary.
    # ------------------------------------------------------------------
    print("\n--- Descriptive per-layer summaries ---")
    descriptive = {}
    all_metric_names = list(FAMILY1_METRICS.keys()) + DESCRIPTIVE_ONLY_METRICS
    for metric in all_metric_names:
        require_nondeg = FAMILY1_METRICS.get(metric, False) or metric in (
            "foreground_area_fraction", "background_area_fraction")
        image_ids, arrays = build_layer_matrix(metric_rows, metric, require_nondeg)
        descriptive[metric] = {}
        for l in LAYERS_DISPLAY:
            vals = arrays[l]
            ci_lo, ci_hi = bootstrap_ci_mean_diff(vals, seed=seed, n_boot=n_boot, alpha=ALPHA)
            descriptive[metric][f"L{l}"] = {
                "n": int(vals.size), "mean": float(vals.mean()), "median": float(np.median(vals)),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
            }

    n_degenerate_images = len({
        r["image_id"] for r in metric_rows if r["is_degenerate_fgbg"] == "True"})
    n_success_images = len({r["image_id"] for r in metric_rows})

    summary = {
        "n_images_total": n_success_images,
        "n_degenerate_images_excluded_from_fgbg_metrics": n_degenerate_images,
        "seed": seed, "n_boot": n_boot, "n_signflip": n_signflip, "alpha": ALPHA,
        "paths_used": {"input_dir": str(input_dir), "output_dir": str(output_dir)},
        "descriptive_per_layer": descriptive,
        "spatial_similarity_descriptive": similarity_descriptive,
        "family1_n_tests": len(family1_records),
        "family2_n_tests": len(family2_records),
    }
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {summary_json}")

    print("\n" + "=" * 70)
    print("  [B-2] Full700 stats DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
