"""
Phase 1 (all-12-layer extension): linear probe on RAW L1..L12 pooled
object features, ALL 700 OSIE images, using the EXACT SAME
lib/osie_text_alignment_probe.py::run_linear_probe function as the
L4/L8/L12 probes (scripts/probe_osie_text_alignment.py /
scripts/probe_osie_full700.py) -- same StratifiedGroupKFold-by-image_id
logic, same StandardScaler+LogisticRegression pipeline, same permutation
methodology.

Two-stage permutation-baseline strategy (per task instructions: run >=20
reps for every attribute x layer; escalate to 100 reps only where compute
allows):
  Stage A: n_permutation_reps=20 for ALL 12 layers x 13 (attribute,
           condition) pairs -- the full, unabridged analysis.
  Stage B: n_permutation_reps=100 for ONLY the "important" layer set per
           (attribute, condition) -- its own peak layer (from Stage A's
           point estimates) union {L1, L4, L8, L12} -- and merges those
           100-rep rows into the final probe_permutation_baseline.csv in
           place of the corresponding 20-rep rows. Every row's
           `n_repetitions` column records which count was actually used.

Additionally computes (using Stage A's out-of-fold predictions):
  - adjacent-layer differences L(i+1)-L(i) for i=1..11, per (attribute,condition)
  - peak-layer vs. {L1,L4,L8,L12} differences
  - Benjamini-Hochberg FDR correction across ALL of the above p-values
    (one family, computed once)
  - peak-layer detection with "near-peak" tie handling (a layer is in the
    peak's tier if its bootstrap CI vs. the peak includes 0)

Writes only under outputs/osie_probe_all12_full700/. Does not touch
outputs/osie_probe_full700/ or any other existing output.

Usage (PowerShell):
    python scripts\\probe_osie_all12_full700.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import itertools
import json
import os
import time

import numpy as np

from lib.osie_text_alignment_probe import (
    benjamini_hochberg_fdr, image_level_bootstrap_auprc_diff, peak_layer_analysis,
    run_linear_probe, save_fold_assignments_csv, save_oof_predictions_csv,
)

# ======================= CONFIG =======================
SEED = 42
LAYERS_ALL12 = tuple(range(1, 13))
REFERENCE_LAYERS = [1, 4, 8, 12]
N_PERM_REPS_STAGE_A = 20
N_PERM_REPS_STAGE_B = 100

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_probe_all12_full700"
OBJECT_FEATURES_NPZ = os.path.join(OUT_DIR, "object_features_all12.npz")
OBJECTS_METADATA_CSV = os.path.join(OUT_DIR, "object_features_metadata.csv")

FOLD_ASSIGNMENTS_CSV = os.path.join(OUT_DIR, "fold_assignments.csv")
OOF_PREDICTIONS_CSV = os.path.join(OUT_DIR, "probe_oof_predictions.csv")
LAYER_DIFF_CSV = os.path.join(OUT_DIR, "probe_layer_differences.csv")
PEAK_LAYERS_CSV = os.path.join(OUT_DIR, "probe_peak_layers.csv")
PERMUTATION_CSV = os.path.join(OUT_DIR, "probe_permutation_baseline.csv")
HYPOTHESIS_JSON = os.path.join(OUT_DIR, "probe_hypothesis_summary.json")
PERM_STAGE_LOG_JSON = os.path.join(OUT_DIR, "permutation_stage_log.json")
# ======================================================


def rename_npz_keys_for_run_linear_probe(object_features_npz, layers):
    """run_linear_probe expects raw_L{l} keys -- object_features_all12.npz
    already uses exactly that naming (raw_L1..raw_L12), so no renaming is
    actually needed; this function exists purely to fail loudly and early
    if that assumption is ever violated."""
    npz = np.load(object_features_npz, allow_pickle=False)
    for l in layers:
        if f"raw_L{l}" not in npz.files:
            raise RuntimeError(f"STOP: {object_features_npz} missing key raw_L{l}")


def main():
    t_start = time.time()
    print("=" * 70)
    print("  Phase 1 (all-12-layer): linear probe, ALL 700 images")
    print("=" * 70)

    for path in (OBJECT_FEATURES_NPZ, OBJECTS_METADATA_CSV):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found -- run the extraction script first")
    rename_npz_keys_for_run_linear_probe(OBJECT_FEATURES_NPZ, LAYERS_ALL12)

    # ------------------------------------------------------------------
    # Stage A: 20 permutation reps, all 12 layers.
    # ------------------------------------------------------------------
    print(f"\n--- Stage A: real probe + {N_PERM_REPS_STAGE_A}-rep permutation, L1-L12 ---")
    t_a0 = time.time()
    fold_rows, summary_rows, perm_rows_a, insufficient, oof_data = run_linear_probe(
        OBJECT_FEATURES_NPZ, OBJECTS_METADATA_CSV, OUT_DIR, layers=LAYERS_ALL12,
        n_permutation_reps=N_PERM_REPS_STAGE_A, seed=SEED, collect_oof=True,
        fold_scores_filename="probe_fold_scores.csv", summary_filename="probe_summary.csv",
        permutation_filename="_stage_a_permutation.csv")
    t_a = time.time() - t_a0
    print(f"  Stage A wall clock: {t_a:.1f}s")
    print(f"  Saved: {os.path.join(OUT_DIR, 'probe_fold_scores.csv')}  ({len(fold_rows)} rows)")
    print(f"  Saved: {os.path.join(OUT_DIR, 'probe_summary.csv')}  "
          f"({len(summary_rows) + len(insufficient)} rows, {len(insufficient)} insufficient_data)")

    n_saved = save_fold_assignments_csv(oof_data, FOLD_ASSIGNMENTS_CSV)
    print(f"  Saved: {FOLD_ASSIGNMENTS_CSV}  ({n_saved} rows)")
    n_oof = save_oof_predictions_csv(oof_data, OOF_PREDICTIONS_CSV)
    print(f"  Saved: {OOF_PREDICTIONS_CSV}  ({n_oof} rows)")

    # ------------------------------------------------------------------
    # Peak-layer + adjacent/reference layer differences (from Stage A OOF)
    # ------------------------------------------------------------------
    print("\n--- Peak-layer detection + adjacent/reference layer differences ---")
    conditions_seen = sorted({k[:2] for k in oof_data.keys()})
    auprc_point = {}  # (attr, cond) -> {layer: auprc}
    for r in summary_rows:
        if r.get("metric") != "auprc" or r["mean"] == "":
            continue
        auprc_point.setdefault((r["attribute"], r["condition"]), {})[int(r["layer"])] = float(r["mean"])

    all_diff_results = {}  # (attr, cond, layer_b, layer_a) -> result dict
    diff_rows = []
    for attr, cond in conditions_seen:
        if (attr, cond) not in auprc_point:
            continue
        # adjacent pairs L(i+1)-L(i)
        adjacent_pairs = [(l + 1, l) for l in LAYERS_ALL12[:-1]]
        for layer_b, layer_a in adjacent_pairs:
            key_b, key_a = (attr, cond, layer_b), (attr, cond, layer_a)
            if key_b not in oof_data or key_a not in oof_data:
                continue
            result = image_level_bootstrap_auprc_diff(oof_data[key_a], oof_data[key_b], seed=SEED)
            all_diff_results[(attr, cond, layer_b, layer_a)] = result
            diff_rows.append({
                "attribute": attr, "condition": cond, "comparison_type": "adjacent",
                "comparison": f"L{layer_b}-L{layer_a}", "layer_minuend": layer_b, "layer_subtrahend": layer_a,
                **{f"observed_{k}": v for k, v in [("layer_a", result["observed_a"]), ("layer_b", result["observed_b"]),
                                                    ("diff", result["observed_diff"])]},
                "ci95_lo_diff": result["ci95_lo"], "ci95_hi_diff": result["ci95_hi"],
                "p_raw": result["p_value"], "n_boot_valid": result["n_boot_valid"],
            })

        # peak vs. reference layers -- peak determined AFTER adjacent pairs
        # exist for the tier logic, but we also need peak-vs-every-layer
        # for the near-peak tier itself.
        peak_layer = max(auprc_point[(attr, cond)], key=auprc_point[(attr, cond)].get)
        for l in LAYERS_ALL12:
            if l == peak_layer:
                continue
            key_peak, key_l = (attr, cond, peak_layer), (attr, cond, l)
            if key_peak not in oof_data or key_l not in oof_data:
                continue
            if (attr, cond, peak_layer, l) in all_diff_results:
                continue  # already computed as an adjacent pair
            result = image_level_bootstrap_auprc_diff(oof_data[key_l], oof_data[key_peak], seed=SEED)
            all_diff_results[(attr, cond, peak_layer, l)] = result
            if l in REFERENCE_LAYERS:
                diff_rows.append({
                    "attribute": attr, "condition": cond, "comparison_type": "peak_vs_reference",
                    "comparison": f"L{peak_layer}-L{l}", "layer_minuend": peak_layer, "layer_subtrahend": l,
                    "observed_layer_a": result["observed_a"], "observed_layer_b": result["observed_b"],
                    "observed_diff": result["observed_diff"],
                    "ci95_lo_diff": result["ci95_lo"], "ci95_hi_diff": result["ci95_hi"],
                    "p_raw": result["p_value"], "n_boot_valid": result["n_boot_valid"],
                })

    # BH-FDR across the ENTIRE family of p-values computed above (one family).
    p_raws = [r["p_raw"] for r in diff_rows]
    p_fdrs = benjamini_hochberg_fdr(p_raws)
    for r, p_fdr in zip(diff_rows, p_fdrs):
        r["p_fdr_bh"] = float(p_fdr) if not np.isnan(p_fdr) else float("nan")

    diff_fields = ["attribute", "condition", "comparison_type", "comparison", "layer_minuend", "layer_subtrahend",
                   "observed_layer_a", "observed_layer_b", "observed_diff", "ci95_lo_diff", "ci95_hi_diff",
                   "p_raw", "p_fdr_bh", "n_boot_valid"]
    with open(LAYER_DIFF_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=diff_fields)
        w.writeheader()
        for r in diff_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  Saved: {LAYER_DIFF_CSV}  ({len(diff_rows)} rows, BH-FDR family size {len(p_raws)})")

    # ------------------------------------------------------------------
    # Peak-layer summary (with near-peak tie tolerance)
    # ------------------------------------------------------------------
    peak_rows = []
    important_layers_by_condition = {}
    for attr, cond in conditions_seen:
        if (attr, cond) not in auprc_point:
            continue
        result = peak_layer_analysis(auprc_point[(attr, cond)], all_diff_results, LAYERS_ALL12)
        peak_rows.append({
            "attribute": attr, "condition": cond, "peak_layer": result["peak_layer"],
            "peak_auprc": result["peak_auprc"], "near_peak_label": result["near_peak_label"],
            "layers_in_peak_tier": ";".join(str(l) for l in result["layers_not_significantly_below_peak"]),
        })
        important = set(result["layers_not_significantly_below_peak"]) | {result["peak_layer"]} | set(REFERENCE_LAYERS)
        important_layers_by_condition[(attr, cond)] = important

    with open(PEAK_LAYERS_CSV, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["attribute", "condition", "peak_layer", "peak_auprc", "near_peak_label", "layers_in_peak_tier"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in peak_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {PEAK_LAYERS_CSV}  ({len(peak_rows)} rows)")
    for row in peak_rows:
        print(f"    [{row['attribute']}/{row['condition']}] peak={row['near_peak_label']} "
              f"(AUPRC={row['peak_auprc']:.3f})")

    # ------------------------------------------------------------------
    # Stage B: escalate permutation reps to 100 for peak + reference layers.
    # ------------------------------------------------------------------
    print(f"\n--- Stage B: escalating to {N_PERM_REPS_STAGE_B}-rep permutation for peak+reference layers ---")
    t_b0 = time.time()
    perm_by_key_a = {(r["attribute"], r["condition"], r["layer"], r["metric"]): r for r in perm_rows_a}
    stage_b_rows_by_key = {}
    stage_b_layer_sets = {}
    for (attr, cond), important in important_layers_by_condition.items():
        layers_needed = tuple(sorted(important))
        stage_b_layer_sets[(attr, cond)] = layers_needed
        print(f"  [{attr}/{cond}] escalating layers {layers_needed} to {N_PERM_REPS_STAGE_B} reps")
        tmp_dir = os.path.join(OUT_DIR, "_stage_b_tmp")
        # build_conditions() only creates the "conditional_on_face" entry
        # when BOTH "Face" and "Emotion" are present in `attributes` -- so
        # for that condition we must pass both (the resulting extra
        # ("Face","all_objects")/("Emotion","all_objects") fits are simply
        # discarded by the filter below).
        probe_attrs = ("Face", "Emotion") if cond == "conditional_on_face" else (attr,)
        _, _, perm_rows_b, _, _ = run_linear_probe(
            OBJECT_FEATURES_NPZ, OBJECTS_METADATA_CSV, tmp_dir, layers=layers_needed,
            attributes=probe_attrs, n_permutation_reps=N_PERM_REPS_STAGE_B,
            seed=SEED, collect_oof=False, verbose=False,
            fold_scores_filename=f"_tmp_fold_{attr}_{cond}.csv",
            summary_filename=f"_tmp_summary_{attr}_{cond}.csv",
            permutation_filename=f"_tmp_perm_{attr}_{cond}.csv")
        for r in perm_rows_b:
            if r["attribute"] == attr and r["condition"] == cond:
                stage_b_rows_by_key[(r["attribute"], r["condition"], r["layer"], r["metric"])] = r
    t_b = time.time() - t_b0
    print(f"  Stage B wall clock: {t_b:.1f}s")

    # Merge: Stage B rows replace Stage A rows where present, else keep Stage A.
    merged_perm_rows = []
    for key, row_a in perm_by_key_a.items():
        merged_perm_rows.append(stage_b_rows_by_key.get(key, row_a))

    perm_fields = ["attribute", "condition", "layer", "metric", "n_repetitions", "n_values",
                   "mean", "std", "random_auprc"]
    with open(PERMUTATION_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=perm_fields)
        w.writeheader()
        for r in merged_perm_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  Saved: {PERMUTATION_CSV}  ({len(merged_perm_rows)} rows, "
          f"{len(stage_b_rows_by_key)} escalated to {N_PERM_REPS_STAGE_B} reps)")

    with open(PERM_STAGE_LOG_JSON, "w", encoding="utf-8") as fh:
        json.dump({
            "stage_a_reps": N_PERM_REPS_STAGE_A, "stage_a_wall_clock_s": t_a,
            "stage_a_layers": "all 12",
            "stage_b_reps": N_PERM_REPS_STAGE_B, "stage_b_wall_clock_s": t_b,
            "stage_b_layer_sets": {f"{a}/{c}": list(l) for (a, c), l in stage_b_layer_sets.items()},
            "n_rows_escalated_to_stage_b": len(stage_b_rows_by_key),
            "n_rows_total": len(merged_perm_rows),
        }, fh, indent=2)
    print(f"  Saved: {PERM_STAGE_LOG_JSON}")

    # ------------------------------------------------------------------
    # Hypothesis summary (H2-relevant, per-attribute; full H1-H4 synthesis
    # happens in outputs/osie_layer_profile_synthesis/)
    # ------------------------------------------------------------------
    monotonic_increase = {}
    for attr, cond in conditions_seen:
        if (attr, cond) not in auprc_point:
            continue
        vals = [auprc_point[(attr, cond)][l] for l in LAYERS_ALL12]
        diffs_sign = np.sign(np.diff(vals))
        monotonic_increase[f"{attr}/{cond}"] = bool(np.all(diffs_sign >= 0))

    hypothesis_summary = {
        "n_attributes": len(set(a for a, c in conditions_seen)),
        "peak_layers": {f"{r['attribute']}/{r['condition']}": r["near_peak_label"] for r in peak_rows},
        "monotonic_nondecreasing_L1_to_L12": monotonic_increase,
        "n_monotonic": sum(monotonic_increase.values()),
        "n_total": len(monotonic_increase),
        "note": "This file summarizes H2 (semantic-attribute readability by layer) only. "
                "The full H1-H4 hypothesis synthesis (including low-level features and "
                "attention/gaze curves) is in outputs/osie_layer_profile_synthesis/hypothesis_decision.json.",
        "permutation_reps_used": {"stage_a_all_layers": N_PERM_REPS_STAGE_A,
                                    "stage_b_peak_and_reference_layers": N_PERM_REPS_STAGE_B},
        "wall_clock_stage_a_s": t_a, "wall_clock_stage_b_s": t_b,
    }
    with open(HYPOTHESIS_JSON, "w", encoding="utf-8") as fh:
        json.dump(hypothesis_summary, fh, indent=2)
    print(f"\n  Saved: {HYPOTHESIS_JSON}")

    print(f"\n  Total wall clock: {time.time() - t_start:.1f}s")
    print("\n" + "=" * 70)
    print("  Phase 1 (all-12-layer) probe: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
