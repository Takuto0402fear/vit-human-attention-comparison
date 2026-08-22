"""
Phase 3: Ridge-regression readout of Phase 2's 8 low-level visual targets
(+ 3 geometry controls, reported separately) from Phase 1's RAW L1..L12
pooled object features, ALL 700 images / 5551 objects.

CLIP is frozen throughout; only StandardScaler+Ridge(alpha=1.0) is fit,
via lib/osie_lowlevel_probe.py::run_ridge_probe (image_id-grouped 5-fold
CV, same folds reused across all 12 layers per target, scaler fit on the
training fold only).

Early/middle/late layer groups (L1-4 / L5-8 / L9-12): for each target,
the GROUP REPRESENTATIVE is the single layer with the highest point-
estimate out-of-fold R2 within that group; group-vs-group differences use
the same paired-by-object, image-level bootstrap as the semantic-attribute
probe (lib.osie_lowlevel_probe.image_level_bootstrap_r2_diff). Adjacent-
layer (L(i+1)-L(i)) and peak-vs-{L1,L4,L8,L12} differences are also
computed; ALL of these p-values (across all targets) form ONE
Benjamini-Hochberg FDR family (lib.osie_text_alignment_probe.benjamini_hochberg_fdr,
reused unmodified -- the correction itself doesn't care whether the
underlying test was AUPRC or R2).

R2 is NOT compared to AUPRC on the same numeric scale anywhere in this
script -- only each metric's OWN peak layer and curve shape are examined
(per task instructions).

Writes only under outputs/osie_lowlevel_probe_all12_full700/. Does not
touch outputs/osie_probe_all12_full700/ or outputs/osie_lowlevel_targets_full700/.

Usage (PowerShell):
    python scripts\\probe_osie_lowlevel_all12_full700.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os
import time

import numpy as np

from lib.osie_lowlevel_probe import (
    DEFAULT_TARGETS, GEOMETRY_CONTROLS, image_level_bootstrap_r2_diff, run_ridge_probe,
)
from lib.osie_text_alignment_probe import benjamini_hochberg_fdr

# ======================= CONFIG =======================
SEED = 42
LAYERS_ALL12 = tuple(range(1, 13))
REFERENCE_LAYERS = [1, 4, 8, 12]
LAYER_GROUPS = {"early": [1, 2, 3, 4], "middle": [5, 6, 7, 8], "late": [9, 10, 11, 12]}
ALL_TARGETS_AND_CONTROLS = list(DEFAULT_TARGETS) + list(GEOMETRY_CONTROLS)

OBJECT_FEATURES_NPZ = r"C:\Users\user\gaze\outputs\osie_probe_all12_full700\object_features_all12.npz"
LOWLEVEL_TARGETS_CSV = r"C:\Users\user\gaze\outputs\osie_lowlevel_targets_full700\lowlevel_object_targets.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_lowlevel_probe_all12_full700"
LAYER_DIFF_CSV = os.path.join(OUT_DIR, "lowlevel_layer_differences.csv")
PEAK_LAYERS_CSV = os.path.join(OUT_DIR, "lowlevel_peak_layers.csv")
CONFIG_JSON = os.path.join(OUT_DIR, "config.json")
# ======================================================


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("  Phase 3: Ridge-regression readout of low-level targets, L1-L12")
    print("=" * 70)

    for path in (OBJECT_FEATURES_NPZ, LOWLEVEL_TARGETS_CSV):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found -- run the prerequisite extraction scripts first")

    print(f"\n--- Ridge probe, {len(ALL_TARGETS_AND_CONTROLS)} targets x 12 layers ---")
    fold_rows, summary_rows, oof_data = run_ridge_probe(
        OBJECT_FEATURES_NPZ, LOWLEVEL_TARGETS_CSV, OUT_DIR, layers=LAYERS_ALL12,
        targets=ALL_TARGETS_AND_CONTROLS, alpha=1.0, seed=SEED, collect_oof=True)
    print(f"  Saved: {os.path.join(OUT_DIR, 'lowlevel_fold_scores.csv')}  ({len(fold_rows)} rows)")
    print(f"  Saved: {os.path.join(OUT_DIR, 'lowlevel_summary.csv')}  ({len(summary_rows)} rows)")
    print(f"  Saved: {os.path.join(OUT_DIR, 'fold_assignments.csv')}")
    print(f"  Saved: {os.path.join(OUT_DIR, 'lowlevel_oof_predictions.csv')}")

    # ------------------------------------------------------------------
    # Peak-layer detection + adjacent/reference/group differences
    # ------------------------------------------------------------------
    print("\n--- Peak-layer detection + layer differences ---")
    r2_point = {}
    for r in summary_rows:
        if r["metric"] != "r2":
            continue
        r2_point.setdefault(r["target"], {})[int(r["layer"])] = float(r["mean"])

    diff_rows = []
    peak_rows = []
    for target in ALL_TARGETS_AND_CONTROLS:
        if target not in r2_point:
            continue
        per_layer = r2_point[target]
        peak_layer = max(per_layer, key=per_layer.get)

        # adjacent pairs
        for l in LAYERS_ALL12[:-1]:
            layer_b, layer_a = l + 1, l
            key_b, key_a = (target, layer_b), (target, layer_a)
            if key_b not in oof_data or key_a not in oof_data:
                continue
            result = image_level_bootstrap_r2_diff(oof_data[key_a], oof_data[key_b], seed=SEED)
            diff_rows.append({
                "target": target, "comparison_type": "adjacent", "comparison": f"L{layer_b}-L{layer_a}",
                "layer_minuend": layer_b, "layer_subtrahend": layer_a,
                "observed_layer_a": result["observed_a"], "observed_layer_b": result["observed_b"],
                "observed_diff": result["observed_diff"], "ci95_lo_diff": result["ci95_lo"],
                "ci95_hi_diff": result["ci95_hi"], "p_raw": result["p_value"], "n_boot_valid": result["n_boot_valid"],
            })

        # peak vs. reference layers
        for l in REFERENCE_LAYERS:
            if l == peak_layer:
                continue
            key_peak, key_l = (target, peak_layer), (target, l)
            if key_peak not in oof_data or key_l not in oof_data:
                continue
            result = image_level_bootstrap_r2_diff(oof_data[key_l], oof_data[key_peak], seed=SEED)
            diff_rows.append({
                "target": target, "comparison_type": "peak_vs_reference", "comparison": f"L{peak_layer}-L{l}",
                "layer_minuend": peak_layer, "layer_subtrahend": l,
                "observed_layer_a": result["observed_a"], "observed_layer_b": result["observed_b"],
                "observed_diff": result["observed_diff"], "ci95_lo_diff": result["ci95_lo"],
                "ci95_hi_diff": result["ci95_hi"], "p_raw": result["p_value"], "n_boot_valid": result["n_boot_valid"],
            })

        # early/middle/late group representatives (best layer within each group)
        group_reps = {}
        for group_name, layers_in_group in LAYER_GROUPS.items():
            available = {l: per_layer[l] for l in layers_in_group if l in per_layer}
            if available:
                group_reps[group_name] = max(available, key=available.get)
        for (g1, g2) in [("early", "middle"), ("middle", "late"), ("early", "late")]:
            if g1 not in group_reps or g2 not in group_reps:
                continue
            l1, l2 = group_reps[g1], group_reps[g2]
            key1, key2 = (target, l1), (target, l2)
            if key1 not in oof_data or key2 not in oof_data:
                continue
            result = image_level_bootstrap_r2_diff(oof_data[key1], oof_data[key2], seed=SEED)
            diff_rows.append({
                "target": target, "comparison_type": "group_representative",
                "comparison": f"{g2}(L{l2})-{g1}(L{l1})", "layer_minuend": l2, "layer_subtrahend": l1,
                "observed_layer_a": result["observed_a"], "observed_layer_b": result["observed_b"],
                "observed_diff": result["observed_diff"], "ci95_lo_diff": result["ci95_lo"],
                "ci95_hi_diff": result["ci95_hi"], "p_raw": result["p_value"], "n_boot_valid": result["n_boot_valid"],
            })

        # near-peak tier (reuse the same tie-tolerance logic conceptually,
        # implemented inline here since peak_layer_analysis is written for
        # dict-shaped diff_results keyed the same way -- build that map)
        tier = [peak_layer]
        for l in LAYERS_ALL12:
            if l == peak_layer:
                continue
            key_peak, key_l = (target, peak_layer), (target, l)
            if key_peak in oof_data and key_l in oof_data:
                result = image_level_bootstrap_r2_diff(oof_data[key_l], oof_data[key_peak], seed=SEED)
                if result["ci95_lo"] <= 0 <= result["ci95_hi"]:
                    tier.append(l)
        tier_sorted = sorted(tier)
        if len(tier_sorted) == 1:
            label = f"L{tier_sorted[0]}"
        elif tier_sorted == list(range(min(tier_sorted), max(tier_sorted) + 1)):
            label = f"L{min(tier_sorted)}-L{max(tier_sorted)}"
        else:
            label = "L" + ",L".join(str(l) for l in tier_sorted)

        peak_rows.append({
            "target": target, "peak_layer": peak_layer, "peak_r2": per_layer[peak_layer],
            "near_peak_label": label, "layers_in_peak_tier": ";".join(str(l) for l in tier_sorted),
            "early_rep_layer": group_reps.get("early", ""), "middle_rep_layer": group_reps.get("middle", ""),
            "late_rep_layer": group_reps.get("late", ""),
        })

    # BH-FDR across the entire family
    p_raws = [r["p_raw"] for r in diff_rows]
    p_fdrs = benjamini_hochberg_fdr(p_raws)
    for r, p_fdr in zip(diff_rows, p_fdrs):
        r["p_fdr_bh"] = float(p_fdr) if not np.isnan(p_fdr) else float("nan")

    diff_fields = ["target", "comparison_type", "comparison", "layer_minuend", "layer_subtrahend",
                   "observed_layer_a", "observed_layer_b", "observed_diff", "ci95_lo_diff", "ci95_hi_diff",
                   "p_raw", "p_fdr_bh", "n_boot_valid"]
    with open(LAYER_DIFF_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=diff_fields)
        w.writeheader()
        for r in diff_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  Saved: {LAYER_DIFF_CSV}  ({len(diff_rows)} rows, BH-FDR family size {len(p_raws)})")

    with open(PEAK_LAYERS_CSV, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["target", "peak_layer", "peak_r2", "near_peak_label", "layers_in_peak_tier",
                      "early_rep_layer", "middle_rep_layer", "late_rep_layer"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in peak_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {PEAK_LAYERS_CSV}  ({len(peak_rows)} rows)")
    for row in peak_rows:
        print(f"    [{row['target']}] peak={row['near_peak_label']} (R2={row['peak_r2']:.4f})")

    config = {
        "seed": SEED, "layers": list(LAYERS_ALL12), "targets": ALL_TARGETS_AND_CONTROLS,
        "alpha": 1.0, "layer_groups": LAYER_GROUPS, "reference_layers": REFERENCE_LAYERS,
        "object_features_source": OBJECT_FEATURES_NPZ, "lowlevel_targets_source": LOWLEVEL_TARGETS_CSV,
        "wall_clock_total_s": time.time() - t_start,
        "note": "R2/Pearson/Spearman/MAE are regression metrics, NEVER compared numerically "
                "against the semantic-attribute probe's AUPRC/AUROC -- only peak-layer position "
                "and curve shape are compared across the two probe families.",
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Saved: {CONFIG_JSON}")

    print(f"\n  Wall clock: {time.time() - t_start:.1f}s")
    print("\n" + "=" * 70)
    print("  Phase 3: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
