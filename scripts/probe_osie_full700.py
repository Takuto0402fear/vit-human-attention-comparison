"""
Phase 4: linear probe (Implementation B) extended to ALL 700 OSIE images.

Calls the EXACT SAME lib/osie_text_alignment_probe.py::run_linear_probe
function as scripts/probe_osie_text_alignment.py (the 50-image pilot) --
same StratifiedGroupKFold-by-image_id logic, same StandardScaler+
LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
random_state=42) pipeline, same permutation-baseline methodology. Only the
object-feature cache (RAW L4/L8/L12, all 700 images) and output directory
differ. This equivalence is asserted by
tests/test_osie_probe_full700.py::test_50_and_700_use_identical_implementation.

Text embeddings, prompt-K stability, and patch similarity maps are NOT
extended here (out of scope per task instructions) -- only the raw pooled
hidden-state features feed this probe.

Additionally computes probe_layer_differences.csv: paired (by held-out
object, via pooled out-of-fold predictions) image-level-bootstrap AUPRC/
AUROC/Balanced-Accuracy differences for L8-L4, L12-L8, L12-L4, per
attribute/condition -- an extension not present in the 50-image pilot.

Writes only under outputs/osie_probe_full700/. Does not modify
outputs/osie_text_alignment_pilot/ or any other existing output.

Usage (PowerShell):
    python scripts\\probe_osie_full700.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os
import time

from lib.osie_text_alignment_probe import image_level_bootstrap_auprc_diff, run_linear_probe

# ======================= CONFIG =======================
SEED = 42
N_PERMUTATION_REPS = 100  # full-scale (spec allows 100 "if compute time permits")
LAYERS_DISPLAY = [4, 8, 12]
LAYER_COMPARISONS = [("L8-L4", 8, 4), ("L12-L8", 12, 8), ("L12-L4", 12, 4)]

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_probe_full700"
OBJECT_FEATURES_NPZ = os.path.join(OUT_DIR, "object_features.npz")
OBJECTS_METADATA_CSV = os.path.join(OUT_DIR, "objects_metadata.csv")
LAYER_DIFF_CSV = os.path.join(OUT_DIR, "probe_layer_differences.csv")
GO_NO_GO_JSON = os.path.join(OUT_DIR, "probe_go_no_go.json")
# ======================================================


def main():
    t_start = time.time()
    print("=" * 70)
    print("  Phase 4: linear probe on ALL 700 images (Implementation B)")
    print("=" * 70)

    for path in (OBJECT_FEATURES_NPZ, OBJECTS_METADATA_CSV):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found -- run the extraction script first")

    fold_rows, summary_rows, perm_rows, insufficient, oof_data = run_linear_probe(
        OBJECT_FEATURES_NPZ, OBJECTS_METADATA_CSV, OUT_DIR,
        layers=LAYERS_DISPLAY, n_permutation_reps=N_PERMUTATION_REPS, seed=SEED, collect_oof=True)

    print(f"\n  Saved: {os.path.join(OUT_DIR, 'probe_fold_scores.csv')}  ({len(fold_rows)} rows)")
    print(f"  Saved: {os.path.join(OUT_DIR, 'probe_summary.csv')}  "
          f"({len(summary_rows) + len(insufficient)} rows, {len(insufficient)} insufficient_data)")
    print(f"  Saved: {os.path.join(OUT_DIR, 'probe_permutation_baseline.csv')}  ({len(perm_rows)} rows)")

    print("\n--- Layer differences (paired-by-object, image-level bootstrap) ---")
    diff_rows = []
    conditions_seen = sorted({k[:2] for k in oof_data.keys()})
    for attr, condition in conditions_seen:
        for comp_name, layer_b, layer_a in LAYER_COMPARISONS:
            key_b, key_a = (attr, condition, layer_b), (attr, condition, layer_a)
            if key_b not in oof_data or key_a not in oof_data:
                continue
            for metric in ("auprc",):  # AUPRC is the primary metric for the diff analysis
                result = image_level_bootstrap_auprc_diff(oof_data[key_a], oof_data[key_b], seed=SEED)
                diff_rows.append({
                    "attribute": attr, "condition": condition, "comparison": comp_name,
                    "layer_minuend": layer_b, "layer_subtrahend": layer_a, "metric": metric,
                    "observed_layer_a": result["observed_a"], "observed_layer_b": result["observed_b"],
                    "observed_diff": result["observed_diff"],
                    "ci95_lo_diff": result["ci95_lo"], "ci95_hi_diff": result["ci95_hi"],
                    "n_boot_valid": result["n_boot_valid"],
                })
    fieldnames = ["attribute", "condition", "comparison", "layer_minuend", "layer_subtrahend", "metric",
                  "observed_layer_a", "observed_layer_b", "observed_diff", "ci95_lo_diff", "ci95_hi_diff",
                  "n_boot_valid"]
    with open(LAYER_DIFF_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in diff_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  Saved: {LAYER_DIFF_CSV}  ({len(diff_rows)} rows)")

    print("\n=== AUPRC L12-L4 diff (image-level bootstrap 95% CI), all_objects ===")
    for r in diff_rows:
        if r["comparison"] == "L12-L4" and r["condition"] == "all_objects":
            print(f"  {r['attribute']:13s} diff={r['observed_diff']:+.3f} "
                  f"[{r['ci95_lo_diff']:+.3f},{r['ci95_hi_diff']:+.3f}]")

    # ------------------------------------------------------------------
    # Go/No-Go
    # ------------------------------------------------------------------
    real_auprc = {(r["attribute"], r["condition"], r["layer"]): r for r in summary_rows if r["metric"] == "auprc"}
    perm_auprc = {(r["attribute"], r["condition"], r["layer"]): r for r in perm_rows if r["metric"] == "auprc"}
    beats_permutation = []
    for key, rr in real_auprc.items():
        pr = perm_auprc.get(key)
        if pr and rr["mean"] != "" and pr["mean"] != "":
            beats_permutation.append(float(rr["mean"]) > float(pr["mean"]) + 2 * float(pr["std"] or 0))
    frac_beats = sum(beats_permutation) / len(beats_permutation) if beats_permutation else float("nan")

    go_no_go = {
        "overall": "Go" if frac_beats > 0.8 else "Partial",
        "frac_attribute_layer_condition_combos_clearly_beating_permutation": frac_beats,
        "n_insufficient_data": len(insufficient),
        "insufficient_data_attributes": [r["attribute"] + "/" + r["condition"] for r in insufficient],
        "note": "linear probe only (Implementation B) -- text alignment (Implementation A) "
                "was NOT extended to 700 images per task instructions; see "
                "outputs/osie_text_alignment_crop_pilot/crop_go_no_go.json for that track's verdict.",
        "no_causal_claim": "Results indicate LINEAR DECODABILITY of attribute information from "
                            "raw hidden states; no claim is made about how or why this information "
                            "is represented.",
    }
    with open(GO_NO_GO_JSON, "w", encoding="utf-8") as fh:
        json.dump(go_no_go, fh, indent=2)
    print(f"\n  Saved: {GO_NO_GO_JSON}")
    print(f"  Go/No-Go: {go_no_go['overall']} "
          f"(fraction clearly beating permutation baseline: {frac_beats:.2f})")

    print(f"\n  Wall clock: {time.time() - t_start:.1f}s")
    print("\n" + "=" * 70)
    print("  Phase 4 linear probe: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
