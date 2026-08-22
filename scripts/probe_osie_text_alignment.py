"""
Implementation B: linear probe on RAW (un-normalized, no ln_post/proj)
patch hidden states at L4/L8/L12, for every OSIE attribute, plus a
Face-conditional probe for Emotion.

Thin wrapper around lib/osie_text_alignment_probe.py::run_linear_probe --
the SAME function is called by scripts/probe_osie_full700.py for the
700-image extension, so both scales are guaranteed to run identical logic
(StratifiedGroupKFold by image_id, StandardScaler+LogisticRegression
pipeline, permutation baseline). See that module's docstring for the full
methodology.

Writes only outputs/osie_text_alignment_pilot/{probe_fold_scores.csv,
probe_summary.csv,probe_permutation_baseline.csv}.

Usage (PowerShell):
    python scripts\\probe_osie_text_alignment.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import os
import time

from lib.osie_text_alignment_probe import run_linear_probe

SEED = 42
N_PERMUTATION_REPS = 20

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot"
OBJECT_FEATURES_NPZ = os.path.join(OUT_DIR, "object_features.npz")
OBJECTS_METADATA_CSV = os.path.join(OUT_DIR, "objects_metadata.csv")


def main():
    t_start = time.time()
    print("=" * 65)
    print("  OSIE Text-Alignment Pilot: linear probe (Implementation B)")
    print("=" * 65)

    for path in (OBJECT_FEATURES_NPZ, OBJECTS_METADATA_CSV):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found -- run the extraction script first")

    fold_rows, summary_rows, perm_rows, insufficient, _ = run_linear_probe(
        OBJECT_FEATURES_NPZ, OBJECTS_METADATA_CSV, OUT_DIR,
        n_permutation_reps=N_PERMUTATION_REPS, seed=SEED, collect_oof=False)

    print(f"\n  Saved: {os.path.join(OUT_DIR, 'probe_fold_scores.csv')}  ({len(fold_rows)} rows)")
    print(f"  Saved: {os.path.join(OUT_DIR, 'probe_summary.csv')}  "
          f"({len(summary_rows) + len(insufficient)} rows, {len(insufficient)} insufficient_data)")
    print(f"  Saved: {os.path.join(OUT_DIR, 'probe_permutation_baseline.csv')}  ({len(perm_rows)} rows)")

    print("\n=== AUPRC: mean (real) vs. permutation-baseline mean, by attribute x layer ===")
    real_auprc = {(r["attribute"], r["condition"], r["layer"]): r for r in summary_rows if r["metric"] == "auprc"}
    perm_auprc = {(r["attribute"], r["condition"], r["layer"]): r for r in perm_rows if r["metric"] == "auprc"}
    for key in sorted(real_auprc):
        rr = real_auprc[key]
        pr = perm_auprc.get(key)
        if pr:
            print(f"  {key[0]:13s} {key[1]:20s} L{key[2]:<3d} real={rr['mean']:.3f}  "
                  f"perm={pr['mean']:.3f}  random={rr['random_auprc']:.3f}")

    print(f"\n  Wall clock: {time.time() - t_start:.1f}s")
    print("\n" + "=" * 65)
    print("  Linear probe: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
