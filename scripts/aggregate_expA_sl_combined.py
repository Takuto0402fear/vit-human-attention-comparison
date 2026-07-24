"""
Exp A (SL ViT-S/16): combined aggregation across the 6 official trials
(supervised/{01..06}/12layers/checkpoint.pth), each already independently
verified by scripts/run_expA_sl_full700.py (700 images x 12 layers,
8400 rows per trial, 0 duplicates/missing/NaN/out-of-range).

Reads ONLY existing per-trial outputs (outputs/expA_sl/trial_{01..06}/
metrics_per_image.csv) and the existing CLIP full700 layer summary
(outputs/expA_clip/healthy/metrics_by_layer.csv, read-only) -- does not
re-run any model, does not touch outputs/expA_sl/trial_*/,
outputs/expA_sl/pilot20/, outputs/expA_sl/validation/, outputs/expA/,
outputs/expA_clip/, outputs/expA_dino_vitb16/, outputs/attn_cache/, or
results/.

Aggregation levels (deliberately NOT collapsed into one another -- see
each function's docstring for why):
  1. metrics_all_trials.csv            -- 6 x 700 x 12 = 50400 raw rows
     (trial, image, layer, relative_depth, NSS, AUC_Judd, sAUC). The full
     data, image-level and trial-level still distinguishable.
  2. metrics_by_layer_and_trial.csv    -- 6 x 12 = 72 rows. Per-trial,
     per-layer mean/std OVER THE 700 IMAGES (i.e. each trial's own
     metrics_by_layer.csv, concatenated with a trial column). This is
     the correct unit for "trial-level" statistics.
  3. metrics_by_layer_across_trials.csv -- 12 rows. Mean/SEM/std of the
     6 TRIAL-LEVEL MEANS from (2), NOT a naive re-pooling of the 4200
     (=6x700) image-level samples per layer as if they were independent.
     The 700 images are the SAME 700 images repeated across all 6
     trials, so treating 50400 (or 4200) rows as independent samples for
     a layer-level statistical summary would understate the true
     between-trial variance and overstate the effective N. N=6 here, by
     design.
  4. metrics_by_image_meanacrosstrials.csv -- 700 x 12 = 8400 rows. Per
     (image, layer), the mean over the 6 trials -- an image-level summary
     comparable in shape to outputs/expA/healthy/metrics_per_image.csv
     (DINO-S) or outputs/expA_clip/healthy/metrics_per_image.csv (CLIP-B),
     which have no trial dimension.

Plots:
  - per_trial_layerwise.png       -- 6 individual trial curves overlaid
  - mean_across_trials_layerwise.png -- cross-trial mean +/- cross-trial
    std band (from level 3, NOT from level 1/2's per-image spread)
  - sl_vs_clip_layerwise.png      -- SL (cross-trial mean) overlaid with
    the existing CLIP-B full700 layer curve (outputs/expA_clip/healthy/
    metrics_by_layer.csv, read-only)

Usage (PowerShell):
    python scripts\\aggregate_expA_sl_combined.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRIALS = [1, 2, 3, 4, 5, 6]
DEPTH = 12
N_IMAGES = 700
METRICS_LIST = ["NSS", "AUC_Judd", "sAUC"]

TRIAL_DIR = r"C:\Users\user\gaze\outputs\expA_sl\trial_{:02d}"
CLIP_LAYER_CSV = r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_by_layer.csv"
OUT_DIR = r"C:\Users\user\gaze\outputs\expA_sl\combined"


def load_trial_records(trial_num):
    path = os.path.join(TRIAL_DIR.format(trial_num), "metrics_per_image.csv")
    records = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "trial": trial_num, "image": row["image"], "layer": int(row["layer"]),
                "relative_depth": float(row["relative_depth"]),
                "NSS": float(row["NSS"]), "AUC_Judd": float(row["AUC_Judd"]),
                "sAUC": float(row["sAUC"]),
            })
    return records


def load_trial_run_config(trial_num):
    path = os.path.join(TRIAL_DIR.format(trial_num), "run_config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_clip_layer_means():
    means = {m: {} for m in METRICS_LIST}
    with open(CLIP_LAYER_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            l = int(row["layer"])
            for m in METRICS_LIST:
                means[m][l] = float(row[f"{m}_mean"])
    return means


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  Exp A (SL ViT-S/16): combined aggregation across 6 trials")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Sanity: all 6 trials fully complete and mechanically clean
    # ------------------------------------------------------------------
    print("\n--- Verifying each trial's own mechanical checks (read-only) ---")
    trial_configs = {}
    for t in TRIALS:
        cfg = load_trial_run_config(t)
        trial_configs[t] = cfg
        mc = cfg["mechanical_checks"]
        if mc["n_rows"] != N_IMAGES * DEPTH or mc["duplicate_pairs"] or mc["missing_images_or_layers"] \
                or mc["nan_inf_in_metrics"] or mc["out_of_range"]:
            raise RuntimeError(f"STOP: trial {t:02d} mechanical_checks not clean: {mc}")
        print(f"  trial {t:02d}: {mc['n_rows']} rows, clean (0 dup/missing/nan/range), "
              f"checkpoint_md5={cfg['checkpoint_md5']}")

    ckpt_hashes = [trial_configs[t]["checkpoint_md5"] for t in TRIALS]
    if len(set(ckpt_hashes)) != len(TRIALS):
        raise RuntimeError(f"STOP: duplicate checkpoint MD5 across trials -- not 6 distinct models: {ckpt_hashes}")
    print(f"  6 distinct checkpoint MD5 hashes confirmed (6 genuinely different trained models)  OK")

    # ------------------------------------------------------------------
    # Level 1: all_trials raw concat (50400 rows)
    # ------------------------------------------------------------------
    print("\n--- Level 1: metrics_all_trials.csv (raw concat) ---")
    all_records = []
    for t in TRIALS:
        all_records.extend(load_trial_records(t))

    expected_total = len(TRIALS) * N_IMAGES * DEPTH
    if len(all_records) != expected_total:
        raise RuntimeError(f"STOP: expected {expected_total} total rows (6x700x12), got {len(all_records)}")
    seen = set((r["trial"], r["image"], r["layer"]) for r in all_records)
    if len(seen) != len(all_records):
        raise RuntimeError("STOP: duplicate (trial,image,layer) tuples found across the combined data")
    print(f"  {len(all_records)} rows (6 x {N_IMAGES} x {DEPTH} = {expected_total})  OK")
    print(f"  unique (trial,image,layer) tuples: {len(seen)}  (no duplicates)  OK")

    all_csv = os.path.join(OUT_DIR, "metrics_all_trials.csv")
    fieldnames1 = ["trial", "image", "layer", "relative_depth"] + METRICS_LIST
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames1)
        w.writeheader()
        w.writerows(all_records)
    print(f"  Saved: {all_csv}")

    # ------------------------------------------------------------------
    # Level 2: per-trial, per-layer means over the 700 images (72 rows)
    # ------------------------------------------------------------------
    print("\n--- Level 2: metrics_by_layer_and_trial.csv (per trial x layer, N=700 images each) ---")
    by_trial_layer = {}  # (trial, layer) -> {metric: [700 values]}
    for r in all_records:
        key = (r["trial"], r["layer"])
        by_trial_layer.setdefault(key, {m: [] for m in METRICS_LIST})
        for m in METRICS_LIST:
            by_trial_layer[key][m].append(r[m])

    layer_trial_rows = []
    for t in TRIALS:
        for l in range(1, DEPTH + 1):
            vals = by_trial_layer[(t, l)]
            row = {"trial": t, "layer": l, "relative_depth": (l - 1) / (DEPTH - 1)}
            for m in METRICS_LIST:
                v = vals[m]
                if len(v) != N_IMAGES:
                    raise RuntimeError(f"STOP: trial {t} layer {l} metric {m} has N={len(v)}, expected {N_IMAGES}")
                row[f"{m}_mean"] = float(np.mean(v))
                row[f"{m}_std"] = float(np.std(v))
            layer_trial_rows.append(row)

    if len(layer_trial_rows) != len(TRIALS) * DEPTH:
        raise RuntimeError(f"STOP: expected {len(TRIALS) * DEPTH} trial x layer rows, got {len(layer_trial_rows)}")

    layer_trial_csv = os.path.join(OUT_DIR, "metrics_by_layer_and_trial.csv")
    fieldnames2 = ["trial", "layer", "relative_depth"] + [f"{m}_{s}" for m in METRICS_LIST for s in ("mean", "std")]
    with open(layer_trial_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames2)
        w.writeheader()
        for row in layer_trial_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  {len(layer_trial_rows)} rows (6 trials x {DEPTH} layers)  OK")
    print(f"  Saved: {layer_trial_csv}")

    # ------------------------------------------------------------------
    # Level 3: cross-trial layer summary -- mean/std OF THE 6 TRIAL-LEVEL
    # MEANS (N=6), not a re-pooling of the 4200 image samples per layer.
    # ------------------------------------------------------------------
    print("\n--- Level 3: metrics_by_layer_across_trials.csv (mean/std OF the 6 trial-level means, N=6) ---")
    trial_means_by_layer = {l: {m: [] for m in METRICS_LIST} for l in range(1, DEPTH + 1)}
    for row in layer_trial_rows:
        for m in METRICS_LIST:
            trial_means_by_layer[row["layer"]][m].append(row[f"{m}_mean"])

    across_trial_rows = []
    for l in range(1, DEPTH + 1):
        row = {"layer": l, "relative_depth": (l - 1) / (DEPTH - 1), "n_trials": len(TRIALS)}
        for m in METRICS_LIST:
            v = trial_means_by_layer[l][m]
            if len(v) != len(TRIALS):
                raise RuntimeError(f"STOP: layer {l} metric {m} has {len(v)} trial-level means, expected {len(TRIALS)}")
            row[f"{m}_mean_across_trials"] = float(np.mean(v))
            row[f"{m}_std_across_trials"] = float(np.std(v))
            row[f"{m}_sem_across_trials"] = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        across_trial_rows.append(row)

    across_trial_csv = os.path.join(OUT_DIR, "metrics_by_layer_across_trials.csv")
    fieldnames3 = ["layer", "relative_depth", "n_trials"] + \
        [f"{m}_{s}" for m in METRICS_LIST for s in ("mean_across_trials", "std_across_trials", "sem_across_trials")]
    with open(across_trial_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames3)
        w.writeheader()
        for row in across_trial_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  {len(across_trial_rows)} rows ({DEPTH} layers, each from N={len(TRIALS)} trial-level means)  OK")
    print(f"  Saved: {across_trial_csv}")

    print(f"\n  {'layer':>5} {'rel_d':>6}  {'NSS':>8}  {'AUC_Judd':>10}  {'sAUC':>8}   (mean across 6 trials)")
    for row in across_trial_rows:
        print(f"  {row['layer']:5d} {row['relative_depth']:6.3f}  {row['NSS_mean_across_trials']:8.4f}  "
              f"{row['AUC_Judd_mean_across_trials']:10.4f}  {row['sAUC_mean_across_trials']:8.4f}")

    # ------------------------------------------------------------------
    # Level 4: image-level, mean across the 6 trials (700 x 12 rows) --
    # comparable in shape to DINO-S/CLIP-B's own (trial-less)
    # metrics_per_image.csv.
    # ------------------------------------------------------------------
    print("\n--- Level 4: metrics_by_image_meanacrosstrials.csv (per image x layer, mean over 6 trials) ---")
    by_image_layer = {}  # (image, layer) -> {metric: [6 values]}
    rel_depth_by_layer = {}
    for r in all_records:
        key = (r["image"], r["layer"])
        by_image_layer.setdefault(key, {m: [] for m in METRICS_LIST})
        for m in METRICS_LIST:
            by_image_layer[key][m].append(r[m])
        rel_depth_by_layer[r["layer"]] = r["relative_depth"]

    image_rows = []
    for (img, l), vals in by_image_layer.items():
        row = {"image": img, "layer": l, "relative_depth": rel_depth_by_layer[l]}
        for m in METRICS_LIST:
            v = vals[m]
            if len(v) != len(TRIALS):
                raise RuntimeError(f"STOP: image {img} layer {l} metric {m} has {len(v)} trial values, "
                                   f"expected {len(TRIALS)}")
            row[f"{m}_mean_across_trials"] = float(np.mean(v))
            row[f"{m}_std_across_trials"] = float(np.std(v))
        image_rows.append(row)

    expected_image_rows = N_IMAGES * DEPTH
    if len(image_rows) != expected_image_rows:
        raise RuntimeError(f"STOP: expected {expected_image_rows} image x layer rows, got {len(image_rows)}")

    image_rows.sort(key=lambda r: (r["image"], r["layer"]))
    image_csv = os.path.join(OUT_DIR, "metrics_by_image_meanacrosstrials.csv")
    fieldnames4 = ["image", "layer", "relative_depth"] + \
        [f"{m}_{s}" for m in METRICS_LIST for s in ("mean_across_trials", "std_across_trials")]
    with open(image_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames4)
        w.writeheader()
        for row in image_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  {len(image_rows)} rows ({N_IMAGES} images x {DEPTH} layers, each mean over {len(TRIALS)} trials)  OK")
    print(f"  Saved: {image_csv}")

    # ------------------------------------------------------------------
    # Plot 1: per-trial layerwise curves overlaid
    # ------------------------------------------------------------------
    print("\n--- Plots ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_defs = [("NSS", "NSS_mean"), ("AUC_Judd", "AUC_Judd_mean"), ("sAUC", "sAUC_mean")]
    layers_arr = np.arange(1, DEPTH + 1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(TRIALS)))
    for ax, (mname, mc) in zip(axes, plot_defs):
        for i, t in enumerate(TRIALS):
            vals = [r[mc] for r in layer_trial_rows if r["trial"] == t]
            ax.plot(layers_arr, vals, "o-", color=colors[i], lw=1.5, ms=4, alpha=0.85, label=f"trial{t:02d}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_title(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_xticks(layers_arr)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-A (SL ViT-S/16): per-trial layerwise curves (6 trials, N=700 images each)", fontsize=13)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "per_trial_layerwise.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ------------------------------------------------------------------
    # Plot 2: mean-across-trials curve with cross-trial std band
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_defs2 = [("NSS", "NSS_mean_across_trials", "NSS_std_across_trials"),
                  ("AUC_Judd", "AUC_Judd_mean_across_trials", "AUC_Judd_std_across_trials"),
                  ("sAUC", "sAUC_mean_across_trials", "sAUC_std_across_trials")]
    for ax, (mname, mc, sc) in zip(axes, plot_defs2):
        means = np.array([r[mc] for r in across_trial_rows])
        stds = np.array([r[sc] for r in across_trial_rows])
        ax.plot(layers_arr, means, "o-", color="firebrick", lw=2, ms=6, label="SL ViT-S/16 (mean of 6 trials)")
        ax.fill_between(layers_arr, means - stds, means + stds, alpha=0.2, color="firebrick",
                         label="+/- 1 SD across trials (N=6)")
        ax.set_xlabel("Layer")
        ax.set_ylabel(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_title(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_xticks(layers_arr)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-A (SL ViT-S/16): mean across 6 trials, band = between-trial SD (N=6, not image-level SD)",
                 fontsize=12)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "mean_across_trials_layerwise.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ------------------------------------------------------------------
    # Plot 3: SL (cross-trial mean) vs existing CLIP-B full700 (read-only)
    # ------------------------------------------------------------------
    clip_means = load_clip_layer_means()
    if any(len(clip_means[m]) != DEPTH for m in METRICS_LIST):
        raise RuntimeError(f"STOP: CLIP layer CSV does not have {DEPTH} layers for every metric")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (mname, mc, sc) in zip(axes, plot_defs2):
        sl_means = np.array([r[mc] for r in across_trial_rows])
        sl_stds = np.array([r[sc] for r in across_trial_rows])
        clip_vals = np.array([clip_means[mname][l] for l in range(1, DEPTH + 1)])

        ax.plot(layers_arr, sl_means, "o-", color="firebrick", lw=2, ms=6, label="SL ViT-S/16 (mean of 6 trials)")
        ax.fill_between(layers_arr, sl_means - sl_stds, sl_means + sl_stds, alpha=0.15, color="firebrick")
        ax.plot(layers_arr, clip_vals, "s--", color="darkorange", lw=2, ms=6, label="CLIP ViT-B/16 (full700)")

        ax.set_xlabel("Layer")
        ax.set_ylabel(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_title(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_xticks(layers_arr)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-A: SL ViT-S/16 (6-trial mean) vs CLIP ViT-B/16 -- full700, layerwise agreement with human gaze",
                 fontsize=12)
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "sl_vs_clip_layerwise.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3}")

    # ------------------------------------------------------------------
    # Combined run_config / provenance log
    # ------------------------------------------------------------------
    combined_config = {
        "model": "sl_vits16", "trials": TRIALS, "depth": DEPTH, "n_images": N_IMAGES,
        "checkpoint_paths": {t: trial_configs[t]["checkpoint_path"] for t in TRIALS},
        "checkpoint_md5": {t: trial_configs[t]["checkpoint_md5"] for t in TRIALS},
        "metrics": METRICS_LIST,
        "row_counts": {
            "metrics_all_trials": len(all_records),
            "expected": expected_total,
            "metrics_by_layer_and_trial": len(layer_trial_rows),
            "metrics_by_layer_across_trials": len(across_trial_rows),
            "metrics_by_image_meanacrosstrials": len(image_rows),
        },
        "aggregation_note": (
            "metrics_by_layer_across_trials.csv's mean/std are computed OVER THE 6 "
            "TRIAL-LEVEL MEANS (N=6), not by re-pooling the 4200 (=6x700) per-image "
            "samples per layer as if independent -- the same 700 images repeat "
            "across all 6 trials, so naive pooling would understate between-trial "
            "variance and overstate the effective sample size."),
        "clip_comparison_source": CLIP_LAYER_CSV,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(OUT_DIR, "combined_run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(combined_config, f, indent=2)
    print(f"\n  Saved: {config_path}")

    print("\n" + "=" * 65)
    print("  Combined aggregation across 6 trials: DONE")
    print("  outputs/expA_sl/trial_*/, outputs/expA/, outputs/expA_clip/, "
          "outputs/expA_dino_vitb16/, outputs/attn_cache/, results/ were NOT touched.")
    print("=" * 65)


if __name__ == "__main__":
    main()
