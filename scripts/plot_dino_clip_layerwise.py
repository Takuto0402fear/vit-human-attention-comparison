"""
DINO ViT-S/16 vs CLIP ViT-B/16 -- layerwise NSS / AUC-Judd / sAUC
comparison plot, plus percentile-based representative-image selection.

Tone-and-manner reference (read, not modified):
  scripts/stats_and_plots_expA.py :: plot_3metrics()
    -> outputs/expA/healthy/layerwise_3metrics_full.png / _full_ci.png
  figsize=(15,5), 1x3 subplots, "o-" steelblue lw=2 ms=5,
  fill_between alpha=0.15, xlabel="Layer", ylabel/title=metric name,
  xticks=1..12, grid(True, alpha=0.3), suptitle fontsize=13,
  tight_layout, savefig(dpi=150, bbox_inches="tight").

This script keeps that exact layout/style for the DINO curve (steelblue,
solid, "o-") and adds a CLIP curve (darkorange, dashed, "s--") on the
same 3 panels. Uncertainty band: 95% CI (bootstrap, seed=42, n_boot=
10000, percentile method) for BOTH models -- DINO's CI is read directly
from its own outputs/expA/healthy/metrics_by_layer.csv (already computed
by stats_and_plots_expA.py); CLIP's CI is freshly computed here with the
identical bootstrap_ci() parameters (duplicated from
stats_and_plots_expA.py, not imported -- matches this repo's per-script
convention), so both models use the same uncertainty methodology.

Also selects 3 representative images (10th / 50th / 90th percentile of
DINO_L10_NSS - CLIP_L4_NSS across all 700 images) for the attention
comparison figures rendered by
scripts/render_dino_clip_attention_comparison.py.

Reads (read-only):
  outputs/expA/healthy/metrics_per_image.csv
  outputs/expA/healthy/metrics_by_layer.csv
  outputs/expA_clip/healthy/metrics_per_image.csv
  outputs/expA_clip/healthy/metrics_by_layer.csv

Writes (new directory only):
  outputs/expA_clip/comparison/dino_clip_layerwise_3metrics.png
  outputs/expA_clip/comparison/dino_clip_layerwise_3metrics.pdf
  outputs/expA_clip/comparison/dino_clip_layerwise_values.csv
  outputs/expA_clip/comparison/plot_config.json
  outputs/expA_clip/comparison/representative_image_selection.csv

Does not modify any existing DINO or CLIP file/output.

Usage (PowerShell):
    python scripts\\plot_dino_clip_layerwise.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
METRICS = ["NSS", "AUC_Judd", "sAUC"]
METRIC_DISPLAY = {"NSS": "NSS", "AUC_Judd": "AUC-Judd", "sAUC": "sAUC"}
N_EXPECTED_IMAGES = 700
SEED = 42
N_BOOT = 10000
ALPHA = 0.05

DINO_PER_IMAGE = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"
DINO_BY_LAYER = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_by_layer.csv"
CLIP_PER_IMAGE = r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_per_image.csv"
CLIP_BY_LAYER = r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_by_layer.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\expA_clip\comparison"
PLOT_PNG = os.path.join(OUT_DIR, "dino_clip_layerwise_3metrics.png")
PLOT_PDF = os.path.join(OUT_DIR, "dino_clip_layerwise_3metrics.pdf")
VALUES_CSV = os.path.join(OUT_DIR, "dino_clip_layerwise_values.csv")
PLOT_CONFIG_JSON = os.path.join(OUT_DIR, "plot_config.json")
SELECTION_CSV = os.path.join(OUT_DIR, "representative_image_selection.csv")

STYLE = {
    "DINO":  {"color": "steelblue",  "marker": "o", "linestyle": "-",  "label": "DINO ViT-S/16"},
    "CLIP":  {"color": "darkorange", "marker": "s", "linestyle": "--", "label": "CLIP ViT-B/16"},
}
# ======================================================


# ======================================================================
# Load + validate
# ======================================================================

def load_per_image_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_and_index(rows, model_name):
    """
    Returns (data, image_ids) where data[metric][layer] = list of values,
    image_ids = sorted set of image stems (no extension).

    Asserts: 700 unique images, 8400 rows, layers 1-12 present exactly
    once per image, no missing/duplicate/NaN/Inf values.
    """
    if len(rows) != N_EXPECTED_IMAGES * 12:
        raise RuntimeError(f"STOP [{model_name}]: expected {N_EXPECTED_IMAGES * 12} rows, got {len(rows)}")

    per_image_layers = {}
    data = {m: {l: [] for l in range(1, 13)} for m in METRICS}
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
            if row[m] is None or row[m] == "":
                raise RuntimeError(f"STOP [{model_name}]: missing {m} for {key}")
            v = float(row[m])
            if math.isnan(v) or math.isinf(v):
                raise RuntimeError(f"STOP [{model_name}]: NaN/Inf {m} for {key}")
            data[m][layer].append(v)

    image_ids = sorted(per_image_layers.keys())
    if len(image_ids) != N_EXPECTED_IMAGES:
        raise RuntimeError(f"STOP [{model_name}]: {len(image_ids)} unique images, expected {N_EXPECTED_IMAGES}")
    bad = [img for img, layers in per_image_layers.items() if layers != set(range(1, 13))]
    if bad:
        raise RuntimeError(f"STOP [{model_name}]: {len(bad)} images missing layers 1-12, e.g. {bad[:3]}")

    return data, image_ids


def validate_by_layer_csv(csv_path, model_name):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if len(rows) != 12:
        raise RuntimeError(f"STOP [{model_name}]: metrics_by_layer.csv has {len(rows)} rows, expected 12")
    layers = sorted(int(r["layer"]) for r in rows)
    if layers != list(range(1, 13)):
        raise RuntimeError(f"STOP [{model_name}]: metrics_by_layer.csv layers {layers} != 1..12")
    return {int(r["layer"]): r for r in rows}


# ======================================================================
# Statistics (bootstrap CI -- identical params to stats_and_plots_expA.py)
# ======================================================================

def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = rng.choice(arr, size=n, replace=True).mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def build_series(model_name, per_image_data, by_layer_rows, ci_source):
    """
    ci_source: "existing" (read ci_lo/ci_hi from by_layer_rows, DINO's
    stats_and_plots_expA.py output) or "bootstrap" (compute fresh, CLIP).
    """
    means, ci_lo, ci_hi = {}, {}, {}
    for m in METRICS:
        means[m] = np.array([float(by_layer_rows[l][f"{m}_mean"]) for l in range(1, 13)])
        if ci_source == "existing":
            ci_lo[m] = np.array([float(by_layer_rows[l][f"{m}_ci_lo"]) for l in range(1, 13)])
            ci_hi[m] = np.array([float(by_layer_rows[l][f"{m}_ci_hi"]) for l in range(1, 13)])
        else:
            los, his = [], []
            for l in range(1, 13):
                lo, hi = bootstrap_ci(per_image_data[m][l])
                los.append(lo); his.append(hi)
            ci_lo[m] = np.array(los)
            ci_hi[m] = np.array(his)
    peak_layer = {m: int(np.argmax(means[m])) + 1 for m in METRICS}
    return {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi, "peak_layer": peak_layer}


# ======================================================================
# Plot
# ======================================================================

def make_plot(dino_series, clip_series, out_png, out_pdf):
    layers_arr = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, mkey in zip(axes, METRICS):
        mname = METRIC_DISPLAY[mkey]
        for model_name, series in [("DINO", dino_series), ("CLIP", clip_series)]:
            st = STYLE[model_name]
            m_mean = series["means"][mkey]
            lo, hi = series["ci_lo"][mkey], series["ci_hi"][mkey]
            ax.plot(layers_arr, m_mean, marker=st["marker"], linestyle=st["linestyle"],
                    color=st["color"], lw=2, ms=5, label=st["label"])
            ax.fill_between(layers_arr, lo, hi, alpha=0.15, color=st["color"])

            peak_l = series["peak_layer"][mkey]
            ax.plot(peak_l, m_mean[peak_l - 1], marker=st["marker"],
                    color=st["color"], ms=9, mfc="none", mew=1.8, linestyle="None")

        ax.set_xlabel("Layer")
        ax.set_ylabel(mname)
        ax.set_title(mname)
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("DINO ViT-S/16 vs CLIP ViT-B/16 (healthy, full700, mean ± 95% CI)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Representative image selection
# ======================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  DINO vs CLIP layerwise comparison + representative-image selection")
    print("=" * 65)

    print("\n--- Loading + validating DINO ---")
    dino_rows = load_per_image_rows(DINO_PER_IMAGE)
    dino_data, dino_ids = validate_and_index(dino_rows, "DINO")
    dino_by_layer = validate_by_layer_csv(DINO_BY_LAYER, "DINO")
    print(f"  {len(dino_rows)} rows, {len(dino_ids)} images  OK")

    print("\n--- Loading + validating CLIP ---")
    clip_rows = load_per_image_rows(CLIP_PER_IMAGE)
    clip_data, clip_ids = validate_and_index(clip_rows, "CLIP")
    clip_by_layer = validate_by_layer_csv(CLIP_BY_LAYER, "CLIP")
    print(f"  {len(clip_rows)} rows, {len(clip_ids)} images  OK")

    if dino_ids != clip_ids:
        raise RuntimeError("STOP: DINO and CLIP image-ID sets differ")
    print(f"  DINO and CLIP image-ID sets are identical ({len(dino_ids)} images)  OK")

    print("\n--- Computing series (DINO: existing CI; CLIP: fresh bootstrap CI) ---")
    dino_series = build_series("DINO", dino_data, dino_by_layer, ci_source="existing")
    clip_series = build_series("CLIP", clip_data, clip_by_layer, ci_source="bootstrap")
    for m in METRICS:
        print(f"  {m:10s} peak -- DINO: L{dino_series['peak_layer'][m]} "
              f"({dino_series['means'][m][dino_series['peak_layer'][m]-1]:.4f})   "
              f"CLIP: L{clip_series['peak_layer'][m]} "
              f"({clip_series['means'][m][clip_series['peak_layer'][m]-1]:.4f})")

    print("\n--- Plotting ---")
    make_plot(dino_series, clip_series, PLOT_PNG, PLOT_PDF)
    print(f"  Saved: {PLOT_PNG}")
    print(f"  Saved: {PLOT_PDF}")

    print("\n--- Saving values CSV ---")
    fieldnames = ["model", "layer"]
    for m in METRICS:
        fieldnames += [f"{m}_mean", f"{m}_ci_lo", f"{m}_ci_hi"]
    with open(VALUES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model_name, series in [("DINO", dino_series), ("CLIP", clip_series)]:
            for l in range(1, 13):
                row = {"model": model_name, "layer": l}
                for m in METRICS:
                    row[f"{m}_mean"] = f"{series['means'][m][l-1]:.6f}"
                    row[f"{m}_ci_lo"] = f"{series['ci_lo'][m][l-1]:.6f}"
                    row[f"{m}_ci_hi"] = f"{series['ci_hi'][m][l-1]:.6f}"
                w.writerow(row)
    print(f"  Saved: {VALUES_CSV}  (24 rows: 2 models x 12 layers)")

    print("\n--- Representative image selection (DINO L10 NSS - CLIP L4 NSS) ---")
    dino_l10_nss = {}
    for row in dino_rows:
        if int(row["layer"]) == 10:
            dino_l10_nss[os.path.splitext(row["image"])[0]] = float(row["NSS"])
    clip_l4_nss = {}
    for row in clip_rows:
        if int(row["layer"]) == 4:
            clip_l4_nss[os.path.splitext(row["image"])[0]] = float(row["NSS"])

    common_ids = sorted(set(dino_l10_nss) & set(clip_l4_nss))
    if len(common_ids) != N_EXPECTED_IMAGES:
        raise RuntimeError(
            f"STOP: {len(common_ids)} images with both DINO L10 and CLIP L4 NSS, "
            f"expected {N_EXPECTED_IMAGES}")

    diffs = np.array([dino_l10_nss[i] - clip_l4_nss[i] for i in common_ids])
    p10, p50, p90 = np.percentile(diffs, [10, 50, 90])
    print(f"  difference percentiles: p10={p10:.4f}  p50={p50:.4f}  p90={p90:.4f}")

    targets = [
        ("dino_favored_p90", p90, 90,
         "difference closest to the 90th percentile of (DINO L10 NSS - CLIP L4 NSS) "
         "across all 700 images -- a DINO-favored example"),
        ("typical_p50", p50, 50,
         "difference closest to the 50th percentile (median) of (DINO L10 NSS - CLIP L4 NSS) "
         "across all 700 images -- a typical example"),
        ("clip_favored_p10", p10, 10,
         "difference closest to the 10th percentile of (DINO L10 NSS - CLIP L4 NSS) "
         "across all 700 images -- a CLIP-favored example"),
    ]

    selection_rows = []
    chosen_ids = set()
    for sel_type, target_val, pct, reason in targets:
        idx = int(np.argmin(np.abs(diffs - target_val)))
        img_id = common_ids[idx]
        if img_id in chosen_ids:
            # extremely unlikely with 700 images and 3 well-separated percentiles,
            # but guard against an accidental collision rather than silently duplicate
            raise RuntimeError(f"STOP: image {img_id} selected for more than one percentile slot")
        chosen_ids.add(img_id)
        selection_rows.append({
            "image_id": img_id,
            "selection_type": sel_type,
            "DINO_L10_NSS": f"{dino_l10_nss[img_id]:.6f}",
            "CLIP_L4_NSS": f"{clip_l4_nss[img_id]:.6f}",
            "difference": f"{diffs[idx]:.6f}",
            "percentile_target": pct,
            "selection_reason": reason,
        })
        print(f"  {sel_type:20s}  image={img_id}  diff={diffs[idx]:.4f}  (target p{pct}={target_val:.4f})")

    with open(SELECTION_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image_id", "selection_type", "DINO_L10_NSS", "CLIP_L4_NSS",
            "difference", "percentile_target", "selection_reason"])
        w.writeheader()
        w.writerows(selection_rows)
    print(f"  Saved: {SELECTION_CSV}")

    print("\n--- Saving plot_config.json ---")
    plot_config = {
        "tonmana_reference": {
            "script": "scripts/stats_and_plots_expA.py :: plot_3metrics()",
            "outputs_referenced": [
                "outputs/expA/healthy/layerwise_3metrics_full.png",
                "outputs/expA/healthy/layerwise_3metrics_full_ci.png",
            ],
            "matched_elements": [
                "figsize=(15,5), 1x3 subplots", "dpi=150, bbox_inches=tight",
                "DINO: color=steelblue, marker='o', linestyle='-', lw=2, ms=5 (unchanged)",
                "fill_between alpha=0.15", "xlabel='Layer', ylabel/title=metric display name",
                "xticks=1..12", "grid(True, alpha=0.3)", "suptitle fontsize=13", "tight_layout",
            ],
            "added_for_clip": [
                "CLIP: color=darkorange, marker='s', linestyle='--', lw=2, ms=5",
                "legend (fontsize=8) -- absent in the single-series original, needed now for 2 series",
                "hollow larger marker (mfc='none', mew=1.8, ms=9) at each curve's peak layer",
            ],
            "uncertainty_band_choice": (
                "95% CI (not SD): both outputs/expA/healthy/layerwise_3metrics_full.png (SD) and "
                "_full_ci.png (95% CI) exist for DINO; 95% CI was chosen here because with N=700 "
                "images per layer the CI band stays narrow enough for the two model curves to "
                "remain visually distinguishable, whereas the SD band (across-image variability) "
                "is wide enough to visually overlap between models and obscure the comparison."
            ),
        },
        "bootstrap_ci": {"n_boot": N_BOOT, "seed": SEED, "alpha": ALPHA, "method": "percentile bootstrap of the mean"},
        "dino_ci_source": "read from outputs/expA/healthy/metrics_by_layer.csv (already computed by stats_and_plots_expA.py)",
        "clip_ci_source": "freshly computed here from outputs/expA_clip/healthy/metrics_per_image.csv with identical bootstrap_ci() parameters",
        "peak_layers": {
            "DINO": dino_series["peak_layer"],
            "CLIP": clip_series["peak_layer"],
        },
        "representative_image_selection": {
            "difference_definition": "DINO Layer-10 NSS minus CLIP Layer-4 NSS, per image",
            "percentiles_used": [10, 50, 90],
            "percentile_values": {"p10": float(p10), "p50": float(p50), "p90": float(p90)},
            "note": "nearest-to-percentile selection (not min/max) to avoid outlier-driven picks",
        },
        "source_files": {
            "dino_per_image": DINO_PER_IMAGE, "dino_by_layer": DINO_BY_LAYER,
            "clip_per_image": CLIP_PER_IMAGE, "clip_by_layer": CLIP_BY_LAYER,
        },
    }
    with open(PLOT_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(plot_config, f, indent=2)
    print(f"  Saved: {PLOT_CONFIG_JSON}")

    print("\n" + "=" * 65)
    print("  Done. Only outputs/expA_clip/comparison/ was written to.")
    print("=" * 65)


if __name__ == "__main__":
    main()
