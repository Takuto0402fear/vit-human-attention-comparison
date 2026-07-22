"""
DINO ViT-S/16 vs DINO ViT-B/16 vs CLIP ViT-B/16 -- descriptive-only
layerwise comparison (NSS / AUC-Judd / sAUC), full 700-image results for
all three models.

Descriptive only: no statistical tests are run here (that is
scripts/stats_dino_clip_layerwise.py's job for the DINO-S/CLIP-B pair;
a three-model statistical comparison is a separate, later step). This
script only reports means, peak layers, and curve shape.

Tone-and-manner: matches scripts/stats_and_plots_expA.py /
scripts/plot_dino_clip_layerwise.py (figsize=(15,5), 1x3 panels,
dpi=150, xlabel="Layer", xticks=1..12, grid(alpha=0.3), suptitle
fontsize=13). DINO-S keeps its existing steelblue/solid/"o"; CLIP-B
keeps its existing darkorange/dashed/"s" (from plot_dino_clip_layerwise.py);
DINO-B is added as seagreen/dash-dot/"^" (same color used in
scripts/run_expA_dino_vitb16.py's pilot20 3-model comparison plot, kept
consistent here).

Reads (read-only, all three FINAL full-700 metrics_by_layer.csv files):
  outputs/expA/healthy/metrics_by_layer.csv               (DINO-S)
  outputs/expA_dino_vitb16/healthy/metrics_by_layer.csv    (DINO-B)
  outputs/expA_clip/healthy/metrics_by_layer.csv           (CLIP-B)

Writes (new directory only): outputs/expA_dino_vitb16/comparison/
  dino_s_dino_b_clip_b_layerwise_3metrics.png
  dino_s_dino_b_clip_b_layerwise_3metrics.pdf
  three_model_layerwise_values.csv
  plot_config.json

Does not modify any existing DINO-S/CLIP-B/DINO-B file or output,
including outputs/expA_clip/comparison/ (the existing DINO-S vs CLIP-B
statistical comparison directory).

Usage (PowerShell):
    python scripts\\plot_three_model_layerwise.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
METRICS = ["NSS", "AUC_Judd", "sAUC"]
METRIC_DISPLAY = {"NSS": "NSS", "AUC_Judd": "AUC-Judd", "sAUC": "sAUC"}

DINO_S_BY_LAYER = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_by_layer.csv"
DINO_B_BY_LAYER = r"C:\Users\user\gaze\outputs\expA_dino_vitb16\healthy\metrics_by_layer.csv"
CLIP_B_BY_LAYER = r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_by_layer.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\expA_dino_vitb16\comparison"
PLOT_PNG = os.path.join(OUT_DIR, "dino_s_dino_b_clip_b_layerwise_3metrics.png")
PLOT_PDF = os.path.join(OUT_DIR, "dino_s_dino_b_clip_b_layerwise_3metrics.pdf")
VALUES_CSV = os.path.join(OUT_DIR, "three_model_layerwise_values.csv")
CONFIG_JSON = os.path.join(OUT_DIR, "plot_config.json")

STYLE = {
    "DINO-S": {"color": "steelblue", "marker": "o", "linestyle": "-"},
    "DINO-B": {"color": "seagreen",  "marker": "^", "linestyle": "-."},
    "CLIP-B": {"color": "darkorange", "marker": "s", "linestyle": "--"},
}
SOURCES = {"DINO-S": DINO_S_BY_LAYER, "DINO-B": DINO_B_BY_LAYER, "CLIP-B": CLIP_B_BY_LAYER}
# ======================================================


def load_by_layer(csv_path, model_name):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if len(rows) != 12:
        raise RuntimeError(f"STOP [{model_name}]: metrics_by_layer.csv has {len(rows)} rows, expected 12")
    layers = sorted(int(r["layer"]) for r in rows)
    if layers != list(range(1, 13)):
        raise RuntimeError(f"STOP [{model_name}]: layers {layers} != 1..12")
    means = {m: {int(r["layer"]): float(r[f"{m}_mean"]) for r in rows} for m in METRICS}
    return means


def describe_shape(means_by_layer):
    """Crude descriptive tag: monotonic-rising-plateau vs M-shaped vs other."""
    vals = np.array([means_by_layer[l] for l in range(1, 13)])
    early_peak_layer = int(np.argmax(vals[:5])) + 1       # peak among L1-5
    late_peak_layer = int(np.argmax(vals[6:])) + 7          # peak among L7-12
    mid_min_layer = int(np.argmin(vals[3:9])) + 4            # min among L4-9
    has_early_high = vals[early_peak_layer - 1] >= 0.9 * vals.max()
    has_mid_dip = vals[mid_min_layer - 1] <= 0.85 * vals[early_peak_layer - 1] if has_early_high else False
    if has_early_high and has_mid_dip:
        return "M-shaped (early peak, mid-layer dip, partial late recovery)"
    return "monotonic-rising / late-layer plateau"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  DINO-S vs DINO-B vs CLIP-B -- descriptive layerwise comparison (full700)")
    print("=" * 65)

    print("\n--- Loading (read-only) ---")
    means = {}
    for model_name, path in SOURCES.items():
        means[model_name] = load_by_layer(path, model_name)
        print(f"  {model_name}: {path}  OK")

    print("\n--- Peak layers ---")
    peak_layers = {}
    for model_name in SOURCES:
        peak_layers[model_name] = {}
        for m in METRICS:
            best_l = max(means[model_name][m], key=lambda l: means[model_name][m][l])
            peak_layers[model_name][m] = {"layer": best_l, "value": means[model_name][m][best_l]}
            print(f"  {model_name:8s} {m:10s} peak: L{best_l}  ({means[model_name][m][best_l]:.4f})")

    print("\n--- Descriptive curve shape (NSS) ---")
    shapes = {}
    for model_name in SOURCES:
        shapes[model_name] = describe_shape(means[model_name]["NSS"])
        print(f"  {model_name}: {shapes[model_name]}")

    dino_b_vs_dino_s_diff = np.mean([
        abs(means["DINO-B"]["NSS"][l] - means["DINO-S"]["NSS"][l]) for l in range(1, 13)])
    dino_b_vs_clip_b_diff = np.mean([
        abs(means["DINO-B"]["NSS"][l] - means["CLIP-B"]["NSS"][l]) for l in range(1, 13)])
    dino_b_resembles = "DINO-S" if dino_b_vs_dino_s_diff < dino_b_vs_clip_b_diff else "CLIP-B"
    print(f"\n  DINO-B mean |NSS diff| vs DINO-S: {dino_b_vs_dino_s_diff:.4f}")
    print(f"  DINO-B mean |NSS diff| vs CLIP-B: {dino_b_vs_clip_b_diff:.4f}")
    print(f"  DINO-B's layer profile more closely resembles: {dino_b_resembles}")

    # ------------------------------------------------------------------
    # Save values CSV
    # ------------------------------------------------------------------
    with open(VALUES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer"] + METRICS)
        w.writeheader()
        for model_name in SOURCES:
            for l in range(1, 13):
                w.writerow({"model": model_name, "layer": l,
                           **{m: f"{means[model_name][m][l]:.6f}" for m in METRICS}})
    print(f"\n  Saved: {VALUES_CSV}  (36 rows: 3 models x 12 layers)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    layers_arr = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m in zip(axes, METRICS):
        for model_name in SOURCES:
            st = STYLE[model_name]
            vals = [means[model_name][m][l] for l in range(1, 13)]
            ax.plot(layers_arr, vals, marker=st["marker"], linestyle=st["linestyle"],
                    color=st["color"], lw=2, ms=5, label=model_name)
        ax.set_xlabel("Layer")
        ax.set_ylabel(METRIC_DISPLAY[m])
        ax.set_title(METRIC_DISPLAY[m])
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("DINO-S vs DINO-B vs CLIP-B (healthy, full700, descriptive only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PLOT_PNG}")
    print(f"  Saved: {PLOT_PDF}")

    # ------------------------------------------------------------------
    # plot_config.json
    # ------------------------------------------------------------------
    config = {
        "descriptive_only_no_statistical_tests": True,
        "tonmana_reference": {
            "scripts": ["scripts/stats_and_plots_expA.py", "scripts/plot_dino_clip_layerwise.py",
                        "scripts/run_expA_dino_vitb16.py (pilot20 3-model plot)"],
            "matched_elements": ["figsize=(15,5), 1x3 subplots", "dpi=150, bbox_inches=tight",
                                 "xlabel='Layer', xticks=1..12", "grid(True, alpha=0.3)",
                                 "suptitle fontsize=13", "tight_layout"],
        },
        "style": STYLE,
        "source_files": SOURCES,
        "peak_layers": peak_layers,
        "descriptive_shape_tags": shapes,
        "dino_b_resembles": dino_b_resembles,
        "mean_abs_nss_diff": {"dino_b_vs_dino_s": float(dino_b_vs_dino_s_diff),
                              "dino_b_vs_clip_b": float(dino_b_vs_clip_b_diff)},
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {CONFIG_JSON}")

    print("\n" + "=" * 65)
    print("  Done. Only outputs/expA_dino_vitb16/comparison/ was written to.")
    print("=" * 65)


if __name__ == "__main__":
    main()
