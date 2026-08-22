"""
Figures for Phase 3 (low-level visual-feature Ridge probe):
  A. target x layer R2 heatmap.
  B. L1-L12 R2 curves for luminance/RGB/saturation/contrast/edge/texture.
  C. early/middle/late group-representative comparison.
  D. geometry-control (area/centroid) curves vs. the core low-level targets.

Reads only outputs/osie_lowlevel_probe_all12_full700/*.csv. Writes only
under outputs/osie_lowlevel_probe_all12_full700/figures/.

Usage (PowerShell):
    python scripts\\plot_osie_lowlevel_probe_all12_full700.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
OUT_DIR = r"C:\Users\user\gaze\outputs\osie_lowlevel_probe_all12_full700"
FIG_DIR = os.path.join(OUT_DIR, "figures")
SUMMARY_CSV = os.path.join(OUT_DIR, "lowlevel_summary.csv")
PEAKS_CSV = os.path.join(OUT_DIR, "lowlevel_peak_layers.csv")

LAYERS = list(range(1, 13))
CORE_TARGETS = [
    "mean_luminance", "mean_red", "mean_green", "mean_blue", "mean_saturation",
    "luminance_contrast", "edge_strength", "fine_texture",
]
GEOMETRY_CONTROLS = ["mask_area_fraction", "centroid_x", "centroid_y"]
LAYER_GROUPS = {"early": [1, 2, 3, 4], "middle": [5, 6, 7, 8], "late": [9, 10, 11, 12]}

INK_PRIMARY, INK_SECONDARY = "#0b0b0b", "#52514e"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# ======================================================


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_r2_matrix():
    rows = read_csv_dicts(SUMMARY_CSV)
    by_target_layer = {}
    for r in rows:
        if r.get("metric") != "r2" or r["mean"] == "":
            continue
        by_target_layer.setdefault(r["target"], {})[int(r["layer"])] = float(r["mean"])
    return by_target_layer


def figure_a_heatmap(by_target_layer):
    targets = [t for t in CORE_TARGETS + GEOMETRY_CONTROLS if t in by_target_layer]
    matrix = np.array([[by_target_layer[t].get(l, np.nan) for l in LAYERS] for t in targets])
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=min(0, np.nanmin(matrix)), vmax=1)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    for i in range(len(targets)):
        for j in range(len(LAYERS)):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.5 else "black", fontsize=7)
    ax.set_title("Low-level Ridge probe R2, targets x 12 layers (geometry controls at bottom)",
                 color=INK_PRIMARY, fontsize=11, loc="left", pad=12)
    plt.colorbar(im, ax=ax, fraction=0.04, label="R2")
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "A_r2_heatmap.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_b_core_curves(by_target_layer):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(CORE_TARGETS):
        if t not in by_target_layer:
            continue
        y = [by_target_layer[t].get(l, np.nan) for l in LAYERS]
        ax.plot(LAYERS, y, "o-", color=cmap(i), markersize=4, linewidth=1.6, label=t)
    ax.axhline(0.0, color=BASELINE, linewidth=1.0, linestyle="--")
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Layer", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("out-of-fold R2", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Low-level visual-feature readout, L1-L12", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "B_core_target_curves.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_c_group_comparison(by_target_layer):
    peaks = read_csv_dicts(PEAKS_CSV)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(CORE_TARGETS))
    width = 0.25
    for i, group in enumerate(["early", "middle", "late"]):
        vals = []
        for t in CORE_TARGETS:
            row = next((r for r in peaks if r["target"] == t), None)
            if row is None:
                vals.append(np.nan)
                continue
            rep_layer = int(row[f"{group}_rep_layer"]) if row[f"{group}_rep_layer"] else None
            vals.append(by_target_layer.get(t, {}).get(rep_layer, np.nan) if rep_layer else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(CORE_TARGETS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("R2 (best layer within group)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Early (L1-4) / Middle (L5-8) / Late (L9-12) group-representative R2",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "C_early_middle_late_comparison.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_d_geometry_controls(by_target_layer):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for t in GEOMETRY_CONTROLS:
        if t not in by_target_layer:
            continue
        y = [by_target_layer[t].get(l, np.nan) for l in LAYERS]
        ax.plot(LAYERS, y, "s--", markersize=4, linewidth=1.4, label=f"{t} (geometry control)")
    core_mean = [np.nanmean([by_target_layer[t].get(l, np.nan) for t in CORE_TARGETS if t in by_target_layer])
                 for l in LAYERS]
    ax.plot(LAYERS, core_mean, "o-", color="#2a78d6", linewidth=2.2, markersize=5, label="mean of 8 core low-level targets")
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Layer", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("out-of-fold R2", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Geometry controls (area/position) vs. core low-level targets",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "D_geometry_controls.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 65)
    print("  Phase 3: Figures")
    print("=" * 65)
    if not os.path.isfile(SUMMARY_CSV):
        raise RuntimeError(f"STOP: {SUMMARY_CSV} not found -- run the probe script first")

    by_target_layer = load_r2_matrix()
    print("\n--- A. Heatmap ---")
    figure_a_heatmap(by_target_layer)
    print("\n--- B. Core target curves ---")
    figure_b_core_curves(by_target_layer)
    print("\n--- C. Early/middle/late comparison ---")
    figure_c_group_comparison(by_target_layer)
    print("\n--- D. Geometry controls ---")
    figure_d_geometry_controls(by_target_layer)

    print("\n" + "=" * 65)
    print("  Figures: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
