"""
Figures for Phase 1 (all-12-layer semantic-attribute probe):
  A. 12 attributes x 12 layers AUPRC heatmap.
  B. Per-attribute L1-L12 AUPRC curves (small multiples).
  C. Attribute-group (direct_visual / emotion / relational / sensory_functional) mean curves.
  D. Real AUPRC vs. permutation-baseline gap, all 12 layers.
  E. Emotion: all_objects vs. conditional_on_face, all 12 layers.
  F. Peak-layer distribution across the 12 attributes (histogram).

Reads only outputs/osie_probe_all12_full700/*.csv. Writes only under
outputs/osie_probe_all12_full700/figures/.

Usage (PowerShell):
    python scripts\\plot_osie_probe_all12_full700.py
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
OUT_DIR = r"C:\Users\user\gaze\outputs\osie_probe_all12_full700"
FIG_DIR = os.path.join(OUT_DIR, "figures")
SUMMARY_CSV = os.path.join(OUT_DIR, "probe_summary.csv")
PERM_CSV = os.path.join(OUT_DIR, "probe_permutation_baseline.csv")
PEAKS_CSV = os.path.join(OUT_DIR, "probe_peak_layers.csv")

LAYERS = list(range(1, 13))
ATTRIBUTES = [
    "Text", "Face", "Emotion", "Sound", "Smell", "Taste", "Touch",
    "Motion", "Operability", "Watchability", "Touched", "Gazed",
]
ATTRIBUTE_GROUPS = {
    "direct_visual": ["Face", "Text"],
    "emotion": ["Emotion"],
    "relational": ["Gazed", "Touched"],
    "sensory_functional": ["Motion", "Sound", "Smell", "Taste", "Touch", "Operability", "Watchability"],
}

INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# ======================================================


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_auprc_matrix():
    rows = read_csv_dicts(SUMMARY_CSV)
    by_attr_layer = {}
    for r in rows:
        if r.get("metric") != "auprc" or r.get("condition") != "all_objects" or r["mean"] == "":
            continue
        by_attr_layer.setdefault(r["attribute"], {})[int(r["layer"])] = float(r["mean"])
    return by_attr_layer


def figure_a_heatmap(by_attr_layer):
    attrs = [a for a in ATTRIBUTES if a in by_attr_layer]
    matrix = np.array([[by_attr_layer[a].get(l, np.nan) for l in LAYERS] for a in attrs])
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_yticks(range(len(attrs)))
    ax.set_yticklabels(attrs)
    for i in range(len(attrs)):
        for j in range(len(LAYERS)):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.6 else "black", fontsize=7)
    ax.set_title("Semantic-attribute probe AUPRC, 12 attributes x 12 layers", color=INK_PRIMARY,
                 fontsize=11, loc="left", pad=12)
    plt.colorbar(im, ax=ax, fraction=0.04, label="AUPRC")
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "A_auprc_heatmap.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_b_small_multiples(by_attr_layer):
    attrs = [a for a in ATTRIBUTES if a in by_attr_layer]
    n_cols = 3
    n_rows = int(np.ceil(len(attrs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes_flat = axes.flatten()
    for ax, attr in zip(axes_flat, attrs):
        ax.set_facecolor(SURFACE)
        y = [by_attr_layer[attr].get(l, np.nan) for l in LAYERS]
        ax.plot(LAYERS, y, "o-", color="#2a78d6", markersize=4, linewidth=1.5)
        ax.set_title(attr, fontsize=10, color=INK_PRIMARY, loc="left")
        ax.set_xticks(LAYERS)
        ax.set_xticklabels([str(l) for l in LAYERS], fontsize=6)
        ax.grid(color=GRIDLINE, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7)
    for ax in axes_flat[len(attrs):]:
        ax.axis("off")
    fig.suptitle("Per-attribute L1-L12 AUPRC curves", fontsize=12, color=INK_PRIMARY, x=0.02, ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(FIG_DIR, "B_per_attribute_curves.png")
    plt.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_c_group_curves(by_attr_layer):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
    for (group, attrs), color in zip(ATTRIBUTE_GROUPS.items(), colors):
        y = []
        for l in LAYERS:
            vals = [by_attr_layer[a][l] for a in attrs if a in by_attr_layer and l in by_attr_layer[a]]
            y.append(np.mean(vals) if vals else np.nan)
        ax.plot(LAYERS, y, "o-", color=color, linewidth=2, markersize=6, label=group)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Layer", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("mean AUPRC", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Attribute-group mean AUPRC curves", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "C_attribute_group_curves.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_d_permutation_gap(by_attr_layer):
    perm_rows = read_csv_dicts(PERM_CSV)
    perm_by = {(r["attribute"], int(r["layer"])): float(r["mean"]) for r in perm_rows
               if r["metric"] == "auprc" and r["condition"] == "all_objects"}
    attrs = [a for a in ATTRIBUTES if a in by_attr_layer]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(attrs))))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y_positions = np.arange(len(attrs))[::-1]
    for attr, y0 in zip(attrs, y_positions):
        for l in LAYERS:
            real = by_attr_layer[attr].get(l)
            perm = perm_by.get((attr, l))
            if real is None or perm is None:
                continue
            x_pos = y0 + (l - 6.5) * 0.06
            ax.plot([perm, real], [x_pos, x_pos], color="#2a78d6", alpha=0.4, linewidth=1)
        ax.plot([], [])
    # simpler: just show mean gap per attribute
    ax.clear()
    ax.set_facecolor(SURFACE)
    gaps = []
    for attr in attrs:
        g = [by_attr_layer[attr][l] - perm_by.get((attr, l), np.nan) for l in LAYERS if l in by_attr_layer[attr]]
        gaps.append(np.nanmean(g))
    ax.barh(attrs, gaps, color="#2a78d6")
    ax.set_xlabel("mean(real AUPRC - permutation AUPRC), averaged over L1-L12", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Real vs. permutation-baseline gap, by attribute", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "D_permutation_gap.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_e_emotion_conditional(by_attr_layer):
    rows = read_csv_dicts(SUMMARY_CSV)
    by_cond = {"all_objects": {}, "conditional_on_face": {}}
    for r in rows:
        if r.get("attribute") != "Emotion" or r.get("metric") != "auprc" or r["mean"] == "":
            continue
        if r["condition"] in by_cond:
            by_cond[r["condition"]][int(r["layer"])] = float(r["mean"])
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for cond, style, color in [("all_objects", "-o", "#2a78d6"), ("conditional_on_face", "--s", "#eb6834")]:
        y = [by_cond[cond].get(l, np.nan) for l in LAYERS]
        ax.plot(LAYERS, y, style, color=color, label=cond, linewidth=2, markersize=6)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Layer", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("AUPRC", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Emotion: all-objects vs. Face=1-conditional, L1-L12", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "E_emotion_conditional.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_f_peak_distribution():
    rows = read_csv_dicts(PEAKS_CSV)
    peaks = [int(r["peak_layer"]) for r in rows if r["condition"] == "all_objects"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(peaks, bins=np.arange(0.5, 13.5, 1), color="#2a78d6", edgecolor=SURFACE)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Peak layer", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Number of attributes", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Peak-layer distribution across 12 semantic attributes", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "F_peak_layer_distribution.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 65)
    print("  Phase 1 (all-12-layer): Figures")
    print("=" * 65)
    if not os.path.isfile(SUMMARY_CSV):
        raise RuntimeError(f"STOP: {SUMMARY_CSV} not found -- run the probe script first")

    by_attr_layer = load_auprc_matrix()
    print("\n--- A. Heatmap ---")
    figure_a_heatmap(by_attr_layer)
    print("\n--- B. Per-attribute curves ---")
    figure_b_small_multiples(by_attr_layer)
    print("\n--- C. Attribute-group curves ---")
    figure_c_group_curves(by_attr_layer)
    print("\n--- D. Permutation gap ---")
    figure_d_permutation_gap(by_attr_layer)
    print("\n--- E. Emotion conditional ---")
    figure_e_emotion_conditional(by_attr_layer)
    print("\n--- F. Peak-layer distribution ---")
    figure_f_peak_distribution()

    print("\n" + "=" * 65)
    print("  Figures: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
