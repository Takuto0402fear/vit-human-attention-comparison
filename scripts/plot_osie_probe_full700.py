"""
Figures for the Phase 4 (700-image) linear probe extension:
  A. 12 attributes x L4/L8/L12 AUPRC (all_objects condition).
  B. Same, AUROC.
  C. Real AUPRC vs. permutation-baseline AUPRC (the gap that matters).
  D. L4 -> L8 -> L12 AUPRC trajectory per attribute (paired lines).
  E. Emotion: all_objects vs. conditional_on_face (Face=1 only).
  F. Attribute-group layer profile (same groups as the crop pilot: Face+Text,
     Emotion, Gazed+Touched, sensory/functional).

Reads only outputs/osie_probe_full700/*.csv. Writes only under
outputs/osie_probe_full700/figures/.

Usage (PowerShell):
    python scripts\\plot_osie_probe_full700.py
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
OUT_DIR = r"C:\Users\user\gaze\outputs\osie_probe_full700"
FIG_DIR = os.path.join(OUT_DIR, "figures")
SUMMARY_CSV = os.path.join(OUT_DIR, "probe_summary.csv")
PERM_CSV = os.path.join(OUT_DIR, "probe_permutation_baseline.csv")
LAYER_DIFF_CSV = os.path.join(OUT_DIR, "probe_layer_differences.csv")

LAYERS_DISPLAY = [4, 8, 12]
COLOR_L4, COLOR_L8, COLOR_L12 = "#2a78d6", "#eb6834", "#1baf7a"
LAYER_COLORS = {4: COLOR_L4, 8: COLOR_L8, 12: COLOR_L12}
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

ATTRIBUTE_GROUPS = {
    "direct_visual": ["Face", "Text"],
    "emotion": ["Emotion"],
    "relational": ["Gazed", "Touched"],
    "sensory_functional": ["Motion", "Sound", "Smell", "Taste", "Touch", "Operability", "Watchability"],
}
# ======================================================


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def forest_plot(by_attr_layer, out_name, title, xlabel, xlim=None):
    attrs = sorted(by_attr_layer.keys(), key=lambda a: by_attr_layer[a].get(12, {}).get("mean", 0), reverse=True)
    fig_h = max(3.5, 0.5 * len(attrs) + 1.8)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y_positions = np.arange(len(attrs))[::-1]
    offsets = {4: 0.22, 8: 0.0, 12: -0.22}
    for attr, y in zip(attrs, y_positions):
        for layer in LAYERS_DISPLAY:
            r = by_attr_layer[attr].get(layer)
            if not r:
                continue
            mean, lo, hi, rand = r["mean"], r["lo"], r["hi"], r.get("random")
            yy = y + offsets[layer]
            ax.plot([lo, hi], [yy, yy], color=LAYER_COLORS[layer], linewidth=1.8, alpha=0.85, zorder=2)
            ax.plot(mean, yy, "o", color=LAYER_COLORS[layer], markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=3)
        rand = by_attr_layer[attr].get(12, {}).get("random")
        if rand is not None:
            ax.plot([rand, rand], [y - 0.35, y + 0.35], color=INK_MUTED, linewidth=1.5, linestyle=":", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(attrs, color=INK_PRIMARY, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    handles = [plt.Line2D([0], [0], marker="o", color=LAYER_COLORS[l], linestyle="", markersize=7, label=f"L{l}")
               for l in LAYERS_DISPLAY]
    ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK_PRIMARY, fontsize=9)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, out_name)
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_ab(metric, out_name, title):
    rows = read_csv_dicts(SUMMARY_CSV)
    by_attr_layer = {}
    for r in rows:
        if r.get("metric") != metric or r.get("condition") != "all_objects" or r["mean"] == "":
            continue
        by_attr_layer.setdefault(r["attribute"], {})[int(r["layer"])] = {
            "mean": float(r["mean"]), "lo": float(r["mean"]) - float(r["std"] or 0),
            "hi": float(r["mean"]) + float(r["std"] or 0), "random": float(r["random_auprc"]),
        }
    forest_plot(by_attr_layer, out_name, title, f"{metric} (mean +/- 1 SD across folds); dotted = random baseline")


def figure_c():
    summary = read_csv_dicts(SUMMARY_CSV)
    perm = read_csv_dicts(PERM_CSV)
    real = {(r["attribute"], r["layer"]): float(r["mean"]) for r in summary
            if r.get("metric") == "auprc" and r.get("condition") == "all_objects" and r["mean"] != ""}
    perm_d = {(r["attribute"], r["layer"]): float(r["mean"]) for r in perm
              if r["metric"] == "auprc" and r["condition"] == "all_objects"}
    attrs = sorted({k[0] for k in real}, key=lambda a: real.get((a, "12"), 0), reverse=True)

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(attrs) + 1.8)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y_positions = np.arange(len(attrs))[::-1]
    offsets = {"4": 0.22, "8": 0.0, "12": -0.22}
    for attr, y in zip(attrs, y_positions):
        for layer in ["4", "8", "12"]:
            r_val = real.get((attr, layer))
            p_val = perm_d.get((attr, layer))
            if r_val is None or p_val is None:
                continue
            yy = y + offsets[layer]
            gap = r_val - p_val
            color = LAYER_COLORS[int(layer)]
            ax.plot([p_val, r_val], [yy, yy], color=color, linewidth=2.5, alpha=0.6, zorder=2)
            ax.plot(p_val, yy, "x", color=INK_MUTED, markersize=5, zorder=3)
            ax.plot(r_val, yy, "o", color=color, markersize=6, markeredgecolor=SURFACE, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(attrs, color=INK_PRIMARY, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("permutation baseline (x) -> real AUPRC (o), by layer", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Real AUPRC vs. permutation-shuffled-label baseline, all_objects",
                 color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    handles = [plt.Line2D([0], [0], marker="o", color=LAYER_COLORS[l], linestyle="", markersize=7, label=f"L{l} real")
               for l in LAYERS_DISPLAY] + [plt.Line2D([0], [0], marker="x", color=INK_MUTED, linestyle="", markersize=6, label="permutation")]
    ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK_PRIMARY, fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "C_real_vs_permutation_auprc.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_d():
    rows = read_csv_dicts(SUMMARY_CSV)
    by_attr = {}
    for r in rows:
        if r.get("metric") != "auprc" or r.get("condition") != "all_objects" or r["mean"] == "":
            continue
        by_attr.setdefault(r["attribute"], {})[int(r["layer"])] = float(r["mean"])

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for attr, vals in by_attr.items():
        y = [vals.get(l, np.nan) for l in LAYERS_DISPLAY]
        ax.plot(LAYERS_DISPLAY, y, "o-", markersize=5, linewidth=1.3, alpha=0.8)
        ax.annotate(attr, (LAYERS_DISPLAY[-1], y[-1]), fontsize=7.5, color=INK_SECONDARY,
                    xytext=(5, 0), textcoords="offset points")
    ax.set_xticks(LAYERS_DISPLAY)
    ax.set_xticklabels([f"L{l}" for l in LAYERS_DISPLAY])
    ax.set_ylabel("AUPRC", color=INK_SECONDARY, fontsize=9)
    ax.set_title("L4 -> L8 -> L12 AUPRC trajectory, all 12 attributes",
                 color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "D_layer_trajectory.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_e():
    rows = read_csv_dicts(SUMMARY_CSV)
    conditions = ["all_objects", "conditional_on_face"]
    by_cond = {c: {} for c in conditions}
    for r in rows:
        if r.get("attribute") != "Emotion" or r.get("metric") != "auprc" or r["mean"] == "":
            continue
        if r["condition"] in conditions:
            by_cond[r["condition"]][int(r["layer"])] = (float(r["mean"]), float(r["random_auprc"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for cond, style in zip(conditions, ["-o", "--s"]):
        y = [by_cond[cond].get(l, (np.nan, np.nan))[0] for l in LAYERS_DISPLAY]
        ax.plot(LAYERS_DISPLAY, y, style, label=cond, linewidth=1.8, markersize=7)
    rand_line = [by_cond["all_objects"].get(l, (np.nan, np.nan))[1] for l in LAYERS_DISPLAY]
    ax.plot(LAYERS_DISPLAY, rand_line, ":", color=INK_MUTED, label="random (all_objects)")
    rand_cond_line = [by_cond["conditional_on_face"].get(l, (np.nan, np.nan))[1] for l in LAYERS_DISPLAY]
    ax.plot(LAYERS_DISPLAY, rand_cond_line, ":", color="#e34948", label="random (conditional_on_face)")
    ax.set_xticks(LAYERS_DISPLAY)
    ax.set_xticklabels([f"L{l}" for l in LAYERS_DISPLAY])
    ax.set_ylabel("AUPRC", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Emotion: all objects vs. Face=1-conditional probe",
                 color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "E_emotion_conditional.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_f():
    rows = read_csv_dicts(SUMMARY_CSV)
    by_attr_layer = {}
    for r in rows:
        if r.get("metric") != "auprc" or r.get("condition") != "all_objects" or r["mean"] == "":
            continue
        by_attr_layer.setdefault(r["attribute"], {})[int(r["layer"])] = float(r["mean"])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
    for (group, attrs), color in zip(ATTRIBUTE_GROUPS.items(), colors):
        y_by_layer = []
        for l in LAYERS_DISPLAY:
            vals = [by_attr_layer[a][l] for a in attrs if a in by_attr_layer and l in by_attr_layer[a]]
            y_by_layer.append(np.mean(vals) if vals else np.nan)
        ax.plot(LAYERS_DISPLAY, y_by_layer, "o-", color=color, linewidth=2, markersize=7, label=group)
    ax.set_xticks(LAYERS_DISPLAY)
    ax.set_xticklabels([f"L{l}" for l in LAYERS_DISPLAY])
    ax.set_ylabel("mean AUPRC across attributes in group", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Attribute-group layer profile (probe AUPRC)", color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "F_attribute_group_profile.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 65)
    print("  Phase 4 (full700 probe): Figures")
    print("=" * 65)
    print("\n--- A. AUPRC by attribute x layer ---")
    figure_ab("auprc", "A_auprc_by_attribute_layer.png", "Linear-probe AUPRC (raw features), attribute x layer, ALL 700 images")
    print("\n--- B. AUROC by attribute x layer ---")
    figure_ab("auroc", "B_auroc_by_attribute_layer.png", "Linear-probe AUROC (raw features), attribute x layer, ALL 700 images")
    print("\n--- C. Real vs. permutation ---")
    figure_c()
    print("\n--- D. Layer trajectory ---")
    figure_d()
    print("\n--- E. Emotion conditional ---")
    figure_e()
    print("\n--- F. Attribute-group profile ---")
    figure_f()
    print("\n" + "=" * 65)
    print("  Figures: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
