"""
Figures for the OSIE text-alignment pilot:

  A. Equal-ensemble Label Alignment Margin, attribute x layer.
  B. Per-prompt median margin (prompt_variant_stability.csv), attribute x layer.
  C. Prompt-count (K) stability: mean margin vs. K=1/2/4/8/16, all 12 attributes,
     one small-multiple panel per attribute, one line per layer.
  D. Linear-probe AUPRC, attribute x layer, with each attribute's random-AUPRC
     baseline marked.
  E. Scatter: text-alignment margin (equal ensemble) vs. probe AUPRC, one
     panel per layer, one point per attribute -- the figure that visually
     separates "text-aligned" from "probe-only-decodable" attributes.
  F. Representative-image panels (original | OSIE mask | L4/L8/L12 CLIP-vs-text
     similarity maps) for 2-3 images containing Face or Text.

Uses the dataviz-skill validated palette (blue/orange/aqua for L4/L8/L12,
consistent with the earlier attribute-grounding pilot's figures).

Reads only the CSVs produced by analyze_osie_text_alignment.py /
probe_osie_text_alignment.py, plus a small re-computation of the patch
similarity maps for the 2-3 representative images (cheap; the full patch
grid was deliberately not cached to keep disk usage small). Writes only
under outputs/osie_text_alignment_pilot/figures/.

Usage (PowerShell):
    python scripts\\plot_osie_text_alignment.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clip_extractor import CLIP_PATCH_SIZE, load_clip_vit_b16, load_osie_image_tensor
from lib.clip_vit import interpolate_patch_pos_embed, patch_grid_from_image_hw
from lib.clip_vit_hidden import project_and_normalize_tokens, visual_forward_with_hidden_states

# ======================= CONFIG =======================
OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot"
FIG_DIR = os.path.join(OUT_DIR, "figures")

SUMMARY_CSV = os.path.join(OUT_DIR, "summary_by_attribute_layer.csv")
VARIANT_CSV = os.path.join(OUT_DIR, "prompt_variant_stability.csv")
K_CSV = os.path.join(OUT_DIR, "prompt_k_stability.csv")
PROBE_SUMMARY_CSV = os.path.join(OUT_DIR, "probe_summary.csv")
TEXT_EMBEDDINGS_NPZ = os.path.join(OUT_DIR, "text_embeddings.npz")

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

IMG_W, IMG_H = 800, 600
LAYERS_DISPLAY = [4, 8, 12]
LAYERS_ZERO_BASED = [3, 7, 11]
K_VALUES = [1, 2, 4, 8, 16]

COLOR_L4, COLOR_L8, COLOR_L12 = "#2a78d6", "#eb6834", "#1baf7a"
LAYER_COLORS = {4: COLOR_L4, 8: COLOR_L8, 12: COLOR_L12}
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

ATTRIBUTE_ORDER = [
    "Text", "Face", "Emotion", "Sound", "Smell", "Taste", "Touch",
    "Motion", "Operability", "Watchability", "Touched", "Gazed",
]
REPRESENTATIVE_CANDIDATES = ["1023", "1077", "1668"]  # contain Face and/or Text
# ======================================================


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def forest_plot(by_attr_layer, value_key, err_lo_key, err_hi_key, title, xlabel, out_basename,
                 ref_line=0.0, ref_label="0"):
    attrs = sorted(by_attr_layer.keys(), key=lambda a: by_attr_layer[a].get(12, {}).get(value_key, 0), reverse=True)
    fig_h = max(3.5, 0.5 * len(attrs) + 1.6)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y_positions = np.arange(len(attrs))[::-1]
    offsets = {4: 0.22, 8: 0.0, 12: -0.22}

    for attr, y in zip(attrs, y_positions):
        for layer in LAYERS_DISPLAY:
            row = by_attr_layer[attr].get(layer)
            if row is None:
                continue
            val = row[value_key]
            lo = row.get(err_lo_key)
            hi = row.get(err_hi_key)
            yy = y + offsets[layer]
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [yy, yy], color=LAYER_COLORS[layer], linewidth=1.8, alpha=0.85, zorder=2)
            ax.plot(val, yy, "o", color=LAYER_COLORS[layer], markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=3)

    ax.axvline(ref_line, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(ref_line, len(attrs) - 0.4, f"  {ref_label}", color=INK_MUTED, fontsize=8, va="bottom", ha="left")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(attrs, color=INK_PRIMARY, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    handles = [plt.Line2D([0], [0], marker="o", color=LAYER_COLORS[l], linestyle="", markersize=7, label=f"L{l}")
               for l in LAYERS_DISPLAY]
    ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK_PRIMARY, fontsize=9)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, f"{out_basename}.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_a_equal_ensemble_margin():
    rows = read_csv_dicts(SUMMARY_CSV)
    by_attr_layer = {}
    for r in rows:
        if r["aggregation"] != "equal_ensemble":
            continue
        n = int(r["n_objects"])
        std = float(r["std_margin"])
        sem = std / max(n, 1) ** 0.5
        mean = float(r["mean_margin"])
        by_attr_layer.setdefault(r["attribute"], {})[int(r["layer"])] = {
            "mean_margin": mean, "lo": mean - sem, "hi": mean + sem,
        }
    forest_plot(by_attr_layer, "mean_margin", "lo", "hi",
                "Equal-ensemble Label Alignment Margin, attribute x layer\n"
                "OSIE text-alignment pilot (n=50 images); error bars = +/- 1 SEM",
                "alignment_margin (positive_sim - mean(negative_sim))",
                "A_equal_ensemble_alignment_margin", ref_line=0.0, ref_label="no margin (=0)")


def figure_b_prompt_median():
    rows = read_csv_dicts(VARIANT_CSV)
    by_attr_layer = {}
    for r in rows:
        by_attr_layer.setdefault(r["attribute"], {})[int(r["layer"])] = {
            "mean_margin": float(r["median_margin_across_templates"]),
            "lo": float(r["min_margin_across_templates"]),
            "hi": float(r["max_margin_across_templates"]),
        }
    forest_plot(by_attr_layer, "mean_margin", "lo", "hi",
                "Per-prompt median Label Alignment Margin, attribute x layer\n"
                "median across the 16 individual prompt templates; range = min-max across templates",
                "median alignment_margin across 16 individual prompts",
                "B_prompt_median_alignment_margin", ref_line=0.0, ref_label="no margin (=0)")


def figure_c_k_stability():
    rows = read_csv_dicts(K_CSV)
    by_attr = {}
    for r in rows:
        by_attr.setdefault(r["attribute"], {}).setdefault(int(r["layer"]), {})[int(r["K"])] = r

    attrs = [a for a in ATTRIBUTE_ORDER if a in by_attr]
    n_cols = 3
    n_rows = int(np.ceil(len(attrs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes_flat = axes.flatten()

    for ax, attr in zip(axes_flat, attrs):
        ax.set_facecolor(SURFACE)
        for layer in LAYERS_DISPLAY:
            means, los, his = [], [], []
            for K in K_VALUES:
                row = by_attr[attr][layer][K]
                means.append(float(row["mean_margin"]))
                los.append(float(row["ci95_lo"]))
                his.append(float(row["ci95_hi"]))
            x = np.arange(len(K_VALUES))
            ax.fill_between(x, los, his, color=LAYER_COLORS[layer], alpha=0.15, linewidth=0)
            ax.plot(x, means, "o-", color=LAYER_COLORS[layer], markersize=4, linewidth=1.5)
        ax.axhline(0.0, color=BASELINE, linewidth=1.0, linestyle="--")
        ax.set_xticks(np.arange(len(K_VALUES)))
        ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=8, color=INK_SECONDARY)
        ax.set_title(attr, fontsize=10, color=INK_PRIMARY, loc="left")
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=7, colors=INK_SECONDARY)

    for ax in axes_flat[len(attrs):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color=LAYER_COLORS[l], marker="o", markersize=5, label=f"L{l}")
               for l in LAYERS_DISPLAY]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Prompt-count (K) stability of the equal-ensemble alignment margin\n"
                 "shaded band = 95% CI across seed=42 K-subsets (exhaustive if C(16,K)<=100, else 100 draws)",
                 fontsize=11, color=INK_PRIMARY, x=0.02, ha="left")
    fig.supxlabel("K (number of prompt templates in the ensemble)", fontsize=9, color=INK_SECONDARY)
    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    out_png = os.path.join(FIG_DIR, "C_prompt_k_stability.png")
    plt.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_d_probe_auprc():
    rows = read_csv_dicts(PROBE_SUMMARY_CSV)
    by_attr_layer = {}
    random_by_attr = {}
    for r in rows:
        if r.get("metric") != "auprc" or r.get("condition") != "all_objects":
            continue
        mean = float(r["mean"])
        std = float(r["std"])
        n = int(float(r["n_folds_valid"])) if r["n_folds_valid"] else 1
        sem = std / max(n, 1) ** 0.5
        by_attr_layer.setdefault(r["attribute"], {})[int(r["layer"])] = {
            "mean_margin": mean, "lo": mean - sem, "hi": mean + sem,
        }
        random_by_attr[r["attribute"]] = float(r["random_auprc"])

    attrs = sorted(by_attr_layer.keys(), key=lambda a: by_attr_layer[a].get(12, {}).get("mean_margin", 0), reverse=True)
    fig_h = max(3.5, 0.5 * len(attrs) + 1.6)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y_positions = np.arange(len(attrs))[::-1]
    offsets = {4: 0.22, 8: 0.0, 12: -0.22}
    for attr, y in zip(attrs, y_positions):
        ax.plot([random_by_attr[attr], random_by_attr[attr]], [y - 0.35, y + 0.35],
                 color=INK_MUTED, linewidth=1.5, linestyle=":", zorder=1)
        for layer in LAYERS_DISPLAY:
            row = by_attr_layer[attr].get(layer)
            if row is None:
                continue
            yy = y + offsets[layer]
            ax.plot([row["lo"], row["hi"]], [yy, yy], color=LAYER_COLORS[layer], linewidth=1.8, alpha=0.85, zorder=2)
            ax.plot(row["mean_margin"], yy, "o", color=LAYER_COLORS[layer], markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(attrs, color=INK_PRIMARY, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_SECONDARY)
    ax.set_xlabel("AUPRC (5-fold, image-grouped CV, mean +/- 1 SEM); dotted = random-baseline AUPRC per attribute",
                  color=INK_SECONDARY, fontsize=9)
    ax.set_title("Linear-probe AUPRC (raw hidden states, no text/projection), attribute x layer",
                 color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    handles = [plt.Line2D([0], [0], marker="o", color=LAYER_COLORS[l], linestyle="", markersize=7, label=f"L{l}")
               for l in LAYERS_DISPLAY]
    ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK_PRIMARY, fontsize=9)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "D_probe_auprc.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_e_scatter_text_vs_probe():
    margin_rows = read_csv_dicts(SUMMARY_CSV)
    probe_rows = read_csv_dicts(PROBE_SUMMARY_CSV)

    margin_by = {}
    for r in margin_rows:
        if r["aggregation"] == "equal_ensemble":
            margin_by[(r["attribute"], int(r["layer"]))] = float(r["mean_margin"])
    auprc_by = {}
    for r in probe_rows:
        if r.get("metric") == "auprc" and r.get("condition") == "all_objects":
            auprc_by[(r["attribute"], int(r["layer"]))] = float(r["mean"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, layer in zip(axes, LAYERS_DISPLAY):
        ax.set_facecolor(SURFACE)
        for attr in ATTRIBUTE_ORDER:
            x = margin_by.get((attr, layer))
            y = auprc_by.get((attr, layer))
            if x is None or y is None:
                continue
            ax.scatter(x, y, color=LAYER_COLORS[layer], s=45, zorder=3, edgecolor=SURFACE, linewidth=0.5)
            ax.annotate(attr, (x, y), fontsize=7.5, color=INK_SECONDARY,
                        xytext=(4, 3), textcoords="offset points")
        ax.axvline(0.0, color=BASELINE, linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(f"L{layer}", color=INK_PRIMARY, fontsize=12)
        ax.set_xlabel("text-alignment margin (equal ensemble)", color=INK_SECONDARY, fontsize=8.5)
        ax.grid(color=GRIDLINE, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    axes[0].set_ylabel("linear-probe AUPRC (raw features)", color=INK_SECONDARY, fontsize=9)
    fig.suptitle("Text alignment (x) vs. linear-probe decodability (y), per attribute\n"
                 "top-left region = information present but not text-aligned (Implementation A low / B high)",
                 fontsize=11, color=INK_PRIMARY, x=0.02, ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    out_png = os.path.join(FIG_DIR, "E_text_alignment_vs_probe_scatter.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def h5_char_str(f, ref):
    return "".join(chr(int(c)) for c in f[ref][()].flatten())


def load_attrs_index(f):
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw = h5_char_str(f, g["img"].id) if False else "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[os.path.splitext(raw)[0]] = g
    return attr_names, index


def union_mask_for_attribute(f, group, mat_idx, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    mask = np.zeros((h, w), dtype=bool)
    any_pos = False
    for j in range(objs_ds.shape[1]):
        obj_g = f[objs_ds[0, j]]
        feat = obj_g["features"][()].flatten()
        if feat[mat_idx] > 0:
            mask |= obj_g["map"][()].T.astype(bool)
            any_pos = True
    return mask if any_pos else None


def figure_f_representative_images():
    if not os.path.isfile(TEXT_EMBEDDINGS_NPZ):
        print("  Skipping F: text_embeddings.npz not found")
        return
    text_npz = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=False)
    attribute_order = list(text_npz["attribute_order"])
    equal_ensemble = {a: text_npz["equal_ensemble_vectors"][i] for i, a in enumerate(attribute_order)}

    f = h5py.File(ATTRS_PATH, "r")
    mat_names, attrs_index = load_attrs_index(f)
    mat_to_task = {"text": "Text", "face": "Face", "emotion": "Emotion", "sound": "Sound",
                   "smell": "Smell", "taste": "Taste", "touch": "Touch", "motion": "Motion",
                   "operability": "Operability", "watchability": "Watchability",
                   "touched": "Touched", "gazed": "Gazed"}

    model, _ = load_clip_vit_b16()
    device = next(model.parameters()).device

    for stem in REPRESENTATIVE_CANDIDATES:
        if stem not in attrs_index:
            continue
        group = attrs_index[stem]
        face_idx = mat_names.index("face")
        text_idx = mat_names.index("text")
        target_attr, target_idx = ("Face", face_idx)
        mask = union_mask_for_attribute(f, group, face_idx, (IMG_H, IMG_W))
        if mask is None:
            mask = union_mask_for_attribute(f, group, text_idx, (IMG_H, IMG_W))
            target_attr, target_idx = ("Text", text_idx)
        if mask is None:
            continue

        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        tensor, orig_hw, pad_hw = load_osie_image_tensor(img_path, CLIP_PATCH_SIZE)
        tensor = tensor.to(device)
        grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
        pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)
        with torch.inference_mode():
            hidden_states, _ = visual_forward_with_hidden_states(
                model.visual, tensor.type(model.dtype), pos_embed=pos_interp,
                layers_zero_based=LAYERS_ZERO_BASED)

        gh, gw = grid_hw
        img = np.array(Image.open(img_path).convert("RGB"))

        fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
        fig.patch.set_facecolor(SURFACE)
        axes[0].imshow(img)
        axes[0].set_title(f"{stem}: original", color=INK_PRIMARY, fontsize=10)
        axes[1].imshow(img)
        axes[1].imshow(np.where(mask, 1.0, np.nan), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title(f"OSIE {target_attr} mask", color=INK_PRIMARY, fontsize=10)

        sim_maps = []
        for l_disp, l_zero in zip(LAYERS_DISPLAY, LAYERS_ZERO_BASED):
            with torch.inference_mode():
                proj = project_and_normalize_tokens(model.visual, hidden_states[l_zero][0, 1:])
            sim = (proj.detach().cpu().numpy() @ equal_ensemble[target_attr]).reshape(gh, gw)
            sim_up = np.array(Image.fromarray(sim.astype(np.float32)).resize((IMG_W, IMG_H), Image.BILINEAR))
            sim_maps.append(sim_up)

        vmin, vmax = min(m.min() for m in sim_maps), max(m.max() for m in sim_maps)
        for l_idx, l_disp in enumerate(LAYERS_DISPLAY):
            ax = axes[2 + l_idx]
            ax.imshow(img)
            ax.imshow(sim_maps[l_idx], cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
            ax.set_title(f"L{l_disp} sim. to \"{target_attr}\" text", color=INK_PRIMARY, fontsize=10)
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        out_png = os.path.join(FIG_DIR, f"F_representative_{stem}_{target_attr}.png")
        plt.savefig(out_png, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        print(f"  Saved: {out_png}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 65)
    print("  OSIE Text-Alignment Pilot: Figures")
    print("=" * 65)

    print("\n--- A. Equal-ensemble alignment margin ---")
    figure_a_equal_ensemble_margin()
    print("\n--- B. Per-prompt median margin ---")
    figure_b_prompt_median()
    print("\n--- C. Prompt-count (K) stability ---")
    figure_c_k_stability()
    print("\n--- D. Probe AUPRC ---")
    figure_d_probe_auprc()
    print("\n--- E. Text alignment vs. probe scatter ---")
    figure_e_scatter_text_vs_probe()
    print("\n--- F. Representative images (Face/Text) ---")
    figure_f_representative_images()

    print("\n" + "=" * 65)
    print("  Figures: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
