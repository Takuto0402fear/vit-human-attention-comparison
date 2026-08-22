"""
Figures for the Phase 1 crop pilot:
  A. Equal-ensemble alignment margin, attribute x crop_condition (image-level
     bootstrap 95% CI).
  B. Attribute-group summary (direct_visual / emotion / relational /
     sensory_functional), crop_condition comparison.
  C. Representative crop examples (2-3 objects): original image with bbox,
     tight_crop, context20_crop, masked_context20 -- actual images, not
     cherry-picked for good scores (deterministic selection rule recorded
     in the figure and report.md).

Reads only outputs/osie_text_alignment_crop_pilot/*.csv (+ a small
re-generation of a few crop thumbnails for panel C). Writes only under
outputs/osie_text_alignment_crop_pilot/figures/.

Usage (PowerShell):
    python scripts\\plot_osie_text_alignment_crop_pilot.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import os

import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.osie_text_alignment import bbox_from_mask, expand_bbox, pad_to_square

# ======================= CONFIG =======================
OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_crop_pilot"
FIG_DIR = os.path.join(OUT_DIR, "figures")
ATTRIBUTE_SUMMARY_CSV = os.path.join(OUT_DIR, "crop_attribute_summary.csv")
GROUP_SUMMARY_CSV = os.path.join(OUT_DIR, "crop_group_summary.csv")

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")
IMG_W, IMG_H = 800, 600

CROP_CONDITIONS = ["global_image", "tight_crop", "context20_crop", "masked_context20"]
COND_COLORS = {"global_image": "#898781", "tight_crop": "#2a78d6",
               "context20_crop": "#eb6834", "masked_context20": "#1baf7a"}
INK_PRIMARY, INK_SECONDARY, GRIDLINE, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Deterministic pick for panel C: first image, its first object with a
# positive Face or Text label -- NOT chosen for a favorable score.
REPRESENTATIVE_IMAGE_IDS = ["1055", "1073", "1249"]
# ======================================================


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def figure_a():
    rows = read_csv_dicts(ATTRIBUTE_SUMMARY_CSV)
    by_attr = {}
    for r in rows:
        by_attr.setdefault(r["attribute"], {})[r["crop_condition"]] = r
    attrs = sorted(by_attr.keys(), key=lambda a: float(by_attr[a].get("tight_crop", {}).get("mean_margin", 0)), reverse=True)

    fig_h = max(4, 0.55 * len(attrs) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y_positions = np.arange(len(attrs))[::-1]
    offsets = {c: (1.5 - i) * 0.18 for i, c in enumerate(CROP_CONDITIONS)}

    for attr, y in zip(attrs, y_positions):
        for cond in CROP_CONDITIONS:
            r = by_attr[attr].get(cond)
            if not r:
                continue
            mean, lo, hi = float(r["mean_margin"]), float(r["ci95_lo"]), float(r["ci95_hi"])
            yy = y + offsets[cond]
            ax.plot([lo, hi], [yy, yy], color=COND_COLORS[cond], linewidth=1.8, alpha=0.85, zorder=2)
            ax.plot(mean, yy, "o", color=COND_COLORS[cond], markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=0.5, zorder=3)

    ax.axvline(0.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(attrs, color=INK_PRIMARY, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_SECONDARY)
    ax.set_xlabel("equal-ensemble alignment margin (official model.encode_image; image-level bootstrap 95% CI)",
                  color=INK_SECONDARY, fontsize=9)
    ax.set_title("Crop pilot: Label Alignment Margin, attribute x crop condition",
                 color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    handles = [plt.Line2D([0], [0], marker="o", color=COND_COLORS[c], linestyle="", markersize=7, label=c)
               for c in CROP_CONDITIONS]
    ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK_PRIMARY, fontsize=8.5)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "A_crop_alignment_margin.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def figure_b():
    rows = read_csv_dicts(GROUP_SUMMARY_CSV)
    groups = sorted({r["attribute_group"] for r in rows})
    by_group = {}
    for r in rows:
        by_group.setdefault(r["attribute_group"], {})[r["crop_condition"]] = r

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y_positions = np.arange(len(groups))[::-1]
    offsets = {c: (1.5 - i) * 0.18 for i, c in enumerate(CROP_CONDITIONS)}
    for group, y in zip(groups, y_positions):
        for cond in CROP_CONDITIONS:
            r = by_group[group].get(cond)
            if not r:
                continue
            mean, lo, hi = float(r["mean_margin"]), float(r["ci95_lo"]), float(r["ci95_hi"])
            yy = y + offsets[cond]
            ax.plot([lo, hi], [yy, yy], color=COND_COLORS[cond], linewidth=2.2, alpha=0.85, zorder=2)
            ax.plot(mean, yy, "o", color=COND_COLORS[cond], markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=0.5, zorder=3)
    ax.axvline(0.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(groups, color=INK_PRIMARY, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_SECONDARY)
    ax.set_xlabel("equal-ensemble alignment margin (image-level bootstrap 95% CI)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Crop pilot: attribute-group summary", color=INK_PRIMARY, fontsize=11, loc="left", pad=14)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    handles = [plt.Line2D([0], [0], marker="o", color=COND_COLORS[c], linestyle="", markersize=7, label=c)
               for c in CROP_CONDITIONS]
    ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK_PRIMARY, fontsize=8.5)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "B_crop_group_summary.png")
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
        raw = "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[os.path.splitext(raw)[0]] = g
    return attr_names, index


def figure_c():
    f = h5py.File(ATTRS_PATH, "r")
    _, attrs_index = load_attrs_index(f)

    fig, axes = plt.subplots(len(REPRESENTATIVE_IMAGE_IDS), 4, figsize=(16, 4 * len(REPRESENTATIVE_IMAGE_IDS)))
    for row_idx, stem in enumerate(REPRESENTATIVE_IMAGE_IDS):
        group = attrs_index[stem]
        objs_ds = group["objs"]
        obj_g = f[objs_ds[0, 0]]  # first object, deterministic -- not cherry-picked
        mask = obj_g["map"][()].T.astype(bool)

        img = np.array(Image.open(os.path.join(STIM_DIR, f"{stem}.jpg")).convert("RGB"))
        bbox = bbox_from_mask(mask)
        y0, y1, x0, x1 = bbox
        tight = pad_to_square(img[y0:y1 + 1, x0:x1 + 1])
        cy0, cy1, cx0, cx1 = expand_bbox(bbox, 0.20, IMG_H, IMG_W)
        context = pad_to_square(img[cy0:cy1 + 1, cx0:cx1 + 1])
        region = img[cy0:cy1 + 1, cx0:cx1 + 1].copy()
        region_mask = mask[cy0:cy1 + 1, cx0:cx1 + 1]
        region[~region_mask] = (123, 117, 104)
        masked = pad_to_square(region)

        ax_row = axes[row_idx] if len(REPRESENTATIVE_IMAGE_IDS) > 1 else axes
        img_with_box = img.copy()
        img_with_box[y0:y1 + 1, [x0, min(x1, IMG_W - 1)]] = [255, 0, 0]
        img_with_box[[y0, min(y1, IMG_H - 1)], x0:x1 + 1] = [255, 0, 0]
        ax_row[0].imshow(img_with_box)
        ax_row[0].set_title(f"{stem}: original + bbox", fontsize=9)
        ax_row[1].imshow(tight)
        ax_row[1].set_title("tight_crop", fontsize=9)
        ax_row[2].imshow(context)
        ax_row[2].set_title("context20_crop", fontsize=9)
        ax_row[3].imshow(masked)
        ax_row[3].set_title("masked_context20", fontsize=9)
        for ax in ax_row:
            ax.axis("off")

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "C_representative_crops.png")
    plt.savefig(out_png, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 65)
    print("  Crop pilot: Figures")
    print("=" * 65)
    print("\n--- A. Alignment margin by attribute x condition ---")
    figure_a()
    print("\n--- B. Attribute-group summary ---")
    figure_b()
    print("\n--- C. Representative crop examples ---")
    figure_c()
    print("\n" + "=" * 65)
    print("  Figures: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
