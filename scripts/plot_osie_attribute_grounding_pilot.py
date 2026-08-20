"""
Figures for the OSIE attribute-grounding pilot
(scripts/run_osie_attribute_grounding_pilot.py):

  A. Attribute x layer comparison: area_normalized_enrichment (mean, 95%
     bootstrap CI) for L4 / L8 / L12, one row per category, sorted by the
     L12 mean. Categorical color assignment uses the first 3 slots of the
     dataviz-skill validated 8-hue palette (blue/orange/aqua), which
     validate all-pairs (CVD delta-E and normal-vision delta-E) in both
     light and dark modes without re-ordering -- see
     ~/.claude "dataviz" skill, references/palette.md.
  B. Representative-image panels: original | OSIE foreground mask |
     L4 attention | L8 attention | L12 attention, for a small,
     deterministically-chosen subset of the pilot images (most attributes
     present, ties broken by ascending image id).

Reads only the CSVs / npz cache produced by run_osie_attribute_grounding_pilot.py
and attrs.mat / data/stimuli (read-only). Does not re-run CLIP inference
(uses the cached grid-resolution attention). Does not modify any existing
output directory. Writes only under
outputs/osie_attribute_grounding_pilot/figures/.

Usage (PowerShell):
    python scripts\\plot_osie_attribute_grounding_pilot.py
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

# ======================= CONFIG =======================
DATA_BASE  = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR   = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

OUT_DIR      = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_pilot"
SUMMARY_CSV  = os.path.join(OUT_DIR, "attribute_layer_summary.csv")
PILOT_LIST_CSV = os.path.join(OUT_DIR, "pilot_image_list.csv")
ATTN_CACHE_PATH = os.path.join(OUT_DIR, "attn_cache", "clip_vitb16_pilot_L4L8L12_patchgrid.npz")
FIG_DIR      = os.path.join(OUT_DIR, "figures")

IMG_W, IMG_H = 800, 600
LAYERS_DISPLAY = [4, 8, 12]
N_REPRESENTATIVE = 3

# dataviz-skill validated palette, first 3 categorical slots (all-pairs
# validated in both light/dark modes -- references/palette.md)
COLOR_L4  = "#2a78d6"  # blue
COLOR_L8  = "#eb6834"  # orange
COLOR_L12 = "#1baf7a"  # aqua
LAYER_COLORS = {4: COLOR_L4, 8: COLOR_L8, 12: COLOR_L12}

INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRIDLINE      = "#e1e0d9"
BASELINE      = "#c3c2b7"
SURFACE       = "#fcfcfb"

LOW_SAMPLE_ATTRS = {"sound", "smell"}
ATTR_NAMES = [
    "text", "face", "emotion", "sound", "smell", "taste", "touch",
    "motion", "operability", "watchability", "touched", "gazed",
]
# ======================================================


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_comparison_figure():
    rows = read_csv_dicts(SUMMARY_CSV)
    by_attr = {}
    for r in rows:
        by_attr.setdefault(r["attribute"], {})[int(r["layer"])] = r

    def l12_mean(attr):
        return float(by_attr[attr][12]["mean"])

    attrs_sorted = sorted(by_attr.keys(), key=l12_mean, reverse=True)

    fig_h = max(4.0, 0.42 * len(attrs_sorted) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y_positions = np.arange(len(attrs_sorted))[::-1]
    offsets = {4: 0.22, 8: 0.0, 12: -0.22}

    for attr, y in zip(attrs_sorted, y_positions):
        for layer in LAYERS_DISPLAY:
            r = by_attr[attr][layer]
            mean = float(r["mean"])
            lo = float(r["ci95_lo"])
            hi = float(r["ci95_hi"])
            n = int(r["valid_image_count"])
            yy = y + offsets[layer]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [yy, yy], color=LAYER_COLORS[layer], linewidth=1.5,
                        solid_capstyle="round", alpha=0.9, zorder=2)
            ax.plot(mean, yy, "o", color=LAYER_COLORS[layer],
                    markersize=6 if n >= 10 else 4.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)

    ax.axvline(1.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(1.0, len(attrs_sorted) - 0.3, "  area-proportional (=1)",
            color=INK_MUTED, fontsize=8, va="bottom", ha="left")

    ylabels = []
    for attr in attrs_sorted:
        n_ref = by_attr[attr][LAYERS_DISPLAY[0]]["valid_image_count"]
        tag = " \u2020" if attr in LOW_SAMPLE_ATTRS else ""
        ylabels.append(f"{attr} (n={n_ref}){tag}")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, color=INK_PRIMARY, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_SECONDARY)

    ax.set_xlabel("area_normalized_enrichment  (attention_mass / area_fraction)",
                  color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        "CLIP ViT-B/16: object/attribute-mask attention enrichment by layer\n"
        "OSIE pilot, n=50 images, seed=42  (\u2020 = low-sample, n\u22486, reference only)",
        color=INK_PRIMARY, fontsize=11, loc="left")

    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    handles = [plt.Line2D([0], [0], marker="o", color=LAYER_COLORS[l], linestyle="",
                           markersize=7, label=f"L{l}") for l in LAYERS_DISPLAY]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              labelcolor=INK_PRIMARY, fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "enrichment_by_attribute_L4_L8_L12.png")
    out_pdf = os.path.join(FIG_DIR, "enrichment_by_attribute_L4_L8_L12.pdf")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.savefig(out_pdf, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    return out_png, out_pdf


def h5_char_str(f, ref):
    arr = f[ref][()]
    return "".join(chr(int(c)) for c in arr.flatten())


def load_attrs_index(f):
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        img_arr = g["img"][()]
        raw_name = "".join(chr(int(c)) for c in img_arr.flatten())
        stem = os.path.splitext(raw_name)[0]
        index[stem] = g
    return attr_names, index


def foreground_mask_for_image(f, group, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    fg = np.zeros((h, w), dtype=bool)
    for j in range(objs_ds.shape[1]):
        mp = f[objs_ds[0, j]]["map"][()].T.astype(bool)
        fg |= mp
    return fg


def select_representative_images(pilot_rows, n=N_REPRESENTATIVE):
    def n_attrs_present(row):
        return sum(int(row[f"has_{name}"]) for name in ATTR_NAMES)

    ranked = sorted(pilot_rows, key=lambda r: (-n_attrs_present(r), r["image_id"]))
    return [r["image_id"] for r in ranked[:n]]


def build_representative_panels():
    pilot_rows = read_csv_dicts(PILOT_LIST_CSV)
    rep_ids = select_representative_images(pilot_rows)
    print(f"  Representative images (most attributes present, tie-break by id): {rep_ids}")

    cache = np.load(ATTN_CACHE_PATH)
    stems = list(cache["stems"])
    attn = cache["attn"]  # (N, 3, 38, 50)
    layers_display = list(cache["layers_display"])
    assert layers_display == LAYERS_DISPLAY, (
        f"STOP: cache layer order {layers_display} != {LAYERS_DISPLAY}")

    f = h5py.File(ATTRS_PATH, "r")
    _, attrs_index = load_attrs_index(f)

    saved = []
    for stem in rep_ids:
        if stem not in stems:
            raise RuntimeError(f"STOP: {stem} not found in attention cache")
        idx = stems.index(stem)
        grid_maps = attn[idx]  # (3, 38, 50)

        t = torch.from_numpy(grid_maps).unsqueeze(1).float()
        up = F.interpolate(t, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
        up = up[:, 0].numpy().astype(np.float32)
        for l in range(3):
            s = up[l].sum()
            if s > 0:
                up[l] /= s

        img = np.array(Image.open(os.path.join(STIM_DIR, f"{stem}.jpg")).convert("RGB"))
        fg_mask = foreground_mask_for_image(f, attrs_index[stem], expected_hw=img.shape[:2])

        fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
        fig.patch.set_facecolor(SURFACE)

        axes[0].imshow(img)
        axes[0].set_title(f"{stem}: original", color=INK_PRIMARY, fontsize=10)

        axes[1].imshow(img)
        axes[1].imshow(np.where(fg_mask, 1.0, np.nan), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title(f"OSIE foreground mask (area={fg_mask.mean():.2f})",
                           color=INK_PRIMARY, fontsize=10)

        vmax = up.max()
        for l_idx, layer_display in enumerate(LAYERS_DISPLAY):
            ax = axes[2 + l_idx]
            ax.imshow(img)
            ax.imshow(up[l_idx], cmap="jet", alpha=0.55, vmin=0, vmax=vmax)
            ax.set_title(f"L{layer_display} attention", color=INK_PRIMARY, fontsize=10)

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        out_png = os.path.join(FIG_DIR, f"representative_{stem}.png")
        plt.savefig(out_png, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        print(f"  Saved: {out_png}")
        saved.append(out_png)

    return rep_ids, saved


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 65)
    print("  OSIE Attribute-Grounding Pilot: Figures")
    print("=" * 65)

    if not os.path.isfile(SUMMARY_CSV):
        raise RuntimeError(
            f"STOP: {SUMMARY_CSV} not found -- run "
            "scripts/run_osie_attribute_grounding_pilot.py first")

    print("\n--- A. Attribute x layer enrichment comparison ---")
    cmp_png, cmp_pdf = build_comparison_figure()

    print("\n--- B. Representative-image panels ---")
    rep_ids, rep_pngs = build_representative_panels()

    manifest = {
        "comparison_figure": {"png": cmp_png, "pdf": cmp_pdf},
        "representative_images": rep_ids,
        "representative_figures": rep_pngs,
        "representative_selection_rule":
            "top-N pilot images by count of present attrs.mat attributes "
            "(descending), ties broken by ascending image id",
        "palette": {"L4": COLOR_L4, "L8": COLOR_L8, "L12": COLOR_L12,
                    "source": "dataviz skill references/palette.md, categorical slots 1-3"},
    }
    manifest_path = os.path.join(FIG_DIR, "figures_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n  Saved: {manifest_path}")

    print("\n" + "=" * 65)
    print("  Figures: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
