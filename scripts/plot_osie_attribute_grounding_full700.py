"""
Figures for the full700 OSIE attribute-grounding extension
(scripts/run_osie_attribute_grounding_pilot.py --all-images +
scripts/stats_osie_attribute_grounding_full700.py):

  A. Attribute x layer enrichment comparison (n=700; same style as the
     pilot figure) -- mean, 95% bootstrap CI, for L4/L8/L12.
  B. Paired diff / effect-size figure: L8-L4, L12-L8, L12-L4 mean_diff
     with 95% bootstrap CI (paired, image-level) and rank-biserial effect
     size, one panel per comparison, all 14 categories.
  C. Focused readable comparison: Face / Text / Emotion / Operability /
     Gazed / Touched only (the attributes named in the research brief),
     L4/L8/L12 enrichment + the L12-L4 effect annotated per row.
  D. Representative-image panels (original | OSIE foreground mask |
     L4 | L8 | L12 attention) for 3 images from the full 700, chosen by
     the same deterministic rule as the pilot (most attributes present,
     ties broken by ascending image id).

Uses the dataviz-skill validated categorical palette (first 3 slots:
blue/orange/aqua) and the blue/red diverging pair for signed diffs --
references/palette.md.

Reads only outputs/osie_attribute_grounding_full700/*.csv and the cached
grid-resolution attention (no GPU re-inference). Does not modify the pilot
directory or any existing output. Writes only under
outputs/osie_attribute_grounding_full700/figures/.

Usage (PowerShell):
    python scripts\\plot_osie_attribute_grounding_full700.py
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

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_full700"
SUMMARY_CSV = os.path.join(OUT_DIR, "attribute_layer_summary.csv")
PAIRED_CSV = os.path.join(OUT_DIR, "paired_layer_tests.csv")
IMAGE_LIST_CSV = os.path.join(OUT_DIR, "full700_image_list.csv")
ATTN_CACHE_PATH = os.path.join(OUT_DIR, "attn_cache", "clip_vitb16_full700_L4L8L12_patchgrid.npz")
FIG_DIR = os.path.join(OUT_DIR, "figures")

IMG_W, IMG_H = 800, 600
LAYERS_DISPLAY = [4, 8, 12]
N_REPRESENTATIVE = 3
FOCUS_ATTRS = ["face", "text", "emotion", "operability", "gazed", "touched"]

# dataviz-skill validated palette (references/palette.md)
COLOR_L4  = "#2a78d6"  # blue (categorical slot 1)
COLOR_L8  = "#eb6834"  # orange (slot 2)
COLOR_L12 = "#1baf7a"  # aqua (slot 3)
LAYER_COLORS = {4: COLOR_L4, 8: COLOR_L8, 12: COLOR_L12}
COLOR_POS = "#2a78d6"   # diverging pair: blue = positive diff
COLOR_NEG = "#e34948"   # diverging pair: red = negative diff (slot 8)

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
COMPARISONS = ["L8-L4", "L12-L8", "L12-L4"]
# ======================================================


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_summary():
    rows = read_csv_dicts(SUMMARY_CSV)
    by_attr = {}
    for r in rows:
        by_attr.setdefault(r["attribute"], {})[int(r["layer"])] = r
    return by_attr


def enrichment_forest_plot(by_attr, attrs, out_basename, title_suffix=""):
    def l12_mean(attr):
        return float(by_attr[attr][12]["mean"])
    attrs_sorted = sorted(attrs, key=l12_mean, reverse=True)

    fig_h = max(3.5, 0.55 * len(attrs_sorted) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
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
                ax.plot([lo, hi], [yy, yy], color=LAYER_COLORS[layer], linewidth=1.8,
                        solid_capstyle="round", alpha=0.9, zorder=2)
            ax.plot(mean, yy, "o", color=LAYER_COLORS[layer],
                    markersize=7.5 if n >= 30 else 5.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)

    ax.axvline(1.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(1.0, 1.01, "  area-proportional (=1)", transform=ax.get_xaxis_transform(),
            color=INK_MUTED, fontsize=8, va="bottom", ha="left")

    ylabels = []
    for attr in attrs_sorted:
        n_ref = by_attr[attr][LAYERS_DISPLAY[0]]["valid_image_count"]
        tag = " \u2020" if attr in LOW_SAMPLE_ATTRS else ""
        ylabels.append(f"{attr} (n={n_ref}){tag}")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, color=INK_PRIMARY, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_SECONDARY)

    ax.set_xlabel("area_normalized_enrichment  (attention_mass / area_fraction)",
                  color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        f"CLIP ViT-B/16: object/attribute-mask attention enrichment by layer{title_suffix}\n"
        "OSIE full700  (\u2020 = low-sample attribute, reference only)",
        color=INK_PRIMARY, fontsize=11, loc="left", pad=16)

    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    handles = [plt.Line2D([0], [0], marker="o", color=LAYER_COLORS[l], linestyle="",
                           markersize=7, label=f"L{l}") for l in LAYERS_DISPLAY]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              labelcolor=INK_PRIMARY, fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, f"{out_basename}.png")
    out_pdf = os.path.join(FIG_DIR, f"{out_basename}.pdf")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.savefig(out_pdf, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    return out_png, out_pdf


def paired_diff_figure():
    rows = read_csv_dicts(PAIRED_CSV)
    by_comp = {c: {} for c in COMPARISONS}
    for r in rows:
        by_comp[r["comparison"]][r["attribute"]] = r

    ref_order = sorted(
        by_comp["L12-L4"].keys(),
        key=lambda a: float(by_comp["L12-L4"][a]["mean_diff"]), reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(wspace=0.55, left=0.14, right=0.90, top=0.80, bottom=0.08)
    y_positions = np.arange(len(ref_order))[::-1]

    for ax, comp in zip(axes, COMPARISONS):
        ax.set_facecolor(SURFACE)
        los, his = [], []
        for attr, y in zip(ref_order, y_positions):
            r = by_comp[comp][attr]
            diff = float(r["mean_diff"])
            lo = float(r["ci95_lo_diff"])
            hi = float(r["ci95_hi_diff"])
            p_holm = float(r["p_holm"])
            eff_r = float(r["rank_biserial_r"])
            los.append(lo); his.append(hi)
            color = COLOR_POS if diff >= 0 else COLOR_NEG
            ax.plot([lo, hi], [y, y], color=color, linewidth=2.2, alpha=0.85, zorder=2)
            ax.plot(diff, y, "o", color=color, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)
            sig = "*" if p_holm < 0.05 else ""
            # Fixed effect-size column to the right of each panel's own data
            # area (axes-fraction x, data-coordinate y) -- never collides
            # with the shared y-tick labels on axes[0]'s left side.
            ax.text(1.04, y, f"r={eff_r:.2f}{sig}",
                    transform=ax.get_yaxis_transform(), clip_on=False,
                    color=INK_SECONDARY, fontsize=8, va="center", ha="left")

        span = max(his) - min(los)
        ax.set_xlim(min(los) - 0.06 * span, max(his) + 0.06 * span)
        ax.axvline(0.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
        ax.set_title(comp, color=INK_PRIMARY, fontsize=12, pad=10)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=8)

    dagger = " \u2020"
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(
        [f"{a}{dagger if a in LOW_SAMPLE_ATTRS else ''}" for a in ref_order],
        color=INK_PRIMARY, fontsize=10)
    axes[0].tick_params(axis="y", length=0)

    fig.suptitle(
        "Paired layer differences in area_normalized_enrichment (image-level, n=700 universe)\n"
        "blue = positive diff, red = negative diff; r = rank-biserial effect size "
        "(* = Holm-corrected p<0.05); dot=mean, line=95% bootstrap CI; sorted by L12-L4",
        color=INK_PRIMARY, fontsize=10.5, x=0.02, ha="left")
    out_png = os.path.join(FIG_DIR, "paired_diffs_L8L4_L12L8_L12L4.png")
    out_pdf = os.path.join(FIG_DIR, "paired_diffs_L8L4_L12L8_L12L4.pdf")
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


def select_representative_images(image_list_rows, n=N_REPRESENTATIVE):
    def n_attrs_present(row):
        return sum(int(row[f"has_{name}"]) for name in ATTR_NAMES)
    ranked = sorted(image_list_rows, key=lambda r: (-n_attrs_present(r), r["image_id"]))
    return [r["image_id"] for r in ranked[:n]]


def build_representative_panels():
    image_list_rows = read_csv_dicts(IMAGE_LIST_CSV)
    rep_ids = select_representative_images(image_list_rows)
    print(f"  Representative images (most attributes present, tie-break by id): {rep_ids}")

    cache = np.load(ATTN_CACHE_PATH)
    stems = list(cache["stems"])
    attn = cache["attn"]
    layers_display = list(cache["layers_display"])
    assert layers_display == LAYERS_DISPLAY, (
        f"STOP: cache layer order {layers_display} != {LAYERS_DISPLAY}")

    f = h5py.File(ATTRS_PATH, "r")
    _, attrs_index = load_attrs_index(f)

    saved = []
    for stem in rep_ids:
        idx = stems.index(stem)
        grid_maps = attn[idx]

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
    print("  OSIE Attribute-Grounding Full700: Figures")
    print("=" * 65)

    for path in (SUMMARY_CSV, PAIRED_CSV):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found -- run the extraction and stats scripts first")

    by_attr = load_summary()
    all_categories = list(by_attr.keys())

    print("\n--- A. Attribute x layer enrichment comparison (all 14 categories) ---")
    fig_a = enrichment_forest_plot(by_attr, all_categories, "enrichment_by_attribute_L4_L8_L12_full700")

    print("\n--- B. Paired diff / effect-size figure ---")
    fig_b = paired_diff_figure()

    print("\n--- C. Focused comparison: Face/Text/Emotion/Operability/Gazed/Touched ---")
    fig_c = enrichment_forest_plot(
        by_attr, FOCUS_ATTRS, "enrichment_focus_face_text_emotion_operability_gazed_touched",
        title_suffix=" -- focus attributes")

    print("\n--- D. Representative-image panels ---")
    rep_ids, rep_pngs = build_representative_panels()

    manifest = {
        "enrichment_all_categories": {"png": fig_a[0], "pdf": fig_a[1]},
        "paired_diffs": {"png": fig_b[0], "pdf": fig_b[1]},
        "enrichment_focus": {"png": fig_c[0], "pdf": fig_c[1]},
        "representative_images": rep_ids,
        "representative_figures": rep_pngs,
        "focus_attributes": FOCUS_ATTRS,
        "representative_selection_rule":
            "top-N full700 images by count of present attrs.mat attributes "
            "(descending), ties broken by ascending image id",
        "palette": {"L4": COLOR_L4, "L8": COLOR_L8, "L12": COLOR_L12,
                    "positive_diff": COLOR_POS, "negative_diff": COLOR_NEG,
                    "source": "dataviz skill references/palette.md"},
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
