"""
Figure 3 (REVISED, main-text version): spatial characterization of CLIP-B
attention at L4/L8/L12, now using LAYER-SPECIFIC color scales in the top
row instead of one color scale shared across L4/L8/L12.

Why revised: the original figure3_spatial_characterization.py used one
shared vmax (= max raw attention over L4/L8/L12) per image, so L4's much
smaller raw attention values rendered as almost uniformly dark/blank next
to L8/L12. That made the figure's own within-layer spatial pattern (L4
broadly spread vs. L8/L12 sharply localized) invisible, even though this
IS the pattern the underlying 700-image statistics (bottom row) support.
This revised script gives each of L4/L8/L12 its own vmin=0/vmax=that
layer's own raw maximum (for that image) and its own colorbar, so the
within-layer spatial distribution -- not the cross-layer intensity
magnitude -- is what the reader compares. See the module docstring note
below and paper_figures/FIGURE_AUDIT.md for the full rationale.

This script does NOT overwrite the original figure3_spatial_characterization.py
or its outputs (paper_figures/figure3_spatial_characterization.*) -- those
remain the "common (shared) color scale" version, additionally re-saved
verbatim as the appendix supplementary figure by
appendix_figure3_common_scale.py. No new model inference, probe training,
or attention extraction is performed; only pre-existing cached attention
(read-only) is re-visualized with a different, but equally reproducible
and non-arbitrary, color-scale convention.

Data sources (identical to the original figure3 script; read-only -- see
paper_figures/FIGURE_AUDIT.md):
  outputs/osie_attribute_grounding_full700/attn_cache/
      clip_vitb16_full700_L4L8L12_patchgrid.npz
  outputs/fixmaps/healthy/{stem}.npz ('heat_all')
  datasets/osie/predicting-human-gaze-beyond-pixels/data/stimuli/{stem}.jpg
  datasets/osie/predicting-human-gaze-beyond-pixels/data/attrs.mat
  outputs/clip_attention_b2_distribution/summary.json

Representative image for the main figure: **1156** (see FIGURE_AUDIT.md
section "Figure 3 revision" for why -- it is the first of the 3 existing
seed=42 pilot-draw representative images, already used throughout this
repo's prior B-2 qualitative figures, and all 3 were checked to show the
same qualitative L4-broad / L8-L12-local pattern under a layer-specific
scale before 1156 was kept as the single main-text example; the other two
(1159, 1213) are shown in the appendix "additional examples" figure so the
main-figure choice is not presented as if it were the only image that
shows the pattern).

Color-scale convention for L4/L8/L12 (identical to the existing
`lib/clip_attention_b2_plots.py::render_per_layer_and_shared_scale_figures`
"per_layer_scale" mode, reused here unchanged): vmin=0.0 (raw attention is
never negative -- verified upstream in the cache health check), vmax =
that layer's own raw maximum value for that image. No percentile clipping
is applied anywhere in this figure. Overlay alpha (0.60) and colormap
(viridis) are identical across L4/L8/L12 and identical to the original
figure3 script. The same foreground-object contour (from OSIE's attrs.mat
masks, patch-coverage > 0.5) is drawn on all three layers for a given
image.

Mandatory caveat (reproduced in the figure itself and in the LaTeX
caption): "Attention maps are visualized with layer-specific color scales
to reveal their within-layer spatial distributions; color intensity
should therefore not be compared directly across layers."
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from PIL import Image

from scripts.paper_figures import common as C
from scripts.paper_figures import figure3_spatial_characterization as F3

CAVEAT_TEXT = (
    "Attention maps are visualized with layer-specific color scales to reveal their "
    "within-layer spatial distributions; color intensity should therefore not be "
    "compared directly across layers."
)

MAIN_STEM = "1156"
ADDITIONAL_STEMS = ["1159", "1213"]


def _open_cache_and_attrs():
    cache = np.load(str(C.ATTN_CACHE_NPZ), allow_pickle=False)
    stems_all = cache["stems"].tolist()
    attn_all = cache["attn"]  # (700, 3, 38, 50)
    layers_display = cache["layers_display"].tolist()
    assert layers_display == F3.LAYERS_DISPLAY, "cache layer order changed unexpectedly"
    attrs_f = h5py.File(str(REPO_ROOT / "datasets" / "osie" /
                            "predicting-human-gaze-beyond-pixels" / "data" / "attrs.mat"), "r")
    attrs_index = F3._load_attrs_index(attrs_f)
    return stems_all, attn_all, attrs_f, attrs_index


def _load_one_image(stem, stems_all, attn_all, attrs_f, attrs_index):
    img_idx = stems_all.index(stem)
    img_path = C.STIMULI_DIR / f"{stem}.jpg"
    img_arr = np.array(Image.open(img_path).convert("RGB"))
    H, W = img_arr.shape[:2]
    assert (W, H) == (F3.IMG_W, F3.IMG_H), f"unexpected size for {stem}: {(W, H)}"

    fg_mask = F3._foreground_mask(attrs_f, attrs_index[stem], (H, W))
    coverage_up = F3._upsample(F3.mask_to_patch_weights(fg_mask, F3.CLIP_PATCH_SIZE),
                                (F3.IMG_W, F3.IMG_H))

    fixmap = np.load(str(C.FIXMAP_DIR / f"{stem}.npz"))
    gaze_heat = fixmap["heat_all"].astype(np.float64)
    if gaze_heat.shape != (F3.IMG_H, F3.IMG_W):
        raise RuntimeError(f"STOP: fixmap shape {gaze_heat.shape} != "
                            f"{(F3.IMG_H, F3.IMG_W)} for {stem}")

    layer_maps = {ld: attn_all[img_idx, i].astype(np.float64)
                  for i, ld in enumerate(F3.LAYERS_DISPLAY)}
    if any((m < 0).any() for m in layer_maps.values()):
        raise RuntimeError(f"STOP: negative raw attention encountered for {stem}")
    return img_arr, coverage_up, gaze_heat, layer_maps


def draw_layer_specific_grid(fig, gs, stems, show_original_and_gaze=True,
                              show_contour=True, panel_letters=None, row_label=True):
    """Render `stems` as rows x [Original, Human gaze, CLIP L4, L8, L12]
    (or just the 3 CLIP columns if show_original_and_gaze=False), each
    CLIP panel using ITS OWN vmin=0/vmax=own-max color scale and its own
    colorbar -- see module docstring for the exact convention."""
    stems_all, attn_all, attrs_f, attrs_index = _open_cache_and_attrs()
    n_clip_cols = len(F3.LAYERS_DISPLAY)
    col_titles_clip = [f"CLIP L{ld}" for ld in F3.LAYERS_DISPLAY]
    col_titles = (["Original image", "Human gaze"] + col_titles_clip) if show_original_and_gaze \
        else col_titles_clip
    letter_i = 0

    for row, stem in enumerate(stems):
        img_arr, coverage_up, gaze_heat, layer_maps = _load_one_image(
            stem, stems_all, attn_all, attrs_f, attrs_index)

        def _title(base_title):
            nonlocal letter_i
            if row != 0:
                letter_i += 1
                return None
            if panel_letters:
                t = f"({panel_letters[letter_i]}) {base_title}"
            else:
                t = base_title
            letter_i += 1
            return t

        col = 0
        if show_original_and_gaze:
            ax0 = fig.add_subplot(gs[row, 0])
            ax0.imshow(img_arr)
            ax0.axis("off")
            title0 = _title(col_titles[0])
            if title0 is not None:
                ax0.set_title(title0, fontsize=C.BASE_FONT_PT, loc="left")
            if row_label:
                ax0.text(-0.10, 0.5, f"image {stem}", transform=ax0.transAxes, rotation=90,
                         ha="center", va="center", fontsize=C.BASE_FONT_PT - 1.5)

            ax1 = fig.add_subplot(gs[row, 1])
            ax1.imshow(gaze_heat, cmap=F3.GAZE_CMAP, alpha=F3.GAZE_ALPHA)
            ax1.axis("off")
            title1 = _title(col_titles[1])
            if title1 is not None:
                ax1.set_title(title1, fontsize=C.BASE_FONT_PT, loc="left")
            col = 2

        for j, layer_display in enumerate(F3.LAYERS_DISPLAY):
            ax = fig.add_subplot(gs[row, col + j])
            ax.imshow(img_arr)
            raw = layer_maps[layer_display]
            attn_up = F3._upsample(raw, (F3.IMG_W, F3.IMG_H))
            vmax = float(raw.max())  # layer's OWN raw max for this image; no clipping
            im = ax.imshow(attn_up, cmap=F3.ATTN_CMAP, alpha=F3.ATTN_ALPHA, vmin=0.0, vmax=vmax)
            if show_contour:
                ax.contour(coverage_up, levels=[0.5], colors="white", linewidths=0.9)
            ax.axis("off")
            base_title = col_titles[col + j] if show_original_and_gaze else col_titles_clip[j]
            titlej = _title(base_title)
            if titlej is not None:
                ax.set_title(titlej, fontsize=C.BASE_FONT_PT, loc="left")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=C.BASE_FONT_PT - 4)
            formatter = ScalarFormatter(useOffset=False, useMathText=False)
            formatter.set_scientific(False)
            cbar.ax.yaxis.set_major_formatter(formatter)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    attrs_f.close()


def _draw_bottom_row_revised(fig, gs_bot, panel_letters=("f", "g", "h")):
    with open(C.B2_SUMMARY_JSON, encoding="utf-8") as fh:
        summary = json.load(fh)
    desc = summary["descriptive_per_layer"]
    sim = summary["spatial_similarity_descriptive"]

    layer_keys = ["L4", "L8", "L12"]
    layer_x = np.arange(3)
    colors = ["#0072B2", "#009E73", "#D55E00"]

    # --- (f) 50%-mass patch count -- neutral title ---
    ax_f = fig.add_subplot(gs_bot[0])
    means = [desc["mass50_n_patches"][k]["mean"] for k in layer_keys]
    los = [desc["mass50_n_patches"][k]["ci_lo"] for k in layer_keys]
    his = [desc["mass50_n_patches"][k]["ci_hi"] for k in layer_keys]
    yerr = np.array([[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]])
    ax_f.bar(layer_x, means, yerr=yerr, color=colors, width=0.6, capsize=3,
             edgecolor="black", linewidth=0.5)
    ax_f.set_xticks(layer_x); ax_f.set_xticklabels(layer_keys)
    ax_f.set_ylabel("patches for 50% attention mass")
    ax_f.set_title(f"({panel_letters[0]}) Number of patches\ncovering 50% attention mass",
                    loc="left", fontsize=C.BASE_FONT_PT - 1.0)
    for x, m in zip(layer_x, means):
        ax_f.text(x, m + max(means) * 0.03, f"{m:.0f}", ha="center", va="bottom",
                   fontsize=C.BASE_FONT_PT - 2.5)

    # --- (g) foreground enrichment -- baseline framed as area-proportional ---
    ax_g = fig.add_subplot(gs_bot[1])
    means = [desc["foreground_enrichment"][k]["mean"] for k in layer_keys]
    los = [desc["foreground_enrichment"][k]["ci_lo"] for k in layer_keys]
    his = [desc["foreground_enrichment"][k]["ci_hi"] for k in layer_keys]
    yerr = np.array([[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]])
    ax_g.axhline(1.0, color="#999999", lw=0.8, ls="--", zorder=0)
    ax_g.text(2.55, 1.03, "area-proportional\nallocation", fontsize=C.BASE_FONT_PT - 3,
               color="#666666", ha="right", va="bottom")
    ax_g.bar(layer_x, means, yerr=yerr, color=colors, width=0.6, capsize=3,
             edgecolor="black", linewidth=0.5)
    ax_g.set_xticks(layer_x); ax_g.set_xticklabels(layer_keys)
    ax_g.set_ylabel("foreground enrichment")
    ax_g.set_title(f"({panel_letters[1]}) Foreground enrichment\nrelative to area",
                    loc="left", fontsize=C.BASE_FONT_PT - 1.0)
    for x, m in zip(layer_x, means):
        ax_g.text(x, m + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=C.BASE_FONT_PT - 2.5)

    # --- (h) same-image spatial-correlation matrix ---
    ax_h = fig.add_subplot(gs_bot[2])
    pairs = {("L4", "L4"): 1.0, ("L8", "L8"): 1.0, ("L12", "L12"): 1.0}
    pairs[("L4", "L8")] = pairs[("L8", "L4")] = sim["L4-L8_pearson"]["same_image_mean"]
    pairs[("L4", "L12")] = pairs[("L12", "L4")] = sim["L4-L12_pearson"]["same_image_mean"]
    pairs[("L8", "L12")] = pairs[("L12", "L8")] = sim["L8-L12_pearson"]["same_image_mean"]
    mat = np.array([[pairs[(a, b)] for b in layer_keys] for a in layer_keys])
    im = ax_h.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_h.set_xticks(layer_x); ax_h.set_xticklabels(layer_keys)
    ax_h.set_yticks(layer_x); ax_h.set_yticklabels(layer_keys)
    for i in range(3):
        for j in range(3):
            txt_color = "white" if abs(mat[i, j]) > 0.6 else "black"
            ax_h.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                      color=txt_color, fontsize=C.BASE_FONT_PT - 2)
    ax_h.set_title(f"({panel_letters[2]}) Same-image spatial\ncorrelation (Pearson $r$)",
                    loc="left", fontsize=C.BASE_FONT_PT - 1.0)
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="Pearson $r$ (same-image)")
    return mat


def make_main_figure():
    C.apply_paper_style()
    fig = plt.figure(figsize=(C.FIG_WIDTH_2COL_IN, 4.75))
    gs_outer = GridSpec(2, 1, height_ratios=[1.0, 1.35], hspace=0.30, figure=fig)
    gs_top = gs_outer[0].subgridspec(1, 5, wspace=0.55)
    gs_bot = gs_outer[1].subgridspec(1, 3, wspace=0.60)

    draw_layer_specific_grid(fig, gs_top, [MAIN_STEM], show_original_and_gaze=True,
                              show_contour=True, panel_letters=list("abcde"), row_label=False)
    _draw_bottom_row_revised(fig, gs_bot, panel_letters=("f", "g", "h"))

    fig.text(0.5, -0.02, CAVEAT_TEXT, ha="center", va="top", fontsize=C.BASE_FONT_PT - 2.5,
              style="italic", wrap=True)
    return fig


def make_additional_examples_figure():
    C.apply_paper_style()
    n = len(ADDITIONAL_STEMS)
    fig = plt.figure(figsize=(C.FIG_WIDTH_2COL_IN, 1.20 * n + 0.55))
    gs = GridSpec(n, 5, wspace=0.55, hspace=0.25, figure=fig)
    draw_layer_specific_grid(fig, gs, ADDITIONAL_STEMS, show_original_and_gaze=True,
                              show_contour=True, row_label=True)
    fig.text(0.5, -0.01, CAVEAT_TEXT + " These are additional representative images "
              "(not the one used in the main-text Figure 3), shown to confirm the pattern "
              "is not specific to a single hand-picked image.",
              ha="center", va="top", fontsize=C.BASE_FONT_PT - 2.5, style="italic", wrap=True)
    return fig


def main():
    fig = make_main_figure()
    C.save_fig(fig, C.FIG_DIR / "figure3_spatial_characterization_revised")
    plt.close(fig)
    print("[figure3-revised] wrote figure3_spatial_characterization_revised.{pdf,png,svg}")

    fig2 = make_additional_examples_figure()
    C.save_fig(fig2, C.APPENDIX_DIR / "figure_attention_layer_specific_additional_examples",
               svg=False)
    plt.close(fig2)
    print("[figure3-revised] wrote appendix/figure_attention_layer_specific_additional_examples.{pdf,png}")


if __name__ == "__main__":
    main()
