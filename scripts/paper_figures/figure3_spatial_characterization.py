"""
Figure 3: spatial characterization of CLIP-B attention at L4/L8/L12 --
same three layers used throughout Figures 1-2, now compared spatially.

Top row: for 3 representative OSIE images, Original / Human gaze /
CLIP L4 / CLIP L8 / CLIP L12 attention, side by side.
Bottom row: the 3 quantitative claims that motivate this figure --
  (1) L4 spreads over many more patches than L8/L12 (50%-mass patch count),
  (2) L8's foreground enrichment drops to ~1 (chance) even though it is
      spatially concentrated, while L4 and L12 are foreground-enriched,
  (3) L8 and L12 attend to nearly the same locations (high spatial
      correlation); L4 does not.

Data sources (read-only; see paper_figures/FIGURE_AUDIT.md section 4/9):
  outputs/osie_attribute_grounding_full700/attn_cache/
      clip_vitb16_full700_L4L8L12_patchgrid.npz   (raw [CLS]->patch
      attention, native 38x50 grid, for all 700 images x {L4,L8,L12})
  outputs/fixmaps/healthy/{stem}.npz ('heat_all')  (human gaze density,
      healthy group, all fixations -- same fixmaps used throughout this
      repo's OSIE analyses)
  datasets/osie/predicting-human-gaze-beyond-pixels/data/stimuli/{stem}.jpg
  datasets/osie/predicting-human-gaze-beyond-pixels/data/attrs.mat
      (object masks, for the thin foreground-outline overlay only)
  outputs/clip_attention_b2_distribution/summary.json
      (descriptive_per_layer.mass50_n_patches,
       descriptive_per_layer.foreground_enrichment;
       both already bootstrap mean + 95% CI over n=700 images,
       seed=42, n_boot=10000 -- identical convention to Figures 1-2)
      (spatial_similarity_descriptive.*_pearson.same_image_mean, for the
       L4-L8 / L4-L12 / L8-L12 layer-similarity matrix)

Representative images: 1156, 1159, 1213 -- the SAME 3 images used by the
existing pilot's seed=42 draw (scripts/report_clip_attention_b2_full700.py:
"Figures reuse the SAME 3 representative images as the pilot (1156, 1159,
1213 -- chosen by the pilot's seed=42 draw, not cherry-picked"). Reused
here verbatim rather than hand-picking new "clean-looking" images -- see
FIGURE_AUDIT.md for this selection-basis note.

Color scale: all 3 CLIP layers for a given image share one linear scale
(vmin=0, vmax=max raw attention over L4/L8/L12 for that image), matching
the existing convention in lib/clip_attention_b2_plots.py
(render_per_layer_and_shared_scale_figures, "shared_scale" mode) so
intensities are comparable across L4/L8/L12. The human-gaze panel uses a
SEPARATE colormap and its own scale (it is a different physical quantity
-- fixation density, not attention mass -- and is never on the same color
scale as the CLIP panels).
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
from PIL import Image

from scripts.paper_figures import common as C
from lib.osie_text_alignment import mask_to_patch_weights

CLIP_PATCH_SIZE = 16
LAYERS_DISPLAY = [4, 8, 12]
IMG_W, IMG_H = 800, 600
ATTN_CMAP = "viridis"
GAZE_CMAP = "inferno"
ATTN_ALPHA = 0.60
GAZE_ALPHA = 0.85


def _upsample(arr_2d, size_wh):
    return np.array(Image.fromarray(arr_2d.astype(np.float32)).resize(size_wh, Image.BILINEAR))


def _load_attrs_index(f):
    def h5_char_str(ref):
        return "".join(chr(int(c)) for c in f[ref][()].flatten())
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw_name = "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[raw_name.rsplit(".", 1)[0]] = g
    return index


def _foreground_mask(f, group, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    n_objs = objs_ds.shape[1]
    fg = np.zeros((h, w), dtype=bool)
    for j in range(n_objs):
        obj_g = f[objs_ds[0, j]]
        mp = obj_g["map"][()].T.astype(bool)
        fg |= mp
    return fg


def _draw_top_row(fig, gs_top, stems):
    cache = np.load(str(C.ATTN_CACHE_NPZ), allow_pickle=False)
    stems_all = cache["stems"].tolist()
    attn_all = cache["attn"]  # (700, 3, 38, 50)
    layers_display = cache["layers_display"].tolist()
    assert layers_display == LAYERS_DISPLAY

    attrs_f = h5py.File(str(REPO_ROOT / "datasets" / "osie" /
                            "predicting-human-gaze-beyond-pixels" / "data" / "attrs.mat"), "r")
    attrs_index = _load_attrs_index(attrs_f)

    col_titles = ["Original image", "Human gaze", "CLIP L4", "CLIP L8", "CLIP L12"]

    for row, stem in enumerate(stems):
        img_idx = stems_all.index(stem)
        img_path = C.STIMULI_DIR / f"{stem}.jpg"
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        H, W = img_arr.shape[:2]
        assert (W, H) == (IMG_W, IMG_H), f"unexpected size for {stem}: {(W, H)}"

        fg_mask = _foreground_mask(attrs_f, attrs_index[stem], (H, W))
        coverage_up = _upsample(mask_to_patch_weights(fg_mask, CLIP_PATCH_SIZE), (IMG_W, IMG_H))

        fixmap = np.load(str(C.FIXMAP_DIR / f"{stem}.npz"))
        gaze_heat = fixmap["heat_all"].astype(np.float64)
        if gaze_heat.shape != (IMG_H, IMG_W):
            raise RuntimeError(f"STOP: fixmap shape {gaze_heat.shape} != {(IMG_H, IMG_W)} for {stem}")

        layer_maps = {ld: attn_all[img_idx, i].astype(np.float64) for i, ld in enumerate(LAYERS_DISPLAY)}
        shared_vmax = max(m.max() for m in layer_maps.values())

        for col in range(5):
            ax = fig.add_subplot(gs_top[row, col])
            ax.imshow(img_arr)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=C.BASE_FONT_PT)
            if col == 0:
                ax.text(-0.06, 0.5, f"image {stem}", transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=C.BASE_FONT_PT - 1.5)
            if col == 1:
                gaze_im = ax.imshow(gaze_heat, cmap=GAZE_CMAP, alpha=GAZE_ALPHA)
            else:
                gaze_im = None
            if col >= 2:
                layer_display = LAYERS_DISPLAY[col - 2]
                attn_up = _upsample(layer_maps[layer_display], (IMG_W, IMG_H))
                attn_im = ax.imshow(attn_up, cmap=ATTN_CMAP, alpha=ATTN_ALPHA,
                                     vmin=0.0, vmax=shared_vmax)
                ax.contour(coverage_up, levels=[0.5], colors="white", linewidths=0.8)
            else:
                attn_im = None
        # one shared colorbar per row for the 3 CLIP panels (columns 2-4)
        row_axes = fig.axes[-3:]
        fig.colorbar(attn_im, ax=row_axes, fraction=0.018, pad=0.01, aspect=32,
                     label="raw attention" if row == len(stems) - 1 else None)
    attrs_f.close()


def _draw_bottom_row(fig, gs_bot):
    with open(C.B2_SUMMARY_JSON, encoding="utf-8") as fh:
        summary = json.load(fh)
    desc = summary["descriptive_per_layer"]
    sim = summary["spatial_similarity_descriptive"]

    layer_keys = ["L4", "L8", "L12"]
    layer_x = np.arange(3)
    colors = ["#0072B2", "#009E73", "#D55E00"]

    # --- (d) 50%-mass patch count ---
    ax_d = fig.add_subplot(gs_bot[0])
    means = [desc["mass50_n_patches"][k]["mean"] for k in layer_keys]
    los = [desc["mass50_n_patches"][k]["ci_lo"] for k in layer_keys]
    his = [desc["mass50_n_patches"][k]["ci_hi"] for k in layer_keys]
    yerr = np.array([[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]])
    ax_d.bar(layer_x, means, yerr=yerr, color=colors, width=0.6, capsize=3,
             edgecolor="black", linewidth=0.5)
    ax_d.set_xticks(layer_x)
    ax_d.set_xticklabels(layer_keys)
    ax_d.set_ylabel("patches for 50% mass")
    ax_d.set_title("(d) L4 spreads widely", loc="left", fontsize=C.BASE_FONT_PT)
    for x, m in zip(layer_x, means):
        ax_d.text(x, m + max(means) * 0.03, f"{m:.0f}", ha="center", va="bottom",
                  fontsize=C.BASE_FONT_PT - 2.5)

    # --- (e) foreground enrichment ---
    ax_e = fig.add_subplot(gs_bot[1])
    means = [desc["foreground_enrichment"][k]["mean"] for k in layer_keys]
    los = [desc["foreground_enrichment"][k]["ci_lo"] for k in layer_keys]
    his = [desc["foreground_enrichment"][k]["ci_hi"] for k in layer_keys]
    yerr = np.array([[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]])
    ax_e.axhline(1.0, color="#999999", lw=0.8, ls="--", zorder=0)
    ax_e.bar(layer_x, means, yerr=yerr, color=colors, width=0.6, capsize=3,
             edgecolor="black", linewidth=0.5)
    ax_e.set_xticks(layer_x)
    ax_e.set_xticklabels(layer_keys)
    ax_e.set_ylabel("foreground enrichment")
    ax_e.set_title("(e) L8 foreground enrichment ~ chance", loc="left", fontsize=C.BASE_FONT_PT)
    for x, m in zip(layer_x, means):
        ax_e.text(x, m + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=C.BASE_FONT_PT - 2.5)

    # --- (f) spatial-correlation matrix ---
    ax_f = fig.add_subplot(gs_bot[2])
    pairs = {("L4", "L4"): 1.0, ("L8", "L8"): 1.0, ("L12", "L12"): 1.0}
    pairs[("L4", "L8")] = pairs[("L8", "L4")] = sim["L4-L8_pearson"]["same_image_mean"]
    pairs[("L4", "L12")] = pairs[("L12", "L4")] = sim["L4-L12_pearson"]["same_image_mean"]
    pairs[("L8", "L12")] = pairs[("L12", "L8")] = sim["L8-L12_pearson"]["same_image_mean"]
    mat = np.array([[pairs[(a, b)] for b in layer_keys] for a in layer_keys])
    im = ax_f.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_f.set_xticks(layer_x); ax_f.set_xticklabels(layer_keys)
    ax_f.set_yticks(layer_x); ax_f.set_yticklabels(layer_keys)
    for i in range(3):
        for j in range(3):
            txt_color = "white" if abs(mat[i, j]) > 0.6 else "black"
            ax_f.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                      color=txt_color, fontsize=C.BASE_FONT_PT - 2)
    ax_f.set_title("(f) L8-L12 attend similarly; L4 differs", loc="left", fontsize=C.BASE_FONT_PT)
    fig.colorbar(im, ax=ax_f, fraction=0.046, pad=0.04, label="Pearson r (same-image)")


def make_figure():
    C.apply_paper_style()
    fig = plt.figure(figsize=(C.FIG_WIDTH_2COL_IN, 6.35))
    gs_outer = GridSpec(2, 1, height_ratios=[1.55, 1.0], hspace=0.22, figure=fig)
    gs_top = gs_outer[0].subgridspec(len(C.REPRESENTATIVE_IMAGE_STEMS), 5,
                                      wspace=0.04, hspace=0.08)
    gs_bot = gs_outer[1].subgridspec(1, 3, wspace=0.55)

    _draw_top_row(fig, gs_top, C.REPRESENTATIVE_IMAGE_STEMS)
    _draw_bottom_row(fig, gs_bot)
    return fig


def main():
    fig = make_figure()
    C.save_fig(fig, C.FIG_DIR / "figure3_spatial_characterization")
    plt.close(fig)
    print("[figure3] wrote figure3_spatial_characterization.{pdf,png,svg}")


if __name__ == "__main__":
    main()
