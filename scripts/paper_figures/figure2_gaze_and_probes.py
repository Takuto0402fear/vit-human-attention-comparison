"""
Figure 2: CLIP-B's "where attention looks" (panel a, gaze-alignment NSS)
vs. "what is linearly readable" (panels b/c, semantic-attribute and
low-level-feature probes) across layers -- showing these two layer-wise
profiles do NOT coincide.

Data sources (read-only; see paper_figures/FIGURE_AUDIT.md section 2 for
full provenance):
  (a) outputs/expA_clip/healthy/metrics_per_image.csv          (CLIP-B NSS,
      n=700 images; 95% CI = image-level bootstrap, seed=42, n_boot=10000,
      same convention as Figure 1)
  (b) outputs/osie_probe_all12_full700/probe_summary.csv        (12 OSIE
      semantic attributes x 12 layers, AUPRC, mean over 5 CV folds;
      condition=all_objects)
  (c) outputs/osie_lowlevel_probe_all12_full700/lowlevel_summary.csv
      (8 low-level visual features x 12 layers, out-of-fold R^2, mean over
      5 CV folds; excludes the 3 geometry-control targets centroid_x/y,
      mask_area_fraction, which are not visual features)

Panels (b) and (c) show fold-level means (no bootstrap-over-images CI is
computed for them -- see FIGURE_AUDIT.md for why that would require new
image-level probe re-fitting, which this task explicitly avoids).
AUC-Judd and sAUC are intentionally NOT overlaid on panel (a); see the
Appendix figures for their full 4-model layer-wise curves and a CLIP-only
3-metric supplementary panel.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from scripts.paper_figures import common as C


def _panel_a(ax):
    _, mat = C.load_model_layer_matrix("CLIP-B", "NSS")
    boot_idx = C.make_boot_indices(mat.shape[0])
    mean, lo, hi = C.bootstrap_mean_ci(mat, boot_idx)
    C.mark_highlight_layers(ax)
    ax.fill_between(C.LAYERS, lo, hi, color=C.PANEL_A_COLOR, alpha=0.15, linewidth=0)
    ax.plot(C.LAYERS, mean, color=C.PANEL_A_COLOR, lw=2.2, ls="-", marker="o", ms=3.6)
    C.style_axes(ax, ylabel="NSS")
    ax.set_title("(a) Human-gaze alignment (CLIP-B)", loc="left", fontsize=C.BASE_FONT_PT)
    return mean


def _panel_b(ax, show_attrs=True):
    df = C.load_semantic_probe_table()
    layer_attr = df.pivot_table(index="layer", columns="attribute", values="mean")
    layer_attr = layer_attr.loc[C.LAYERS, C.SEMANTIC_ATTRIBUTES]
    if show_attrs:
        for attr in C.SEMANTIC_ATTRIBUTES:
            ax.plot(C.LAYERS, layer_attr[attr].to_numpy(), **C.AUX_LINE_STYLE)
    group_mean = layer_attr.mean(axis=1).to_numpy()
    C.mark_highlight_layers(ax)
    ax.plot(C.LAYERS, group_mean, color=C.PANEL_B_COLOR, lw=2.2, ls="-", marker="s", ms=3.6,
            zorder=5)
    C.style_axes(ax, ylabel="AUPRC")
    ax.set_title("(b) Semantic-attribute probe (12-attribute mean)", loc="left",
                 fontsize=C.BASE_FONT_PT)
    return group_mean


def _panel_c(ax, show_feats=True):
    df = C.load_lowlevel_probe_table()
    layer_feat = df.pivot_table(index="layer", columns="target", values="mean")
    layer_feat = layer_feat.loc[C.LAYERS, C.LOWLEVEL_FEATURES]
    if show_feats:
        for feat in C.LOWLEVEL_FEATURES:
            ax.plot(C.LAYERS, layer_feat[feat].to_numpy(), **C.AUX_LINE_STYLE)
    group_mean = layer_feat.mean(axis=1).to_numpy()
    C.mark_highlight_layers(ax)
    ax.axhline(0.0, color="#aaaaaa", lw=0.6, ls="-", zorder=0)
    ax.plot(C.LAYERS, group_mean, color=C.PANEL_C_COLOR, lw=2.2, ls="-", marker="D", ms=3.4,
            zorder=5)
    C.style_axes(ax, ylabel="Out-of-fold $R^2$")
    ax.set_title("(c) Low-level visual-feature probe (8-feature mean)", loc="left",
                 fontsize=C.BASE_FONT_PT)
    return group_mean


def make_figure(width_in, panel_h_in=2.05):
    C.apply_paper_style()
    fig, axes = plt.subplots(3, 1, figsize=(width_in, panel_h_in * 3), constrained_layout=True,
                              sharex=True)
    _panel_a(axes[0])
    _panel_b(axes[1])
    _panel_c(axes[2])
    for ax in axes:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Layer")
    return fig


def main():
    fig = make_figure(C.FIG_WIDTH_2COL_IN)
    C.save_fig(fig, C.FIG_DIR / "figure2_gaze_and_probes")
    plt.close(fig)

    fig1 = make_figure(C.FIG_WIDTH_1COL_IN, panel_h_in=2.0)
    C.save_fig(fig1, C.FIG_DIR / "figure2_gaze_and_probes_1col")
    plt.close(fig1)

    fig2 = make_figure(C.FIG_WIDTH_2COL_IN, panel_h_in=2.05)
    C.save_fig(fig2, C.FIG_DIR / "figure2_gaze_and_probes_2col")
    plt.close(fig2)

    print("[figure2] wrote figure2_gaze_and_probes.{pdf,png,svg} (+ _1col/_2col variants)")


if __name__ == "__main__":
    main()
