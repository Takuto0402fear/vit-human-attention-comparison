"""
Figure 1: layer-wise NSS comparison across 4 models (CLIP-B, DINO-S, DINO-B,
DeiT/SL). CLIP-B shows the "N-shape" profile (early-layer rise, mid-layer
dip, late-layer recovery); the other 3 models do not.

Data sources (read-only; see paper_figures/FIGURE_AUDIT.md section 1 for the
full provenance chain):
  outputs/expA_clip/healthy/metrics_per_image.csv          (CLIP-B, n=700)
  outputs/expA/healthy/metrics_per_image.csv               (DINO-S, n=700)
  outputs/expA_dino_vitb16/healthy/metrics_per_image.csv   (DINO-B, n=700)
  outputs/expA_sl/combined/metrics_by_image_meanacrosstrials.csv
                                                            (DeiT/SL, n=700,
                                                             each value is the
                                                             per-image mean
                                                             across the
                                                             existing 6 trials)

95% CI: image-ID-level bootstrap (seed=42, n_boot=10000), computed
independently per model from the per-image tables above -- see
common.bootstrap_mean_ci and FIGURE_AUDIT.md for why this is the one
unified method usable for all 4 models (their existing precomputed
mean/std tables are NOT on a common footing: DeiT/SL's own std reflects
across-TRIAL spread while the other 3 models' std reflects across-IMAGE
spread; re-bootstrapping every model over its own per-image rows removes
that mismatch).

No causal language is used anywhere in this figure or its caption; the
profile is described purely as an observed layer-wise pattern.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from scripts.paper_figures import common as C

ANNOTATIONS = {
    4: ("early peak", (0, 14)),
    8: ("intermediate trough", (0, -22)),
    12: ("late recovery", (6, 14)),
}


def draw(ax, curves, annotate=True, legend=True, legend_loc="right"):
    C.mark_highlight_layers(ax)
    for model_key in C.MODEL_ORDER:
        mean, lo, hi = curves[model_key]
        st = C.MODEL_STYLE[model_key]
        if model_key == "CLIP-B":
            ax.fill_between(C.LAYERS, lo, hi, color=st["color"], alpha=0.15,
                             linewidth=0, zorder=st["zorder"] - 1)
        ax.plot(C.LAYERS, mean, color=st["color"], lw=st["lw"], ls=st["ls"],
                 marker=st["marker"], ms=st["ms"], zorder=st["zorder"],
                 label=f"{model_key} ({C.MODEL_SOURCES[model_key]['display']})",
                 markerfacecolor=st["color"], markeredgecolor=st["color"])
        if model_key != "CLIP-B":
            ax.fill_between(C.LAYERS, lo, hi, color=st["color"], alpha=0.10,
                             linewidth=0, zorder=st["zorder"] - 1)

    C.style_axes(ax, ylabel="NSS")

    if annotate:
        clip_mean = curves["CLIP-B"][0]
        for layer, (label, offset) in ANNOTATIONS.items():
            y = clip_mean[layer - 1]
            ax.annotate(
                f"{label}\n(L{layer})", xy=(layer, y), xytext=offset,
                textcoords="offset points", ha="center", fontsize=C.BASE_FONT_PT - 2.5,
                color="#000000",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5, shrinkA=2, shrinkB=4),
            )

    if legend:
        if legend_loc == "right":
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                       frameon=False, handlelength=2.4)
        else:  # "below" -- for narrow (1-column) layouts
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), borderaxespad=0.0,
                       frameon=False, handlelength=2.0, ncol=2, columnspacing=1.0)
    return ax


def make_figure(width_in, height_in, annotate=True, legend_loc="right"):
    C.apply_paper_style()
    curves = C.load_all_model_curves("NSS")
    fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=True)
    draw(ax, curves, annotate=annotate, legend_loc=legend_loc)
    return fig


def main():
    # Primary (matches the exact filenames requested for the paper).
    fig = make_figure(C.FIG_WIDTH_2COL_IN, 3.4)
    C.save_fig(fig, C.FIG_DIR / "figure1_model_comparison_nss")
    plt.close(fig)

    # Explicit column-width variants for layout checking in the manuscript.
    fig1 = make_figure(C.FIG_WIDTH_1COL_IN, 4.1, annotate=False, legend_loc="below")
    C.save_fig(fig1, C.FIG_DIR / "figure1_model_comparison_nss_1col")
    plt.close(fig1)

    fig2 = make_figure(C.FIG_WIDTH_2COL_IN, 3.4, annotate=True, legend_loc="right")
    C.save_fig(fig2, C.FIG_DIR / "figure1_model_comparison_nss_2col")
    plt.close(fig2)

    print("[figure1] wrote figure1_model_comparison_nss.{pdf,png,svg} (+ _1col/_2col variants)")


if __name__ == "__main__":
    main()
