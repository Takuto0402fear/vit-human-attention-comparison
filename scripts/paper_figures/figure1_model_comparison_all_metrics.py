"""
Figure 1 (3-panel revision): layer-wise (a) NSS, (b) AUC-Judd, (c) sAUC
comparison across 4 models (CLIP-B, DINO-S, DINO-B, DeiT/SL), showing that
CLIP-B's N-shape layer-wise profile (early-layer rise, mid-layer dip,
late-layer recovery) is not an artifact of a single gaze-alignment metric.

This script performs NO new computation: it reuses, unmodified, the exact
same data-loading and bootstrap-CI function already used and audited for
the original (NSS-only) Figure 1 and for the Appendix's separate AUC-Judd
and sAUC 4-model comparison figures --
`common.load_all_model_curves(metric)` (image-ID-level bootstrap, seed=42,
n_boot=10000, alpha=0.05, over each model's own 700-image per-image table;
see common.py and paper_figures/FIGURE_AUDIT.md sections 1/3 for the full
provenance and the reason this exact method -- not each model's
pre-existing, non-uniform `_std` column -- is used). Calling this same
deterministic function again for AUC-Judd/sAUC produces byte-identical
numbers to the existing appendix_aucjudd_model_comparison.* /
appendix_sauc_model_comparison.* figures; this is a re-use of an existing
audited computation, not a new bootstrap run.

Does NOT overwrite the original figure1_model_comparison_nss.* (single-
panel NSS) files -- this is a new, separate 3-panel figure.

No causal language; "M_contrast"/"M-shape" (the internal code/CSV name)
is referred to as "N-shape profile" in this figure and its caption, per
the paper's terminology.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from scripts.paper_figures import common as C

METRICS = [("NSS", "NSS"), ("AUC_Judd", "AUC-Judd"), ("sAUC", "sAUC")]
PANEL_LETTERS = ["a", "b", "c"]


def _draw_panel(ax, metric, curves, letter):
    C.mark_highlight_layers(ax)
    for model_key in C.MODEL_ORDER:
        mean, lo, hi = curves[model_key]
        st = C.MODEL_STYLE[model_key]
        ax.fill_between(C.LAYERS, lo, hi, color=st["color"], alpha=0.15, linewidth=0,
                         zorder=st["zorder"] - 1)
        ax.plot(C.LAYERS, mean, color=st["color"], lw=st["lw"], ls=st["ls"],
                 marker=st["marker"], ms=st["ms"], zorder=st["zorder"],
                 label=f"{model_key} ({C.MODEL_SOURCES[model_key]['display']})")
    C.style_axes(ax)
    ax.set_ylabel({"NSS": "NSS", "AUC_Judd": "AUC-Judd", "sAUC": "sAUC"}[metric])
    ax.margins(y=0.08)
    ax.set_title(f"({letter})", loc="left", fontsize=C.BASE_FONT_PT, fontweight="bold")
    return ax


def make_figure(width_in, layout="horizontal", legend=True):
    """layout='horizontal': (a)(b)(c) side by side (2-column-width figure).
    layout='vertical': (a)/(b)/(c) stacked -- used for the 1-column-width
    check, since 3 side-by-side panels do not fit meaningfully in a single
    narrow (~3.4in) column; stacking is the same solution already used by
    Figure 2's 1-column variant."""
    C.apply_paper_style()
    curves_by_metric = {metric: C.load_all_model_curves(metric) for metric, _ in METRICS}

    if layout == "horizontal":
        fig, axes = plt.subplots(1, 3, figsize=(width_in, width_in * 0.34),
                                  constrained_layout=True)
        legend_kwargs = dict(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(width_in, width_in * 1.9),
                                  constrained_layout=True)
        legend_kwargs = dict(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2)

    for ax, (metric, _), letter in zip(axes, METRICS, PANEL_LETTERS):
        _draw_panel(ax, metric, curves_by_metric[metric], letter)
        if layout == "vertical":
            ax.set_xlabel("")
    if layout == "vertical":
        axes[-1].set_xlabel("Layer")

    handles, labels = axes[0].get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, frameon=False, handlelength=2.2, columnspacing=1.2,
                    bbox_transform=fig.transFigure, **legend_kwargs)
    return fig, curves_by_metric


def sanity_check(curves_by_metric):
    """Re-derives the same numbers already checked in verify_figures.py
    (CLIP-B NSS peak at L4, etc.) plus basic shape/data-integrity checks,
    without recomputing anything new -- these are read directly off the
    curves already produced by common.load_all_model_curves."""
    problems = []
    for metric, _ in METRICS:
        for model_key in C.MODEL_ORDER:
            mean, lo, hi = curves_by_metric[metric][model_key]
            if len(mean) != 12:
                problems.append(f"{metric}/{model_key}: {len(mean)} layers, expected 12")
            if np.any(lo > mean + 1e-9) or np.any(hi < mean - 1e-9):
                problems.append(f"{metric}/{model_key}: CI does not bracket mean")
    return problems


def main():
    fig2, curves = make_figure(C.FIG_WIDTH_2COL_IN)
    problems = sanity_check(curves)
    if problems:
        raise RuntimeError("STOP: sanity check failed:\n" + "\n".join(problems))
    C.save_fig(fig2, C.FIG_DIR / "figure1_model_comparison_all_metrics_2col")
    plt.close(fig2)

    fig1, _ = make_figure(C.FIG_WIDTH_1COL_IN, layout="vertical")
    C.save_fig(fig1, C.FIG_DIR / "figure1_model_comparison_all_metrics_1col", svg=False)
    plt.close(fig1)

    print("[figure1-3panel] wrote figure1_model_comparison_all_metrics_2col.{pdf,png,svg} "
          "and _1col.{pdf,png}")

    print("\n--- sanity: peak layers / values per metric ---")
    for metric, _ in METRICS:
        for model_key in C.MODEL_ORDER:
            mean, _, _ = curves[metric][model_key]
            peak_l = int(np.argmax(mean)) + 1
            print(f"  {metric:9s} {model_key:8s} peak L{peak_l} = {mean[peak_l-1]:.4f}   "
                  f"L4={mean[3]:.4f} L8={mean[7]:.4f} L12={mean[11]:.4f}")


if __name__ == "__main__":
    main()
