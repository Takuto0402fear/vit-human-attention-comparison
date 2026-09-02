"""
Appendix figures (same style/provenance conventions as Figures 1-2).

1. appendix_aucjudd_model_comparison  -- 4-model layer-wise AUC-Judd
2. appendix_sauc_model_comparison     -- 4-model layer-wise sAUC
3. appendix_clip_three_metrics        -- CLIP-B only, NSS/AUC-Judd/sAUC
                                          in one 3-panel supplementary figure
4. appendix_semantic_attributes_individual -- 12 small multiples, one per
                                          OSIE semantic attribute (AUPRC)
5. appendix_lowlevel_features_individual   -- 8 small multiples, one per
                                          low-level visual feature (R^2)

Data sources: identical to figure1_model_comparison_nss.py and
figure2_gaze_and_probes.py -- see those modules' docstrings and
paper_figures/FIGURE_AUDIT.md. Appendix (4) and (5) additionally show each
attribute/feature's mean +/- 1 SD ACROSS THE 5 CV FOLDS (not an image-level
bootstrap CI) -- this is a different unit of uncertainty than Figures 1-2's
image-level bootstrap bands, and is labeled as such in each panel's caption
text and in FIGURE_AUDIT.md.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from scripts.paper_figures import common as C
from scripts.paper_figures import figure1_model_comparison_nss as F1


def appendix_model_comparison(metric, ylabel, out_name, title_note):
    C.apply_paper_style()
    curves = C.load_all_model_curves(metric)
    fig, ax = plt.subplots(figsize=(C.FIG_WIDTH_2COL_IN, 3.4), constrained_layout=True)
    C.mark_highlight_layers(ax)
    for model_key in C.MODEL_ORDER:
        mean, lo, hi = curves[model_key]
        st = C.MODEL_STYLE[model_key]
        ax.fill_between(C.LAYERS, lo, hi, color=st["color"],
                         alpha=0.15 if model_key == "CLIP-B" else 0.10, linewidth=0)
        ax.plot(C.LAYERS, mean, color=st["color"], lw=st["lw"], ls=st["ls"],
                 marker=st["marker"], ms=st["ms"], zorder=st["zorder"],
                 label=f"{model_key} ({C.MODEL_SOURCES[model_key]['display']})")
    C.style_axes(ax, ylabel=ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False, handlelength=2.4)
    C.save_fig(fig, C.APPENDIX_DIR / out_name)
    plt.close(fig)
    print(f"[appendix] wrote {out_name}.{{pdf,png,svg}} ({title_note})")


def appendix_clip_three_metrics():
    C.apply_paper_style()
    metrics = [("NSS", "NSS"), ("AUC_Judd", "AUC-Judd"), ("sAUC", "sAUC")]
    fig, axes = plt.subplots(1, 3, figsize=(C.FIG_WIDTH_2COL_IN, 2.5), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        _, mat = C.load_model_layer_matrix("CLIP-B", metric)
        boot_idx = C.make_boot_indices(mat.shape[0])
        mean, lo, hi = C.bootstrap_mean_ci(mat, boot_idx)
        C.mark_highlight_layers(ax)
        ax.fill_between(C.LAYERS, lo, hi, color=C.PANEL_A_COLOR, alpha=0.15, linewidth=0)
        ax.plot(C.LAYERS, mean, color=C.PANEL_A_COLOR, lw=2.0, marker="o", ms=3.2)
        C.style_axes(ax, ylabel=ylabel)
    C.save_fig(fig, C.APPENDIX_DIR / "appendix_clip_three_metrics")
    plt.close(fig)
    print("[appendix] wrote appendix_clip_three_metrics.{pdf,png,svg}")


def _small_multiples(items, table, value_col_getter, ylabel, ncols, out_name, ref_zero=False):
    C.apply_paper_style()
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(C.FIG_WIDTH_2COL_IN, 2.05 * nrows),
                              constrained_layout=True, sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    for i, item in enumerate(items):
        ax = axes_flat[i]
        mean, std = value_col_getter(table, item)
        C.mark_highlight_layers(ax)
        if ref_zero:
            ax.axhline(0.0, color="#aaaaaa", lw=0.5, zorder=0)
        ax.fill_between(C.LAYERS, mean - std, mean + std, color=C.PANEL_B_COLOR, alpha=0.20,
                         linewidth=0)
        ax.plot(C.LAYERS, mean, color=C.PANEL_B_COLOR, lw=1.4, marker="o", ms=2.2)
        ax.set_title(item, fontsize=C.BASE_FONT_PT - 1)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        C.style_axes(ax, xlabel="")
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    for ax in axes_flat[max(0, n - ncols):n]:
        ax.set_xlabel("Layer")
    C.save_fig(fig, C.APPENDIX_DIR / out_name)
    plt.close(fig)
    print(f"[appendix] wrote {out_name}.{{pdf,png,svg}}")


def appendix_semantic_attributes_individual():
    df = C.load_semantic_probe_table()
    pivot_mean = df.pivot_table(index="layer", columns="attribute", values="mean").loc[C.LAYERS]
    pivot_std = df.pivot_table(index="layer", columns="attribute", values="std").loc[C.LAYERS]

    def getter(_, attr):
        return pivot_mean[attr].to_numpy(), pivot_std[attr].to_numpy()

    _small_multiples(C.SEMANTIC_ATTRIBUTES, None, getter, "AUPRC", ncols=4,
                      out_name="appendix_semantic_attributes_individual")


def appendix_lowlevel_features_individual():
    df = C.load_lowlevel_probe_table()
    pivot_mean = df.pivot_table(index="layer", columns="target", values="mean").loc[C.LAYERS]
    pivot_std = df.pivot_table(index="layer", columns="target", values="std").loc[C.LAYERS]

    def getter(_, feat):
        return pivot_mean[feat].to_numpy(), pivot_std[feat].to_numpy()

    _small_multiples(C.LOWLEVEL_FEATURES, None, getter, "$R^2$", ncols=4,
                      out_name="appendix_lowlevel_features_individual", ref_zero=True)


def main():
    appendix_model_comparison("AUC_Judd", "AUC-Judd", "appendix_aucjudd_model_comparison",
                               "4-model AUC-Judd")
    appendix_model_comparison("sAUC", "sAUC", "appendix_sauc_model_comparison", "4-model sAUC")
    appendix_clip_three_metrics()
    appendix_semantic_attributes_individual()
    appendix_lowlevel_features_individual()


if __name__ == "__main__":
    main()
