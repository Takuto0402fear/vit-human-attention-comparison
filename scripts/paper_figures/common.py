"""
Shared constants, styling, data-loading and statistics helpers for the
LAU-conference paper figures (Figures 1-3 + Appendix).

This module reads ONLY existing files under outputs/ and datasets/ --
it never re-runs model inference, probe training, or attention extraction.
All aggregation performed here (bootstrap CI over images) uses the SAME
convention already established elsewhere in this repository:
  seed=42, n_boot=10000, alpha=0.05, image-ID-level resampling
  (see scripts/stats_three_model_layerwise.py:BOOT_SEED/N_BOOT/ALPHA and
  outputs/clip_attention_b2_distribution/config.json/summary.json).

See paper_figures/FIGURE_AUDIT.md for the full provenance of every number
plotted by the scripts in this directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs"
DATASET_DIR = REPO_ROOT / "datasets" / "osie" / "predicting-human-gaze-beyond-pixels"
STIMULI_DIR = DATASET_DIR / "data" / "stimuli"

FIG_DIR = REPO_ROOT / "paper_figures"
APPENDIX_DIR = FIG_DIR / "appendix"
FIG_DIR.mkdir(parents=True, exist_ok=True)
APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Repo-wide bootstrap convention (reused, not reinvented -- see docstring).
# ---------------------------------------------------------------------------
BOOT_SEED = 42
N_BOOT = 10000
ALPHA = 0.05
LAYERS = list(range(1, 13))
HIGHLIGHT_LAYERS = (4, 8, 12)

# ---------------------------------------------------------------------------
# Canonical per-model source files (identified in FIGURE_AUDIT.md section 1).
# Each is a per-image, per-layer long table with columns
# ["image", "layer", "NSS", "AUC_Judd"/"AUC-Judd", "sAUC", ...] except the
# DeiT/SL file, whose per-image NSS/AUC_Judd/sAUC columns are already the
# mean across its 6 trials for that image (the repo's own established
# primary-analysis convention -- see outputs/model_comparison_statistics/
# statistics_summary.md, line 3: "DeiT/SL primary analysis uses its 6-trial
# MEAN, never the pooled 6x700 rows").
# ---------------------------------------------------------------------------
MODEL_SOURCES = {
    "CLIP-B": {
        "display": "CLIP ViT-B/16",
        "per_image_csv": OUT_DIR / "expA_clip" / "healthy" / "metrics_per_image.csv",
        "col_map": {"NSS": "NSS", "AUC_Judd": "AUC_Judd", "sAUC": "sAUC"},
    },
    "DINO-S": {
        "display": "DINO ViT-S/16",
        "per_image_csv": OUT_DIR / "expA" / "healthy" / "metrics_per_image.csv",
        "col_map": {"NSS": "NSS", "AUC_Judd": "AUC_Judd", "sAUC": "sAUC"},
    },
    "DINO-B": {
        "display": "DINO ViT-B/16",
        "per_image_csv": OUT_DIR / "expA_dino_vitb16" / "healthy" / "metrics_per_image.csv",
        "col_map": {"NSS": "NSS", "AUC_Judd": "AUC_Judd", "sAUC": "sAUC"},
    },
    "DeiT/SL": {
        "display": "DeiT/SL ViT-S/16",
        "per_image_csv": OUT_DIR / "expA_sl" / "combined" / "metrics_by_image_meanacrosstrials.csv",
        "col_map": {
            "NSS": "NSS_mean_across_trials",
            "AUC_Judd": "AUC_Judd_mean_across_trials",
            "sAUC": "sAUC_mean_across_trials",
        },
    },
}
MODEL_ORDER = ["CLIP-B", "DINO-S", "DINO-B", "DeiT/SL"]

SEMANTIC_PROBE_CSV = OUT_DIR / "osie_probe_all12_full700" / "probe_summary.csv"
SEMANTIC_ATTRIBUTES = [
    "Text", "Face", "Emotion", "Sound", "Smell", "Taste",
    "Touch", "Motion", "Operability", "Watchability", "Touched", "Gazed",
]

LOWLEVEL_PROBE_CSV = OUT_DIR / "osie_lowlevel_probe_all12_full700" / "lowlevel_summary.csv"
LOWLEVEL_FEATURES = [
    "mean_luminance", "mean_red", "mean_green", "mean_blue",
    "mean_saturation", "edge_strength", "fine_texture", "luminance_contrast",
]
LOWLEVEL_FEATURES_EXCLUDED_GEOMETRY = ["centroid_x", "centroid_y", "mask_area_fraction"]

B2_DIR = OUT_DIR / "clip_attention_b2_distribution"
B2_SUMMARY_JSON = B2_DIR / "summary.json"
B2_CONFIG_JSON = B2_DIR / "config.json"
ATTN_CACHE_NPZ = (
    OUT_DIR / "osie_attribute_grounding_full700" / "attn_cache"
    / "clip_vitb16_full700_L4L8L12_patchgrid.npz"
)
FIXMAP_DIR = OUT_DIR / "fixmaps" / "healthy"
REPRESENTATIVE_IMAGE_STEMS = ["1156", "1159", "1213"]  # see FIGURE_AUDIT.md for selection basis

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Okabe & Ito, 2008) + distinct linestyles/markers
# so every curve remains distinguishable in black-and-white print.
# ---------------------------------------------------------------------------
MODEL_STYLE = {
    "CLIP-B": dict(color="#000000", lw=2.6, ls="-", marker="o", ms=4.2, zorder=5),
    "DINO-S": dict(color="#0072B2", lw=1.2, ls="--", marker="^", ms=3.2, zorder=3),
    "DINO-B": dict(color="#009E73", lw=1.2, ls="-.", marker="s", ms=3.0, zorder=3),
    "DeiT/SL": dict(color="#D55E00", lw=1.2, ls=":", marker="D", ms=3.0, zorder=3),
}
PANEL_A_COLOR = MODEL_STYLE["CLIP-B"]["color"]
PANEL_B_COLOR = "#0072B2"   # semantic-attribute probe (blue)
PANEL_C_COLOR = "#D55E00"   # low-level-feature probe (vermillion)
AUX_LINE_STYLE = dict(color="#999999", lw=0.6, alpha=0.55, zorder=1)

FIG_WIDTH_1COL_IN = 3.4
FIG_WIDTH_2COL_IN = 7.0
BASE_FONT_PT = 9

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Georgia", "DejaVu Serif"],
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": BASE_FONT_PT - 1,
        "ytick.labelsize": BASE_FONT_PT - 1,
        "legend.fontsize": BASE_FONT_PT - 1.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,   # embed as real (vector/searchable) TrueType, not paths
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep text as text in SVG output
    })


def mark_highlight_layers(ax, layers=HIGHLIGHT_LAYERS, color="#777777", alpha=0.08):
    for l in layers:
        ax.axvspan(l - 0.32, l + 0.32, color=color, alpha=alpha, zorder=0, lw=0)


def style_axes(ax, xlabel="Layer", ylabel=None):
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(LAYERS)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)


def save_fig(fig, out_path_noext: Path, svg=True, dpi=300):
    out_path_noext = Path(out_path_noext)
    out_path_noext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path_noext) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path_noext) + ".png", dpi=dpi, bbox_inches="tight")
    if svg:
        fig.savefig(str(out_path_noext) + ".svg", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Bootstrap CI (image-ID-level resampling; reused convention, see docstring).
# ---------------------------------------------------------------------------

def make_boot_indices(n_images, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_images, size=(n_boot, n_images))


def bootstrap_mean_ci(values_2d, boot_indices=None, alpha=ALPHA):
    """values_2d: (n_images, n_layers) -> (mean, ci_lo, ci_hi) each (n_layers,)."""
    values_2d = np.asarray(values_2d, dtype=np.float64)
    n_images = values_2d.shape[0]
    if boot_indices is None:
        boot_indices = make_boot_indices(n_images)
    mean = values_2d.mean(axis=0)
    boot_means = values_2d[boot_indices].mean(axis=1)  # (n_boot, n_layers)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_model_layer_matrix(model_key, metric):
    """Return (image_ids sorted, (n_images, 12) array) for one model/metric,
    read directly from that model's canonical per-image CSV (see
    MODEL_SOURCES / FIGURE_AUDIT.md)."""
    spec = MODEL_SOURCES[model_key]
    df = pd.read_csv(spec["per_image_csv"])
    col = spec["col_map"][metric]
    df = df[["image", "layer", col]].copy()
    df = df.sort_values(["image", "layer"])
    images = sorted(df["image"].unique())
    n_images = len(images)
    pivot = df.pivot(index="image", columns="layer", values=col).loc[images, LAYERS]
    if pivot.isna().any().any():
        raise RuntimeError(f"STOP: missing values in {spec['per_image_csv']} for metric {metric}")
    return images, pivot.to_numpy()


def load_all_model_curves(metric):
    """Return {model_key: (mean(12,), lo(12,), hi(12,))} for one metric,
    each model bootstrapped independently (image-level, seed=42, n_boot=10000)
    over its own 700 images -- see FIGURE_AUDIT.md for why this is a
    methodologically unified but per-model-independent resampling."""
    out = {}
    for model_key in MODEL_ORDER:
        _, mat = load_model_layer_matrix(model_key, metric)
        boot_idx = make_boot_indices(mat.shape[0])
        mean, lo, hi = bootstrap_mean_ci(mat, boot_idx)
        out[model_key] = (mean, lo, hi)
    return out


def load_semantic_probe_table():
    df = pd.read_csv(SEMANTIC_PROBE_CSV)
    df = df[(df["metric"] == "auprc") & (df["condition"] == "all_objects")]
    missing = set(SEMANTIC_ATTRIBUTES) - set(df["attribute"].unique())
    if missing:
        raise RuntimeError(f"STOP: semantic attributes missing from {SEMANTIC_PROBE_CSV}: {missing}")
    return df


def load_lowlevel_probe_table():
    df = pd.read_csv(LOWLEVEL_PROBE_CSV)
    df = df[df["metric"] == "r2"]
    missing = set(LOWLEVEL_FEATURES) - set(df["target"].unique())
    if missing:
        raise RuntimeError(f"STOP: low-level features missing from {LOWLEVEL_PROBE_CSV}: {missing}")
    return df
