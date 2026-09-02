"""
Figure 2 redraw: same 3-panel structure as figure2_gaze_and_probes.py
-- (a) CLIP-B human-gaze alignment (NSS), (b) semantic-attribute probe
AUPRC, (c) low-level visual-feature probe R^2 -- but panels (b)/(c) now
let individual attribute/feature curves be identified, not just the
group mean.

Produces two figures from the SAME underlying data:
  - "compact": for the main text. Mean line kept bold; only the top-3 and
    bottom-3 attributes/features (by mean AUPRC/R^2 across L1-L12) are
    drawn in color with a distinct linestyle/marker and directly labeled;
    the remaining attributes/features are thin light-gray background
    lines (no individual legend entries).
  - "full_labels": for the Appendix. All 12 attributes / 8 features are
    individually colored, distinctly styled (color + linestyle + marker,
    not color alone), and legended.

No new probe training, no new CLIP feature extraction, no bootstrap
re-run: panel (a) is imported unchanged from figure2_gaze_and_probes.py
(same NSS values, same 95% CI, same L4/L8/L12 highlighting). Panels
(b)/(c) read the exact same two canonical CSVs already used by the
existing Figure 2 (`probe_summary.csv`, `lowlevel_summary.csv`) plus, for
the ranking-CSV's baseline-corrected column only, the semantic probe's
existing permutation-baseline CSV (`probe_permutation_baseline.csv` --
no permutation baseline exists for the low-level probe, so that probe's
`permutation_mean`/`mean_above_permutation` are left blank, not
estimated).

Data sources (read-only; see paper_figures/FIGURE_AUDIT.md and this
figure's own figure2_probe_redraw_report.md for full provenance):
  outputs/expA_clip/healthy/metrics_per_image.csv           (panel a, via
      figure2_gaze_and_probes._panel_a, unchanged)
  outputs/osie_probe_all12_full700/probe_summary.csv         (panel b,
      metric=="auprc", condition=="all_objects", 12 attributes x 12
      layers, mean over 5 CV folds; also random_auprc = prevalence)
  outputs/osie_probe_all12_full700/probe_permutation_baseline.csv
      (ranking CSV only: permutation-shuffled-label AUPRC baseline, same
      filter, 12 attributes x 12 layers)
  outputs/osie_lowlevel_probe_all12_full700/lowlevel_summary.csv
      (panel c, metric=="r2", 8 visual-feature targets x 12 layers, mean
      over 5 CV folds; geometry-control targets centroid_x/y,
      mask_area_fraction excluded, same convention as the original
      Figure 2)

Ranking caveat (stated in the figure's caption and in the redraw report,
not just in code comments): AUPRC depends on each attribute's positive
prevalence, so "top"/"bottom" by raw mean AUPRC is a descriptive ranking
of THIS experiment's probe performance, not a claim about which
attributes CLIP is intrinsically better "suited" to, and not a claim that
high-AUPRC attributes are causally used by CLIP.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.paper_figures import common as C
from scripts.paper_figures import figure2_gaze_and_probes as F2

PERMUTATION_BASELINE_CSV = (
    C.OUT_DIR / "osie_probe_all12_full700" / "probe_permutation_baseline.csv"
)

MIN_LABEL_FONT_PT = 7.0
MEAN_LINE_COLOR = "#000000"  # kept distinct from every individual attribute/feature color

# 12 visually distinct (color, linestyle, marker) combinations -- color is
# never the only distinguishing channel (linestyle/marker always differ
# too). Colors are the colorblind-safe Okabe-Ito set, EXCLUDING black
# (reserved exclusively for the bold "Mean" line, so no individual curve
# ever shares its color) and pure yellow (too low-contrast as a line
# color on white).
STYLE_SLOTS = [
    dict(color="#0072B2", ls="-", marker="o"),
    dict(color="#D55E00", ls="-", marker="s"),
    dict(color="#009E73", ls="-", marker="^"),
    dict(color="#CC79A7", ls="-", marker="D"),
    dict(color="#E69F00", ls="-", marker="v"),
    dict(color="#56B4E9", ls="-", marker="P"),
    dict(color="#0072B2", ls="--", marker="s"),
    dict(color="#D55E00", ls="--", marker="o"),
    dict(color="#009E73", ls="--", marker="D"),
    dict(color="#CC79A7", ls="--", marker="^"),
    dict(color="#E69F00", ls="--", marker="P"),
    dict(color="#56B4E9", ls="--", marker="v"),
]


def style_slots(n):
    if n > len(STYLE_SLOTS):
        raise ValueError(f"need {n} style slots, only {len(STYLE_SLOTS)} defined")
    return STYLE_SLOTS[:n]


# ---------------------------------------------------------------------------
# Ranking computation (also drives compact-panel top3/bottom3 selection)
# ---------------------------------------------------------------------------

def compute_semantic_ranking():
    summ = C.load_semantic_probe_table()  # metric==auprc, condition==all_objects
    perm = pd.read_csv(PERMUTATION_BASELINE_CSV)
    perm = perm[(perm["metric"] == "auprc") & (perm["condition"] == "all_objects")]

    rows = []
    for attr in C.SEMANTIC_ATTRIBUTES:
        s = summ[summ["attribute"] == attr].set_index("layer")["mean"].loc[C.LAYERS]
        prevalence = summ[summ["attribute"] == attr]["random_auprc"].iloc[0]
        p = perm[perm["attribute"] == attr].set_index("layer")["mean"]
        p = p.loc[C.LAYERS] if set(C.LAYERS).issubset(p.index) else None

        mean_raw = float(s.mean())
        peak_layer = int(s.idxmax())
        peak_score = float(s.max())
        if p is not None:
            permutation_mean = float(p.mean())
            mean_above_permutation = float((s - p).mean())
        else:
            permutation_mean = np.nan
            mean_above_permutation = np.nan

        rows.append(dict(
            probe_type="semantic_attribute", label=attr,
            mean_score_L1_L12=mean_raw, peak_score=peak_score, peak_layer=peak_layer,
            prevalence=float(prevalence), permutation_mean=permutation_mean,
            mean_above_permutation=mean_above_permutation,
        ))
    df = pd.DataFrame(rows)
    df["rank_raw"] = df["mean_score_L1_L12"].rank(ascending=False).astype(int)
    df["rank_baseline_corrected"] = df["mean_above_permutation"].rank(ascending=False).astype(int)
    return df


def compute_lowlevel_ranking():
    low = C.load_lowlevel_probe_table()  # metric==r2
    rows = []
    for feat in C.LOWLEVEL_FEATURES:
        s = low[low["target"] == feat].set_index("layer")["mean"].loc[C.LAYERS]
        rows.append(dict(
            probe_type="lowlevel_feature", label=feat,
            mean_score_L1_L12=float(s.mean()), peak_score=float(s.max()),
            peak_layer=int(s.idxmax()), prevalence=np.nan,
            permutation_mean=np.nan, mean_above_permutation=np.nan,
        ))
    df = pd.DataFrame(rows)
    df["rank_raw"] = df["mean_score_L1_L12"].rank(ascending=False).astype(int)
    df["rank_baseline_corrected"] = np.nan  # no permutation baseline exists for this probe
    return df


def assign_compact_group(df, n_top=3, n_bottom=3):
    """compact_group assigned from the RAW mean-score ranking (per task
    spec: "上位・下位の順位は、原則として...観測AUPRC/R^2の平均で決定")."""
    df = df.sort_values("mean_score_L1_L12", ascending=False).reset_index(drop=True)
    n = len(df)
    group = np.array(["other"] * n, dtype=object)
    group[:n_top] = "top3"
    group[n - n_bottom:] = "bottom3"
    df["compact_group"] = group
    return df


def build_ranking_table():
    sem = assign_compact_group(compute_semantic_ranking())
    low = assign_compact_group(compute_lowlevel_ranking())
    cols = ["probe_type", "label", "mean_score_L1_L12", "peak_score", "peak_layer",
            "prevalence", "permutation_mean", "mean_above_permutation", "compact_group",
            "rank_raw", "rank_baseline_corrected"]
    return pd.concat([sem[cols], low[cols]], ignore_index=True)


# ---------------------------------------------------------------------------
# Label de-collision for end-of-line direct labels (no adjustText dependency)
# ---------------------------------------------------------------------------

def declash_positions(values, min_gap):
    """values: array of target y positions (data units), sorted order not
    assumed. Returns adjusted y positions with a minimum vertical gap,
    nudging outward from the median while preserving the original order."""
    order = np.argsort(values)
    sorted_vals = np.array(values, dtype=float)[order]
    adjusted = sorted_vals.copy()
    for i in range(1, len(adjusted)):
        if adjusted[i] - adjusted[i - 1] < min_gap:
            adjusted[i] = adjusted[i - 1] + min_gap
    # re-center the adjusted block on the original block's center to avoid drift
    if len(adjusted):
        adjusted += (sorted_vals.mean() - adjusted.mean())
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _draw_panel_bc_compact(ax, layer_table, labels, ranking_df, mean_color, marker,
                            ylabel, title, y_ref_zero=False):
    top3 = ranking_df[ranking_df["compact_group"] == "top3"]["label"].tolist()
    bottom3 = ranking_df[ranking_df["compact_group"] == "bottom3"]["label"].tolist()
    highlighted = top3 + bottom3
    others = [l for l in labels if l not in highlighted]

    C.mark_highlight_layers(ax)
    if y_ref_zero:
        ax.axhline(0.0, color="#aaaaaa", lw=0.6, ls="-", zorder=0)
    for lab in others:
        ax.plot(C.LAYERS, layer_table[lab].to_numpy(), **C.AUX_LINE_STYLE)

    slots = style_slots(len(highlighted))
    end_vals = []
    for lab, st in zip(highlighted, slots):
        y = layer_table[lab].to_numpy()
        ax.plot(C.LAYERS, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=3.2, lw=1.3, zorder=4)
        end_vals.append(y[-1])
    end_vals = np.array(end_vals)
    y_span = layer_table[labels].to_numpy().max() - layer_table[labels].to_numpy().min()
    label_y = declash_positions(end_vals, min_gap=max(y_span * 0.045, 1e-6))
    for lab, st, ly in zip(highlighted, slots, label_y):
        ax.annotate(lab, xy=(12, layer_table[lab].to_numpy()[-1]),
                    xytext=(12.3, ly), fontsize=MIN_LABEL_FONT_PT, color=st["color"],
                    va="center", ha="left", annotation_clip=False)

    group_mean = layer_table[labels].mean(axis=1).to_numpy()
    ax.plot(C.LAYERS, group_mean, color=mean_color, lw=2.4, ls="-", marker=marker,
            ms=3.6, zorder=5, label="Mean")
    C.style_axes(ax, ylabel=ylabel)
    ax.set_xlim(0.5, 13.6)  # extra right margin for end-of-line labels (overrides style_axes default)
    ax.set_title(title, loc="left", fontsize=C.BASE_FONT_PT)
    ax.legend(loc="upper left" if not y_ref_zero else "lower left", frameon=False,
              fontsize=MIN_LABEL_FONT_PT, handlelength=1.6)
    return group_mean


def _draw_panel_bc_full(ax, layer_table, labels, mean_color, marker, ylabel, title,
                         y_ref_zero=False, legend_ncol=4):
    C.mark_highlight_layers(ax)
    if y_ref_zero:
        ax.axhline(0.0, color="#aaaaaa", lw=0.6, ls="-", zorder=0)
    slots = style_slots(len(labels))
    for lab, st in zip(labels, slots):
        ax.plot(C.LAYERS, layer_table[lab].to_numpy(), color=st["color"], ls=st["ls"],
                marker=st["marker"], ms=2.8, lw=1.1, zorder=3, label=lab)
    group_mean = layer_table[labels].mean(axis=1).to_numpy()
    ax.plot(C.LAYERS, group_mean, color=mean_color, lw=2.6, ls="-", marker=marker,
            ms=3.8, zorder=5, label="Mean")
    C.style_axes(ax, ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=C.BASE_FONT_PT)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False,
              fontsize=MIN_LABEL_FONT_PT, handlelength=1.6, ncol=1, borderaxespad=0.0)
    return group_mean


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def make_compact_figure(ranking_df):
    C.apply_paper_style()
    sem_df = C.load_semantic_probe_table()
    sem_table = sem_df.pivot_table(index="layer", columns="attribute", values="mean").loc[C.LAYERS]
    low_df = C.load_lowlevel_probe_table()
    low_table = low_df.pivot_table(index="layer", columns="target", values="mean").loc[C.LAYERS]

    fig, axes = plt.subplots(3, 1, figsize=(C.FIG_WIDTH_2COL_IN, 2.35 * 3),
                              constrained_layout=True, sharex=False)
    F2._panel_a(axes[0])
    axes[0].set_xlim(0.5, 13.6)

    sem_rank = ranking_df[ranking_df["probe_type"] == "semantic_attribute"]
    _draw_panel_bc_compact(axes[1], sem_table, C.SEMANTIC_ATTRIBUTES, sem_rank,
                            MEAN_LINE_COLOR, "s", "AUPRC",
                            "(b) Semantic-attribute probe (top 3 / bottom 3 labeled)")

    low_rank = ranking_df[ranking_df["probe_type"] == "lowlevel_feature"]
    _draw_panel_bc_compact(axes[2], low_table, C.LOWLEVEL_FEATURES, low_rank,
                            MEAN_LINE_COLOR, "D", "Out-of-fold $R^2$",
                            "(c) Low-level visual-feature probe (top 3 / bottom 3 labeled)",
                            y_ref_zero=True)

    for ax in axes:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Layer")
    return fig


def make_full_label_figure():
    C.apply_paper_style()
    sem_df = C.load_semantic_probe_table()
    sem_table = sem_df.pivot_table(index="layer", columns="attribute", values="mean").loc[C.LAYERS]
    low_df = C.load_lowlevel_probe_table()
    low_table = low_df.pivot_table(index="layer", columns="target", values="mean").loc[C.LAYERS]

    fig, axes = plt.subplots(3, 1, figsize=(C.FIG_WIDTH_2COL_IN, 2.55 * 3),
                              constrained_layout=True, sharex=False)
    F2._panel_a(axes[0])

    _draw_panel_bc_full(axes[1], sem_table, C.SEMANTIC_ATTRIBUTES, MEAN_LINE_COLOR, "s",
                         "AUPRC", "(b) Semantic-attribute probe (all 12 attributes)")
    _draw_panel_bc_full(axes[2], low_table, C.LOWLEVEL_FEATURES, MEAN_LINE_COLOR, "D",
                         "Out-of-fold $R^2$", "(c) Low-level visual-feature probe (all 8 features)",
                         y_ref_zero=True)

    for ax in axes:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Layer")
    return fig


# ---------------------------------------------------------------------------
# Sanity checks against already-established canonical values
# ---------------------------------------------------------------------------

def run_sanity_checks(ranking_df):
    """Cross-checks against the already-established canonical values (see
    paper_figures/FIGURE_AUDIT.md and results_fact_check_report.md) --
    does not overwrite or recompute those values, only confirms this
    script's own aggregation reproduces them from the same source files."""
    problems = []

    def check(name, cond, detail=""):
        if not cond:
            problems.append(f"{name}: {detail}")

    sem_df = C.load_semantic_probe_table()
    sem_table = sem_df.pivot_table(index="layer", columns="attribute", values="mean").loc[C.LAYERS]
    sem_mean = sem_table[C.SEMANTIC_ATTRIBUTES].mean(axis=1)
    check("semantic mean AUPRC L1 ~ 0.22", abs(sem_mean.loc[1] - 0.22) < 0.01,
          f"got {sem_mean.loc[1]:.4f}")
    check("semantic mean AUPRC L10 ~ 0.60", abs(sem_mean.loc[10] - 0.60) < 0.01,
          f"got {sem_mean.loc[10]:.4f}")

    sem_rank = ranking_df[ranking_df["probe_type"] == "semantic_attribute"]
    n_peak_8_11 = sem_rank["peak_layer"].between(8, 11).sum()
    check("10 of 12 semantic attributes peak in L8-L11", n_peak_8_11 == 10,
          f"got {n_peak_8_11}: {sem_rank[['label', 'peak_layer']].to_dict('records')}")
    peak_by_label = sem_rank.set_index("label")["peak_layer"]
    check("Touch peaks at L12", peak_by_label.get("Touch") == 12,
          f"got L{peak_by_label.get('Touch')}")
    check("Sound peaks at L12", peak_by_label.get("Sound") == 12,
          f"got L{peak_by_label.get('Sound')}")

    low_df = C.load_lowlevel_probe_table()
    low_table = low_df.pivot_table(index="layer", columns="target", values="mean").loc[C.LAYERS]
    low_mean = low_table[C.LOWLEVEL_FEATURES].mean(axis=1)
    check("low-level mean R2 at L4 ~ 0.80", abs(low_mean.loc[4] - 0.80) < 0.01,
          f"got {low_mean.loc[4]:.4f}")

    low_rank = ranking_df[ranking_df["probe_type"] == "lowlevel_feature"]
    n_peak_1_4 = low_rank["peak_layer"].between(1, 4).sum()
    check("7 of 8 low-level features peak in L1-L4", n_peak_1_4 == 7,
          f"got {n_peak_1_4}: {low_rank[['label', 'peak_layer']].to_dict('records')}")
    lc_peak = low_rank.set_index("label").loc["luminance_contrast", "peak_layer"]
    check("luminance_contrast peaks near L6-L8 (not L1-L4)", lc_peak in (6, 7, 8),
          f"got L{lc_peak}")

    return problems


def main():
    ranking_df = build_ranking_table()

    problems = run_sanity_checks(ranking_df)
    if problems:
        raise RuntimeError("STOP: sanity check failed:\n" + "\n".join(problems))
    print("[figure2-redraw] sanity checks vs. established canonical values: all passed")

    ranking_csv = C.FIG_DIR / "figure2_probe_label_ranking.csv"
    ranking_df.to_csv(ranking_csv, index=False)
    print(f"[figure2-redraw] wrote {ranking_csv}")

    fig_compact = make_compact_figure(ranking_df)
    C.save_fig(fig_compact, C.FIG_DIR / "figure2_gaze_probe_profiles_compact")
    plt.close(fig_compact)

    fig_full = make_full_label_figure()
    C.save_fig(fig_full, C.FIG_DIR / "figure2_gaze_probe_profiles_full_labels")
    plt.close(fig_full)

    print("[figure2-redraw] wrote figure2_gaze_probe_profiles_compact.{pdf,png,svg} "
          "and figure2_gaze_probe_profiles_full_labels.{pdf,png,svg}")
    return ranking_df


if __name__ == "__main__":
    main()
