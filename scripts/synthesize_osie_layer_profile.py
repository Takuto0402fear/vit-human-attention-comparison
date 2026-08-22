"""
Phase 4 + 5: integrates (1) the canonical CLIP ViT-B/16 / OSIE-700 /
"healthy" human-gaze agreement profile, (2) the semantic-attribute probe
(Phase 1, outputs/osie_probe_all12_full700/), and (3) the low-level
visual-feature probe (Phase 3, outputs/osie_lowlevel_probe_all12_full700/)
into one layer-profile table, computes exploratory (n=12 layers) curve
correlations, and renders the final H1-H4 hypothesis judgment.

Canonical attention/gaze source (identified by explicit provenance chain,
not by filename guessing -- see input_sources.json for the full audit
trail): outputs/expA_clip/healthy/metrics_by_layer.csv. This file's
provenance was confirmed via:
  - its own run_config.json ("model": "clip_vitb16", 700 images, "healthy")
  - scripts/run_expA_clip_full700.py's docstring (explicitly the
    production 700-image run; pilot20 is a separate, smaller run that
    never writes here)
  - outputs/model_comparison_layerwise/comparison_sources.json, which
    cites this exact path as the CLIP-B source for its own multi-model
    comparison and explicitly EXCLUDES pilot20 to avoid double-counting

Does not re-run any CLIP attention extraction -- reads the existing
result only. Writes only under outputs/osie_layer_profile_synthesis/.

Usage (PowerShell):
    python scripts\\synthesize_osie_layer_profile.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
ATTENTION_CSV = r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_by_layer.csv"
ATTENTION_RUN_CONFIG = r"C:\Users\user\gaze\outputs\expA_clip\healthy\run_config.json"

SEMANTIC_SUMMARY_CSV = r"C:\Users\user\gaze\outputs\osie_probe_all12_full700\probe_summary.csv"
SEMANTIC_PEAKS_CSV = r"C:\Users\user\gaze\outputs\osie_probe_all12_full700\probe_peak_layers.csv"
LOWLEVEL_SUMMARY_CSV = r"C:\Users\user\gaze\outputs\osie_lowlevel_probe_all12_full700\lowlevel_summary.csv"
LOWLEVEL_PEAKS_CSV = r"C:\Users\user\gaze\outputs\osie_lowlevel_probe_all12_full700\lowlevel_peak_layers.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_layer_profile_synthesis"
FIG_DIR = os.path.join(OUT_DIR, "figures")

SEMANTIC_ATTRIBUTES = [
    "Text", "Face", "Emotion", "Sound", "Smell", "Taste", "Touch",
    "Motion", "Operability", "Watchability", "Touched", "Gazed",
]
LOWLEVEL_TARGETS = [
    "mean_luminance", "mean_red", "mean_green", "mean_blue", "mean_saturation",
    "luminance_contrast", "edge_strength", "fine_texture",
]
LAYER_GROUPS = {"early": [1, 2, 3, 4], "middle": [5, 6, 7, 8], "late": [9, 10, 11, 12]}

INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR_NSS, COLOR_AUCJ, COLOR_SAUC = "#2a78d6", "#eb6834", "#1baf7a"
COLOR_SEMANTIC, COLOR_LOWLEVEL = "#4a3aa7", "#e34948"
# ======================================================


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_attention_curve():
    rows = read_csv_dicts(ATTENTION_CSV)
    by_layer = {int(r["layer"]): r for r in rows}
    if sorted(by_layer.keys()) != list(range(1, 13)):
        raise RuntimeError(f"STOP: {ATTENTION_CSV} does not have exactly layers 1-12")
    return by_layer


def load_semantic_curves():
    if not os.path.isfile(SEMANTIC_SUMMARY_CSV):
        return None
    rows = read_csv_dicts(SEMANTIC_SUMMARY_CSV)
    by_attr_layer = {}
    for r in rows:
        if r.get("metric") != "auprc" or r.get("condition") != "all_objects" or r["mean"] == "":
            continue
        by_attr_layer.setdefault(r["attribute"], {})[int(r["layer"])] = float(r["mean"])
    return by_attr_layer


def load_lowlevel_curves():
    if not os.path.isfile(LOWLEVEL_SUMMARY_CSV):
        return None
    rows = read_csv_dicts(LOWLEVEL_SUMMARY_CSV)
    by_target_layer = {}
    for r in rows:
        if r.get("metric") != "r2" or r["mean"] == "":
            continue
        by_target_layer.setdefault(r["target"], {})[int(r["layer"])] = float(r["mean"])
    return by_target_layer


def build_input_sources_json(attention_by_layer, semantic_curves, lowlevel_curves):
    sources = {
        "attention_human_gaze": {
            "path": ATTENTION_CSV, "columns": "layer,NSS_mean,NSS_std,AUC_Judd_mean,AUC_Judd_std,sAUC_mean,sAUC_std",
            "n_rows": 12, "model": "clip_vitb16", "group": "healthy", "n_images": 700,
            "provenance": [
                "run_config.json in the same directory confirms model=clip_vitb16, 700 images, group=healthy",
                "scripts/run_expA_clip_full700.py docstring: production 700-image run (not pilot20)",
                "outputs/model_comparison_layerwise/comparison_sources.json cites this exact path for "
                "'clip_b' and explicitly excludes pilot20 to avoid double-counting",
            ],
        },
        "semantic_attribute_probe": {
            "path": SEMANTIC_SUMMARY_CSV, "available": semantic_curves is not None,
            "n_attributes": len(SEMANTIC_ATTRIBUTES), "metric_used": "auprc, condition=all_objects",
            "source_script": "scripts/probe_osie_all12_full700.py",
        },
        "lowlevel_feature_probe": {
            "path": LOWLEVEL_SUMMARY_CSV, "available": lowlevel_curves is not None,
            "n_targets": len(LOWLEVEL_TARGETS), "metric_used": "r2 (out-of-fold)",
            "source_script": "scripts/probe_osie_lowlevel_all12_full700.py",
        },
    }
    if os.path.isfile(ATTENTION_RUN_CONFIG):
        with open(ATTENTION_RUN_CONFIG, encoding="utf-8") as f:
            sources["attention_human_gaze"]["run_config_excerpt"] = json.load(f).get("model")
    with open(os.path.join(OUT_DIR, "input_sources.json"), "w", encoding="utf-8") as fh:
        json.dump(sources, fh, indent=2)
    return sources


def build_layer_profile_table(attention_by_layer, semantic_curves, lowlevel_curves):
    rows = []
    mean_semantic = {l: np.mean([semantic_curves[a][l] for a in SEMANTIC_ATTRIBUTES
                                  if a in semantic_curves and l in semantic_curves[a]])
                      for l in range(1, 13)} if semantic_curves else {}
    mean_lowlevel = {l: np.mean([lowlevel_curves[t][l] for t in LOWLEVEL_TARGETS
                                  if t in lowlevel_curves and l in lowlevel_curves[t]])
                      for l in range(1, 13)} if lowlevel_curves else {}

    for l in range(1, 13):
        a = attention_by_layer[l]
        row = {
            "layer": l,
            "NSS_mean": float(a["NSS_mean"]), "AUC_Judd_mean": float(a["AUC_Judd_mean"]),
            "sAUC_mean": float(a["sAUC_mean"]),
            "semantic_probe_mean_auprc": float(mean_semantic.get(l, np.nan)),
            "lowlevel_probe_mean_r2": float(mean_lowlevel.get(l, np.nan)),
        }
        if semantic_curves:
            for attr in SEMANTIC_ATTRIBUTES:
                row[f"semantic_auprc_{attr}"] = semantic_curves.get(attr, {}).get(l, np.nan)
        if lowlevel_curves:
            for t in LOWLEVEL_TARGETS:
                row[f"lowlevel_r2_{t}"] = lowlevel_curves.get(t, {}).get(l, np.nan)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "layer_profile_table.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {os.path.join(OUT_DIR, 'layer_profile_table.csv')}")
    return rows, mean_semantic, mean_lowlevel


def build_curve_correlations(attention_by_layer, mean_semantic, mean_lowlevel):
    layers = list(range(1, 13))
    curves = {
        "NSS": [float(attention_by_layer[l]["NSS_mean"]) for l in layers],
        "AUC_Judd": [float(attention_by_layer[l]["AUC_Judd_mean"]) for l in layers],
        "sAUC": [float(attention_by_layer[l]["sAUC_mean"]) for l in layers],
    }
    if mean_semantic:
        curves["semantic_probe_mean_auprc"] = [mean_semantic.get(l, np.nan) for l in layers]
    if mean_lowlevel:
        curves["lowlevel_probe_mean_r2"] = [mean_lowlevel.get(l, np.nan) for l in layers]

    pairs = []
    for a_name in ("NSS", "AUC_Judd", "sAUC"):
        if "semantic_probe_mean_auprc" in curves:
            pairs.append((a_name, "semantic_probe_mean_auprc"))
        if "lowlevel_probe_mean_r2" in curves:
            pairs.append((a_name, "lowlevel_probe_mean_r2"))
    if "semantic_probe_mean_auprc" in curves and "lowlevel_probe_mean_r2" in curves:
        pairs.append(("semantic_probe_mean_auprc", "lowlevel_probe_mean_r2"))

    rows = []
    for x_name, y_name in pairs:
        x, y = np.array(curves[x_name]), np.array(curves[y_name])
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            continue
        pear_r, pear_p = pearsonr(x[valid], y[valid])
        spear_r, spear_p = spearmanr(x[valid], y[valid])
        rows.append({
            "curve_x": x_name, "curve_y": y_name, "n_layers": int(valid.sum()),
            "pearson_r": float(pear_r), "pearson_p": float(pear_p),
            "spearman_r": float(spear_r), "spearman_p": float(spear_p),
            "note": "exploratory -- n=12 layers, not a hypothesis test with adequate power",
        })

    fieldnames = ["curve_x", "curve_y", "n_layers", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "note"]
    with open(os.path.join(OUT_DIR, "curve_correlations.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {os.path.join(OUT_DIR, 'curve_correlations.csv')}  ({len(rows)} pairs)")
    return rows, curves


def build_figures(attention_by_layer, semantic_curves, lowlevel_curves, mean_semantic, mean_lowlevel):
    layers = list(range(1, 13))
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    nss = [float(attention_by_layer[l]["NSS_mean"]) for l in layers]
    ax.plot(layers, nss, "o-", color=COLOR_NSS, label="NSS")
    ax.set_ylabel("NSS", color=COLOR_NSS, fontsize=9)
    ax2 = ax.twinx()
    aucj = [float(attention_by_layer[l]["AUC_Judd_mean"]) for l in layers]
    sauc = [float(attention_by_layer[l]["sAUC_mean"]) for l in layers]
    ax2.plot(layers, aucj, "s--", color=COLOR_AUCJ, label="AUC-Judd")
    ax2.plot(layers, sauc, "^:", color=COLOR_SAUC, label="sAUC")
    ax2.set_ylabel("AUC-Judd / sAUC", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Human-gaze agreement (raw metrics, separate axes), CLIP ViT-B/16, OSIE-700",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_xticks(layers)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=8)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    if semantic_curves:
        for attr in SEMANTIC_ATTRIBUTES:
            if attr in semantic_curves:
                y = [semantic_curves[attr].get(l, np.nan) for l in layers]
                ax.plot(layers, y, "-", color=INK_MUTED, alpha=0.35, linewidth=1)
        mean_y = [mean_semantic.get(l, np.nan) for l in layers]
        ax.plot(layers, mean_y, "o-", color=COLOR_SEMANTIC, linewidth=2.5, label="mean across 12 attributes")
    ax.set_ylabel("AUPRC", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Semantic-attribute linear probe (12 attributes, faint; mean, bold)",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_xticks(layers)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    ax = axes[2]
    ax.set_facecolor(SURFACE)
    if lowlevel_curves:
        for t in LOWLEVEL_TARGETS:
            if t in lowlevel_curves:
                y = [lowlevel_curves[t].get(l, np.nan) for l in layers]
                ax.plot(layers, y, "-", color=INK_MUTED, alpha=0.35, linewidth=1)
        mean_y = [mean_lowlevel.get(l, np.nan) for l in layers]
        ax.plot(layers, mean_y, "o-", color=COLOR_LOWLEVEL, linewidth=2.5, label="mean across 8 targets")
    ax.axhline(0.0, color=BASELINE, linewidth=1.0, linestyle="--")
    ax.set_ylabel("R2 (out-of-fold)", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("Layer", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Low-level visual-feature Ridge probe (8 targets, faint; mean, bold)",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_xticks(layers)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY)

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "layer_profile_three_panel.png")
    plt.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # ---- exploratory z-score overlay ----
    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    def zscore(vals):
        arr = np.array(vals, dtype=float)
        valid = np.isfinite(arr)
        arr = arr.copy()
        arr[~valid] = np.nanmean(arr)
        return (arr - arr.mean()) / (arr.std() if arr.std() > 0 else 1)

    ax.plot(layers, zscore(aucj), "o-", color=COLOR_AUCJ, label="AUC-Judd (z)")
    if semantic_curves:
        ax.plot(layers, zscore([mean_semantic.get(l, np.nan) for l in layers]), "s-", color=COLOR_SEMANTIC,
                label="semantic probe mean AUPRC (z)")
    if lowlevel_curves:
        ax.plot(layers, zscore([mean_lowlevel.get(l, np.nan) for l in layers]), "^-", color=COLOR_LOWLEVEL,
                label="low-level probe mean R2 (z)")
    ax.set_title("EXPLORATORY ONLY: z-score-normalized overlay -- no effect-size/superiority claims made",
                 color="#e34948", fontsize=10, loc="left")
    ax.set_xlabel("Layer", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("z-score", color=INK_SECONDARY, fontsize=9)
    ax.set_xticks(layers)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    out_png2 = os.path.join(FIG_DIR, "EXPLORATORY_zscore_overlay.png")
    plt.savefig(out_png2, dpi=150, facecolor=SURFACE)
    plt.close(fig2)
    print(f"  Saved: {out_png2}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 70)
    print("  Phase 4+5: layer-profile synthesis")
    print("=" * 70)

    if not os.path.isfile(ATTENTION_CSV):
        raise RuntimeError(f"STOP: canonical attention CSV not found: {ATTENTION_CSV}")
    attention_by_layer = load_attention_curve()
    semantic_curves = load_semantic_curves()
    lowlevel_curves = load_lowlevel_curves()
    print(f"  Attention/gaze source: {ATTENTION_CSV}")
    print(f"  Semantic probe available: {semantic_curves is not None}")
    print(f"  Low-level probe available: {lowlevel_curves is not None}")

    build_input_sources_json(attention_by_layer, semantic_curves, lowlevel_curves)
    rows, mean_semantic, mean_lowlevel = build_layer_profile_table(attention_by_layer, semantic_curves, lowlevel_curves)
    corr_rows, curves = build_curve_correlations(attention_by_layer, mean_semantic, mean_lowlevel)
    build_figures(attention_by_layer, semantic_curves, lowlevel_curves, mean_semantic, mean_lowlevel)

    print("\n  Layer profile summary:")
    print(f"  {'layer':>5s} {'NSS':>8s} {'AUCJ':>8s} {'sAUC':>8s} {'sem_auprc':>10s} {'low_r2':>8s}")
    for row in rows:
        print(f"  {row['layer']:>5d} {row['NSS_mean']:8.3f} {row['AUC_Judd_mean']:8.3f} "
              f"{row['sAUC_mean']:8.3f} {row['semantic_probe_mean_auprc']:10.3f} {row['lowlevel_probe_mean_r2']:8.3f}")

    print("\n" + "=" * 70)
    print("  Phase 4 synthesis: DONE (see write_hypothesis_decision.py for Phase 5)")
    print("=" * 70)


if __name__ == "__main__":
    main()
