"""
Model-comparison layerwise plots: DeiT/SL vs DINO-S vs DINO-B vs CLIP-B,
all evaluated on the SAME OSIE 700-image healthy-group fixation data with
the SAME metrics.py implementation (NSS / AUC-Judd / sAUC).

Read-only with respect to every existing result file -- this script only
reads outputs/expA*/**/metrics_by_layer*.csv and run_config*.json (never
writes to them) and writes exclusively under outputs/model_comparison_
layerwise/ (a new directory).

Sources actually found and inspected (not guessed) -- see the printed
inventory below and comparison_sources.json for the full detail:

  DeiT / SL   outputs/expA_sl/combined/{metrics_by_layer_across_trials.csv,
              metrics_by_layer_and_trial.csv}
              model=sl_vits16, 6 trials (supervised/{01..06}/12layers),
              depth=12, patch=16, n_images=700, checkpoint MD5s confirmed
              distinct per trial (see outputs/expA_sl/trial_*/run_config.json)
  DINO-S      outputs/expA/healthy/metrics_by_layer.csv
              model=dino_vits16 (Yamamoto-paper-trained backbone,
              trial=1), depth=12, patch=16, n_images=700, single run
              (no trial/seed sweep was run for this backbone on OSIE)
  DINO-B      outputs/expA_dino_vitb16/healthy/metrics_by_layer.csv
              model=dino_vitb16 (official facebookresearch pretrained
              backbone, not Yamamoto-trained), depth=12, patch=16,
              n_images=700, single run
  CLIP-B      outputs/expA_clip/healthy/metrics_by_layer.csv
              model=clip_vitb16 (OpenAI), depth=12, patch=16,
              n_images=700, single run

DINOv2 was searched for across the entire repo (grep -ri "dinov2" over
all .py files, plus a full outputs/ directory listing) and NOT FOUND --
no checkpoint, no loader, no run output exists for DINOv2 in this
project. It is excluded from every comparison below; nothing is
fabricated in its place.

outputs/expB*/ (fixation-subset conditions: all/firstk/randk/lastk, on
DINO-S only) is a DIFFERENT experiment (early-vs-late saccade analysis),
not a cross-model layerwise comparison, and is excluded here.

All four included models share: OSIE 700 images (same intersection,
independently re-derived by each script from stimuli inter fixmaps inter
DINO-S's own metrics_per_image.csv image list), group="healthy",
patch_size=16, depth=12, fixation points read from the SAME outputs/
fixmaps/healthy/*.npz files (generated once by scripts/generate_fixmaps.py
at SIGMA_PX=24; that sigma only feeds the heat_all Gaussian map used for
CC/SIM in expA/healthy's older 5-metric run -- NSS/AUC-Judd/sAUC read raw
fixation POINTS directly and are unaffected by sigma), and metrics.py's
nss/auc_judd/sauc functions unmodified.

Usage (PowerShell):
    python scripts\\generate_model_comparison_layerwise.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_LIST = ["NSS", "AUC_Judd", "sAUC"]
DEPTH = 12
N_IMAGES = 700
OUT_DIR = r"C:\Users\user\gaze\outputs\model_comparison_layerwise"

# ======================================================================
# Model registry -- built from files actually inspected above, not from
# assumed naming. Each entry's `layer_csv` / column names were verified
# by direct `head -1` inspection (see this script's module docstring and
# the investigation preceding it).
# ======================================================================
MODELS = [
    {
        "key": "deit_sl", "display": "DeiT/SL ViT-S/16 (Yamamoto-paper, 6-trial mean)",
        "short": "DeiT (SL)", "color": "firebrick", "marker": "o", "ls": "-", "lw": 2.6, "ms": 7,
        "n_layers": 12, "patch_size": 16, "n_images": 700, "n_trials": 6,
        "layer_csv": r"C:\Users\user\gaze\outputs\expA_sl\combined\metrics_by_layer_across_trials.csv",
        "mean_col": "{m}_mean_across_trials", "std_col": "{m}_std_across_trials",
        "source_note": "mean +/- SD of the 6 trial-level means (N=6), from outputs/expA_sl/combined/",
    },
    {
        "key": "dino_s", "display": "DINO ViT-S/16 (Yamamoto-paper trained, trial01)",
        "short": "DINO-S", "color": "steelblue", "marker": "^", "ls": "-.", "lw": 2.0, "ms": 7,
        "n_layers": 12, "patch_size": 16, "n_images": 700, "n_trials": 1,
        "layer_csv": r"C:\Users\user\gaze\outputs\expA\healthy\metrics_by_layer.csv",
        "mean_col": "{m}_mean", "std_col": None,
        "source_note": "single run (no trial/seed sweep computed for this backbone on OSIE)",
    },
    {
        "key": "dino_b", "display": "DINO ViT-B/16 (official facebookresearch pretrained)",
        "short": "DINO-B", "color": "seagreen", "marker": "D", "ls": "--", "lw": 2.0, "ms": 6,
        "n_layers": 12, "patch_size": 16, "n_images": 700, "n_trials": 1,
        "layer_csv": r"C:\Users\user\gaze\outputs\expA_dino_vitb16\healthy\metrics_by_layer.csv",
        "mean_col": "{m}_mean", "std_col": None,
        "source_note": "single run (official ImageNet-pretrained backbone, not a Yamamoto-trial checkpoint)",
    },
    {
        "key": "clip_b", "display": "CLIP ViT-B/16 (OpenAI)",
        "short": "CLIP-B", "color": "darkorange", "marker": "s", "ls": ":", "lw": 2.2, "ms": 7,
        "n_layers": 12, "patch_size": 16, "n_images": 700, "n_trials": 1,
        "layer_csv": r"C:\Users\user\gaze\outputs\expA_clip\healthy\metrics_by_layer.csv",
        "mean_col": "{m}_mean", "std_col": None,
        "source_note": "single run",
    },
]

EXCLUDED = [
    {
        "candidate": "DINOv2",
        "reason": "not found anywhere in the repository -- no checkpoint, no loader module, no "
                  "outputs/ directory, no reference in any .py file (grep -ri 'dinov2' over all "
                  ".py files returned zero matches). Never implemented or run; excluded, not "
                  "fabricated.",
    },
    {
        "candidate": "outputs/expB, expB_control, expB_lastk (DINO-S, fixation-subset conditions)",
        "reason": "a different experiment (early-vs-late saccade / first-k vs last-k fixation "
                  "analysis on the SAME DINO-S backbone), not an independent model's layerwise "
                  "attention-vs-gaze comparison. Excluded from the cross-model comparison.",
    },
    {
        "candidate": "outputs/expA_clip/pilot20, outputs/expA_dino_vitb16/pilot20, outputs/expA_sl/pilot20",
        "reason": "20-image pilot subsets of the SAME models already included via their full-700 "
                  "results; superseded by the full700 data, excluded to avoid double-counting.",
    },
    {
        "candidate": "outputs/expA_sl/trial_{02..06} individually",
        "reason": "not excluded from the comparison, but folded into the 6-trial mean/SD used for "
                  "the 'DeiT/SL' series (outputs/expA_sl/combined/metrics_by_layer_across_trials.csv) "
                  "rather than plotted as 6 separate series in the MAIN comparison figures -- the "
                  "per-trial spread is still shown explicitly as a band around the DeiT/SL line.",
    },
]


def load_layer_means(model):
    rows = list(csv.DictReader(open(model["layer_csv"], encoding="utf-8")))
    if len(rows) != model["n_layers"]:
        raise RuntimeError(f"STOP: {model['key']} layer_csv has {len(rows)} rows, expected {model['n_layers']}")
    layers = [int(r["layer"]) for r in rows]
    if layers != list(range(1, model["n_layers"] + 1)):
        raise RuntimeError(f"STOP: {model['key']} layers not 1..{model['n_layers']} in order: {layers}")

    data = {"layer": np.array(layers), "relative_depth": (np.array(layers) - 1) / (model["n_layers"] - 1)}
    for m in METRICS_LIST:
        mean_col = model["mean_col"].format(m=m)
        data[f"{m}_mean"] = np.array([float(r[mean_col]) for r in rows])
        if model["std_col"] is not None:
            std_col = model["std_col"].format(m=m)
            data[f"{m}_std"] = np.array([float(r[std_col]) for r in rows])
        else:
            data[f"{m}_std"] = None
    return data


def compute_ylims(all_data):
    ylims = {}
    for m in METRICS_LIST:
        vals = []
        for d in all_data.values():
            vals.extend(d[f"{m}_mean"].tolist())
            if d[f"{m}_std"] is not None:
                vals.extend((d[f"{m}_mean"] + d[f"{m}_std"]).tolist())
                vals.extend((d[f"{m}_mean"] - d[f"{m}_std"]).tolist())
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.10
        ylims[m] = (lo - pad, hi + pad)
    return ylims


def metric_label(m):
    return "AUC-Judd" if m == "AUC_Judd" else m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  Model-comparison layerwise plots: DeiT/SL vs DINO-S vs DINO-B vs CLIP-B")
    print("=" * 65)

    print("\n--- Model inventory (verified from actual files) ---")
    for model in MODELS:
        print(f"  {model['short']:10s} layers={model['n_layers']:2d} patch={model['patch_size']:2d} "
              f"n_images={model['n_images']:3d} trials={model['n_trials']}  file={model['layer_csv']}")
    print("\n--- Excluded / out of scope ---")
    for e in EXCLUDED:
        print(f"  - {e['candidate']}: {e['reason'][:90]}...")

    all_data = {m["key"]: load_layer_means(m) for m in MODELS}
    ylims = compute_ylims(all_data)
    layers_arr = np.arange(1, DEPTH + 1)

    peak_info = {}
    for model in MODELS:
        d = all_data[model["key"]]
        peak_info[model["key"]] = {}
        for m in METRICS_LIST:
            idx = int(np.argmax(d[f"{m}_mean"]))
            peak_info[model["key"]][m] = {"layer": int(d["layer"][idx]), "value": float(d[f"{m}_mean"][idx])}

    print("\n--- Peak layer per model per metric ---")
    for model in MODELS:
        pk = peak_info[model["key"]]
        print(f"  {model['short']:10s} NSS: L{pk['NSS']['layer']:2d} ({pk['NSS']['value']:.4f})   "
              f"AUC-Judd: L{pk['AUC_Judd']['layer']:2d} ({pk['AUC_Judd']['value']:.4f})   "
              f"sAUC: L{pk['sAUC']['layer']:2d} ({pk['sAUC']['value']:.4f})")

    # ------------------------------------------------------------------
    # A. 12-layer direct comparison, 3 metrics x subplots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor="white")
    for ax, m in zip(axes, METRICS_LIST):
        for model in MODELS:
            d = all_data[model["key"]]
            means = d[f"{m}_mean"]
            ax.plot(layers_arr, means, color=model["color"], marker=model["marker"], linestyle=model["ls"],
                    lw=model["lw"], ms=model["ms"], label=model["short"], zorder=3)
            if d[f"{m}_std"] is not None:
                ax.fill_between(layers_arr, means - d[f"{m}_std"], means + d[f"{m}_std"],
                                 color=model["color"], alpha=0.15, zorder=1)
            pk = peak_info[model["key"]][m]
            ax.plot(pk["layer"], pk["value"], marker="*", color=model["color"], markersize=15,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_label(m))
        ax.set_title(metric_label(m))
        ax.set_xticks(layers_arr)
        ax.set_ylim(*ylims[m])
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("Layerwise agreement with human gaze (OSIE 700, healthy) -- 12-layer models\n"
                 "(stars = peak layer per model per metric; band = SD across trials where available)",
                 fontsize=12)
    fig.tight_layout()
    pA = os.path.join(OUT_DIR, "all_models_12layer_3metrics.png")
    fig.savefig(pA, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {pA}")

    # ------------------------------------------------------------------
    # B. Single-metric large figures -- one per metric (NSS, AUC-Judd,
    # sAUC), each following the same "big, uncluttered overlay" template.
    # ------------------------------------------------------------------
    single_metric_files = {"NSS": "all_models_12layer_nss.png",
                            "AUC_Judd": "all_models_12layer_aucjudd.png",
                            "sAUC": "all_models_12layer_sauc.png"}
    for m in METRICS_LIST:
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
        for model in MODELS:
            d = all_data[model["key"]]
            means = d[f"{m}_mean"]
            ax.plot(layers_arr, means, color=model["color"], marker=model["marker"], linestyle=model["ls"],
                    lw=model["lw"] + 0.4, ms=model["ms"] + 1, label=f"{model['display']}", zorder=3)
            if d[f"{m}_std"] is not None:
                ax.fill_between(layers_arr, means - d[f"{m}_std"], means + d[f"{m}_std"],
                                 color=model["color"], alpha=0.15, zorder=1)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel(metric_label(m), fontsize=12)
        ax.set_title(f"{metric_label(m)} vs Layer -- DeiT/SL vs DINO-S vs DINO-B vs CLIP-B (OSIE 700, healthy)",
                     fontsize=13)
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        loc = "upper right" if m == "NSS" else "lower right"
        ax.legend(fontsize=10, loc=loc, framealpha=0.9)
        fig.tight_layout()
        pB = os.path.join(OUT_DIR, single_metric_files[m])
        fig.savefig(pB, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {pB}")

    # ------------------------------------------------------------------
    # C. Relative-depth comparison (all current models have 12 layers, so
    # this is a rescaled x-axis view of A -- kept for future comparability
    # once a different-depth model is added; observed points stay as
    # markers, no interpolation is drawn as if it were data).
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor="white")
    for ax, m in zip(axes, METRICS_LIST):
        for model in MODELS:
            d = all_data[model["key"]]
            means = d[f"{m}_mean"]
            ax.plot(d["relative_depth"], means, color=model["color"], marker=model["marker"],
                    linestyle=model["ls"], lw=model["lw"], ms=model["ms"], label=model["short"], zorder=3)
            if d[f"{m}_std"] is not None:
                ax.fill_between(d["relative_depth"], means - d[f"{m}_std"], means + d[f"{m}_std"],
                                 color=model["color"], alpha=0.15, zorder=1)
        ax.set_xlabel("Relative depth = (layer-1)/(num_layers-1)")
        ax.set_ylabel(metric_label(m))
        ax.set_title(metric_label(m))
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(*ylims[m])
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("Relative-depth comparison -- for cross-depth comparability "
                 "(all 4 models here happen to have 12 layers; markers = actual observed layers, "
                 "no interpolation)", fontsize=11)
    fig.tight_layout()
    pC = os.path.join(OUT_DIR, "all_models_relative_depth_3metrics.png")
    fig.savefig(pC, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pC}")

    # ------------------------------------------------------------------
    # D. Individual model plots (3 metrics, same y-lims as A, peak annotated)
    # ------------------------------------------------------------------
    indiv_names = {"deit_sl": "individual_deit_3metrics.png", "dino_s": "individual_dino_s_3metrics.png",
                    "dino_b": "individual_dino_b_3metrics.png", "clip_b": "individual_clip_3metrics.png"}
    for model in MODELS:
        d = all_data[model["key"]]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
        for ax, m in zip(axes, METRICS_LIST):
            means = d[f"{m}_mean"]
            ax.plot(layers_arr, means, color=model["color"], marker=model["marker"], linestyle="-",
                    lw=2.2, ms=7)
            if d[f"{m}_std"] is not None:
                ax.fill_between(layers_arr, means - d[f"{m}_std"], means + d[f"{m}_std"],
                                 color=model["color"], alpha=0.18)
            pk = peak_info[model["key"]][m]
            ax.plot(pk["layer"], pk["value"], marker="*", color=model["color"], markersize=16,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=4)
            ax.annotate(f"peak L{pk['layer']}\n{pk['value']:.3f}", xy=(pk["layer"], pk["value"]),
                        xytext=(6, 8), textcoords="offset points", fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_label(m))
            ax.set_title(metric_label(m))
            ax.set_xticks(layers_arr)
            ax.set_ylim(*ylims[m])
            ax.grid(True, alpha=0.25, linewidth=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.suptitle(f"{model['display']} -- layerwise agreement with human gaze (OSIE 700, healthy)",
                     fontsize=12)
        fig.tight_layout()
        pD = os.path.join(OUT_DIR, indiv_names[model["key"]])
        fig.savefig(pD, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {pD}")

    # ------------------------------------------------------------------
    # E. Difference from CLIP (same-depth models only -- all 4 are 12
    # layers here, so a direct per-layer subtraction is valid for all 3)
    # ------------------------------------------------------------------
    clip_data = all_data["clip_b"]
    diff_models = [m for m in MODELS if m["key"] != "clip_b"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor="white")
    for ax, m in zip(axes, METRICS_LIST):
        ax.axhline(0.0, color="black", lw=1.0, ls="-", zorder=1)
        for model in diff_models:
            d = all_data[model["key"]]
            diff = d[f"{m}_mean"] - clip_data[f"{m}_mean"]
            ax.plot(layers_arr, diff, color=model["color"], marker=model["marker"], linestyle=model["ls"],
                    lw=model["lw"], ms=model["ms"], label=f"{model['short']} - CLIP-B")
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"{metric_label(m)} (model - CLIP-B)")
        ax.set_title(metric_label(m))
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("Difference from CLIP-B, per layer (all models 12-layer; direct layer-aligned subtraction)",
                 fontsize=12)
    fig.tight_layout()
    pE = os.path.join(OUT_DIR, "clip_difference_3metrics.png")
    fig.savefig(pE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pE}")

    # ------------------------------------------------------------------
    # comparison_values.csv
    # ------------------------------------------------------------------
    csv_rows = []
    for model in MODELS:
        d = all_data[model["key"]]
        for i, layer in enumerate(d["layer"]):
            row = {
                "model": model["short"], "model_display": model["display"], "layer": int(layer),
                "relative_depth": float(d["relative_depth"][i]),
                "n_images": model["n_images"], "n_trials": model["n_trials"], "patch_size": model["patch_size"],
                "source_file": model["layer_csv"],
            }
            for m in METRICS_LIST:
                row[f"{m}_mean"] = float(d[f"{m}_mean"][i])
                row[f"{m}_std"] = float(d[f"{m}_std"][i]) if d[f"{m}_std"] is not None else ""
            csv_rows.append(row)

    values_csv = os.path.join(OUT_DIR, "comparison_values.csv")
    fieldnames = ["model", "model_display", "layer", "relative_depth", "n_images", "n_trials", "patch_size",
                  "source_file"] + [f"{m}_{s}" for m in METRICS_LIST for s in ("mean", "std")]
    with open(values_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n  Saved: {values_csv}  ({len(csv_rows)} rows)")

    # ------------------------------------------------------------------
    # comparison_sources.json
    # ------------------------------------------------------------------
    sources = {
        "included_models": [
            {
                "key": model["key"], "display": model["display"], "checkpoint_or_source":
                    model["layer_csv"], "n_layers": model["n_layers"], "patch_size": model["patch_size"],
                "n_images": model["n_images"], "n_trials": model["n_trials"], "note": model["source_note"],
                "peak_layer": peak_info[model["key"]],
            }
            for model in MODELS
        ],
        "excluded": EXCLUDED,
        "shared_evaluation_conditions": {
            "dataset": "OSIE 700 images (identical intersection re-derived independently by each "
                      "experiment script: stimuli images inter fixmaps inter DINO-S's metrics_per_image.csv)",
            "group": "healthy",
            "fixation_source": "outputs/fixmaps/healthy/*.npz ('points' key), shared unmodified across "
                               "all 4 included models",
            "fixmap_generation_sigma": 24,
            "metrics_implementation": "metrics.py :: nss / auc_judd / sauc, unmodified, shared across all "
                                      "4 included models' experiment scripts",
            "attn_convention": "[CLS] token -> patch tokens, mean over heads, bilinear-upsampled to "
                              "(600,800)",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    sources_path = os.path.join(OUT_DIR, "comparison_sources.json")
    with open(sources_path, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)
    print(f"  Saved: {sources_path}")

    print("\n" + "=" * 65)
    print("  Model-comparison layerwise plots: DONE")
    print("  No existing outputs/expA*/ result file was modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
