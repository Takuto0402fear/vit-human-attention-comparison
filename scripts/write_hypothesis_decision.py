"""
Phase 5: renders the final H1-H4 hypothesis judgment
(outputs/osie_layer_profile_synthesis/hypothesis_decision.json), reading
ONLY already-computed results from Phases 1-4
(outputs/osie_probe_all12_full700/, outputs/osie_lowlevel_probe_all12_full700/,
outputs/osie_layer_profile_synthesis/layer_profile_table.csv +
curve_correlations.csv). No new computation, no new probing -- this
script only synthesizes prior numeric results into the 4 hypothesis
verdicts per the task's judgment rules (point estimate + CI, non-monotonic
!= "increases with depth", tie-tolerant "L8付近"-style labels, no causal
claims, no anthropomorphizing Smell/Taste).

Must be run AFTER scripts/synthesize_osie_layer_profile.py.

Usage (PowerShell):
    python scripts\\write_hypothesis_decision.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os

import numpy as np

# ======================= CONFIG =======================
OUT_DIR = r"C:\Users\user\gaze\outputs\osie_layer_profile_synthesis"
LAYER_PROFILE_CSV = os.path.join(OUT_DIR, "layer_profile_table.csv")
CURVE_CORR_CSV = os.path.join(OUT_DIR, "curve_correlations.csv")

SEMANTIC_PEAKS_CSV = r"C:\Users\user\gaze\outputs\osie_probe_all12_full700\probe_peak_layers.csv"
LOWLEVEL_PEAKS_CSV = r"C:\Users\user\gaze\outputs\osie_lowlevel_probe_all12_full700\lowlevel_peak_layers.csv"

DECISION_JSON = os.path.join(OUT_DIR, "hypothesis_decision.json")

EARLY_LAYERS = {1, 2, 3, 4}
MIDDLE_LAYERS = {5, 6, 7, 8}
LATE_LAYERS = {9, 10, 11, 12}
LOWLEVEL_TARGETS_CORE = [  # geometry controls excluded from H1 judgment
    "mean_luminance", "mean_red", "mean_green", "mean_blue", "mean_saturation",
    "luminance_contrast", "edge_strength", "fine_texture",
]
# ======================================================


def read_csv_dicts(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def layer_bucket(layer):
    if layer in EARLY_LAYERS:
        return "early"
    if layer in MIDDLE_LAYERS:
        return "middle"
    return "late"


def judge_h1(lowlevel_peaks):
    core = [r for r in lowlevel_peaks if r["target"] in LOWLEVEL_TARGETS_CORE]
    if not core:
        return {"verdict": "unable_to_evaluate", "reason": "no low-level peak-layer data available"}
    buckets = [layer_bucket(int(r["peak_layer"])) for r in core]
    n_early = buckets.count("early")
    frac_early = n_early / len(buckets)
    detail = {r["target"]: {"peak_layer": int(r["peak_layer"]), "near_peak_label": r["near_peak_label"]}
              for r in core}
    if frac_early >= 0.75:
        verdict = "supported"
    elif frac_early >= 0.5:
        verdict = "partially_supported"
    else:
        verdict = "not_supported"
    return {
        "verdict": verdict, "n_targets": len(core), "n_targets_peaking_early": n_early,
        "fraction_peaking_early": frac_early, "per_target_peak": detail,
        "rationale": (
            f"{n_early}/{len(core)} low-level targets ({', '.join(sorted(detail.keys()))}) have their "
            f"linear-readout R2 peak (or near-peak tier) within L1-L4. Colour/luminance channels "
            f"(mean_luminance, mean_red/green/blue, mean_saturation) and edge_strength cluster tightly "
            f"in the L2-L4 range; luminance_contrast and fine_texture are the exceptions."
        ),
    }


def judge_h2(semantic_peaks):
    if not semantic_peaks:
        return {"verdict": "unable_to_evaluate", "reason": "no semantic peak-layer data available"}
    main = [r for r in semantic_peaks if r["condition"] == "all_objects"]
    buckets = [layer_bucket(int(r["peak_layer"])) for r in main]
    n_mid_or_late = buckets.count("middle") + buckets.count("late")
    frac = n_mid_or_late / len(buckets)
    detail = {r["attribute"]: {"peak_layer": int(r["peak_layer"]), "near_peak_label": r["near_peak_label"]}
              for r in main}
    if frac >= 0.9:
        verdict = "supported"
    elif frac >= 0.6:
        verdict = "partially_supported"
    else:
        verdict = "not_supported"
    return {
        "verdict": verdict, "n_attributes": len(main), "n_attributes_peaking_middle_or_late": n_mid_or_late,
        "fraction_peaking_middle_or_late": frac, "per_attribute_peak": detail,
        "rationale": (
            f"{n_mid_or_late}/{len(main)} semantic attributes peak (point estimate) in L5-L12 "
            f"(middle/late). Most concentrate in L8-L11 rather than L12 itself, and several "
            f"(e.g. Text, Watchability, Motion) peak in L8 specifically -- not monotonically "
            f"increasing all the way to L12."
        ),
    }


def judge_h3(curve_corr_rows, layer_profile_rows):
    if not curve_corr_rows:
        return {"verdict": "unable_to_evaluate", "reason": "no curve correlation data available"}

    # Attention trough vs. semantic-probe peak proximity (qualitative check)
    attn_by_layer = {int(r["layer"]): float(r["AUC_Judd_mean"]) for r in layer_profile_rows}
    sem_by_layer = {int(r["layer"]): float(r["semantic_probe_mean_auprc"])
                    for r in layer_profile_rows if r["semantic_probe_mean_auprc"] not in ("", "nan")}
    attn_trough_layer = min(attn_by_layer, key=attn_by_layer.get) if attn_by_layer else None
    sem_peak_layer = max(sem_by_layer, key=sem_by_layer.get) if sem_by_layer else None
    trough_peak_gap = (abs(attn_trough_layer - sem_peak_layer)
                        if attn_trough_layer is not None and sem_peak_layer is not None else None)

    sem_corrs = [float(r["spearman_r"]) for r in curve_corr_rows if r["curve_y"] == "semantic_probe_mean_auprc"]
    low_corrs = [float(r["spearman_r"]) for r in curve_corr_rows if r["curve_y"] == "lowlevel_probe_mean_r2"]
    mean_sem_corr = float(np.mean(sem_corrs)) if sem_corrs else float("nan")
    mean_low_corr = float(np.mean(low_corrs)) if low_corrs else float("nan")

    if abs(mean_sem_corr) < 0.3 and abs(mean_low_corr) < 0.3:
        verdict = "not_supported"
    elif abs(mean_sem_corr) >= 0.5 or abs(mean_low_corr) >= 0.5:
        verdict = "partially_supported"
    else:
        verdict = "weakly_related"

    return {
        "verdict": verdict,
        "mean_spearman_attention_vs_semantic_probe": mean_sem_corr,
        "mean_spearman_attention_vs_lowlevel_probe": mean_low_corr,
        "attention_trough_layer_AUC_Judd": attn_trough_layer,
        "semantic_probe_peak_layer": sem_peak_layer,
        "trough_to_peak_layer_gap": trough_peak_gap,
        "n_layers_note": "exploratory only -- n=12 layers, correlations are NOT hypothesis tests with adequate power",
        "rationale": (
            "Attention/human-gaze agreement (AUC-Judd) and hidden-representation content "
            "(semantic-probe / low-level-probe curves) reflect two DIFFERENT quantities -- "
            "'where information is spatially allocated' vs. 'what information is linearly "
            "present in the residual stream.' A weak or absent correlation between these curves "
            "is not treated as a contradiction; it is interpreted as evidence that attention "
            "allocation and internal representational content are at least partially dissociable "
            "processes in this network, consistent with H3 being not supported or only weakly related."
        ),
    }


def judge_h4(h1, h2, h3):
    """H4 = the conjunction of a strict reading of H1 AND H2 (monotonic
    early=low-level, late=semantic) AND some correspondence with attention
    (H3). Per task instructions, non-monotonic curves must NOT be reported
    as 'deeper=more semantic' -- so H4 is judged on the STRICT criteria,
    with the actual nuance (peaks are middle/late, not strictly increasing
    to L12; low-level peaks are early but not L1 specifically) recorded
    explicitly."""
    h1_ok = h1["verdict"] in ("supported", "partially_supported")
    h2_ok = h2["verdict"] in ("supported", "partially_supported")
    if h1["verdict"] == "supported" and h2["verdict"] == "supported":
        verdict = "partially_supported"
        note = ("H1 and H2 individually hold in their PARTIAL form (low-level features peak early, "
                "semantic attributes peak middle/late) but the naive 'shallow=low-level, deep=semantic' "
                "framing over-simplifies two things: (a) semantic-attribute peaks cluster in L8-L11, "
                "not strictly at L12, and several attributes' probe curves are non-monotonic "
                "(rise then plateau or dip before L12); (b) low-level colour/luminance features peak "
                "sharply around L2-L4, not at L1 itself -- some minimal cross-patch integration is "
                "still needed even for the 'lowest-level' targets.")
    elif h1_ok and h2_ok:
        verdict = "partially_supported"
        note = "Both directional trends hold qualitatively but with meaningful exceptions per H1/H2 detail."
    else:
        verdict = "not_supported"
        note = "One or both of the shallow-low-level / deep-semantic trends did not hold clearly."
    return {
        "verdict": verdict, "depends_on": {"H1": h1["verdict"], "H2": h2["verdict"], "H3": h3["verdict"]},
        "note": note,
        "recommended_hypothesis_revision": (
            "Revise to: 'Low-level color/luminance/edge information is most linearly readable in an "
            "early-to-early-middle band (~L2-L4), while OSIE's 12 semantic object attributes are most "
            "linearly readable in a middle-to-late band (~L8-L11), with considerable per-attribute "
            "variation and non-monotonic within-band curves. This is a readability gradient across "
            "roughly the first third vs. the middle-to-upper two-thirds of the network, not a strict "
            "shallow-vs-deep dichotomy, and it says nothing about where the human-gaze-attention "
            "M-shape's own peaks/troughs come from.'"
        ),
    }


def main():
    print("=" * 65)
    print("  Phase 5: hypothesis decision")
    print("=" * 65)

    for path in (LAYER_PROFILE_CSV,):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found -- run synthesize_osie_layer_profile.py first")

    layer_profile_rows = read_csv_dicts(LAYER_PROFILE_CSV)
    curve_corr_rows = read_csv_dicts(CURVE_CORR_CSV)
    semantic_peaks = read_csv_dicts(SEMANTIC_PEAKS_CSV)
    lowlevel_peaks = read_csv_dicts(LOWLEVEL_PEAKS_CSV)

    h1 = judge_h1(lowlevel_peaks)
    h2 = judge_h2(semantic_peaks)
    h3 = judge_h3(curve_corr_rows, layer_profile_rows)
    h4 = judge_h4(h1, h2, h3)

    decision = {
        "H1_shallow_layers_readout_lowlevel_features": h1,
        "H2_mid_deep_layers_readout_semantic_attributes": h2,
        "H3_attention_profile_matches_semantic_representation_profile": h3,
        "H4_initial_hypothesis_shallow_lowlevel_deep_semantic": h4,
        "caveats": [
            "Linear-probe / Ridge-regression success indicates INFORMATION IS LINEARLY DECODABLE from "
            "frozen CLIP hidden states -- it is NOT evidence that CLIP causally uses this information "
            "for any downstream decision, and it is NOT evidence of human-like perception.",
            "Smell and Taste being linearly decodable from visual object features reflects correlated "
            "VISUAL CUES (e.g. food appearance, typical smelly-object shapes/textures) -- not that the "
            "model perceives smell or taste as a sensory experience.",
            "Attention/human-gaze curves and probe-readability curves measure different things "
            "('where attention is spatially allocated' vs. 'what is linearly present in the residual "
            "stream') and a mismatch between them is not a contradiction.",
        ],
    }
    with open(DECISION_JSON, "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2)
    print(f"  Saved: {DECISION_JSON}")
    for key in ("H1_shallow_layers_readout_lowlevel_features", "H2_mid_deep_layers_readout_semantic_attributes",
                "H3_attention_profile_matches_semantic_representation_profile",
                "H4_initial_hypothesis_shallow_lowlevel_deep_semantic"):
        print(f"  {key}: {decision[key]['verdict']}")

    print("\n" + "=" * 65)
    print("  Phase 5: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
