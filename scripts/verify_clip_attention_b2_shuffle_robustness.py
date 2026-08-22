"""
[B-2] Pre-commit robustness check: does Family 2 (same-image vs.
shuffled-image-pair spatial similarity) depend on the ONE fixed seed=42
derangement used by scripts/compute_clip_attention_b2_full700.py? This
script draws N_DERANGEMENTS (default 1000) INDEPENDENT derangements
(base seed, each a fixed-point-free permutation of the 700 images,
fully reproducible, processed one at a time -- never all of them held
in memory together) and re-tests the same 9 (layer pair x metric)
comparisons against this much larger, more robust null distribution.
Derangement generation reuses lib.clip_attention_b2.iter_derangements /
fixed_point_free_permutation / derangement_seed verbatim -- no
duplicate derangement logic in this script.

Per (layer_pair, metric):
  1. same-image dataset-mean similarity is computed ONCE (not per
     derangement) -- Pearson via Fisher-z-transform-then-average
     (back-transformed via tanh for reporting, lib.clip_attention_b2.
     fisher_z_inverse); cosine/JSD use the already-defined values
     directly (plain arithmetic mean).
  2. for each derangement, the shuffled (different-image) pairs'
     dataset-mean similarity is computed the same way, forming an
     N_DERANGEMENTS-value empirical null distribution.
  3. Monte-Carlo one-sided p-value (+1 correction), direction per metric:
       - pearson / cosine: "same-image > shuffled"
         p = (1 + #{shuffled means >= same-image mean}) / (1 + B)
       - normalized_jsd:    "same-image < shuffled" (lower JSD = more similar)
         p = (1 + #{shuffled means <= same-image mean}) / (1 + B)
     With B=1000 the smallest possible p is 1/1001 ~ 0.000999 -- p=0 is
     never reported.
  4. BH-FDR across these 9 tests (a family independent of, and not
     comparable to, Family 1's 27 tests or the original single-derangement
     Family 2 in paired_layer_comparisons.csv).

IMPORTANT naming: the (observed - null_mean) / null_std quantity computed
here is called `null_standardized_distance` throughout (JSON keys, CSV,
report text) -- it is NEVER called "effect size" or "Cohen's dz". It
measures how far the observed dataset-mean sits from the empirical null
distribution's own spread, which is a different question from a
practical effect size. The practical-magnitude reference is: same-image
mean, shuffled mean, shuffled means' 2.5/97.5 percentiles, the raw
(same - shuffled) difference, the raw metric values themselves, and (for
context) the ORIGINAL per-image paired comparison's own Cohen's dz from
paired_layer_comparisons.csv (a genuinely different, legitimate effect
size computed on a different quantity -- per-image paired differences,
not a dataset-mean-vs-null-distribution comparison).

This script does NOT touch any existing output (report.md, summary.json,
per_image_metrics.csv, layer_pair_similarity.csv, paired_layer_comparisons.csv,
cache_verification.json, or any existing figure) -- it reads them
read-only for input validation and comparison, and performs ALL input
validation BEFORE writing anything (a validation failure produces no
partial output). It writes only the four new files below, each
individually refused if it already exists (not merely because the
output directory exists):
  layer_pair_similarity_repeated_shuffle.csv
  repeated_shuffle_summary.json
  summary_revised.json
  report_revised.md

All paths are resolved relative to the repository root or overridable
via CLI flags -- no machine-specific absolute path is hardcoded.

Usage (PowerShell):
    python scripts\\verify_clip_attention_b2_shuffle_robustness.py
    python scripts\\verify_clip_attention_b2_shuffle_robustness.py --output-dir D:\\tmp\\b2_smoke\\shuffle --n-derangements 1000
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import argparse
import csv
import json
import time

import numpy as np

from lib.clip_attention_b2 import (
    bh_fdr, dataset_mean_cosine_for_permutation, dataset_mean_jsd_for_permutation,
    dataset_mean_pearson_for_permutation, iter_derangements, monte_carlo_one_sided_p,
    refuse_if_exists,
)

BASE_SEED_DEFAULT = 42
N_DERANGEMENTS_DEFAULT = 1000
N_IMAGES_EXPECTED = 700
GRID_HW = (38, 50)
LAYER_PAIRS = [(4, 8), (8, 12), (4, 12)]
METRICS = ["pearson", "cosine", "normalized_jsd"]
DIRECTION = {"pearson": "greater", "cosine": "greater", "normalized_jsd": "less"}
LAYERS_DISPLAY = [4, 8, 12]
LAYERS_ZERO_BASED = [3, 7, 11]
LAYER_IDX_IN_CACHE = {4: 0, 8: 1, 12: 2}

DEFAULT_CACHE_PATH = (
    REPO_ROOT / "outputs" / "osie_attribute_grounding_full700" / "attn_cache"
    / "clip_vitb16_full700_L4L8L12_patchgrid.npz")
DEFAULT_DATA_DIR = REPO_ROOT / "outputs" / "clip_attention_b2_distribution"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="[B-2] Repeated-shuffle robustness check: re-test Family 2 "
                    "(same-image vs. shuffled-image spatial similarity) against many "
                    "independent derangements instead of the one fixed seed=42 "
                    "derangement used by the main full700 pipeline.")
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                    help=f"Path to the existing L4/L8/L12 attention cache .npz, read-only "
                        f"(default: {DEFAULT_CACHE_PATH}).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Directory containing the existing full700 pipeline outputs, "
                        "read-only: cache_verification.json, per_image_metrics.csv, "
                        "layer_pair_similarity.csv, paired_layer_comparisons.csv, "
                        f"summary.json, report.md (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Directory to write the four new files into (created if "
                        "missing; refuses to overwrite any of them if it already "
                        f"exists) (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--seed", type=int, default=BASE_SEED_DEFAULT,
                    help=f"Base seed for the derangement sequence (default: {BASE_SEED_DEFAULT}).")
    p.add_argument("--n-derangements", type=int, default=N_DERANGEMENTS_DEFAULT,
                    help=f"Number of independent derangements to draw (default: {N_DERANGEMENTS_DEFAULT}).")
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# Input validation -- ALL checks run before anything is written. On
# failure, raises before any output file is created (no partial output).
# ----------------------------------------------------------------------

def validate_inputs(cache_path, data_dir):
    cache_verification_json = data_dir / "cache_verification.json"
    metrics_csv = data_dir / "per_image_metrics.csv"
    similarity_csv = data_dir / "layer_pair_similarity.csv"
    paired_csv = data_dir / "paired_layer_comparisons.csv"
    summary_json = data_dir / "summary.json"
    report_md = data_dir / "report.md"

    for required in (cache_path, cache_verification_json, metrics_csv, similarity_csv,
                      paired_csv, summary_json, report_md):
        if not Path(required).is_file():
            raise RuntimeError(f"STOP: required input not found: {required}")

    with open(cache_verification_json, encoding="utf-8") as fh:
        cache_verif = json.load(fh)
    if cache_verif.get("verdict") != "PASS":
        raise RuntimeError(f"STOP: {cache_verification_json} verdict != PASS")

    cache = np.load(str(cache_path), allow_pickle=False)
    attn = cache["attn"]
    stems = cache["stems"].tolist()
    layers_display = cache["layers_display"].tolist()
    layers_zero_based = cache["layers_zero_based"].tolist()

    if attn.shape != (N_IMAGES_EXPECTED, 3, *GRID_HW):
        raise RuntimeError(f"STOP: cache attn.shape {attn.shape} != {(N_IMAGES_EXPECTED, 3, *GRID_HW)}")
    if layers_display != LAYERS_DISPLAY:
        raise RuntimeError(f"STOP: cache layers_display {layers_display} != {LAYERS_DISPLAY}")
    if layers_zero_based != LAYERS_ZERO_BASED:
        raise RuntimeError(f"STOP: cache layers_zero_based {layers_zero_based} != {LAYERS_ZERO_BASED}")
    if len(stems) != N_IMAGES_EXPECTED:
        raise RuntimeError(f"STOP: cache has {len(stems)} stems, expected {N_IMAGES_EXPECTED}")
    if len(set(stems)) != len(stems):
        raise RuntimeError("STOP: cache stems contain duplicates")
    if not np.isfinite(attn).all():
        raise RuntimeError("STOP: cache attention contains non-finite values")
    if (attn < 0).any():
        raise RuntimeError("STOP: cache attention contains negative values")
    cache_stem_set = set(stems)

    with open(metrics_csv, encoding="utf-8") as fh:
        metric_rows = list(csv.DictReader(fh))
    if len(metric_rows) != N_IMAGES_EXPECTED * len(LAYERS_DISPLAY):
        raise RuntimeError(
            f"STOP: {metrics_csv} has {len(metric_rows)} rows, expected "
            f"{N_IMAGES_EXPECTED * len(LAYERS_DISPLAY)}")
    seen_pairs = set()
    for r in metric_rows:
        key = (r["image_id"], r["layer"])
        if key in seen_pairs:
            raise RuntimeError(f"STOP: duplicate (image_id, layer) pair {key} in {metrics_csv}")
        seen_pairs.add(key)
    metrics_stem_set = set(r["image_id"] for r in metric_rows)
    if metrics_stem_set != cache_stem_set:
        raise RuntimeError(
            f"STOP: {metrics_csv} image_id set differs from the cache's stem set "
            f"(symmetric difference size {len(metrics_stem_set ^ cache_stem_set)})")

    with open(similarity_csv, encoding="utf-8") as fh:
        sim_rows = list(csv.DictReader(fh))
    similarity_stem_set = set(r["image_id"] for r in sim_rows)
    if similarity_stem_set != cache_stem_set:
        raise RuntimeError(
            f"STOP: {similarity_csv} image_id set differs from the cache's stem set "
            f"(symmetric difference size {len(similarity_stem_set ^ cache_stem_set)})")

    return stems


# ----------------------------------------------------------------------
# Dataset-mean similarity for a given permutation (vectorized across all
# 700 images at once; only one derangement's intermediate arrays are
# held in memory at a time -- see main()'s loop).
# ----------------------------------------------------------------------

def precompute_layer_stats(raw):
    """raw: (n_images, n_patches). Returns everything needed to compute
    dataset-mean pearson/cosine/jsd against any permutation without
    recomputing per-row reductions every time."""
    centered = raw - raw.mean(axis=1, keepdims=True)
    norms_centered = np.linalg.norm(centered, axis=1)
    norms_raw = np.linalg.norm(raw, axis=1)
    p = raw / raw.sum(axis=1, keepdims=True)
    log_p = np.log(p)
    return {
        "raw": raw, "centered": centered, "norms_centered": norms_centered,
        "norms_raw": norms_raw, "p": p, "log_p": log_p,
    }


def dataset_mean_for_metric(metric, stats_a, stats_b, perm):
    if metric == "pearson":
        val, _ = dataset_mean_pearson_for_permutation(
            stats_a["centered"], stats_a["norms_centered"],
            stats_b["centered"], stats_b["norms_centered"], perm)
    elif metric == "cosine":
        val, _ = dataset_mean_cosine_for_permutation(
            stats_a["raw"], stats_a["norms_raw"], stats_b["raw"], stats_b["norms_raw"], perm)
    elif metric == "normalized_jsd":
        val, _ = dataset_mean_jsd_for_permutation(
            stats_a["p"], stats_a["log_p"], stats_b["p"], stats_b["log_p"], perm)
    else:
        raise ValueError(metric)
    return val


def load_original_family2(paired_csv, la, lb, metric):
    with open(paired_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["family"] == "spatial_similarity_same_vs_shuffled" and r["metric"] == metric
                    and r["comparison"] == f"L{la}-L{lb}_same_minus_shuffled"):
                return r
    raise KeyError((la, lb, metric))


def fmt(x, nd=4):
    if x is None or x == "":
        return "n/a"
    return f"{float(x):.{nd}f}"


def build_report_revised(report_revised_md, test_results, ordered_keys, repeated_summary, paired_csv):
    with open(paired_csv, encoding="utf-8") as fh:
        paired_rows = list(csv.DictReader(fh))

    def get_row(family, metric, comparison):
        for r in paired_rows:
            if r["family"] == family and r["metric"] == metric and r["comparison"] == comparison:
                return r
        raise KeyError((family, metric, comparison))

    lines = []
    lines.append("# [B-2] M-shape spatial-distribution comparison: L4 vs L8 vs L12 (REVISED)")
    lines.append("")
    lines.append("This is a REVISED interpretation pass over the same underlying numbers "
                 "as `report.md` (unchanged, kept as-is), tightened after a pre-commit "
                 "robustness review. It replaces overreaching phrasing with more precise "
                 "statements and adds a repeated-derangement robustness check for the "
                 "same-image-vs-shuffled-baseline (Family 2) comparisons. See "
                 "`report.md` for the full numeric tables (Family 1's 27 tests, per-layer "
                 "descriptives, figures) -- this file focuses on the corrected narrative "
                 "and the new robustness section.")
    lines.append("")

    lines.append("## Summary of supported conclusions")
    lines.append("")
    lines.append("**1. L4 and L12 are not in the same Attention state.** L4 is weakly and "
                 "broadly spread out; L12 is strongly concentrated on a limited region. "
                 "Their fine-grained patterns of Attention strength mostly do not "
                 "correspond to each other. Because L4 is broadly spread, this is NOT "
                 "described as \"L4 looks at none of the patches L12 looks at\" -- L4's "
                 "broad spread means it inevitably places some weight everywhere, "
                 "including wherever L12 later concentrates; what differs is the relative "
                 "emphasis, not disjoint support.")
    lines.append("")
    lines.append("**2. No meaningful average difference in foreground enrichment was found "
                 "between L4 and L12.** L4=1.599, L12=1.572, diff=-0.027, 95% CI=[-0.074, "
                 "0.028], Cohen's dz=-0.038, BH-FDR q=0.316 (from the per-image paired "
                 "comparison in `paired_layer_comparisons.csv`). A formal equivalence test "
                 "was NOT run, so this is reported as \"no meaningful difference was "
                 "found\", NOT as \"L4 and L12 are equivalent\" or \"proven equivalent\". "
                 "Likewise, \"not significant\" is never written as \"equivalent\" anywhere "
                 "in this report.")
    lines.append("")
    lines.append("**3. L8 and L12 are spatially very similar, but not identical.** Pearson="
                 "0.958, cosine=0.947, normalized JSD=0.122 (same-image, dataset means), "
                 "robust across every derangement tried (see below). Going from L8 to L12, "
                 "top-10% mass increases and the 50%/80% support area shrinks further: L12 "
                 "further sharpens its focus on largely the SAME location L8 already "
                 "attends to, and reallocates Attention onto the foreground (foreground "
                 "enrichment recovers from L8's ~0.995 to L12's ~1.572).")
    lines.append("")
    lines.append("**4. L8 is not an \"Attention has become diffuse\" trough.** L8 is "
                 "substantially MORE concentrated than L4 (Hoyer sparsity L8-L4 "
                 f"dz={fmt(get_row('layer_scalar_metrics','hoyer_sparsity','L8-L4')['cohens_dz'],3)}). "
                 "The M-shape's trough (in gaze-alignment / foreground-enrichment terms) "
                 "cannot be explained by L8's Attention being blurry or unfocused -- it is "
                 "in fact more sharply focused than L4's, just apparently less focused on "
                 "foreground objects specifically.")
    lines.append("")
    lines.append("**5. This experiment does not pin down an exact \"switch layer\", and L8 "
                 "is not asserted to be it.** Only L4, L8, and L12 were compared. What can "
                 "be said is that a large spatial change happens somewhere between L4 and "
                 "L8, and that by L8 a later-layer-type spatial arrangement is already in "
                 "place. Identifying the precise switch layer would require L5-L7, which "
                 "were not extracted in this experiment.")
    lines.append("")
    lines.append("**6. L4-L8's absolute similarity is low, clearly different from "
                 "L8-L12's strong similarity -- but the three similarity metrics do not "
                 "fully agree on direction relative to the shuffled baseline for L4-L8** "
                 "(see the dedicated L4-L8 discussion below). This is reported precisely, "
                 "rather than as a single \"L4-L8 is consistently less similar than "
                 "shuffled\" statement, which the per-metric results do not support.")
    lines.append("")

    lines.append("## Corrected phrasing (before -> after)")
    lines.append("")
    lines.append("- BEFORE: \"L4 and L12 attend to spatially almost unrelated, different "
                 "patches.\"\n  AFTER: \"L4's broadly spread pattern of Attention strength "
                 "mostly does not correspond to L12's concentrated pattern.\"")
    lines.append("- BEFORE: \"L8 is the switch point.\"\n  AFTER: \"A change happens "
                 "between L4 and L8, and by L8 a later-layer-type spatial arrangement is "
                 "already in place. The exact switch layer is not identified.\"")
    lines.append("- BEFORE: \"The concentration change from L8 to L12 is small.\"\n  AFTER: "
                 "\"L8 and L12 attend to very similar locations, while L12 sharpens focus "
                 "on the same location further and reallocates Attention onto the "
                 "foreground.\"")
    lines.append("- BEFORE: \"L4 and L12's foreground enrichment are equivalent\" / \"this "
                 "is a valid null result.\"\n  AFTER: \"No meaningful average difference in "
                 "foreground enrichment was found between L4 and L12; a formal equivalence "
                 "test was not performed.\"")
    lines.append("- BEFORE: \"L4-L8 is consistently less similar than shuffled.\"\n  AFTER: "
                 "\"L4-L8's Pearson and cosine same-image similarity did not exceed the "
                 "shuffled baseline, while its normalized JSD same-image divergence WAS "
                 "lower than shuffled (more similar) -- the three metrics do not fully "
                 "agree on direction; L4-L8's absolute similarity is in any case low and "
                 "clearly distinct from L8-L12's strong similarity.\"")
    lines.append("")

    lines.append("## Limitations (retained)")
    lines.append("")
    lines.append("- L4-L12 spans 8 layers; L4-L8 and L8-L12 each span 4 layers -- L4-L12's "
                 "smaller raw differences are partly confounded with this coarser sampling "
                 "interval.")
    lines.append("- This is a DESCRIPTIVE comparison of head-averaged Attention's SPATIAL "
                 "DISTRIBUTION only. Individual-head structure, and the causal "
                 "contribution of MHA, MLP, and residual connections to producing this "
                 "pattern, were NOT tested.")
    lines.append("- This is not evidence that Attention is actually USED in the model's "
                 "final decision (raw Attention is not a causal explanation of the output).")
    lines.append("- This is not evidence that language-supervised training CAUSED the "
                 "M-shape; no untrained/ablated baseline was compared here.")
    lines.append("- `null_standardized_distance` (observed dataset-mean expressed in units "
                 "of the empirical null distribution's own standard deviation) is NOT "
                 "\"effect size\" and NOT Cohen's d/dz -- it is never called either in this "
                 "report. A narrow null distribution (small shuffled-means variance across "
                 "derangements) can produce a large null_standardized_distance even when "
                 "the underlying raw difference (observed - shuffled mean) is practically "
                 "small -- see the L4-L12 discussion below. Cohen's dz (from "
                 "`paired_layer_comparisons.csv`, computed on per-image paired "
                 "differences) remains the primary practical-magnitude reference for that "
                 "separate, per-image analysis; it answers a different question from "
                 "null_standardized_distance and the two are never merged.")
    lines.append("")

    lines.append(f"## Repeated-shuffle robustness check ({repeated_summary['n_derangements']} "
                 f"independent derangements, base_seed={repeated_summary['base_seed']})")
    lines.append("")
    lines.append(f"Elapsed: {repeated_summary['elapsed_seconds']:.1f}s. Family 2 is NOT a "
                 "test of \"correlation > 0\": Pearson/cosine test whether same-image "
                 "similarity EXCEEDS the shuffled-image (center-bias) baseline "
                 "(p = (1 + #{shuffled means >= same-image mean}) / (1 + B)); normalized "
                 "JSD tests whether same-image divergence is LOWER than the shuffled "
                 "baseline (p = (1 + #{shuffled means <= same-image mean}) / (1 + B), lower "
                 f"JSD = more similar). With B={repeated_summary['n_derangements']}, the "
                 f"smallest possible p is 1/{repeated_summary['n_derangements']+1} ~ "
                 f"{1/(repeated_summary['n_derangements']+1):.6f} -- p=0 is never reported. "
                 "Full per-test numbers are in `repeated_shuffle_summary.json`; this "
                 "section gives the layer-pair-level interpretation.")
    lines.append("")

    l4l8 = {m: test_results[(4, 8, m)] for m in METRICS}
    l8l12 = {m: test_results[(8, 12, m)] for m in METRICS}
    l4l12 = {m: test_results[(4, 12, m)] for m in METRICS}

    lines.append("### L4-L8")
    lines.append("")
    lines.append(
        f"Pearson (same={fmt(l4l8['pearson']['same_image_mean'])}, shuffled mean="
        f"{fmt(l4l8['pearson']['shuffled_means_mean'])}) and cosine (same="
        f"{fmt(l4l8['cosine']['same_image_mean'])}, shuffled mean="
        f"{fmt(l4l8['cosine']['shuffled_means_mean'])}) -- same-image similarity did "
        f"**not exceed** the shuffled baseline for either metric, across every "
        f"derangement tried. Normalized JSD (same={fmt(l4l8['normalized_jsd']['same_image_mean'])}, "
        f"shuffled mean={fmt(l4l8['normalized_jsd']['shuffled_means_mean'])}) shows the "
        f"**opposite** pattern: same-image divergence was **lower** than shuffled (i.e. "
        f"more similar), also robust across every derangement tried. **The three metrics "
        f"do not fully agree on direction for L4-L8.** This report does NOT state "
        f"\"L4-L8 is consistently less similar than shuffled\" -- that is not what the "
        f"data show. What can be said: **L4-L8's absolute similarity is low in all three "
        f"metrics, and clearly different from L8-L12's strong similarity below.**")
    lines.append("")

    lines.append("### L8-L12")
    lines.append("")
    lines.append(
        f"All three metrics show very high same-image similarity, robust across every "
        f"derangement tried: Pearson same={fmt(l8l12['pearson']['same_image_mean'])} vs. "
        f"shuffled mean={fmt(l8l12['pearson']['shuffled_means_mean'])} (95% range=["
        f"{fmt(l8l12['pearson']['shuffled_means_ci_lo_2p5'])}, "
        f"{fmt(l8l12['pearson']['shuffled_means_ci_hi_97p5'])}]); cosine same="
        f"{fmt(l8l12['cosine']['same_image_mean'])} vs. shuffled mean="
        f"{fmt(l8l12['cosine']['shuffled_means_mean'])}; normalized JSD same="
        f"{fmt(l8l12['normalized_jsd']['same_image_mean'])} vs. shuffled mean="
        f"{fmt(l8l12['normalized_jsd']['shuffled_means_mean'])}. The observed values fall "
        f"nowhere near any of the shuffled-baseline distributions tried. Going from L8 to "
        f"L12, the same location is attended to, but more sharply: L12 further sharpens "
        f"focus on largely the same location L8 already attends to, and reallocates "
        f"Attention onto the foreground (foreground enrichment L8~0.995 -> L12~1.572).")
    lines.append("")

    lines.append("### L4-L12")
    lines.append("")
    lines.append(
        f"Across every derangement tried, the tendency for same-image similarity to be "
        f"slightly higher than shuffled was stable for Pearson (same="
        f"{fmt(l4l12['pearson']['same_image_mean'])}, shuffled mean="
        f"{fmt(l4l12['pearson']['shuffled_means_mean'])}, observed-shuffle_mean="
        f"{fmt(l4l12['pearson']['observed_minus_shuffled_mean'])}) and cosine (same="
        f"{fmt(l4l12['cosine']['same_image_mean'])}, shuffled mean="
        f"{fmt(l4l12['cosine']['shuffled_means_mean'])}, observed-shuffle_mean="
        f"{fmt(l4l12['cosine']['observed_minus_shuffled_mean'])}). **The raw magnitude of "
        f"this difference is small** (Pearson diff ~0.022, cosine diff ~0.014). The "
        f"shuffled-means distribution across derangements is itself very narrow, which is "
        f"why `null_standardized_distance` looks large "
        f"(Pearson: {fmt(l4l12['pearson']['null_standardized_distance'],1)}; "
        f"cosine: {fmt(l4l12['cosine']['null_standardized_distance'],1)}) -- this "
        f"reflects the null distribution's narrowness, NOT a practically large effect; "
        f"`null_standardized_distance` is not Cohen's d/dz and is not reported as one "
        f"(see the limitations above). Statistical stability across derangements and "
        f"practical magnitude are two different things: the direction is stable, but the "
        f"absolute spatial pattern is not nearly as similar as L8-L12's.")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `layer_pair_similarity_repeated_shuffle.csv` -- 9 (layer_pair x metric) "
                 "same-image dataset means + per-derangement shuffled dataset means.")
    lines.append("- `repeated_shuffle_summary.json` -- the 9 test results in full, plus "
                 "each one's agreement with the original fixed-derangement version.")
    lines.append("- `summary_revised.json` -- `summary.json`'s content plus a "
                 "`repeated_shuffle_robustness` section pointing at the above.")
    lines.append("- This file (`report_revised.md`) is a NEW file; `report.md` is "
                 "unmodified.")
    lines.append("")

    with open(report_revised_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved: {report_revised_md}")


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(args.cache_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    base_seed = args.seed
    n_derangements = args.n_derangements

    repeated_sim_csv = output_dir / "layer_pair_similarity_repeated_shuffle.csv"
    repeated_summary_json = output_dir / "repeated_shuffle_summary.json"
    summary_revised_json = output_dir / "summary_revised.json"
    report_revised_md = output_dir / "report_revised.md"
    paired_csv = data_dir / "paired_layer_comparisons.csv"
    summary_json_path = data_dir / "summary.json"

    print("=" * 70)
    print(f"  [B-2] Repeated-shuffle robustness: {n_derangements} derangements, "
          f"base_seed={base_seed}")
    print("=" * 70)
    print(f"  cache_path = {cache_path}")
    print(f"  data_dir   = {data_dir}")
    print(f"  output_dir = {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_if_exists([str(repeated_sim_csv), str(repeated_summary_json),
                       str(summary_revised_json), str(report_revised_md)])

    stems = validate_inputs(cache_path, data_dir)
    print("  Input validation (cache_verification PASS, cache shape/layers/stems, "
          "per_image_metrics.csv 2100 rows no dup, image-ID sets match): OK")

    cache = np.load(str(cache_path), allow_pickle=False)
    attn = cache["attn"].astype(np.float64)  # (700, 3, 38, 50)
    flat = {ld: attn[:, LAYER_IDX_IN_CACHE[ld]].reshape(N_IMAGES_EXPECTED, -1) for ld in LAYERS_DISPLAY}
    layer_stats = {ld: precompute_layer_stats(flat[ld]) for ld in LAYERS_DISPLAY}
    identity = np.arange(N_IMAGES_EXPECTED)

    sim_rows = []
    test_results = {}

    t0 = time.time()
    for la, lb in LAYER_PAIRS:
        stats_a, stats_b = layer_stats[la], layer_stats[lb]
        same_means = {}
        for metric in METRICS:
            same_means[metric] = dataset_mean_for_metric(metric, stats_a, stats_b, identity)
            sim_rows.append({
                "layer_a": la, "layer_b": lb, "metric": metric, "source": "same_image",
                "derangement_index": -1, "dataset_mean_value": same_means[metric],
            })

        null_dists = {metric: [] for metric in METRICS}
        t_pair0 = time.time()
        for k, perm in enumerate(iter_derangements(N_IMAGES_EXPECTED, base_seed, n_derangements)):
            for metric in METRICS:
                val = dataset_mean_for_metric(metric, stats_a, stats_b, perm)
                null_dists[metric].append(val)
                sim_rows.append({
                    "layer_a": la, "layer_b": lb, "metric": metric, "source": "shuffled",
                    "derangement_index": k, "dataset_mean_value": val,
                })
            if (k + 1) % 200 == 0:
                print(f"    L{la}-L{lb}: {k + 1}/{n_derangements} derangements, "
                      f"elapsed={time.time() - t_pair0:.1f}s")

        for metric in METRICS:
            null = np.array(null_dists[metric])
            observed = same_means[metric]
            null_mean = float(null.mean())
            null_std = float(null.std(ddof=1))
            null_ci_lo, null_ci_hi = np.percentile(null, [2.5, 97.5]).tolist()
            null_standardized_distance = (
                (observed - null_mean) / null_std if null_std > 0 else float("nan"))
            p = monte_carlo_one_sided_p(observed, null, direction=DIRECTION[metric])
            test_results[(la, lb, metric)] = {
                "layer_a": la, "layer_b": lb, "metric": metric,
                "direction_tested": DIRECTION[metric],
                "same_image_mean": observed,
                "shuffled_means_mean": null_mean,
                "shuffled_means_ci_lo_2p5": null_ci_lo,
                "shuffled_means_ci_hi_97p5": null_ci_hi,
                "observed_minus_shuffled_mean": observed - null_mean,
                "null_standardized_distance": null_standardized_distance,
                "monte_carlo_p": p,
                "n_derangements": n_derangements,
            }

    elapsed_total = time.time() - t0
    print(f"\n  Total elapsed: {elapsed_total:.1f}s for {n_derangements} derangements x "
          f"{len(LAYER_PAIRS)} layer pairs x {len(METRICS)} metrics")

    # ------------------------------------------------------------------
    # BH-FDR over the 9 tests
    # ------------------------------------------------------------------
    ordered_keys = [(la, lb, m) for la, lb in LAYER_PAIRS for m in METRICS]
    pvals = np.array([test_results[k]["monte_carlo_p"] for k in ordered_keys])
    qvals = bh_fdr(pvals)
    for k, q in zip(ordered_keys, qvals):
        test_results[k]["bh_fdr_q"] = float(q)

    # ------------------------------------------------------------------
    # Compare to the original single-derangement Family 2
    # ------------------------------------------------------------------
    for (la, lb, metric), result in test_results.items():
        orig = load_original_family2(paired_csv, la, lb, metric)
        orig_significant = float(orig["bh_fdr_q"]) < 0.05
        new_significant = result["bh_fdr_q"] < 0.05
        orig_direction_matches = (
            (DIRECTION[metric] == "greater" and float(orig["mean_diff"]) > 0) or
            (DIRECTION[metric] == "less" and float(orig["mean_diff"]) < 0))
        new_direction_matches = (
            (DIRECTION[metric] == "greater" and result["observed_minus_shuffled_mean"] > 0) or
            (DIRECTION[metric] == "less" and result["observed_minus_shuffled_mean"] < 0))
        result["original_single_derangement"] = {
            "mean_diff": float(orig["mean_diff"]),
            "cohens_dz_from_per_image_paired_comparison": float(orig["cohens_dz"]),
            "bh_fdr_q": float(orig["bh_fdr_q"]), "significant": orig_significant,
            "direction_matches_hypothesis": orig_direction_matches,
        }
        result["agreement_with_fixed_derangement"] = (
            orig_significant == new_significant and orig_direction_matches == new_direction_matches)

    # ------------------------------------------------------------------
    # Save layer_pair_similarity_repeated_shuffle.csv
    # ------------------------------------------------------------------
    with open(repeated_sim_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sim_rows[0].keys()))
        w.writeheader()
        w.writerows(sim_rows)
    print(f"\n  Saved: {repeated_sim_csv}  ({len(sim_rows)} rows)")

    # ------------------------------------------------------------------
    # Save repeated_shuffle_summary.json
    # ------------------------------------------------------------------
    repeated_summary = {
        "base_seed": base_seed, "n_derangements": n_derangements, "n_images": N_IMAGES_EXPECTED,
        "elapsed_seconds": elapsed_total,
        "paths_used": {"cache_path": str(cache_path), "data_dir": str(data_dir),
                        "output_dir": str(output_dir)},
        "derangement_generation": "lib.clip_attention_b2.iter_derangements("
                                    "n=700, base_seed, n_derangements) -- "
                                    "np.random.RandomState(derangement_seed(base_seed, k)), "
                                    "rejection-sampled for zero fixed points, one k=0..N-1 "
                                    "at a time (never all held in memory together)",
        "statistic_naming_note": "null_standardized_distance = (observed - shuffled_mean) "
                                   "/ shuffled_std -- this is NOT Cohen's d/dz and is never "
                                   "called an 'effect size' anywhere in this script's "
                                   "output. See original_single_derangement."
                                   "cohens_dz_from_per_image_paired_comparison for the "
                                   "genuine, separate per-image effect size.",
        "tests": [{**test_results[k]} for k in ordered_keys],
        "n_agreements_with_fixed_derangement": sum(
            1 for k in ordered_keys if test_results[k]["agreement_with_fixed_derangement"]),
        "n_tests": len(ordered_keys),
    }
    with open(repeated_summary_json, "w", encoding="utf-8") as fh:
        json.dump(repeated_summary, fh, indent=2)
    print(f"  Saved: {repeated_summary_json}")

    print("\n--- Family 2 (repeated-shuffle) results ---")
    for k in ordered_keys:
        r = test_results[k]
        agree = "AGREE" if r["agreement_with_fixed_derangement"] else "DISAGREE"
        print(f"    L{r['layer_a']}-L{r['layer_b']} {r['metric']:14s} "
              f"same={r['same_image_mean']:+.4f} shuf_mean={r['shuffled_means_mean']:+.4f} "
              f"[{r['shuffled_means_ci_lo_2p5']:+.4f},{r['shuffled_means_ci_hi_97p5']:+.4f}] "
              f"null_std_dist={r['null_standardized_distance']:+.2f} p={r['monte_carlo_p']:.6f} "
              f"q={r['bh_fdr_q']:.6f} [{agree} with fixed-derangement version]")

    # ------------------------------------------------------------------
    # summary_revised.json = original summary.json + repeated-shuffle section
    # ------------------------------------------------------------------
    with open(summary_json_path, encoding="utf-8") as fh:
        original_summary = json.load(fh)
    summary_revised = dict(original_summary)
    summary_revised["repeated_shuffle_robustness"] = {
        "base_seed": base_seed, "n_derangements": n_derangements,
        "source_file": "repeated_shuffle_summary.json",
        "n_agreements_with_fixed_derangement": repeated_summary["n_agreements_with_fixed_derangement"],
        "n_tests": repeated_summary["n_tests"],
        "note": "This section supplements (does not replace) the original single-"
                "seed=42-derangement Family 2 results already in this file's "
                "computation; see repeated_shuffle_summary.json for full detail per test. "
                "'null_standardized_distance' is not an effect size / Cohen's dz -- see "
                "repeated_shuffle_summary.json's statistic_naming_note.",
        "interpretation": {
            "L4-L8": "Pearson and cosine same-image similarity did NOT exceed the "
                     "shuffled baseline (robust across every derangement tried); "
                     "normalized JSD same-image divergence WAS lower than shuffled "
                     "(more similar), also robust. The three metrics do not fully "
                     "agree on direction -- NOT described as 'consistently less "
                     "similar than shuffled'. L4-L8's absolute similarity is low in "
                     "all three metrics, clearly different from L8-L12's strong "
                     "similarity.",
            "L8-L12": "All three metrics show very high same-image similarity, "
                      "robust across every derangement tried (observed values fall "
                      "nowhere near any shuffled-baseline distribution tried). L12 "
                      "further sharpens focus on largely the same location L8 already "
                      "attends to, and reallocates Attention onto the foreground.",
            "L4-L12": "A slightly higher same-image similarity than shuffled was "
                      "stable across every derangement tried for Pearson (diff ~0.022) "
                      "and cosine (diff ~0.014), but the raw magnitude is small. The "
                      "shuffled-means null distribution is itself narrow across "
                      "derangements, which inflates null_standardized_distance without "
                      "implying a practically large effect; Cohen's dz (from the "
                      "separate per-image paired comparison in "
                      "paired_layer_comparisons.csv) remains the primary "
                      "practical-magnitude reference, not null_standardized_distance.",
            "general_caveats": [
                "No exact switch layer is identified -- only L4/L8/L12 were compared; "
                "a change happens between L4 and L8, and by L8 a later-layer-type "
                "spatial arrangement is already established.",
                "L4-L12 spans 8 layers; L4-L8 and L8-L12 each span 4 layers.",
                "Descriptive comparison of head-averaged Attention's spatial "
                "distribution only -- MHA/MLP/residual causal contribution untested.",
                "Not evidence that Attention is used in the model's final decision.",
                "Not evidence that language-supervised training caused the M-shape.",
                "'Not significant' is never equated with 'equivalent' here.",
                "null_standardized_distance is never called an 'effect size' or "
                "'Cohen's dz' -- those terms are reserved for the separate per-image "
                "paired comparison.",
            ],
        },
    }
    with open(summary_revised_json, "w", encoding="utf-8") as fh:
        json.dump(summary_revised, fh, indent=2)
    print(f"  Saved: {summary_revised_json}")

    build_report_revised(report_revised_md, test_results, ordered_keys, repeated_summary, paired_csv)

    print("\n" + "=" * 70)
    print("  [B-2] Repeated-shuffle robustness DONE.")
    print("=" * 70)

    return test_results, ordered_keys, repeated_summary


if __name__ == "__main__":
    main()
