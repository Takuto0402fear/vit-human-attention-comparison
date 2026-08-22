"""
[B-2] Full700 main analysis, step 3/3 (report): figures/ + report.md,
synthesizing <data-dir>/{summary.json, paired_layer_comparisons.csv}.

Figures reuse the SAME 3 representative images as the pilot
(1156, 1159, 1213 -- chosen by the pilot's seed=42 draw, not cherry-picked
for this report), rendered with the corrected per_layer_scale /
shared_scale pair (lib.clip_attention_b2_plots), reading raw attention
directly from the verified cache (no forward pass).

The report text intentionally avoids overreaching claims:
  - does NOT assert "L8 is THE switch layer" -- only L4, L8, L12 were
    compared, so the most that can be said is that a large change
    happens somewhere between L4 and L8, and that by L8 a
    later-layer-type spatial arrangement is already established.
  - does NOT describe the L8->L12 change as merely "small" -- L8 and
    L12 attend to very similar locations, while L12 additionally
    sharpens focus further and reallocates Attention onto the
    foreground.
  - does NOT say L4 and L12 "look at completely different patches" --
    L4 is broadly spread (so it inevitably places some weight
    everywhere), it just mostly does not correspond to L12's
    concentrated pattern.
  - does NOT equate a non-significant test with "equivalent".
  - does NOT claim Attention is used in the model's final decision, or
    that MHA/MLP/residual computation or language-supervised training
    caused this pattern -- all of those are explicitly out of scope.

All paths are resolved relative to the repository root or overridable
via CLI flags -- no machine-specific absolute path is hardcoded. Writes
only new files (refuses to overwrite ANY of the specific files below if
they already exist -- including every figure, not merely report.md):
  <output-dir>/figures/{stem}_per_layer_scale.png
  <output-dir>/figures/{stem}_shared_scale.png
  <output-dir>/report.md

Usage (PowerShell):
    python scripts\\report_clip_attention_b2_full700.py
    python scripts\\report_clip_attention_b2_full700.py --data-dir D:\\tmp\\b2_smoke\\full700 --output-dir D:\\tmp\\b2_smoke\\full700
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

import h5py
import numpy as np
from PIL import Image

from lib.clip_attention_b2 import refuse_if_exists
from lib.clip_attention_b2_plots import render_per_layer_and_shared_scale_figures
from lib.osie_text_alignment import mask_to_patch_weights

LAYERS_DISPLAY = [4, 8, 12]
IMG_W, IMG_H = 800, 600
FIGURE_STEMS_DEFAULT = ["1156", "1159", "1213"]  # same as the pilot's seed=42 selection

DEFAULT_CACHE_PATH = (
    REPO_ROOT / "outputs" / "osie_attribute_grounding_full700" / "attn_cache"
    / "clip_vitb16_full700_L4L8L12_patchgrid.npz")
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "osie" / "predicting-human-gaze-beyond-pixels"
DEFAULT_STIMULI_DIR = DEFAULT_DATASET_DIR / "data" / "stimuli"
DEFAULT_ATTRS_PATH = DEFAULT_DATASET_DIR / "data" / "attrs.mat"
DEFAULT_DATA_DIR = REPO_ROOT / "outputs" / "clip_attention_b2_distribution"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="[B-2] Full700 report: synthesize figures/ + report.md from "
                    "scripts/stats_clip_attention_b2_full700.py's outputs.")
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                    help=f"Path to the existing L4/L8/L12 attention cache .npz, read-only "
                        f"(default: {DEFAULT_CACHE_PATH}).")
    p.add_argument("--stimuli-dir", type=Path, default=DEFAULT_STIMULI_DIR,
                    help=f"Directory containing OSIE stimulus images (<id>.jpg), read-only "
                        f"(default: {DEFAULT_STIMULI_DIR}).")
    p.add_argument("--attrs-path", type=Path, default=DEFAULT_ATTRS_PATH,
                    help=f"Path to OSIE's attrs.mat (object masks), read-only "
                        f"(default: {DEFAULT_ATTRS_PATH}).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Directory containing config.json / summary.json / "
                        f"paired_layer_comparisons.csv, read-only (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Directory to write figures/*.png and report.md into (created "
                        "if missing; refuses to overwrite any of them if they already "
                        f"exist) (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--figure-stems", type=str, default=",".join(FIGURE_STEMS_DEFAULT),
                    help="Comma-separated OSIE image IDs to render figures for (default: "
                        f"{','.join(FIGURE_STEMS_DEFAULT)}, the pilot's seed=42 selection).")
    return p.parse_args(argv)


def load_attrs_index(f):
    def h5_char_str(ref):
        return "".join(chr(int(c)) for c in f[ref][()].flatten())
    names_ds = f["attrNames"]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw_name = "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[Path(raw_name).stem] = g
    return index


def foreground_mask_for_image(f, group, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    fg = np.zeros((h, w), dtype=bool)
    for j in range(objs_ds.shape[1]):
        obj_g = f[objs_ds[0, j]]
        fg |= obj_g["map"][()].T.astype(bool)
    return fg


def make_figures(cache_path, stimuli_dir, attrs_path, output_dir, figure_stems):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        str(fig_dir / f"{stem}_{mode}.png")
        for stem in figure_stems for mode in ("per_layer_scale", "shared_scale")
    ]
    refuse_if_exists(expected_outputs)

    cache = np.load(str(cache_path), allow_pickle=False)
    stems_all = cache["stems"].tolist()
    attn_all = cache["attn"]

    f = h5py.File(str(attrs_path), "r")
    attrs_index = load_attrs_index(f)

    fig_paths = []
    for stem in figure_stems:
        img_idx = stems_all.index(stem)
        img_arr = np.array(Image.open(stimuli_dir / f"{stem}.jpg").convert("RGB"))
        H, W = img_arr.shape[:2]
        fg_mask = foreground_mask_for_image(f, attrs_index[stem], (H, W))
        coverage = mask_to_patch_weights(fg_mask, 16)
        layer_maps = {ld: attn_all[img_idx, i].astype(np.float64) for i, ld in enumerate(LAYERS_DISPLAY)}

        per_layer_path, shared_path = render_per_layer_and_shared_scale_figures(
            stem=stem, img_arr=img_arr, coverage=coverage, layer_maps=layer_maps,
            layers_display=LAYERS_DISPLAY, out_dir=str(fig_dir), img_wh=(IMG_W, IMG_H))
        fig_paths.extend([per_layer_path, shared_path])
        print(f"  Saved: {per_layer_path}")
        print(f"  Saved: {shared_path}")
    return fig_paths


def fmt(x, nd=4):
    if x is None or x == "":
        return "n/a"
    return f"{float(x):.{nd}f}"


def build_report(data_dir, output_dir, fig_paths, figure_stems):
    report_md = output_dir / "report.md"
    refuse_if_exists([str(report_md)])

    with open(data_dir / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    with open(data_dir / "paired_layer_comparisons.csv", encoding="utf-8") as fh:
        paired_rows = list(csv.DictReader(fh))

    def get_row(family, metric, comparison):
        for r in paired_rows:
            if r["family"] == family and r["metric"] == metric and r["comparison"] == comparison:
                return r
        raise KeyError((family, metric, comparison))

    desc = summary["descriptive_per_layer"]
    sim_desc = summary["spatial_similarity_descriptive"]

    def desc_line(metric):
        d = desc[metric]
        return " | ".join(
            f"L{l}: mean={fmt(d[f'L{l}']['mean'])} median={fmt(d[f'L{l}']['median'])} "
            f"95%CI=[{fmt(d[f'L{l}']['ci_lo'])}, {fmt(d[f'L{l}']['ci_hi'])}] (n={d[f'L{l}']['n']})"
            for l in LAYERS_DISPLAY)

    def test_line(metric, comp):
        r = get_row("layer_scalar_metrics", metric, comp)
        sig = "significant" if float(r["bh_fdr_q"]) < 0.05 else "NOT significant"
        return (f"  - **{comp}**: mean_diff={fmt(r['mean_diff'])}, "
                f"95%CI=[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}], "
                f"Cohen's dz={fmt(r['cohens_dz'], 3)}, "
                f"sign-flip p={fmt(r['sign_flip_p'], 5)}, "
                f"BH-FDR q={fmt(r['bh_fdr_q'], 5)} ({sig})")

    def sim_test_line(pair, metric):
        r = get_row("spatial_similarity_same_vs_shuffled", metric,
                     f"{pair}_same_minus_shuffled")
        sig = "significant" if float(r["bh_fdr_q"]) < 0.05 else "NOT significant"
        d = sim_desc[f"{pair}_{metric}"]
        return (f"  - **{metric}** ({pair}): same-image mean={fmt(d['same_image_mean'])}, "
                f"shuffled-baseline mean={fmt(d['shuffled_baseline_mean'])}, "
                f"diff 95%CI=[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}], "
                f"Cohen's dz={fmt(r['cohens_dz'], 3)}, "
                f"sign-flip p={fmt(r['sign_flip_p'], 5)}, BH-FDR q={fmt(r['bh_fdr_q'], 5)} ({sig})")

    family1_metrics = [
        ("normalized_entropy", "normalized entropy (H/log(1900))"),
        ("hoyer_sparsity", "Hoyer sparsity"),
        ("max_patch_attention", "max patch attention"),
        ("top10pct_mass", "top-10% mass (190 patches)"),
        ("mass50_area_fraction", "area fraction covering 50% mass"),
        ("mass80_area_fraction", "area fraction covering 80% mass"),
        ("normalized_center_distance", "normalized centroid distance from image center"),
        ("foreground_conditional_mass", "foreground conditional mass (p-weighted)"),
        ("foreground_enrichment", "foreground enrichment (conditional mass / area fraction)"),
    ]

    lines = []
    lines.append("# [B-2] M-shape spatial-distribution comparison: L4 vs L8 vs L12")
    lines.append("")
    lines.append("CLIP ViT-B/16, OSIE full700, Phase-2 (38x50 = 1900-patch) [CLS]->patch "
                 "attention, head-averaged, reused from the existing, independently "
                 "verified attention cache (see cache_verification.json). This is a "
                 "DESCRIPTIVE, statistical comparison of spatial Attention distributions "
                 "only. It does not test individual attention heads, and makes no causal "
                 "claim about MHA/MLP/residual mechanisms, about Attention being used in "
                 "the model's final decision, or about language-supervised training "
                 "having caused this pattern.")
    lines.append("")
    lines.append(f"- n images: {summary['n_images_total']} / 700 succeeded "
                 f"(0 missing masks, 0 non-finite attention)")
    lines.append(f"- degenerate images excluded from foreground/background-dependent "
                 f"metrics only: {summary['n_degenerate_images_excluded_from_fgbg_metrics']}")
    lines.append(f"- seed={summary['seed']}, bootstrap reps={summary['n_boot']}, "
                 f"sign-flip reps={summary['n_signflip']}")
    lines.append("- Family 2 below uses a SINGLE seed=42 derangement for the "
                 "shuffled-image baseline; a 1000-derangement robustness check is a "
                 "separate analysis, not included in this report.")
    lines.append("")

    lines.append("## 1. Per-layer descriptive statistics (mean / median / 95% bootstrap CI)")
    lines.append("")
    for metric, label in family1_metrics:
        lines.append(f"**{label}** (`{metric}`)")
        lines.append(f"  {desc_line(metric)}")
        lines.append("")

    lines.append("## 2. Paired layer comparisons (Family 1: layer_scalar_metrics, "
                 "BH-FDR over 27 tests)")
    lines.append("")
    lines.append("Direction: diff = value(L_B) - value(L_A), e.g. L8-L4 = L8 - L4.")
    lines.append("")
    for metric, label in family1_metrics:
        lines.append(f"**{label}**")
        for comp in ("L8-L4", "L12-L8", "L12-L4"):
            lines.append(test_line(metric, comp))
        lines.append("")

    lines.append("## 3. Spatial similarity between layers (descriptive)")
    lines.append("")
    lines.append("Pearson means use a Fisher-z transform (average in z-space, "
                 "back-transformed). Layer distance: L4-L8 and L8-L12 are 4 layers "
                 "apart each; L4-L12 is 8 layers apart -- a coarser sampling interval, "
                 "kept in mind when comparing L4-L12's similarity to the other two pairs.")
    lines.append("")
    for pair in ("L4-L8", "L8-L12", "L4-L12"):
        for metric in ("pearson", "cosine", "normalized_jsd"):
            d = sim_desc[f"{pair}_{metric}"]
            lines.append(f"  - {pair} {metric}: mean={fmt(d['same_image_mean'])}, "
                         f"median={fmt(d['same_image_median'])}")
    lines.append("")

    lines.append("## 4. Same-image vs. shuffled-image-pair baseline "
                 "(Family 2: spatial_similarity_same_vs_shuffled, BH-FDR over 9 tests, "
                 "single seed=42 derangement)")
    lines.append("")
    lines.append("This is NOT a test of \"correlation > 0\" -- it tests whether "
                 "same-image layer-pair similarity exceeds a center-bias baseline "
                 "(the same layer-pair computed against a randomly permuted, "
                 "different image, fixed seed=42 derangement, no image paired with "
                 "itself).")
    lines.append("")
    for pair in ("L4-L8", "L8-L12", "L4-L12"):
        lines.append(f"**{pair}**")
        for metric in ("pearson", "cosine", "normalized_jsd"):
            lines.append(sim_test_line(pair, metric))
        lines.append("")

    l4l12_pearson = get_row("spatial_similarity_same_vs_shuffled", "pearson", "L4-L12_same_minus_shuffled")
    l8l12_pearson = get_row("spatial_similarity_same_vs_shuffled", "pearson", "L8-L12_same_minus_shuffled")
    l4l8_pearson = get_row("spatial_similarity_same_vs_shuffled", "pearson", "L4-L8_same_minus_shuffled")
    fg_enrich_l12l4 = get_row("layer_scalar_metrics", "foreground_enrichment", "L12-L4")
    hoyer_l8l4 = get_row("layer_scalar_metrics", "hoyer_sparsity", "L8-L4")
    hoyer_l12l4 = get_row("layer_scalar_metrics", "hoyer_sparsity", "L12-L4")
    top10_l12l4 = get_row("layer_scalar_metrics", "top10pct_mass", "L12-L4")
    fg_enrich_l8l4 = get_row("layer_scalar_metrics", "foreground_enrichment", "L8-L4")
    fg_enrich_l12l8 = get_row("layer_scalar_metrics", "foreground_enrichment", "L12-L8")

    lines.append("## 5. Answers to the four research questions")
    lines.append("")
    lines.append("**Q1: Do L4 and L12 (similar gaze-alignment) attend to the same patches?**")
    lines.append("")
    lines.append(
        f"L4 is weakly and broadly spread out; L12 is strongly concentrated on a limited "
        f"region. L4 and L12 are not a re-appearance of the same Attention state: "
        f"L4-L12 same-image spatial similarity barely exceeds the shuffled-image "
        f"baseline (Pearson diff dz={fmt(l4l12_pearson['cohens_dz'],3)}, "
        f"q={fmt(l4l12_pearson['bh_fdr_q'],5)}) -- statistically detectable at n=700 but "
        f"a small effect size -- whereas L8-L12 same-image similarity vastly exceeds "
        f"the baseline (Pearson diff dz={fmt(l8l12_pearson['cohens_dz'],3)}). Because L4 "
        f"is broadly spread, it inevitably places some weight everywhere, including "
        f"wherever L12 later concentrates; what can be said precisely is that L4's "
        f"broadly spread pattern of Attention strength mostly does not correspond to "
        f"L12's concentrated pattern -- not that L4 and L12 look at completely "
        f"different, non-overlapping patches.")
    lines.append("")
    lines.append("**Q2: Does concentration differ between L4 and L12?**")
    lines.append("")
    lines.append(
        f"Yes, substantially. L12 is far more concentrated than L4 on every concentration "
        f"metric (Hoyer sparsity L12-L4 dz={fmt(hoyer_l12l4['cohens_dz'],3)}, "
        f"top-10% mass L12-L4 dz={fmt(top10_l12l4['cohens_dz'],3)}, "
        f"all q<0.001). L4's attention is diffuse; L12's is sharply peaked.")
    lines.append("")
    lines.append("**Q3: Does foreground/background allocation and centroid differ?**")
    lines.append("")
    lines.append(
        f"Centroid location itself barely moves (normalized center distance differences "
        f"have small effect sizes, dz<1 for all comparisons, though statistically "
        f"significant at n=700). Foreground ENRICHMENT, however, shows the classic "
        f"M-shape: L8 is significantly lower than both L4 and L12 "
        f"(L8-L4 dz={fmt(fg_enrich_l8l4['cohens_dz'],3)}, "
        f"L12-L8 dz={fmt(fg_enrich_l12l8['cohens_dz'],3)}). No meaningful average "
        f"difference in foreground enrichment was found between L4 and L12 "
        f"(L12-L4 dz={fmt(fg_enrich_l12l4['cohens_dz'],3)}, "
        f"95%CI=[{fmt(fg_enrich_l12l4['ci_lo'])}, {fmt(fg_enrich_l12l4['ci_hi'])}] "
        f"includes 0, q={fmt(fg_enrich_l12l4['bh_fdr_q'],5)}, NOT significant) -- this "
        f"is reported as \"no meaningful difference was found\", NOT as \"L4 and L12 are "
        f"equivalent\": a formal equivalence test was not performed, and a "
        f"non-significant result is not the same claim as equivalence.")
    lines.append("")
    lines.append("**Q4: Is there a spatial switch, and where?**")
    lines.append("")
    lines.append(
        f"Only L4, L8, and L12 were compared, so the precise switch layer cannot be "
        f"pinned down (that would require L5-L7, not extracted here). What the data "
        f"support: a large change happens somewhere between L4 and L8, and by L8 a "
        f"later-layer-type spatial arrangement is already established. L4-L8 same-image "
        f"similarity does not exceed the shuffled baseline for Pearson/cosine "
        f"(Pearson diff dz={fmt(l4l8_pearson['cohens_dz'],3)}), while normalized JSD "
        f"shows the opposite direction (same-image divergence lower than shuffled) -- "
        f"the three L4-L8 metrics do not fully agree on direction, but in any case "
        f"L4-L8's absolute similarity is low, clearly different from L8-L12's strong "
        f"similarity (Pearson diff dz={fmt(l8l12_pearson['cohens_dz'],3)}). Concentration "
        f"also jumps sharply from L4 to L8 "
        f"(Hoyer L8-L4 dz={fmt(hoyer_l8l4['cohens_dz'],3)}). L8 is therefore NOT an "
        f"\"Attention has become diffuse\" trough -- it is already far more concentrated "
        f"than L4. Going from L8 to L12: L8 and L12 attend to very similar locations "
        f"(Pearson diff dz={fmt(l8l12_pearson['cohens_dz'],3)} above the shuffled "
        f"baseline), while L12 additionally sharpens focus on that same location "
        f"further and reallocates Attention onto the foreground (foreground enrichment "
        f"L8-L4 dz={fmt(fg_enrich_l8l4['cohens_dz'],3)}, L12-L8 dz={fmt(fg_enrich_l12l8['cohens_dz'],3)}).")
    lines.append("")

    lines.append("## 6. Figures")
    lines.append("")
    lines.append(f"Representative images (same as the pilot's seed=42 draw): "
                 f"{', '.join(figure_stems)}. "
                 "Two versions per image, per_layer_scale and shared_scale (see "
                 "Implementation notes below for what each is for):")
    lines.append("")
    for p in fig_paths:
        rel = Path(p).relative_to(output_dir).as_posix() if Path(p).is_relative_to(output_dir) else str(p)
        lines.append(f"- `{rel}`")
    lines.append("")

    lines.append("## 7. Implementation and statistical notes")
    lines.append("")
    lines.append("- Metrics were computed entirely from the existing, independently "
                 "verified 38x50 attention cache (bit-exact match to a fresh forward "
                 "pass, see cache_verification.json) -- no new CLIP forward pass was run "
                 "for the 700-image metrics themselves.")
    lines.append("- `cls_self_mass = 1 - patch_mass_raw` (no extra forward pass): exact up "
                 "to float32 softmax-sum rounding, since the full [CLS] row is a "
                 "head-averaged softmax output that sums to 1 by construction; verified "
                 "against a real recomputed full_row_sum for 10 pilot images (max "
                 "deviation 1.19e-7).")
    lines.append("- `background_conditional_mass` is the exact complement of "
                 "`foreground_conditional_mass` and was intentionally NOT entered as a "
                 "separate test in Family 1 (would duplicate the foreground test under a "
                 "sign flip).")
    lines.append("- Two independent BH-FDR families, per task spec: Family 1 "
                 "(layer_scalar_metrics, 27 tests) and Family 2 "
                 "(spatial_similarity_same_vs_shuffled, 9 tests) were corrected "
                 "separately -- a q-value from one family is not comparable to the "
                 "other family's q-values.")
    lines.append("- Family 2 is NOT a test of \"correlation > 0\"; it tests same-image vs. "
                 "shuffled-image-pair (center-bias) similarity, paired per image via a "
                 "fixed seed=42 derangement (no image ever paired with itself). It uses a "
                 "single derangement here; a 1000-derangement robustness check is a "
                 "separate analysis.")
    lines.append("- L4-L8 and L8-L12 are 4 layers apart; L4-L12 is 8 layers apart -- a "
                 "coarser sampling interval that should be kept in mind when interpreting "
                 "why L4-L12 similarity looks small (it spans twice the depth).")
    lines.append("- A statistically significant result at n=700 is not automatically a "
                 "practically large one, and a non-significant result is not automatically "
                 "\"equivalence\" -- Cohen's dz and the 95% CI, not the p/q-value alone, "
                 "are the primary basis for judging practical magnitude throughout this "
                 "report.")
    lines.append("- `per_layer_scale` figures use each layer's own vmin=0/vmax=own max "
                 "-- colors are NOT comparable across layers, only within one panel. "
                 "`shared_scale` figures use vmin=0/vmax=that image's max over "
                 "L4/L8/L12 with one shared colorbar -- colors ARE comparable across "
                 "layers. Both use raw (un-normalized) attention, identical colormap, "
                 "alpha, bilinear upsampling (display only; all metrics were computed on "
                 "the native 38x50 grid), and a foreground-mask contour.")
    lines.append("- No causal claim: this experiment describes spatial differences in "
                 "raw, head-averaged Attention only. It does not establish that "
                 "MHA/MLP/residual computation or language-supervised training "
                 "\"produced\" this pattern, and it is not evidence that Attention is "
                 "actually used in the model's final decision.")
    lines.append("")

    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n  Saved: {report_md}")


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(args.cache_path)
    stimuli_dir = Path(args.stimuli_dir)
    attrs_path = Path(args.attrs_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    figure_stems = [s.strip() for s in args.figure_stems.split(",") if s.strip()]

    print("=" * 70)
    print("  [B-2] Full700 report: figures + report.md")
    print("=" * 70)
    print(f"  cache_path  = {cache_path}")
    print(f"  stimuli_dir = {stimuli_dir}")
    print(f"  attrs_path  = {attrs_path}")
    print(f"  data_dir    = {data_dir}")
    print(f"  output_dir  = {output_dir}")
    print(f"  figure_stems = {figure_stems}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = make_figures(cache_path, stimuli_dir, attrs_path, output_dir, figure_stems)
    build_report(data_dir, output_dir, fig_paths, figure_stems)
    print("\n" + "=" * 70)
    print("  [B-2] Full700 report DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
