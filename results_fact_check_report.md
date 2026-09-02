# Results Fact-Check Report (LAU-conference submission)

Audit date: 2026-09-01. Scope: verify every number and claim quoted in the
task brief's "現在のResults草稿" against this repository's existing, saved
outputs (CSV/JSON/Markdown reports) and the scripts that produced them. No
new model inference, attention re-extraction, or full metric recomputation
was performed. A small number of **cheap, existing-data-only re-aggregations**
were run during this audit (all listed explicitly below, all deterministic
and reproducible from already-cached per-image CSVs) — these are flagged
wherever used, and are distinguished from genuinely pre-existing canonical
outputs. No canonical file, figure, or script was modified while producing
this report.

**Note on scope**: no manuscript/Results draft file (`.tex`/`.md`/`.docx`)
exists inside this repository (confirmed by `find . -iname "*results*"`,
`-iname "*.docx"`, `-iname "*draft*"`, `-iname "*manuscript*"` — all return
no manuscript file, only `outputs/`/`results/` data directories). The
"現在のResults草稿の数値" referenced in the task brief are therefore treated
as the numbers quoted directly in the task brief itself, verified against
this repository's data.

---

## 1. Executive Summary

**Overall verdict: PASS WITH MINOR FIXES.**

The Results draft's core empirical claims are well-supported by the
repository's canonical outputs. Three issues require a text/number fix
before submission; none require new analysis.

**Critical/must-fix before submission:**

1. **"Pearson r = −0.84" (Figure 2, low-level vs. semantic curve
   correlation) is mislabeled.** The canonical file
   (`outputs/osie_layer_profile_synthesis/curve_correlations.csv`) reports
   **Pearson r = −0.7824** and **Spearman r = −0.8392** for this exact
   pair (12 layers). **−0.84 is the Spearman value, not Pearson.** Either
   relabel the statistic as Spearman (r = −0.84) or, if Pearson is
   required, correct the number to r = −0.78. Both are statistically
   significant at the exploratory n=12 sample size (Pearson p=0.0026,
   Spearman p=0.0006) but the file's own note states this is "exploratory
   -- n=12 layers, not a hypothesis test with adequate power," and that
   qualifier should be preserved in the Results text.
2. **50%-mass patch counts: the canonical values are 563 / 121 / 68 (L4/L8/L12), not 562 / 122 / 68.**
   The 562/122/68 figure is a **double-rounding artifact** (see Figure 3
   Audit, section 5, for the exact mechanism) and does not correspond to
   any independently-computed alternate result. Use 563/121/68 throughout
   (this is what the current Figure 3 already uses).
3. **"L4 and L12 show comparable gaze alignment" is only supported for
   AUC-Judd and sAUC, not for NSS** (the paper's own primary metric). A
   newly-run, cheap paired bootstrap (this audit; see section on Audit
   Target 4) shows CLIP-B's NSS at L4 (1.388) is significantly and
   substantially higher than at L12 (0.654) -- a ~2x difference, 95% CI of
   the difference [0.684, 0.782], excludes zero, Wilcoxon p=2.6e-89. Under
   AUC-Judd/sAUC the two layers ARE close (difference <0.01, Wilcoxon
   p=0.83/0.87, not significant). **Recommend wording option B** (frames
   L4/L12 as the early-peak/late-recovery pair of the N-shape, without
   asserting they are at "the same level" of alignment) **over option A**,
   per the task's own fallback rule.

**No other numerical discrepancies were found.** All other quoted numbers
(m_contrast values and CIs, NSS/AUC-Judd/sAUC per-layer values, low-level
R^2 peak layers, semantic AUPRC peak layers, foreground enrichment,
spatial correlations, Holm p-values) matched the canonical files exactly
or to the stated rounding precision.

**Confirmed but requiring careful phrasing (not numerical errors):**
sAUC's negative pool does not exclude the evaluated image's own fixations
(confirmed in code); "L8 = switch to language processing" has no
supporting evidence in any existing output and should not appear in the
Results text; the exact switch layer between L4 and L8 cannot be
identified without L5-L7 data, which does not exist in any cache.

**Could not verify:** whether "M-shape"/"M字型" appears anywhere in the
actual Results manuscript text, since that file does not exist in this
repository (see section 10).

---

## 2. Source Inventory

| Result | Code | CSV/JSON | Markdown report | Figure script | Final figure | Cache |
|---|---|---|---|---|---|---|
| 4-model NSS/AUC-Judd/sAUC per-image | `scripts/run_expA_clip_full700.py`, `run_expA.py`, `run_expA_dino_vitb16_full700.py`, `run_expA_sl_full700.py`; `metrics.py` | `outputs/expA_clip/healthy/metrics_per_image.csv`, `outputs/expA/healthy/metrics_per_image.csv`, `outputs/expA_dino_vitb16/healthy/metrics_per_image.csv`, `outputs/expA_sl/combined/metrics_by_image_meanacrosstrials.csv` | -- | `scripts/paper_figures/figure1_model_comparison_nss.py`, `figure1_model_comparison_all_metrics.py` | `paper_figures/figure1_model_comparison_nss.*`, `figure1_model_comparison_all_metrics_2col.*` | -- |
| N-shape contrast (`m_contrast`) | `scripts/stats_model_comparison_mcontrast.py`, `stats_three_model_layerwise.py` | `outputs/model_comparison_statistics/m_contrast_summary.csv`, `m_contrast_model_comparisons.csv`, `m_contrast_per_image.csv`, `layerwise_paired_comparisons.csv` | `outputs/model_comparison_statistics/statistics_summary.md` | -- | -- | -- |
| Semantic-attribute probe (12 attrs) | `scripts/extract_osie_probe_all12_full700_features.py`, `probe_osie_all12_full700.py`; `lib/osie_text_alignment_probe.py`, `lib/clip_vit_hidden.py` | `outputs/osie_probe_all12_full700/probe_summary.csv`, `probe_layer_differences.csv`, `probe_peak_layers.csv` | `outputs/osie_probe_all12_full700/report.md` | `scripts/paper_figures/figure2_gaze_and_probes.py` | `paper_figures/figure2_gaze_and_probes.*` | `object_features_all12.npz` |
| Low-level visual-feature probe (8 feats) | `scripts/extract_osie_lowlevel_targets.py`, `probe_osie_lowlevel_all12_full700.py`; `lib/osie_lowlevel_probe.py`, `lib/osie_lowlevel_features.py` | `outputs/osie_lowlevel_probe_all12_full700/lowlevel_summary.csv`, `lowlevel_peak_layers.csv`, `lowlevel_layer_differences.csv`; `outputs/osie_lowlevel_targets_full700/sanity_checks.json` | `outputs/osie_lowlevel_probe_all12_full700/report.md` | `scripts/paper_figures/figure2_gaze_and_probes.py` | `paper_figures/figure2_gaze_and_probes.*` | `object_features_all12.npz` (shared with semantic probe) |
| Layer-profile synthesis (correlations, H1-H4) | `scripts/synthesize_osie_layer_profile.py` | `outputs/osie_layer_profile_synthesis/layer_profile_table.csv`, `curve_correlations.csv`, `hypothesis_decision.json` | `outputs/osie_layer_profile_synthesis/report.md` | -- | -- | -- |
| B-2 spatial characterization (L4/L8/L12) | `scripts/compute_clip_attention_b2_full700.py`, `stats_clip_attention_b2_full700.py`, `verify_clip_attention_b2_shuffle_robustness.py`; `lib/clip_attention_b2.py` | `outputs/clip_attention_b2_distribution/per_image_metrics.csv`, `layer_pair_similarity.csv`, `layer_pair_similarity_repeated_shuffle.csv`, `summary.json`, `summary_revised.json`, `config.json` | `outputs/clip_attention_b2_distribution/report.md`, `report_revised.md` | `scripts/paper_figures/figure3_spatial_characterization_revised.py` (+ original `figure3_spatial_characterization.py`) | `paper_figures/figure3_spatial_characterization_revised.*` | `outputs/osie_attribute_grounding_full700/attn_cache/clip_vitb16_full700_L4L8L12_patchgrid.npz` |
| Pre-existing figure audits (this project's own prior work) | -- | -- | `paper_figures/FIGURE_AUDIT.md`, `METHOD_AUDIT.md` | -- | -- | -- |

Commit hash: not captured (this audit was run against the working tree
state as of 2026-09-01; `git status --porcelain` showed only new,
previously-added untracked files under `paper_figures/`,
`scripts/paper_figures/`, and `METHOD_AUDIT.md` -- no tracked file has
local modifications, so the HEAD commit accurately reflects all data
files read here).

---

## 3. Figure 1 Audit

| Item | 論文記載値 | 再計算値 | 元ファイル | 計算方法 | 判定 |
|---|---:|---:|---|---|---|
| CLIP-B N-shape contrast (NSS) | +0.883 | +0.8826 (rounds to +0.883) | `outputs/model_comparison_statistics/m_contrast_summary.csv` (row: model=CLIP-B, metric=NSS, measure=m_contrast) | `m_contrast = mean(NSS, L3-L5)/2 + mean(NSS, L11-L12)/2 - mean(NSS, L7-L9)`, computed per image (n=700), then bootstrap mean | MATCH |
| 95% CI | [+0.859, +0.907] | [+0.8590, +0.9065] | same file, `ci_lo`/`ci_hi` columns | percentile bootstrap, seed=42, n_boot=10000, over 700 images | MATCH |
| DeiT/SL contrast | −0.293 | −0.293392 | same file, model=DeiT-SL | same formula/method | MATCH |
| DINO-S contrast | −0.218 | −0.217619 | same file, model=DINO-S | same formula/method | MATCH |
| DINO-B contrast | −0.030 | −0.030277 | same file, model=DINO-B | same formula/method | MATCH |
| CLIP vs. 3 models, Holm p | all < 1e-100 | 8.60e-116 (DeiT-SL), 8.60e-116 (DINO-S), 8.60e-116 (DINO-B) | `outputs/model_comparison_statistics/m_contrast_model_comparisons.csv` (`p_holm` column) **for the printed value**; `statistics_summary.md` (`.2e`-formatted text) **for the precise exponent** | Wilcoxon signed-rank on paired per-image `m_contrast` (CLIP vs. other model), Holm-corrected within a 3-test family (`holm_family` column) | MATCH, with a caveat: see note below |
| CLIP-B NSS at L4 | 1.388 | 1.387560 | `outputs/expA_clip/healthy/metrics_by_layer.csv` (`NSS_mean`, layer=4); independently re-derived from `outputs/expA_clip/healthy/metrics_per_image.csv` (mean of 700 images) | plain mean over 700 images | MATCH |
| DINO-S max NSS | L10, 1.918 | L10, 1.918458 | `outputs/model_comparison_layerwise/comparison_sources.json` ("dino_s" peak_layer); re-derived from `outputs/expA/healthy/metrics_per_image.csv` | plain mean over 700 images, argmax over 12 layers | MATCH |
| DINO-B max NSS | L11, 1.755 | L11, 1.755134 | same JSON ("dino_b"); re-derived from `outputs/expA_dino_vitb16/healthy/metrics_per_image.csv` | same | MATCH |

**Caveat on the Holm p-value check (item 6 in the task brief):**
`m_contrast_model_comparisons.csv` itself stores `p_raw`/`p_holm` **rounded
to 6 decimal places at write time** (`scripts/stats_model_comparison_
mcontrast.py:490`, `f"{v:.6f}"`), so for a p-value as small as ~1e-116 the
CSV literally contains `0.000000` -- this is a **display/precision
artifact of the CSV writer, not evidence the true p-value is exactly
zero or was fabricated.** The true, full-precision value is preserved in
`statistics_summary.md`'s prose (written via `.2e` formatting from the
same in-memory float, before the lossy CSV write). This audit
independently re-derived the raw (pre-Holm) p-value directly from
`outputs/model_comparison_statistics/m_contrast_per_image.csv` using
`scipy.stats.wilcoxon` (a cheap, existing-data-only recomputation): CLIP-B
vs. DeiT-SL p=2.87e-116, vs. DINO-S p=2.87e-116, vs. DINO-B p=2.97e-116.
Multiplying by the family size (3, per Holm's first-step correction)
gives ≈8.6e-116 -- matching `statistics_summary.md`'s reported Holm
p-values almost exactly. **The `p<1e-100` claim is genuine and
independently reproducible, not a rounding artifact** -- but the paper
should cite it from `statistics_summary.md` (or recompute as this audit
did), not from the CSV's `p_holm` column, which has lost the precision.

**Figure 1 vs. final aggregation files**: `paper_figures/figure1_model_
comparison_nss.*` and `figure1_model_comparison_all_metrics_2col.*` both
read directly from the same 4 per-image CSVs listed in the Source
Inventory (via `scripts/paper_figures/common.py::load_all_model_curves`)
and were independently confirmed (in this project's own prior
`FIGURE_AUDIT.md`, section 5/11) to match `metrics_by_layer.csv` to 1e-6.
No discrepancy found.

**Figure 1 error bars**: confirmed to be image-ID-level bootstrap 95% CI
(seed=42, n_boot=10000, percentile method) computed independently per
model from each model's own 700-image per-image table
(`common.py::bootstrap_mean_ci`) -- this is the same convention already
established as the repo-wide standard (`scripts/stats_three_model_
layerwise.py:77-81`: `BOOT_SEED=42, N_BOOT=10000, ALPHA=0.05`) and matches
the Method description exactly ("画像単位bootstrap, 10,000回").

**Draft claim (item 9): "NSSに限らずAUC-JuddとsAUCでも主要な層別傾向が確認された。ただし3指標が完全に同一であることを意味しない."**

Verdict: **Supported.** CLIP-B's per-layer values (re-derived from the
same 3 per-image CSVs, no new computation beyond a plain mean):

```
NSS       L1..L12: 0.592 1.065 1.301 1.388 1.313 0.691 0.160 0.056 0.030 0.229 0.536 0.654
AUC-Judd  L1..L12: 0.678 0.760 0.795 0.799 0.790 0.687 0.682 0.696 0.700 0.767 0.800 0.806
sAUC      L1..L12: 0.659 0.732 0.749 0.767 0.766 0.738 0.734 0.703 0.698 0.741 0.765 0.772
```

All 3 metrics show the same qualitative 3-phase shape (early rise to a
local peak L3-L5, mid-layer dip L7-L9, late-layer recovery L10-L12). They
are **not identical in one respect that must be stated precisely**: for
NSS, the L12 recovery (0.654) reaches only ~47% of the L4 peak (1.388) --
a **partial** recovery. For AUC-Judd and sAUC, L12 (0.806 / 0.772)
slightly **exceeds** L4 (0.799 / 0.767) -- recovery that (mildly) surpasses
the early peak. The draft's hedge ("does not mean the three curves are
identical") is appropriate and should be kept; do not claim the recovery
magnitude is metric-independent.

**sAUC phrasing**: confirmed the negative pool does not exclude the
evaluated image's own fixations (`scripts/run_expA_clip_full700.py`
docstring: "no self-image exclusion, metrics.py's internal RandomState(0)
untouched"; confirmed identical across all 4 models' extraction scripts
via `grep -rn "sauc(" scripts/*.py` -- no override anywhere). The
draft should use "sAUC reduces sensitivity to dataset-level positional
bias," never "completely removes"/"fully controls for" center bias.

---

## 4. Figure 2 Audit

### Low-level visual features (8), peak layer per feature

| Feature | Peak layer | Peak R² | In L1-L4? |
|---|---|---:|---|
| mean_luminance | L3 | 0.9192 | Yes |
| mean_red | L3 | 0.9218 | Yes |
| mean_green | L3 | 0.9165 | Yes |
| mean_blue | L3 | 0.9169 | Yes |
| mean_saturation | L3 | 0.8737 | Yes |
| edge_strength | L4 | 0.8444 | Yes |
| fine_texture | L2 | 0.7646 | Yes |
| **luminance_contrast** | **L6** | 0.3557 | **No (exception)** |

Source: `outputs/osie_lowlevel_probe_all12_full700/lowlevel_summary.csv`
(`metric=="r2"`, `mean` column, argmax per `target` across `layer`).
**"7 of 8 peak in L1-L4" is confirmed exactly; the sole exception is
`luminance_contrast`, peaking at L6** (consistent with `outputs/
osie_lowlevel_probe_all12_full700/report.md`'s own text: "luminance_
contrastのみL6〜L8で例外的にピーク").

**8-feature mean R² at L4**: recomputed as the unweighted mean, across
the 8 features above, of `lowlevel_summary.csv`'s per-feature/per-layer
`mean` (each already itself a mean over 5 CV folds) = **0.7956** at L4 →
rounds to **0.80**, matching the draft. Full 12-layer curve: `0.7556
0.7807 0.7915 0.7956 0.7768 0.7442 0.6994 0.6378 0.5834 0.5181 0.4824
0.4190`.

### Semantic attributes (12), peak layer per attribute

| Attribute | Peak layer | Peak AUPRC | In L8-L11? |
|---|---|---:|---|
| Text | L8 | 0.9325 | Yes |
| Face | L10 | 0.9659 | Yes |
| Emotion | L9 | 0.5282 | Yes |
| Sound | **L12** | 0.4099 | **No (exception)** |
| Smell | L11 | 0.3895 | Yes |
| Taste | L9 | 0.7700 | Yes |
| Touch | **L12** | 0.4345 | **No (exception)** |
| Motion | L8 | 0.5125 | Yes |
| Operability | L11 | 0.5835 | Yes |
| Watchability | L8 | 0.8619 | Yes |
| Touched | L9 | 0.6720 | Yes |
| Gazed | L9 | 0.3903 | Yes |

Source: `outputs/osie_probe_all12_full700/probe_summary.csv`
(`metric=="auprc"`, `condition=="all_objects"`, `mean` column, argmax per
`attribute` across `layer`). **"10 of 12 peak in L8-L11" is confirmed
exactly; the 2 exceptions are `Sound` and `Touch`, both peaking at L12**
(matching `outputs/osie_layer_profile_synthesis/report.md`'s own text:
"L12単独でピークとなる属性は少数(Sound, Touchのみ)").

**12-attribute mean AUPRC**: L1 = **0.2167** (rounds to 0.22), L10 =
**0.5983** (rounds to 0.60), matching the draft. Full 12-layer curve:
`0.2167 0.2785 0.3341 0.3821 0.4280 0.4787 0.5165 0.5638 0.5970 0.5983
0.5812 0.5720`. Note L9 (0.5970) and L10 (0.5983) are nearly tied (a
near-plateau, not a sharp single-layer peak) -- fine detail worth keeping
in mind but not a contradiction of "rises to L10."

### r = −0.84 correlation

**Discrepancy found (see Executive Summary item 1).** Recomputed directly
(`np.corrcoef` on the two 12-point mean curves above) gives **Pearson r =
−0.7824**, matching `outputs/osie_layer_profile_synthesis/curve_
correlations.csv`'s stored `pearson_r` for
`semantic_probe_mean_auprc` vs. `lowlevel_probe_mean_r2` (n_layers=12)
exactly. That same CSV row's `spearman_r` = **−0.8392**, which rounds to
**−0.84**. **The draft's "−0.84" is the Spearman value; if the draft
labels it "Pearson," that label is incorrect.** Both are nominally
significant at this n=12 exploratory sample (Pearson p=0.0026, Spearman
p=0.0006), and the source file's own note ("exploratory -- n=12 layers,
not a hypothesis test with adequate power") should be preserved.

### Averaging / fold-aggregation method (items 10-11)

**Confirmed**: "mean R²"/"mean AUPRC" = **simple (unweighted) arithmetic
mean across the 8 features / 12 attributes**, where each feature's/
attribute's own per-layer value is **already the mean across its 5 CV
folds** (fold-mean computed first, upstream, in `lib/osie_lowlevel_
probe.py::run_ridge_probe` / `lib/osie_text_alignment_probe.py::
run_linear_probe`; group-mean-across-features/attributes computed
downstream in `scripts/synthesize_osie_layer_profile.py:139-144`:
`np.mean([...curves[a][l] for a in ATTRIBUTES/TARGETS])`, no sample-count
weighting anywhere). This exact aggregation order (fold-mean, then
simple mean across features/attributes) is the same order used
independently by `scripts/paper_figures/figure2_gaze_and_probes.py`
(confirmed in this project's own prior `verify_figures.py`, matching
`layer_profile_table.csv` to 1e-6 for all 12 layers).

**Figure 2 source check**: `figure2_gaze_and_probes.py` reads
`probe_summary.csv` / `lowlevel_summary.csv` directly -- the same files
cited above and in `layer_profile_table.csv`'s own provenance. No
discrepancy.

### Claims (items 13-14)

- "低次視覚特徴と意味属性の線形な読み出しやすさは、層に対して異なる方向に変化する": **Supported** -- one curve falls monotonically-ish from L1 (low-level) while the other rises to a mid/late plateau; Spearman r=−0.84 (Pearson r=−0.78) quantifies this trade-off, exploratory power caveat noted above.
- "低次から意味情報への読み出し可能性の変化だけでは、Attentionと人間視線のN字型プロファイルを説明できない": **Supported with qualification** -- `outputs/osie_layer_profile_synthesis/report.md`'s own H3 analysis found NSS's Spearman correlation with the probe curves is exploratory-only and inconsistent with AUC-Judd/sAUC (which show near-zero/non-significant correlation with the same probe curves) -- i.e., the probe curves' monotonic-ish trade-off does not itself have a mid-layer dip/late-layer-recovery shape matching the N-shape's 3-phase structure, so a "probe-explains-attention" story is not supported by these correlations. This is an *indirect* argument (shape mismatch + inconsistent correlation across the 3 gaze metrics), not a formal mediation/ablation test -- phrase as "not explained by" rather than "disproven."
- Linear-probe-is-not-causal-use caveat: **confirmed necessary and already present** in `outputs/osie_layer_profile_synthesis/report.md` ("probe成功は情報が線形に読み出せることの証拠であり、CLIPがその情報を因果的に利用している証明ではない") and in this project's own `METHOD_AUDIT.md`. Keep this caveat in the Results text.

---

## 5. Figure 3 Audit

| Item | 論文記載値 | 再計算値 | 元ファイル | 計算方法 | 判定 |
|---|---:|---:|---|---|---|
| 50%-mass patches, L4 | 563 | 563.2214 (round -> 563) | `outputs/clip_attention_b2_distribution/summary.json` (`descriptive_per_layer.mass50_n_patches.L4.mean`); independently re-derived as the plain mean of `mass50_n_patches` in `per_image_metrics.csv` (n=700) | mean of per-image integer patch counts, then rounded once | MATCH (563) |
| 50%-mass patches, L8 | 121 | 121.4986 (round -> 121) | same | same | MATCH (121) |
| 50%-mass patches, L12 | 68 | 67.6343 (round -> 68) | same | same | MATCH (68) |
| Foreground enrichment, L4/L8/L12 | 1.60 / 0.99 / 1.57 | 1.5986 / 0.9949 / 1.5719 | `summary.json` `descriptive_per_layer.foreground_enrichment`; independently re-derived from `per_image_metrics.csv` | mean of per-image `foreground_conditional_mass / foreground_area_fraction` (n=700) | MATCH |
| Pearson, L4-L8 | −0.048 | −0.0482 (Fisher-z mean) | `summary.json` `spatial_similarity_descriptive`; independently re-derived from `layer_pair_similarity.csv` (`pair_type=="same_image"`) | Fisher-z-transform each of the 700 per-image Pearson r, average, inverse-transform | MATCH |
| Pearson, L8-L12 | 0.958 | 0.9584 (Fisher-z mean; **plain mean = 0.9431, notably different**) | same | same | MATCH (Fisher-z method only) |
| Pearson, L4-L12 | 0.015 | 0.0146 (Fisher-z mean) | same | same | MATCH |

### Priority item: the 562/122/68 vs. 563/121/68 discrepancy -- RESOLVED

**Both numbers trace to the exact same canonical file**
(`outputs/clip_attention_b2_distribution/summary.json`) and the **exact
same underlying 700-image data** -- there is no separate/older/different
computation behind either one. The difference is a **rounding-order
artifact**, reproduced exactly in this audit:

- `summary.json`'s `descriptive_per_layer.mass50_n_patches.{L4,L8,L12}.mean`
  = **563.2214 / 121.4986 / 67.6343**.
- **563 / 121 / 68** = rounding this mean **once**, directly (the
  mathematically correct approach, and what Figure 3's panel (f)
  currently uses).
- **562 / 122 / 68** = taking the companion statistic
  `mass50_area_fraction.mean` (**0.296432 / 0.063947 / 0.035597**),
  rounding *that* to 3 decimal places first (0.296 / 0.064 / 0.036 -- as
  it might appear in a compact results table), *then* multiplying by
  1900 patches and rounding *again* (0.296×1900=562.4→562;
  0.064×1900=121.6→**122**; 0.036×1900=68.4→68). This **double rounding**
  reproduces 562/122/68 to the exact digit, confirmed by direct
  calculation in this audit. L12 happens to round to 68 either way, which
  is why only L4 and L8 visibly differ between the two conventions.

**Which is canonical**: `mass50_n_patches.mean`, rounded once -- **563 /
121 / 68**. This is what the current `figure3_spatial_
characterization_revised.py` panel (f) already displays and what
`FIGURE_AUDIT.md` already documents. **Recommendation: use 563/121/68
throughout the paper (body text, Figure 3, and its caption); do not use
562/122/68**, and if 562/122/68 appears anywhere in an existing draft or
external report, correct it to 563/121/68 (or, at minimum, add a footnote
noting both are the same underlying result under different rounding
conventions).

### Other Figure 3 checks

- **All 700 images included, no exclusions/NaN/Inf**: confirmed directly
  on `per_image_metrics.csv` (700 unique `image_id`, every image has
  exactly layers {4,8,12}, `is_degenerate_fgbg` all False, `mask_missing`
  all False, 0 NaN, 0 Inf across all numeric columns). `config.json`
  independently confirms `n_degenerate_images: 0`.
- **Same experimental conditions as the original N-shape experiment**:
  confirmed via `config.json` -- `cache_path` points to the same
  `clip_vitb16_full700_L4L8L12_patchgrid.npz` used throughout this
  project's OSIE analyses, `grid_hw: [38, 50]` (= 1900 patches),
  `layers_display: [4, 8, 12]`, `n_images: 700`.
- **Layer-specific color scale in the top row**: confirmed --
  `figure3_spatial_characterization_revised.py` uses `vmin=0.0`,
  `vmax = that layer's own raw max for that image`, independently for
  L4/L8/L12, each with its own colorbar (see `paper_figures/
  FIGURE_AUDIT.md` section 10/11 for the full rendering-convention
  audit). Color intensity is therefore explicitly NOT comparable across
  layers in this figure (stated in the figure and caption); the
  cross-layer-intensity comparison lives only in the separate
  common-scale appendix figure.
- **"L4 spreads widely" supported by the 700-image aggregate, not only
  the representative image**: confirmed -- the mass50/enrichment/
  correlation numbers above are all 700-image statistics; the
  representative-image row is explicitly captioned as illustrative only
  (`paper_figures/FIGURE_AUDIT.md` section 10: "the dataset-level claims
  ... are based entirely on the 700-image statistics ..., not on the
  appearance of any single representative image").

### The 9 spatial claims

| Claim | 判定 | 根拠 |
|---|---|---|
| L4ではAttentionが広い範囲に分散している | Supported | mass50=563/1900≈29.6% of all patches vs. L8 6.4%, L12 3.6%; `hoyer_sparsity` L4=0.137 (low = spread out) |
| L8ではAttentionが急激に集中する | Supported | mass50 563→121; `hoyer_sparsity` L4=0.137 → L8=0.722 (dz=7.951, `report_revised.md` point 4) |
| L8ではforegroundへの相対的配分が弱い | Supported | `foreground_enrichment` L8=0.995 ≈ 1.0 (area-proportional baseline) |
| L12ではさらに狭い領域へ集中する | Supported | mass50 L8=121→L12=68; `top10pct_mass` L8=0.559→L12=0.692; `normalized_entropy` L8=0.848→L12=0.796 (all consistent with further concentration) |
| L12ではforeground enrichmentが再び高くなる | Supported | 0.995 → 1.572 |
| L8とL12は強調する画像位置が非常に近い | Supported | Pearson (Fisher-z mean) = 0.958, robust across 1000 independent derangements (`layer_pair_similarity_repeated_shuffle.csv`, `repeated_shuffle_summary.json`) |
| L4とL12の空間的な強弱パターンはほとんど対応しない | Supported | Pearson = 0.015 (near zero); `report_revised.md`: raw magnitude small but the *tendency* is stable across derangements -- do not overstate as "exactly zero," but "little spatial correspondence" is accurate |
| 中間層の視線一致度低下はAttentionが単純に拡散したためではない | Supported | L8 is MORE concentrated than L4 (Hoyer dz=7.951), not diffuse; `report_revised.md` point 4 states this explicitly |
| L8の時点で後半層に類似した空間配置が形成されている | Supported | L8-L12 Pearson=0.958, robust across derangements; `report_revised.md`: "by L8 a later-layer-type spatial arrangement is already in place" |

**L8 as a "switch to language processing" layer**: **Not supported / no
evidence exists.** No linguistic or language-processing-specific analysis
was performed anywhere in the B-2 spatial pipeline (it measures only raw
attention geometry: patch counts, foreground overlap, cross-layer spatial
correlation). `report_revised.md` explicitly disclaims exactly this
framing: "This experiment does not pin down an exact 'switch layer', and
L8 is not asserted to be it." This phrase must not appear in the Results
text.

**Which layer the spatial change happens at (L4→L8) cannot be
pinpointed**: **Confirmed as an explicit, stated limitation**, not merely
an oversight -- `report_revised.md` point 5: "Identifying the precise
switch layer would require L5-L7, which were not extracted in this
experiment." No cache or output file in this repository contains L5, L6,
or L7 attention maps for the B-2 spatial metrics. Any Results-text claim
about "the exact layer of transition" beyond "somewhere between L4 and
L8" is unsupported.

---

## 6. Audit Target 4: "L4 and L12 show comparable gaze alignment"

**This audit ran one new, cheap computation** (paired bootstrap and
Wilcoxon test on already-loaded per-image CLIP-B data, seed=42,
n_boot=10000 -- same convention as everywhere else in this project; no
new model inference or feature extraction):

| Metric | L4 mean | L12 mean | Diff (L4−L12) | 95% CI of diff | Wilcoxon p |
|---|---:|---:|---:|---|---:|
| NSS | 1.3876 | 0.6540 | **+0.7335** | [+0.684, +0.782] (excludes 0) | 2.6e-89 |
| AUC-Judd | 0.7994 | 0.8059 | −0.0065 | [−0.0117, −0.0015] (excludes 0, tiny magnitude) | 0.83 (n.s.) |
| sAUC | 0.7667 | 0.7724 | −0.0057 | [−0.0114, ~0.0000] (borders 0) | 0.87 (n.s.) |

**No pre-existing file in this repository contains a formal L4-vs-L12
within-CLIP-B comparison** (`layerwise_paired_comparisons.csv` only
contains cross-MODEL comparisons at each layer, e.g. "DINO-S − CLIP-B at
L4," never within-model layer-vs-layer). This audit's numbers above are
therefore new (but cheap, deterministic, existing-data-only)
computations, not a re-statement of a previously-saved result.

**Interpretation**: under **NSS** (the paper's stated primary metric),
L4 and L12 are **not** at a comparable level -- L4 is roughly double L12,
a large and highly significant difference. Under **AUC-Judd and sAUC**,
L4 and L12 **are** close (raw difference <0.01) and the Wilcoxon test does
not reject equality (though this is a non-significant *difference* test,
not a formal *equivalence* test, so "statistically equivalent" would
still overstate it).

**No formal equivalence test (TOST or similar) was run anywhere for this
comparison** -- consistent with the B-2 spatial analysis's own explicit
practice elsewhere ("a formal equivalence test was NOT run" for the L4
vs. L12 foreground-enrichment comparison, `report_revised.md` point 2).

**Verdict**: **wording option A** ("同程度の視線一致度を示すL4とL12は、
同一のAttention状態ではない") **is not well-supported as a
metric-independent claim**, since it is false for NSS specifically (the
metric most prominently reported for L4 in the current draft, 1.388).
**Recommend wording option B** ("N字型プロファイルの前半の山と後半の回復
を代表するL4とL12は、異なるAttention状態を示した") -- this frames L4/L12
by their *role in the N-shape* (early peak, late recovery) rather than by
an unqualified "same level" claim, and is accurate regardless of which
metric is being discussed.

---

## 7. Claim Audit (Section "現在のResultsの中心的な結論")

| # | 主張 | 判定 | 根拠 | 推奨表現 |
|---|---|---|---|---|
| 1 | CLIP-B: 初期上昇→中間低下→後期再上昇のN字型プロファイル | Supported | `outputs/expA_clip/healthy/metrics_by_layer.csv`; `m_contrast_summary.csv` (+0.883, CI excludes 0, p<1e-100 vs. other models) | Keep as-is |
| 2 | この傾向はNSSだけでなくAUC-Judd/sAUCでも確認 | Supported with qualification | Section 3 above (3-phase shape common to all 3; recovery-vs-peak magnitude differs by metric) | Keep, but do not claim the 3 curves are identical in shape magnitude |
| 3 | DINO/DeiTでは同じN字型contrastが観察されない | Supported | `m_contrast_summary.csv`: DeiT −0.293, DINO-S −0.218, DINO-B −0.030, all negative vs. CLIP-B's +0.883; Holm p<1e-100 for all 3 vs. CLIP-B | Keep as-is |
| 4 | 低次特徴=浅層、意味属性=中〜深層で相対的に読み出しやすい | Supported | Section 4 above (7/8 low-level features peak L1-L4; 10/12 semantic attributes peak L8-L11) | Keep, name the 3 exceptions (luminance_contrast, Sound, Touch) if space allows |
| 5 | probeの層別変化は視線一致度のN字型変化と一致しない | Supported | `outputs/osie_layer_profile_synthesis/report.md` H3: attention-vs-probe correlations are weak/inconsistent across NSS/AUC-Judd/sAUC (exploratory, n=12) | Keep, retain "exploratory" qualifier |
| 6 | N字型は低次→意味情報への移行だけでは説明できない | Supported with qualification | Indirect argument via shape mismatch + inconsistent cross-metric correlation, not a mediation/ablation test | Phrase as "not explained by," not "disproven" |
| 7 | L4とL12は異なるAttention空間分布を持つ | Supported | Figure 3: Pearson(L4,L12)=0.015 (near-zero spatial correspondence); mass50 563 vs. 68; foreground_enrichment both ~1.6 but via very different spatial support | Keep as-is; do not conflate with "same gaze-alignment level" (see item below) |
| 8 | L8の谷はAttention拡散ではなく、集中したAttentionのforeground配分が弱い状態 | Supported | Hoyer sparsity L8 > L4 (more concentrated, not diffuse); foreground_enrichment L8≈1.0 (weak relative foreground allocation) | Keep as-is |
| 9 | 観察された関連であり、Attentionの因果的役割・probe情報の実利用・対照学習の因果効果を示すものではない | Supported (this is a scope caveat, verified consistent with all existing reports' own stated limitations) | `outputs/clip_attention_b2_distribution/report_revised.md` limitations section; `outputs/osie_layer_profile_synthesis/report.md` closing caveats | Keep as-is; this framing already matches how every underlying analysis report describes its own limitations |
| -- | "L4とL12は同程度の視線一致度" (separate from claim 7) | **Not supported for NSS; supported for AUC-Judd/sAUC** | Section 6 above | Use wording option B, not A (see section 6) |
| -- | "r = −0.84 (Pearson)" | **Not supported as labeled** | Section 4 above | Relabel as Spearman r=−0.84, or correct to Pearson r=−0.78 |
| -- | "50% mass: 562/122/68" | **Not supported as a distinct result; use 563/121/68** | Section 5 above | Use 563/121/68 |

---

## 8. Cross-section Consistency

1. **Figure 1-3 numbers vs. Results text**: all match to stated precision
   except the two flagged discrepancies (r=−0.84 mislabeled; 562/122/68
   double-rounding).
2. **Results vs. Method aggregation description**: consistent. Bootstrap
   = image-level, seed=42, n_boot=10000, alpha=0.05 throughout NSS/AUC-
   Judd/sAUC comparisons, N-shape contrast, and B-2 spatial analysis
   (confirmed independently in `METHOD_AUDIT.md`, sections H and P).
3. **Holm correction for N-shape model comparisons**: confirmed
   (`scripts/stats_model_comparison_mcontrast.py::holm_correction`,
   `holm_family` column explicitly documents each family's composition).
4. **BH-FDR for spatial-analysis multiple comparisons**: confirmed
   (`lib/clip_attention_b2.py::bh_fdr`, used in `scripts/stats_clip_
   attention_b2_full700.py` Family 1/2 and in the repeated-shuffle
   robustness check's 9-test family).
5. **Figure 3's Pearson correlation is explicitly same-image**: confirmed
   -- `layer_pair_similarity.csv`'s `pair_type=="same_image"` rows are
   the ones used for `summary.json`'s `spatial_similarity_descriptive`;
   the figure/caption in `paper_figures/FIGURE_AUDIT.md` already states
   "same-image Pearson correlation" explicitly.
6. **Figure 1-3 reference the latest canonical outputs, not stale ones**:
   confirmed -- e.g. the semantic/low-level probe figures use `_all12_
   full700` (12-layer) outputs, not the older `osie_probe_full700`
   (L4/L8/L12-only) directory; the B-2 spatial figure uses the full700
   run, not the 20-image pilot; the model-comparison figures use full700
   runs for all 4 models, not `pilot20`. (All of this provenance is
   independently re-confirmed in this project's own prior `FIGURE_
   AUDIT.md`, sections 1 and 4.)
7. **"M-shape"/"M字型" residue check**: none found in this project's own
   figure-generation scripts or `FIGURE_AUDIT.md`/`METHOD_AUDIT.md`
   (both consistently use "N-shape profile" in all captions; the only 2
   occurrences of the string "M-shape" in these files are explicit
   terminology-mapping notes, e.g. "'M-shape' / `M_contrast` is referred
   to as the 'N-shape profile'"). **Could not check the actual Results
   manuscript text**, since no such file exists in this repository (see
   Executive Summary / section 10 below) -- this check must be repeated
   directly on the manuscript wherever it is maintained (e.g. Overleaf).
8. **Code/variable/directory names leaking into prose**: none found in
   this project's own figure captions or audit documents (deliberately
   avoided per this project's own prior instructions -- e.g. "OSIE 700,
   healthy," internal file/directory names, and `M_contrast`/
   `m_contrast` are kept out of all captions). Could not check the actual
   manuscript body text for the same reason as item 7.

---

## 9. Exact Recommended Corrections

- **現在**: "低次視覚特徴と意味属性の層別曲線のPearson相関：r = −0.84"
  **修正案**: "低次視覚特徴と意味属性の層別曲線のSpearman相関：r = −0.84
  (Pearson r = −0.78, p=0.0026; Spearman p=0.0006; n=12層のexploratory
  な相関であり、検定力は限定的)"
  **修正理由**: `outputs/osie_layer_profile_synthesis/curve_
  correlations.csv`はこの層ペアについてPearson r=−0.7824、
  Spearman r=−0.8392を保存しており、−0.84はSpearman値である。

- **現在**: "50% Attention massに必要なpatch数：L4=562, L8=122, L12=68"
  （もし本文にこの表記がある場合）
  **修正案**: "50% Attention massに必要なpatch数：L4=563, L8=121, L12=68"
  **修正理由**: `summary.json`の`mass50_n_patches.mean`（563.22/121.50/
  67.63）を一度だけ丸めた値が正しい。562/122/68は面積比を先に3桁へ丸めて
  から1900倍して再度丸めた二重丸め誤差であり、独立した結果ではない。

- **現在**: "同程度の視線一致度を示すL4とL12は、同一のAttention状態では
  ない"
  **修正案**: "N字型プロファイルの前半の山（L4）と後半の回復（L12）を代表
  するこれら2層は、異なるAttention空間状態を示した（AUC-Judd/sAUCでは両層
  の値はほぼ一致するが、主指標のNSSではL4がL12のおよそ2倍であり、両層が
  「同程度」なのはAUC-Judd/sAUCに限られる）"
  **修正理由**: NSSでのL4-L12差は+0.734（95%CI [+0.684,+0.782]、
  Wilcoxon p=2.6e-89）であり、主指標での「同程度」という表現は支持されない。
  AUC-Judd/sAUCでは差が0.01未満で非有意（p=0.83/0.87）であり、こちらは
  支持される。

- **現在**: （もしResults文中に）"L8は言語処理へ切り替わる層である"
  **修正案**: 該当箇所を削除するか、"L8までに後半層に近い空間配置がすでに
  形成されている（L8-L12のPearson相関=0.958）が、これが言語処理への切り替
  えを意味するという証拠はない" へ書き換える。
  **修正理由**: `report_revised.md`が明示的に「L8がswitch layerであるとは
  主張しない」としており、言語処理に関する分析は本研究のどこにも存在しない。

---

## 10. Final Verified Values (copy-paste table for the manuscript)

| Quantity | Value | Source |
|---|---:|---|
| CLIP-B N-shape contrast (NSS) | +0.883, 95% CI [+0.859, +0.907] | `m_contrast_summary.csv` |
| DeiT/SL contrast | −0.293 | `m_contrast_summary.csv` |
| DINO-S contrast | −0.218 | `m_contrast_summary.csv` |
| DINO-B contrast | −0.030 | `m_contrast_summary.csv` |
| CLIP vs. each of 3 models, Holm p | < 1e-100 (≈8.60e-116 each) | `statistics_summary.md` (see CSV precision caveat, section 3) |
| CLIP-B NSS: L4 / L8 / L12 | 1.388 / 0.056 / 0.654 | `metrics_by_layer.csv` |
| CLIP-B AUC-Judd: L4 / L8 / L12 | 0.799 / 0.696 / 0.806 | re-derived from `metrics_per_image.csv` |
| CLIP-B sAUC: L4 / L8 / L12 | 0.767 / 0.703 / 0.772 | re-derived from `metrics_per_image.csv` |
| DINO-S max NSS | L10, 1.918 | `comparison_sources.json` |
| DINO-B max NSS | L11, 1.755 | `comparison_sources.json` |
| Low-level 8-feature mean R² at L4 | 0.80 (0.7956) | `lowlevel_summary.csv` |
| Low-level peak-in-L1-L4 count | 7 of 8 (exception: luminance_contrast, L6) | `lowlevel_summary.csv` |
| Semantic 12-attribute mean AUPRC: L1 / L10 | 0.22 (0.2167) / 0.60 (0.5983) | `probe_summary.csv` |
| Semantic peak-in-L8-L11 count | 10 of 12 (exceptions: Sound, Touch, both L12) | `probe_summary.csv` |
| Low-level vs. semantic curve correlation | **Spearman r = −0.84** (−0.8392); Pearson r = −0.78 (−0.7824) | `curve_correlations.csv` |
| 50%-mass patches: L4 / L8 / L12 | **563 / 121 / 68** | `summary.json` (single rounding of the mean) |
| Foreground enrichment: L4 / L8 / L12 | 1.60 / 0.99 / 1.57 | `summary.json` |
| Same-image Pearson: L4-L8 / L8-L12 / L4-L12 | −0.048 / 0.958 / 0.015 | `summary.json` (Fisher-z-averaged for Pearson) |
| NSS L4 vs. L12 difference (CLIP-B) | +0.734, 95% CI [+0.684,+0.782], p=2.6e-89 | computed in this audit (see section 6) |
| AUC-Judd/sAUC L4 vs. L12 difference (CLIP-B) | −0.006 / −0.006, both n.s. (p=0.83/0.87) | computed in this audit (see section 6) |

---

## 11. Remaining Unknowns

1. **The actual Results manuscript text was not available for direct
   inspection** (no `.tex`/`.md`/`.docx` draft exists in this
   repository). All "draft vs. data" comparisons in this report are
   therefore against the numbers/claims as quoted in the audit task
   brief itself, not against a source file this audit could open and
   grep directly. The "M-shape residue" and "code/variable names in
   prose" checks (Cross-section Consistency items 7-8) could only be
   performed on this project's own figure-generation artifacts
   (`FIGURE_AUDIT.md`, `METHOD_AUDIT.md`, caption drafts), not on the
   manuscript itself -- **these two checks must be repeated directly on
   the actual manuscript wherever it is maintained.**
2. **sAUC's fixed `RandomState(0)`** (distinct from the `seed=42`
   convention used everywhere else) -- confirmed present in `metrics.py`
   but no comment/commit explains why 0 rather than 42 was chosen
   (already flagged in `METHOD_AUDIT.md`, section "確認できなかった点").
   Does not affect any number in this report's tables (sAUC results were
   still cross-checked against the saved per-image CSVs, which already
   reflect this fixed seed), but worth knowing if a reviewer asks why
   sAUC's seed differs from the paper's stated "seed=42" convention
   elsewhere.
3. **Whether luminance_contrast's true peak spans L6-L8 as a plateau or
   is a sharp single-layer peak at L6**: this audit found peak
   *exactly* at L6 (0.3557); the L6-L8 range wording used in `report.md`
   likely refers to a broader near-peak tier rather than a single point
   -- not independently re-verified layer-by-layer for this specific
   nuance (low priority, does not change the "7 of 8 / exception"
   conclusion).
4. **No formal statistical equivalence test (TOST-style) exists anywhere
   in this repository** for any of the "L4 ≈ L12" or "L4 ≈ L12 foreground
   enrichment" claims. Every "no significant difference" statement in
   this report (and in the pre-existing `report_revised.md`) is a
   standard null-hypothesis-test non-rejection, not a demonstrated
   equivalence -- this distinction should be preserved in the manuscript
   language (as the task brief itself already anticipates).
