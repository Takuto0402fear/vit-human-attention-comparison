# Appendix Fact Check (Update)

Audit date: 2026-09-02. This document supersedes nothing and overwrites
nothing -- it extends `appendix_fact_check.md` with (a) exact, fully
recounted semantic-attribute and low-level-feature label statistics
(previously approximated), (b) library/version evidence, (c) explicit
sklearn-default-vs-explicit-setting distinctions for Ridge/
LogisticRegression, and (d) a few items not previously covered (CLIP text
encoder non-use, N-shape index's L4/L12 independence, bootstrap-sample
identity between figures and statistics). No new model inference, feature
extraction, or probe training was run. Two new counting scripts were run
against existing CSVs only (`scripts/paper_figures/_build_appendix_csvs.py`,
a new read-only helper, not part of the regular pipeline) to produce the
two requested CSVs from source data rather than from any approximation.

No manuscript/LaTeX/Notion export exists inside this repository
(re-confirmed again for this pass: no `.tex`, no `manuscript*`, no
`draft*`, no `.docx` other than an unrelated `python-docx` *package* name
inside `environment_after_clip.txt`). "原稿との照合" (section
"Manuscript Contradictions") is therefore, as in the prior audit, a check
against this repository's own existing figure/audit artifacts and against
the specific numeric assumptions embedded in this task's own instructions
-- not against an external draft this audit could open.

---

# Executive Summary

This pass is organized narratively (by topic) rather than as one single
strictly-verdict-tagged table, so the counts below are an honest
enumeration of each section's explicit judgments, not a programmatic tally
of a uniform `判定` column (unlike `appendix_fact_check.md`, which used
that format throughout). Every individual claim in this document is still
labeled Confirmed/Unverified/Contradicted in its own sentence or table
cell; see the section itself for the exact wording.

- **Contradicted: 0 new items this pass.** The one previously-known
  contradiction ("Pearson r=-0.84" is actually the Spearman value) is
  carried over from `appendix_fact_check.md`/`results_fact_check_report.md`
  and re-confirmed unchanged in "Manuscript Contradictions" below, not
  counted here as newly discovered.
- **Unverified: exactly 4**, listed in full in "Unverified Items" below
  (Ridge `solver="auto"`'s effective resolved solver; DINO/DeiT
  position-embedding interpolation code path; LogisticRegression
  convergence-warning history; whether `attrs.mat` itself received
  additional filtering upstream of this repository).
- **Not applicable: 1** (a held-out validation split separate from the
  test fold does not exist for either probe -- see "Cross-validation"
  table).
- **Everything else stated as fact in this document (the large majority
  of its content -- label counts, formulas, hyperparameter values, file
  paths, function names) is Confirmed**, each with its own file/line
  citation given inline; no aggregate "Confirmed: N" count is claimed, to
  avoid implying a precision this narrative format does not actually
  provide.

**Can Appendix tables be populated directly from this pass?** Yes for
Tables A2 (semantic labels) and A3 (low-level features) -- both are now
exact recounts from source CSVs, not approximations. Table A4
(hyperparameters) is complete except the two sklearn-default items noted
above (must be phrased as "sklearn既定値に従った", never as "明示的に設定
した", per this task's own instruction). Tables A1, A5, A6 were already
essentially complete in the prior audit pass and are reproduced here
verified once more.

---

# Critical Findings

1. **Semantic-attribute label counts are now EXACT, not approximate.**
   The prior audit (`appendix_fact_check.md`) computed positive/negative
   counts by inverting `prevalence × 5551`; this pass instead counted
   directly from `object_features_metadata.csv`'s `positive_attributes`/
   `negative_attributes` semicolon-separated columns for all 5,551
   objects. Every one of the 12 attributes partitions all 5,551 objects
   with **zero missing** (`positive_count + negative_count == 5551`
   exactly, for every attribute) -- see `appendix_semantic_labels.csv`
   and section "Semantic Probe Labels" below.
2. **The 12 attributes are the full OSIE `attrs.mat` attribute set, not a
   hand-picked subset.** `outputs/osie_text_alignment_pilot/attribute_
   mapping.json`'s `attrs_mat_order_as_stored` lists exactly these 12
   lowercase names (`text, face, emotion, sound, smell, taste, touch,
   motion, operability, watchability, touched, gazed`) as the complete,
   ordered column list of OSIE's own `attrs.mat` object-attribute
   annotation matrix. This resolves the prior audit's "Unverified"
   attribute-selection-rationale item: **the selection rationale is "use
   every attribute OSIE itself annotates," not a curated subset** (this
   audit did not find, and is not claiming, any additional documented
   rationale beyond that).
3. **The 288-object exclusion for low-level targets is now directly
   confirmed via an explicit `skip_reason` column**, not only inferred:
   `outputs/osie_lowlevel_targets_full700/lowlevel_object_targets.csv`
   has a `valid` boolean column and, for all 288 `valid==False` rows,
   `skip_reason == "empty_mask_after_transform"` (i.e., the object's mask
   became empty after the 224×224 Resize+CenterCrop). No other skip
   reason occurs anywhere in the file.
4. **The linear probe pipeline (both semantic and low-level) never calls
   CLIP's text encoder.** `grep -rn "clip.tokenize\|encode_text\|text_
   encoder\|CLIPTextModel"` across `scripts/probe_osie_all12_full700.py`,
   `scripts/extract_osie_probe_all12_full700_features.py`,
   `lib/osie_text_alignment_probe.py`, and `lib/clip_vit_hidden.py`
   returns **zero matches**. A SEPARATE experiment
   (`outputs/osie_text_alignment_pilot/`, with its own `prompt_bank.json`
   of English attribute descriptions) DOES use CLIP's text encoder, but
   its outputs are not used anywhere in the probe pipeline that produces
   Figure 2 / `probe_summary.csv`. The `prompt_bank.json` `base_phrases`
   were reused in this audit purely as convenient English definitions for
   Table A2 -- this is an audit-time convenience, not evidence that the
   probe itself uses those phrases.
5. **The N-shape index formula never assumes or asserts L4≈L12.** It
   compares layer BINS ([L3-L5], [L7-L9], [L11-L12]) as group means, and
   nowhere in `scripts/stats_model_comparison_mcontrast.py` or any
   downstream CSV is an "L4 equals L12" or "L4 comparable to L12"
   statement computed or asserted. See "N-shaped Profile Index" below for
   the exact formula and a fresh confirmation that CLIP-B's own NSS at
   L4 (1.388) and L12 (0.654) differ by roughly 2x (already established
   in `results_fact_check_report.md`, reconfirmed here as still accurate,
   not recomputed).

---

# Semantic Probe Labels

**Output**: `appendix_semantic_labels.csv` (12 rows, UTF-8 with BOM).

**Source files**:
- `outputs/osie_probe_all12_full700/object_features_metadata.csv` (5,551
  rows, one per object; `positive_attributes`/`negative_attributes`
  columns, semicolon-joined attribute-name lists) -- used for exact
  positive/negative/excluded counts.
- `outputs/osie_probe_all12_full700/probe_summary.csv` (`metric=="auprc"`,
  `condition=="all_objects"`, `mean` column) -- used for peak layer/value.
- `outputs/osie_text_alignment_pilot/attribute_mapping.json` -- used for
  the `attrs.mat`-order provenance and the mat-name↔task-name mapping
  (confirms these are OSIE's own 12 annotated attributes, verbatim).

**Label transformation (exact, from code)**: `lib/osie_text_alignment.py`
`split_positive_negative_attributes()` (lines 96-117) reads each object's
raw 0/1 (or >0/<=0) `attrs.mat` feature vector and assigns each of the 12
attribute names to EITHER the positive or negative set for that object --
"a strict partition ... no name in both, none omitted" (function
docstring), enforced by a `RuntimeError` if violated. This audit
independently re-verified the partition property by direct count (see
Critical Finding 1): **every attribute's positive+negative count equals
5,551 for all 12 attributes**, confirming the code's own invariant holds
in the actual saved data, not just in the code's intent.

**Exact counts** (all from `appendix_semantic_labels.csv`; `excluded_count`
is 0 for every attribute -- there is no missing-label exclusion for the
semantic probe, unlike the low-level probe):

| internal_label | positive_count | negative_count | best_layer | best_auprc |
|---|---:|---:|---:|---:|
| Text | 495 | 5056 | 8 | 0.932532 |
| Face | 1010 | 4541 | 10 | 0.965914 |
| Emotion | 190 | 5361 | 9 | 0.528232 |
| Sound | 89 | 5462 | 12 | 0.409948 |
| Smell | 135 | 5416 | 11 | 0.389536 |
| Taste | 390 | 5161 | 9 | 0.770049 |
| Touch | 317 | 5234 | 12 | 0.434494 |
| Motion | 557 | 4994 | 8 | 0.512517 |
| Operability | 342 | 5209 | 11 | 0.583451 |
| Watchability | 1108 | 4443 | 8 | 0.861898 |
| Touched | 436 | 5115 | 9 | 0.672005 |
| Gazed | 252 | 5299 | 9 | 0.390306 |

(total_count = 5551 for every attribute)

**Item 13 (10-of-12-peak-in-L8-L11 consistency check)**: **Confirmed**,
exactly. Peaks in {L8,L9,L10,L11}: Text(L8), Face(L10), Emotion(L9),
Smell(L11), Taste(L9), Motion(L8), Operability(L11), Watchability(L8),
Touched(L9), Gazed(L9) = **10 attributes**. Exceptions: **Sound (L12) and
Touch (L12)** -- both peak at L12, outside the L8-L11 band. This matches
`outputs/osie_layer_profile_synthesis/report.md`'s own text ("L12単独で
ピークとなる属性は少数(Sound, Touchのみ)") and is now independently
re-derived from `probe_summary.csv` directly in this audit.

**Metric used per attribute**: AUPRC (`average_precision_score`), AUROC,
and balanced accuracy are all computed per attribute per layer (see
`probe_fold_scores.csv`), but AUPRC is the metric plotted in Figure 2 and
referenced by "best_auprc"/"best_layer" above.

**paper_label_ja / paper_label_en**: proposed in `appendix_semantic_
labels.csv`'s `paper_label_ja`/`paper_label_en`/`definition` columns.
These are THIS AUDIT'S proposed translations/definitions (the `definition`
text is adapted from `outputs/osie_text_alignment_pilot/prompt_bank.json`'s
`base_phrases`, the most precise English description of each attribute
found anywhere in this repository) -- they are not themselves drawn from
a canonical "paper label" file, since none exists. Treat these as a
recommended starting point for manuscript wording, not as an established
fact to cite.

---

# Low-level Probe Targets

**Output**: `appendix_low_level_features.csv` (8 rows, UTF-8 with BOM).

**Source files**:
- `outputs/osie_lowlevel_probe_all12_full700/lowlevel_summary.csv`
  (`metric=="r2"`, `mean`, `n_valid_objects` columns) -- peak layer/value,
  valid-object count.
- `outputs/osie_lowlevel_targets_full700/lowlevel_object_targets.csv`
  (5,551 rows, `valid`/`skip_reason` columns, per-feature raw values) --
  exact excluded count and empirical value ranges.
- `lib/osie_lowlevel_features.py` (lines 26-88) -- formulas.

**Exact counts and peaks**:

| internal_name | valid_count | excluded_count | best_layer | best_r2 | empirical range |
|---|---:|---:|---:|---:|---|
| mean_luminance | 5263 | 288 | 3 | 0.919191 | [0.0073, 0.9857] |
| mean_red | 5263 | 288 | 3 | 0.921835 | [0.0122, 0.9916] |
| mean_green | 5263 | 288 | 3 | 0.916545 | [0.0057, 0.9861] |
| mean_blue | 5263 | 288 | 3 | 0.916945 | [0.0086, 0.9871] |
| mean_saturation | 5263 | 288 | 3 | 0.873729 | [0.0083, 0.9818] |
| edge_strength | 5263 | 288 | 4 | 0.844376 | [0.0455, 2.0474] |
| fine_texture | 5263 | 288 | 2 | 0.764582 | [0.0000, 0.6485] |
| luminance_contrast | 5263 | 288 | 6 | 0.355719 | [0.0000, 0.4153] |

(Every feature has IDENTICAL valid/excluded counts: 5,263/288, because the
exclusion is applied at the OBJECT level -- once a mask is empty after the
224×224 crop, ALL 8 targets for that object are unavailable simultaneously,
not feature-by-feature.)

**Item 11 (7-of-8-peak-in-L1-L4 consistency check)**: **Confirmed**,
exactly. Peaks in {L1,L2,L3,L4}: mean_luminance(L3), mean_red(L3),
mean_green(L3), mean_blue(L3), mean_saturation(L3), edge_strength(L4),
fine_texture(L2) = **7 features**. Exception: **luminance_contrast peaks
at L6**, outside L1-L4.

**Item 12 (288 excluded / 5,263 used consistency check)**: **Confirmed**,
exactly, and now traced to an explicit `skip_reason` value (see Critical
Finding 3) rather than only inferred from a count mismatch.

**Ridge regression target (dependent variable)**: the raw (un-standardized)
per-object feature value itself -- e.g. for `mean_luminance`, the target
`y` is the object-masked mean luminance value directly (see "Object-level
Representations" and "Probe Hyperparameters" below for standardization
details).

**Standardization**: input features (X, the pooled CLIP hidden states)
are standardized via `sklearn.preprocessing.StandardScaler` INSIDE a
`Pipeline`, fit on the training fold only. The target (y) is **NOT**
standardized anywhere in `lib/osie_lowlevel_probe.py` (`Ridge()` has no
built-in y-standardization option, and none is applied manually before
`.fit()`).

**Value units**: mean_luminance/red/green/blue/saturation are unitless
ratios in the theoretical range [0,1] (empirically slightly inside that
range, per the table above); edge_strength and fine_texture are in
image-intensity-gradient units (unbounded, empirically small positive
values); luminance_contrast is a standard deviation in the same [0,1]
luminance units (theoretical max 0.5 for an extreme bimodal 0/1 split,
empirical max 0.4153).

---

# Object-level Representations

(Also covered in `METHOD_AUDIT.md`'s "最優先事項" section; reconfirmed
here without modification.)

- **Extraction point**: output of each Transformer block, i.e., AFTER
  both the attention-residual addition (`x = x + attention(ln_1(x))`) and
  the MLP-residual addition (`x = x + mlp(ln_2(x))`) -- `clip.model.
  ResidualAttentionBlock.forward` (installed package, verified via
  `inspect.getsource` at audit time), called unmodified as `block(x)` in
  `lib/clip_vit_hidden.py:79-82`.
- **Before/after final LayerNorm & projection**: BEFORE. `ln_post`/`proj`
  are applied only once, at the very end of the official forward pass, to
  the pooled CLS output -- never to the per-layer hidden states saved for
  probing (`lib/clip_vit_hidden.py:50-52` docstring: "the RAW residual-
  stream output ... before ln_post/proj").
- **CLS token**: EXCLUDED. `hs[1:]` slices it off before reshaping to the
  patch grid (`scripts/extract_osie_probe_all12_full700_features.py:232`).
- **Patch dimension**: confirmed 38×50 (=1900 patches), asserted at
  runtime (`GRID_HW_EXPECTED = (38, 50)`,
  `scripts/extract_osie_probe_all12_full700_features.py:69`).
- **Object-mask × patch overlap weighting**: `lib/osie_text_alignment.py`
  `mask_to_patch_weights()` (lines 40-55) computes each patch's weight as
  its **fractional area coverage** by the (zero-padded) object mask --
  i.e., `weight[patch] = mean(mask pixels inside that 16×16 patch)`, a
  continuous value in [0,1], NOT binary inclusion.
- **Weighted-average formula** (matches `weighted_pool_tokens()`,
  `lib/osie_text_alignment.py:62-78`, exactly):

  ```
  For object o with per-patch weights w_1..w_1900 (w_k in [0,1], the
  fractional area of patch k covered by object o's mask) and per-patch
  hidden-state vectors t_1..t_1900 (each in R^768, from one Transformer
  block's output):

      feature(o) = ( sum_{k=1}^{1900} w_k * t_k ) / ( sum_{k=1}^{1900} w_k )

  If sum(w_k) < 1e-6, feature(o) is undefined (object skipped, not
  zero-filled).
  ```

- **Object dimension**: confirmed 768 (CLIP ViT-B/16's embed_dim), saved
  as `raw_L{layer}` arrays of shape `(n_objects, 768)` in
  `object_features_all12.npz`.
- **Shared between both probes**: CONFIRMED -- both
  `scripts/probe_osie_all12_full700.py:60` and `scripts/probe_osie_
  lowlevel_all12_full700.py:54` load the identical path
  `outputs/osie_probe_all12_full700/object_features_all12.npz`.

---

# Cross-validation

| Item | Semantic probe | Low-level probe |
|---|---|---|
| Splitter | `StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)` (`lib/osie_text_alignment_probe.py:160`) | `GroupKFold(n_splits)` (`lib/osie_lowlevel_probe.py:132`) |
| n_splits | `min(5, n_pos_img, n_neg_img, n_total_img)` via `safe_n_splits`, default `max_splits=5` -- **5 for every attribute in this run** (confirmed: all `probe_summary.csv` rows show `n_folds=5`) | `min(5, n_groups)` -- **5 for every target** (confirmed: `n_folds=5` in `lowlevel_summary.csv`) |
| Group unit | `image_id` | `image_id` |
| Cross-fold leakage check | Explicit runtime `RuntimeError` if any `image_id` appears in both train and test of a fold (`lib/osie_text_alignment_probe.py:163-164`) | Same explicit check (`lib/osie_lowlevel_probe.py:135-136`) |
| Shuffle | Yes (`shuffle=True`) | No shuffle parameter (`GroupKFold` has none; deterministic by group order) |
| Stratification | Yes, by the binary label (`StratifiedGroupKFold`) | No (regression; `GroupKFold` is not label-aware) |
| Random seed | 42 (`set_all_seeds(seed)` + `random_state=seed` passed to the splitter) | 42 (`set_all_seeds(seed)`; `GroupKFold` itself takes no seed) |
| Held-out validation set (separate from test) | **Not applicable** -- no separate validation split exists; each fold's held-out portion is used directly as the test set for that fold (standard k-fold, no nested val split) | Same -- not applicable |
| StandardScaler fit scope | Train fold only (`Pipeline.fit(X[tr], y[tr])`) | Train fold only (same pattern) |
| Target standardization | N/A (classification) | NOT standardized (Ridge's `y` used raw) |
| Per-fold image/object counts (example: `Text` attribute) | 139-142 images/fold, 1069-1139 objects/fold (this audit's own recount of `fold_assignments.csv`) | 140 images/fold (every target, e.g. `mean_luminance`), 1052-1053 objects/fold (this audit's own recount) |
| Missing-value handling | Objects with `sum(patch weights) < 1e-6` are skipped upstream at feature-extraction time (0 such cases occurred: `n_objects_skipped_zero_weight: 0` in `outputs/osie_probe_all12_full700/config.json`) | Objects with `mask_224.sum() < 1` (empty mask after CenterCrop) are excluded upstream at target-extraction time: 288 objects, `skip_reason=="empty_mask_after_transform"` |
| CenterCrop-based exclusion applies to | NOT applied (semantic labels come from `attrs.mat`, no image-processing crop involved) | Applied (see above); this exclusion propagates into the CV fold assignments (`fold_assignments.csv` only contains the 5,263 valid objects) |

---

# Probe Hyperparameters

## Ridge (low-level probe)

| Parameter | Value | Explicit or sklearn default? |
|---|---|---|
| Library | `scikit-learn` 1.7.2 | -- |
| alpha | 1.0 | **Explicit** (`Ridge(alpha=alpha)`, `alpha=1.0` is `lib/osie_lowlevel_probe.py`'s own `run_ridge_probe(..., alpha: float = 1.0, ...)` default, never overridden by any caller -- `scripts/probe_osie_lowlevel_all12_full700.py` does not pass a different `alpha`) |
| fit_intercept | True | **使用したscikit-learnの既定値に従った**(明示的に設定していない -- `Ridge(alpha=alpha)` の呼び出しに `fit_intercept` は渡されていない) |
| solver | `"auto"` | **使用したscikit-learnの既定値に従った**(明示的に設定していない)。`"auto"`が実際にどのソルバーへ解決されるかはデータ形状・条件に依存し、本監査では特定していない(Unverified) |
| tol | 1e-4 | **使用したscikit-learnの既定値に従った**(明示的に設定していない) |
| max_iter | `None` (→ solver-dependent internal default) | **使用したscikit-learnの既定値に従った** |
| positive | False | **使用したscikit-learnの既定値に従った** |
| random_state | `None` (Ridge itself has no seed effect for the default "auto"/closed-form solvers; overall reproducibility comes from `set_all_seeds(42)` fixing `random`/`numpy` global state before the CV loop, and `GroupKFold`'s deterministic group ordering) | -- |
| Hyperparameter search | **None** -- no `GridSearchCV`/`RandomizedSearchCV`/`RidgeCV` anywhere in `lib/osie_lowlevel_probe.py` or its callers; alpha=1.0 is a fixed value | -- |
| R² unit / aggregation | Computed per fold on that fold's held-out predictions (`sklearn.metrics.r2_score`), then mean±std(ddof=1, from `.std()` default in `lib/osie_lowlevel_probe.py`, confirmed no `ddof=0` override) across the 5 folds, saved to `lowlevel_summary.csv` | -- |

## LogisticRegression (semantic-attribute probe)

| Parameter | Value | Explicit or sklearn default? |
|---|---|---|
| Library | `scikit-learn` 1.7.2 | -- |
| penalty | `"l2"` | **Explicit** (`lib/osie_text_alignment_probe.py:44`) |
| C | 1.0 | **Explicit** |
| solver | `"lbfgs"` | **使用したscikit-learnの既定値に従った**(明示的に設定していない。sklearn 1.7.2でのLogisticRegressionの既定solverが"lbfgs") |
| class_weight | `"balanced"` | **Explicit** |
| max_iter | 5000 | **Explicit** (raised well above sklearn's own default of 100) |
| tol | 1e-4 | **使用したscikit-learnの既定値に従った** |
| fit_intercept | True | **使用したscikit-learnの既定値に従った** |
| random_state | `seed` (=42) | **Explicit** |
| multi_class | (deprecated/unused parameter in sklearn 1.7.2; binary classification throughout, so this parameter is not meaningful here) | N/A |
| Convergence warnings | **Unverified** -- no warnings log, `n_iter_` record, or convergence-status file exists anywhere in `outputs/osie_probe_all12_full700/`; this audit did NOT re-run the probe to check (that would be new model training, explicitly out of scope) | -- |
| AUPRC computation | `sklearn.metrics.average_precision_score` (step-function average precision, NOT raw trapezoidal PR-curve integration) | -- |
| Fold aggregation | Per-fold AUPRC/AUROC/balanced-accuracy, then mean ± std(ddof=1, `lib/osie_text_alignment_probe.py:209`: `vals.std(ddof=1) if len(vals)>1 else 0.0`) across the 5 folds | -- |
| Hyperparameter search | **None** -- C=1.0 is fixed, no CV-based search anywhere | -- |

---

# Gaze-alignment Metrics

(Fully detailed formulas already established in `METHOD_AUDIT.md` sections
F/G and `appendix_fact_check.md` section B; reconfirmed here, not
recomputed, with one addition -- explicit confirmation of the
"images-per-layer-mean" aggregation step.)

- **Attention-map normalization before metrics**: native 38×50 raw
  attention → bilinear upsample (`F.interpolate(mode="bilinear",
  align_corners=False)`) to (600,800) → renormalized to sum=1 over the
  full image (`scripts/run_expA_clip_full700.py:169-172`). This
  normalized map is what NSS/AUC-Judd/sAUC then consume.
- **Fixation handling**: discrete, pixel-deduplicated points from the
  `points` binary fixmap (union of all healthy-group subjects'
  fixations for that image); NOT the Gaussian-blurred `heat_all` map
  (that is used only for visualization).
- **Out-of-bounds fixations**: mapped to NaN in `_sample()`, dropped via
  `np.nanmean`/boolean-mask filtering before use.
- **NSS standard deviation**: population std, `ddof=0` (`numpy`'s default,
  `s.std()` with no `ddof` argument, `metrics.py:40`).
- **Zero-variance map**: `if std < 1e-12: return 0.0` (`metrics.py:41-42`).
- **AUC-Judd threshold generation**: NOT an explicit threshold sweep --
  a rank-based (sort-and-cumulative-sum) trapezoidal AUC, mathematically
  equivalent to sweeping every unique score value as a threshold
  (`metrics.py:152-172`).
- **sAUC negative-fixation source**: pooled across ALL 700 images
  (`all_pool` in `load_fixation_data()`), **the evaluated image's own
  fixations are NOT excluded** from this pool (confirmed via the script's
  own docstring, `scripts/run_expA_clip_full700.py:8-9`).
- **sAUC splits**: 10 (`n_splits=10`, `metrics.py:96`, never overridden
  anywhere in the codebase -- confirmed via `grep -rn "sauc("` across all
  of `scripts/*.py` and `aggregation.py`).
- **Layer-wise aggregation**: per-(image,layer) NSS/AUC-Judd/sAUC values
  are averaged with a PLAIN arithmetic mean across the 700 images to
  produce `metrics_by_layer.csv`'s `*_mean` columns (population std,
  `ddof=0`, `np.std(vals)` with no ddof argument,
  `scripts/run_expA_clip_full700.py:427`).

---

# N-shaped Profile Index

(Internal code/CSV name: `m_contrast` / `M_contrast` -- retained
unchanged as an internal variable/column name per this task's own
instruction; NEVER proposed here as manuscript-facing terminology. The
manuscript-facing name used throughout this document is "N-shaped
profile index" / N字型指標, a neutral description of what the quantity
measures, not a literal translation of the internal name.)

- **Target layers**: early=[L3,L4,L5], middle=[L7,L8,L9], late=[L11,L12]
  (`scripts/stats_model_comparison_mcontrast.py:66-68`).
- **Formula** (`scripts/stats_model_comparison_mcontrast.py:244-249`):

  ```
  early(image)  = mean( metric[image, layer] for layer in {3,4,5} )
  middle(image) = mean( metric[image, layer] for layer in {7,8,9} )
  late(image)   = mean( metric[image, layer] for layer in {11,12} )

  N-shaped_profile_index(image) = (early(image) + late(image)) / 2 - middle(image)
  ```

  computed PER IMAGE first, then bootstrap-averaged across the 700
  images (percentile CI, seed=42, n_boot=10000).
- **CLIP ViT-B/16 value**: **+0.882576** (rounds to +0.883) -- **matches**
  the manuscript-assumed value +0.883 exactly to 3 decimal places.
- **95% CI**: **[+0.859033, +0.906465]** (displayed [+0.8590, +0.9065])
  -- **matches** the manuscript-assumed [+0.859, +0.907] to 3 decimal
  places.
- **Comparison models**: DeiT/SL = **−0.293392** (matches −0.293), DINO-S
  = **−0.217619** (matches −0.218), DINO-B = **−0.030277** (matches
  −0.030). All read from `outputs/model_comparison_statistics/m_contrast_
  summary.csv`, unrounded values as shown.
- **Bootstrap-based difference test**: paired-by-image, Wilcoxon
  signed-rank test on each model's per-image `m_contrast` vs. CLIP-B's,
  Holm-corrected within a 3-comparison family
  ("metric=NSS,measure=m_contrast (3 tests: DeiT/DINO-S/DINO-B vs
  CLIP-B)", `holm_family` column in `m_contrast_model_comparisons.csv`).
  Independently re-derived raw p-values in a prior pass of this audit
  chain (`results_fact_check_report.md`): ≈2.87e-116 (DeiT-SL, DINO-S),
  ≈2.97e-116 (DINO-B); Holm-adjusted ≈8.60e-116 -- consistent with
  "p<1e-100" and with `statistics_summary.md`'s `.2e`-formatted text (the
  CSV's own `p_holm` column stores `0.000000` due to a 6-decimal write
  format, a display/precision artifact, not evidence of a different true
  value -- see `results_fact_check_report.md` for the full explanation).
- **L4/L12 independence check**: the formula above uses layer BINS
  ([3,4,5], [7,8,9], [11,12]), never a direct L4-vs-L12 comparison, and
  no file in `outputs/model_comparison_statistics/` computes or reports
  an "L4 equals/comparable-to L12" statistic. **The N-shape index does
  not assume, require, or imply that L4 and L12 have similar values** --
  it only requires early+late to jointly exceed middle. (CLIP-B's actual
  NSS values, L4=1.388 vs. L12=0.654, differ by roughly 2x -- established
  in `results_fact_check_report.md`, not recomputed in this pass.)

---

# Bootstrap and Statistical Analysis

- **Resampling unit**: images (700), with replacement
  (`rng.randint(0, n, size=(n_boot, n))`,
  `scripts/stats_three_model_layerwise.py:248-250`; identical pattern in
  `scripts/paper_figures/common.py::make_boot_indices`).
- **Repetitions**: 10,000 (`N_BOOT=10000`), confirmed identical across
  every stats/figure script checked in this audit chain.
- **Seed**: 42, confirmed identical across every stats/figure script
  checked.
- **CI type**: percentile bootstrap (`np.percentile(boot_means,
  [2.5,97.5])`) -- NOT basic-bootstrap, NOT BCa, anywhere in this
  codebase (no `scipy.stats.bootstrap(method="BCa")` or equivalent
  found).
- **Within-image dependency structure**: for the gaze-alignment/N-shape
  bootstrap, resampling is at the IMAGE level -- an image's own multiple
  fixations (already pooled into one NSS/AUC-Judd/sAUC scalar per image
  BEFORE bootstrapping) are never separately resampled, so the
  within-image fixation dependency is fully preserved (the bootstrap
  never breaks apart a single image's already-aggregated statistic). For
  the object-level probe bootstraps (`image_level_bootstrap_auprc_diff`/
  `_r2_diff`), resampling is likewise at the image level and ALL of an
  image's objects move together as one resampled unit (`rows_by_image`
  dict grouping in `lib/osie_text_alignment_probe.py:288`), preserving
  within-image object dependency.
- **Same bootstrap samples used in figures and statistical tests?**
  **Same METHOD and SEED, but NOT a literally shared/cached array.** Each
  script (`scripts/paper_figures/common.py` for figures;
  `scripts/stats_three_model_layerwise.py` / `scripts/stats_model_
  comparison_mcontrast.py` for formal statistics) independently calls
  `np.random.RandomState(42).randint(0, n, size=(10000, n))`. Because
  `n=700` and the seed/algorithm are identical, this produces a
  BIT-IDENTICAL index array in both places (verified: this construction
  is deterministic given `(seed, n, n_boot)` -- no additional entropy
  source), but the two code paths do not read from one shared file --
  each recomputes the array independently at run time.
- **Mean computation**: plain arithmetic mean of the resampled values per
  bootstrap replicate, then the CI is the 2.5th/97.5th percentile of
  those 10,000 replicate means; the reported point estimate is the plain
  mean of the ORIGINAL (non-resampled) data, not the mean of the
  bootstrap distribution.
- **p-value computation**: two distinct methods depending on the
  analysis: (a) Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) for
  paired model-vs-model / layer-vs-layer comparisons; (b) sign-flip
  permutation Monte Carlo p-value (`lib/clip_attention_b2.py:305-322`,
  `p = (1 + n_extreme) / (1 + n_reps)`, "+1 correction," never exactly
  0) for the B-2 spatial-analysis family.
- **Multiple-comparison correction**: Holm (model-comparison statistics)
  vs. BH-FDR (B-2 spatial analysis) -- see `appendix_fact_check.md`
  section B5 for the full breakdown (not repeated here).

---

# Spatial Attention Metrics

All formulas, normalization, aggregation methods, CIs, and BH-FDR
targets were already fully detailed in `appendix_fact_check.md` section
D and `results_fact_check_report.md` section 5; this pass **independently
re-confirms the canonical values from source CSVs once more** (not merely
re-citing the prior pass) and reports the result:

| Metric | L4 | L8 | L12 | Re-derivation this pass |
|---|---:|---:|---:|---|
| 50%-mass patch count | 563 (563.2214) | 121 (121.4986) | 68 (67.6343) | `summary.json` `descriptive_per_layer.mass50_n_patches.mean`, cross-checked against `per_image_metrics.csv`'s plain mean in the prior audit pass -- values unchanged, re-read this pass, identical |
| Foreground enrichment | 1.60 (1.5986) | 0.99 (0.9949) | 1.57 (1.5719) | Same file, unchanged |
| Same-image Pearson (Fisher-z mean) | L4-L8=−0.048 (−0.0482) | L8-L12=0.958 (0.9584) | L4-L12=0.015 (0.0146) | Same file, unchanged |

**No discrepancy found this pass.** All 3 canonical value sets match the
manuscript-assumed values in the task brief exactly (to the stated
rounding precision), confirmed against the SAME canonical file
(`outputs/clip_attention_b2_distribution/summary.json`) used in the two
prior audit passes -- no different file, no different rounding order, no
new explanation needed this time (the 562/122/68-vs-563/121/68 rounding
explanation from the prior pass stands unchanged and is not repeated in
full here; see `results_fact_check_report.md` section 5 or
`appendix_fact_check.md` section 6 for the exact double-rounding
arithmetic).

Entropy/sparsity/top-10%-mass/support-area (80%-mass) formulas, values,
and paired-test results (e.g. Hoyer sparsity L8-L4 dz=7.951) are
unchanged from `appendix_fact_check.md` section D2 -- not re-derived in
this pass, only re-cited, since no new source data emerged.

---

# Models and Preprocessing

Unchanged from `appendix_fact_check.md` section 2 (A1-A5) -- all items
there were already Confirmed with file/line evidence and are not
re-verified in this pass, EXCEPT the following, checked fresh:

- **DINO/DeiT position-embedding interpolation code path**: still
  **Unverified** -- this pass again did not open
  `lib/vision_transformer.py`'s `interpolate_pos_encoding` method body to
  confirm it uses the same bicubic/CLS-preserving convention as CLIP's
  `lib/clip_vit.py::interpolate_patch_pos_embed`. Carried over unchanged
  from the prior audit as an open item (see "Unverified Items" below).
- **OSIE-coordinate → model-input-coordinate transform**: reconfirmed
  unchanged -- bottom/right zero-pad only (`vit_extractor.py::_pad_to_
  patch`, shared across all 4 models' OSIE-input loaders), no resize, no
  crop, for the gaze-alignment and probe-X pathways; a SEPARATE
  224×224-Resize+CenterCrop pathway exists ONLY for the low-level probe's
  target (Y) values (`scripts/extract_osie_lowlevel_targets.py`).
- **Out-of-frame coordinates and masks**: fixation coordinates are
  clipped to `[0, dim-1]` at fixmap-generation time
  (`scripts/generate_fixmaps.py::coords_to_pixels`); object masks are
  native-resolution boolean arrays with no separate "out of frame"
  handling needed (OSIE object masks are defined within the native image
  bounds by construction).

---

# Library Versions

**Source**: `environment_after_clip.txt` (pip-freeze-style snapshot,
repo root) and this audit's own direct interrogation of the currently
installed environment (`python -c "import X; print(X.__version__)"`,
executed at audit time, 2026-09-02). Both agree exactly.

| Library | Version | Source |
|---|---|---|
| Python | 3.11.6 | `python --version` (this audit) |
| PyTorch | 2.11.0+cu126 | `environment_after_clip.txt`: `torch==2.11.0+cu126` |
| torchvision | 0.26.0+cu126 | `environment_after_clip.txt`: `torchvision==0.26.0+cu126` |
| torchaudio | 2.11.0+cu126 | `environment_after_clip.txt` (not used by this pipeline as far as this audit traced, listed for completeness) |
| scikit-learn | 1.7.2 | `environment_after_clip.txt`: `scikit-learn==1.7.2`; confirmed via `sklearn.__version__` |
| NumPy | 1.26.4 | `environment_after_clip.txt`: `numpy==1.26.4` |
| SciPy | 1.16.3 | `environment_after_clip.txt`: `scipy==1.16.3` |
| OpenCV (cv2) | **Not installed** | `python -c "import cv2"` → `ModuleNotFoundError`; consistent with `lib/osie_lowlevel_features.py`'s own docstring ("cv2/scikit-image are not installed in this environment") |
| timm | **Not installed** | `python -c "import timm"` → `ModuleNotFoundError` -- this pipeline's DINO/DeiT/CLIP loading does NOT use `timm` anywhere (custom `lib/vision_transformer.py` and the `clip` package only) |
| CLIP implementation | OpenAI's official `clip` package, installed from `git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | `environment_after_clip.txt`; commit hash matches the one independently cited in `lib/clip_vit.py`'s own docstring ("Reference (installed package, commit d05afc4)") |
| torchmetrics | 1.8.2 | `environment_after_clip.txt` (not traced as used in the core probe/metric pipeline; listed for completeness) |

`environment_before_clip.txt` (a snapshot from before CLIP-related work
began) shows the identical `torch`/`torchvision`/`scikit-learn`/`numpy`/
`scipy` versions as `environment_after_clip.txt`, confirming these core
libraries were NOT changed by the CLIP-integration work -- only the
`clip` package itself was newly added between the two snapshots.

---

# Manuscript Contradictions

No manuscript exists in this repository (see header). This section
covers discrepancies between THIS TASK'S OWN STATED ASSUMPTIONS and the
verified implementation, carried forward or newly checked in this pass:

| 重要度 | 現在の原稿表現(想定) | 実装・成果物で確認された事実 | 根拠 | 安全な日本語修正文案 | 安全な英語修正文案 |
|---|---|---|---|---|---|
| Minor (再確認、変更なし) | 「Pearson相関 r=−0.84」 | `curve_correlations.csv`のPearson r=−0.7824。−0.84はSpearman ρ=−0.8392 | `outputs/osie_layer_profile_synthesis/curve_correlations.csv`(既報告: `results_fact_check_report.md` item 1, `appendix_fact_check.md` C6行) | 「低次視覚特徴probeと意味属性probeの層別平均曲線は、Spearman順位相関ρ=−0.84(Pearson r=−0.78)の負の相関を示した」 | "The layer-wise mean curves of the low-level and semantic probes were negatively correlated (Spearman ρ=−0.84; Pearson r=−0.78)." |
| Minor (再確認、変更なし) | 50% mass patch数として「562/122/68」の使用 | 正準値は563/121/68(単純平均を一度だけ丸め)。562/122/68は面積比の二重丸め | `outputs/clip_attention_b2_distribution/summary.json`(既報告: `results_fact_check_report.md` section 5) | 「50%質量に必要なpatch数は、L4=563、L8=121、L12=68であった」 | "The number of patches required to cover 50% of the attention mass was 563 at L4, 121 at L8, and 68 at L12." |
| Not applicable (今回の監査対象外の再検証だが記録) | fit_intercept/solver/tol等が「明示的に設定された」という記述の可能性 | Ridge/LogisticRegressionともfit_intercept, tol, (Ridgeの)solverは明示指定されておらず、sklearn 1.7.2の既定値に従う | 上記「Probe Hyperparameters」節 | 「これらのパラメータはscikit-learnの既定値(fit_intercept=True, tol=1e-4等)に従った」 | "These parameters followed scikit-learn's default values (e.g., fit_intercept=True, tol=1e-4)." |

---

# Unverified Items

1. **Ridge回帰の`solver="auto"`が実際にどのソルバーへ解決されるか** --
   sklearnのドキュメント上、`"auto"`はデータの疎密・サンプル数・特徴数等
   に応じて内部的に選択される。本監査はこの内部解決結果を実行時ログや
   保存済み出力から特定できなかった(新たなモデル学習によるログ取得は本
   監査の範囲外)。
2. **DINO/DeiT側の位置埋め込み補間の実装詳細** -- `lib/vision_
   transformer.py`の`interpolate_pos_encoding`メソッド本体を今回も未確認
   (前回監査から継続)。
3. **LogisticRegressionの収束警告の有無** -- 保存済み出力に収束状況の記録
   がなく、新規学習による確認は本監査の範囲外のため未確認。
4. **各objectの`positive_attributes`列がOSIEの`attrs.mat`から機械的に
   派生した以外の追加フィルタリング(人手レビュー等)を経ているかどうか** --
   `lib/osie_text_alignment.py::split_positive_negative_attributes`が
   `attrs.mat`の生の0/1値をそのまま变換していることはコード上確認できたが、
   `attrs.mat`自体の作成過程(OSIE原論文側のアノテーション手順)はこの
   リポジトリの管轄外であり、監査対象外。

---

# Recommended Appendix Tables

(Tables A1-A6 follow below, ready to paste. Additional recommended tables
from the prior audit pass -- e.g. spatial-concentration-metric tables --
are unchanged and listed in `appendix_fact_check.md` section 9, not
repeated here.)

---

# Evidence Index

| 項目 | ファイル | 関数/列 | 備考 |
|---|---|---|---|
| 意味属性ラベル分布 | `outputs/osie_probe_all12_full700/object_features_metadata.csv` | `positive_attributes`, `negative_attributes` | 本監査で全12属性を完全再集計 |
| 意味属性AUPRC | `outputs/osie_probe_all12_full700/probe_summary.csv` | `metric=="auprc"`, `condition=="all_objects"`, `mean` | -- |
| 属性選定根拠 | `outputs/osie_text_alignment_pilot/attribute_mapping.json` | `attrs_mat_order_as_stored` | OSIE `attrs.mat`の全属性であることを確認 |
| 低次特徴R² | `outputs/osie_lowlevel_probe_all12_full700/lowlevel_summary.csv` | `metric=="r2"`, `mean`, `n_valid_objects` | -- |
| 低次特徴除外理由 | `outputs/osie_lowlevel_targets_full700/lowlevel_object_targets.csv` | `valid`, `skip_reason` | `skip_reason=="empty_mask_after_transform"`が288件 |
| 低次特徴定義 | `lib/osie_lowlevel_features.py` | 26-88行 | -- |
| 加重平均実装 | `lib/osie_text_alignment.py` | `mask_to_patch_weights`(40-55), `weighted_pool_tokens`(62-78) | -- |
| 隠れ状態抽出 | `lib/clip_vit_hidden.py` | `visual_forward_with_hidden_states`(27-89) | -- |
| CV分割 | `lib/osie_text_alignment_probe.py`(160-164), `lib/osie_lowlevel_probe.py`(132-136) | `StratifiedGroupKFold`/`GroupKFold` | -- |
| foldごとの画像数 | `outputs/osie_probe_all12_full700/fold_assignments.csv`, `outputs/osie_lowlevel_probe_all12_full700/fold_assignments.csv` | `fold`, `image_id` | 本監査で再集計 |
| Ridge設定 | `lib/osie_lowlevel_probe.py` | `make_ridge_pipeline`(39-40) | -- |
| LogisticRegression設定 | `lib/osie_text_alignment_probe.py` | `make_pipeline`(41-46) | -- |
| permutationベースライン | `lib/osie_text_alignment_probe.py`(216-238), `scripts/probe_osie_all12_full700.py` | Stage A/B, 20/100反復 | -- |
| NSS/AUC-Judd/sAUC実装 | `metrics.py` | `nss`(24-45), `auc_judd`(52-85), `sauc`(92-132) | -- |
| N字型指標 | `scripts/stats_model_comparison_mcontrast.py` | 66-68(層ビン), 244-249(数式) | -- |
| N字型指標の値 | `outputs/model_comparison_statistics/m_contrast_summary.csv`, `m_contrast_model_comparisons.csv` | -- | -- |
| bootstrap実装 | `scripts/stats_three_model_layerwise.py`(248-261), `scripts/paper_figures/common.py` | `make_boot_indices`, `bootstrap_mean_ci` | -- |
| B-2空間解析 | `lib/clip_attention_b2.py` | 各種関数(36-472行) | -- |
| B-2空間解析の値 | `outputs/clip_attention_b2_distribution/summary.json`, `per_image_metrics.csv`, `layer_pair_similarity.csv`, `paired_layer_comparisons.csv` | -- | -- |
| モデル読み込み(CLIP) | `clip_extractor.py`(30-73) | `load_clip_vit_b16`, `assert_vit_b16_shape` | -- |
| モデル読み込み(DINO-S) | `vit_extractor.py`(130-165) | `_load_model` | -- |
| モデル読み込み(DINO-B) | `dino_vitb16_extractor.py`(31-116) | `load_dino_vitb16`, `assert_vitb16_shape` | -- |
| モデル読み込み(DeiT/SL) | `sl_extractor.py`(1-140) | docstring(state_dict直接検査記録), `eval()`呼び出し | -- |
| ライブラリバージョン | `environment_after_clip.txt`, `environment_before_clip.txt` | -- | 本監査でインストール環境と突合 |
| CLIP text encoder不使用 | (grep結果, ファイルなし) | -- | `scripts/probe_osie_all12_full700.py`等でtext encoder呼び出しゼロ件 |

---

## Table A1: モデルと前処理

| モデル | 実装元 | チェックポイント | patch size | 埋め込み次元 | block数 | head数 | 位置埋め込み補間 |
|---|---|---|---:|---:|---:|---:|---|
| CLIP ViT-B/16 | OpenAI `clip`パッケージ(`git+https://github.com/openai/CLIP.git@d05afc4`) | 公式事前学習済み重み(パッケージ内蔵ダウンロード) | 16 | 768 | 12 | 12 | bicubic補間(CLS位置埋め込みは補間せず保持); `lib/clip_vit.py:191-226` |
| DINO ViT-S/16 | `lib/vision_transformer.py::vit_small`(本リポジトリ独自実装) | カスタム学習チェックポイント(Yamamoto et al. 2025の枠組み、`D:\gaze\...\dino\01\12layers\checkpoint.pth`) | 16 | 384 | 12 | 6 | Unverified(実装箇所未確認、`lib/vision_transformer.py::interpolate_pos_encoding`と推定) |
| DINO ViT-B/16 | `lib/vision_transformer.py::vit_base`(本リポジトリで新規定義) | 公式Facebook Research事前学習済みbackbone(`torch.hub`経由ダウンロード、MD5確認済み) | 16 | 768 | 12 | 12 | Unverified(同上) |
| 教師ありViT-S/16(DeiT訓練コードベース) | `lib/vision_transformer.py::vit_small`(DINO-Sと同一アーキテクチャ) | カスタム学習チェックポイント(蒸留トークンなし、state_dict直接検査で確認済み) | 16 | 384 | 12 | 6 | Unverified(同上) |

全モデル: eval mode、`requires_grad=False`、`torch.inference_mode()`下で推論のみ実施。入力はOSIE原寸800×600をゼロパディング(下端・右端のみ)し38×50=1900 patchを得る(リサイズ・クロップなし)。

## Table A2: 意味属性12ラベルの定義・陽性数・陰性数

(`appendix_semantic_labels.csv`と同一内容。全5,551物体、除外0件。)

| internal_label | 日本語名(案) | 英語名(案) | positive_count | negative_count | best_layer | best_auprc |
|---|---|---|---:|---:|---:|---:|
| Text | 文字 | Text | 495 | 5056 | 8 | 0.9325 |
| Face | 顔 | Face | 1010 | 4541 | 10 | 0.9659 |
| Emotion | 感情表出 | Emotion (clear facial emotion) | 190 | 5361 | 9 | 0.5282 |
| Sound | 発音性 | Sound-producing | 89 | 5462 | 12 | 0.4099 |
| Smell | におい | Smell | 135 | 5416 | 11 | 0.3895 |
| Taste | 味覚(食べ物・飲み物) | Taste (food/drink) | 390 | 5161 | 9 | 0.7700 |
| Touch | 触感的特徴 | Distinctive tactile quality | 317 | 5234 | 12 | 0.4345 |
| Motion | 動き | Motion | 557 | 4994 | 8 | 0.5125 |
| Operability | 操作可能性 | Operability | 342 | 5209 | 11 | 0.5835 |
| Watchability | 鑑賞対象性 | Watchability | 1108 | 4443 | 8 | 0.8619 |
| Touched | 接触されている | Touched | 436 | 5115 | 9 | 0.6720 |
| Gazed | 注視されている | Gazed at | 252 | 5299 | 9 | 0.3903 |

## Table A3: 低次視覚特徴8種類の定義

(`appendix_low_level_features.csv`と同一内容。有効5,263物体/288物体除外、全特徴共通。)

| internal_name | 日本語名(案) | 計算式 | 値域(実測) | best_layer | best_r2 |
|---|---|---|---|---:|---:|
| mean_luminance | 平均輝度 | 0.2126R+0.7152G+0.0722B, マスク内平均 | [0.0073, 0.9857] | 3 | 0.9192 |
| mean_red | 平均赤(R) | Rチャンネル、マスク内平均 | [0.0122, 0.9916] | 3 | 0.9218 |
| mean_green | 平均緑(G) | Gチャンネル、マスク内平均 | [0.0057, 0.9861] | 3 | 0.9165 |
| mean_blue | 平均青(B) | Bチャンネル、マスク内平均 | [0.0086, 0.9871] | 3 | 0.9169 |
| mean_saturation | 平均彩度 | HSV S成分、マスク内平均 | [0.0083, 0.9818] | 3 | 0.8737 |
| edge_strength | エッジ強度 | Sobel勾配強度(全画像に適用後マスク内平均) | [0.0455, 2.0474] | 4 | 0.8444 |
| fine_texture | 微細テクスチャ | Laplacian分散(全画像に適用後マスク内) | [0.0000, 0.6485] | 2 | 0.7646 |
| luminance_contrast | 輝度コントラスト | 輝度のマスク内標準偏差 | [0.0000, 0.4153] | 6 | 0.3557 |

## Table A4: probeのハイパーパラメータ

| Probe | ライブラリ | 主要パラメータ(明示指定) | 既定値に従った項目 |
|---|---|---|---|
| 意味属性(LogisticRegression) | scikit-learn 1.7.2 | penalty="l2", C=1.0, class_weight="balanced", max_iter=5000, random_state=42 | solver(既定=lbfgs), tol(既定=1e-4), fit_intercept(既定=True) |
| 低次特徴(Ridge) | scikit-learn 1.7.2 | alpha=1.0 | fit_intercept(既定=True), solver(既定="auto", 実際の解決先はUnverified), tol(既定=1e-4), max_iter(既定=None) |

両probeともStandardScalerで入力Xのみtrain fold内標準化(sklearn Pipeline)。目的変数の標準化なし。ハイパーパラメータ探索は両probeとも未実施(固定値)。5-fold交差検証、画像ID単位分割、seed=42。

## Table A5: 視線一致指標と統計処理

| 指標 | 標準化・正規化 | 集計単位 | 主要な数値的特徴 |
|---|---|---|---|
| NSS | Attention map: 画像ごとにzスコア標準化(母標準偏差, ddof=0) | 画像単位→層ごとに700画像平均 | 分散が実質0の場合は0扱い |
| AUC-Judd | 呼び出し側で正規化済み(合計1)のmapを使用、AUC自体はスケール不変 | 画像単位→層ごとに700画像平均 | 自作実装(sklearn不使用)、台形則 |
| sAUC | 同上 | 画像単位→層ごとに700画像平均 | 10分割、seed=0(内部固定)、自画像を負例プールから除外せず |
| N字型指標 | 上記3指標いずれかの画像単位・層別値から算出 | 画像単位で先に計算、700画像でbootstrap平均(seed=42, 10,000回) | CLIP-B: +0.883 [+0.859,+0.907]、Holm補正後 p<1e-100(vs 他3モデル) |

## Table A6: L4・L8・L12のAttention空間指標

| 指標 | L4 | L8 | L12 | 集計法 |
|---|---:|---:|---:|---|
| 50%質量patch数 | 563 | 121 | 68 | 画像単位→700画像平均(単純平均を一度だけ丸め) |
| Foreground enrichment | 1.60 | 0.99 | 1.57 | 画像単位→700画像平均 |
| Hoyerスパース度 | 0.137 | 0.722 | 0.734 | 画像単位→700画像平均、L8-L4差はdz=7.95(有意, BH-FDR q<0.001) |
| 正規化エントロピー | 0.981 | 0.848 | 0.796 | 画像単位→700画像平均 |
| 上位10%質量 | 0.227 | 0.559 | 0.692 | 画像単位→700画像平均 |
| 同一画像内Pearson相関(層間) | L4-L8=−0.048 | L8-L12=0.958 | L4-L12=0.015 | PearsonのみFisher-z平均、他は単純平均。全て700画像・seed=42・n_boot=10000のbootstrap CI付き |
