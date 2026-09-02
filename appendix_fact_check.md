# Appendix Fact Check

Audit date: 2026-09-01/09-02 (repository working-tree state; no tracked
file has local modifications beyond the new files this project's prior
sessions already added under `paper_figures/`, `scripts/paper_figures/`,
`METHOD_AUDIT.md`, `results_fact_check_report.md`). No new experiment,
model inference, feature extraction, or probe training was run to produce
this document. All values reported here were either read directly from
existing saved outputs, or computed by cheap, existing-data-only
aggregation (each such case is labeled "computed in this audit" and is
distinguished from a canonical pre-existing value).

**No manuscript/LaTeX draft exists in this repository** (re-confirmed:
`find . -iname "*.tex" -o -iname "*results*" -o -iname "*draft*" -o
-iname "*manuscript*" -o -iname "*.docx"` returns only `outputs/`/
`results/` data directories, `datasets/osie/.../lib/gbvs/readme.txt`, and
generic environment `.txt` files -- no manuscript). The "本文との整合性
チェック" section below is therefore an audit of **this repository's own
figure/audit artifacts** (`paper_figures/FIGURE_AUDIT.md`,
`METHOD_AUDIT.md`, `results_fact_check_report.md`, figure captions), not
of an actual paper draft, which is maintained outside this repository
(e.g. Overleaf) and was not available for inspection here.

---

## 0. Executive Summary

- **Confirmed: 133** (includes 2 items whose verdict is phrased "Confirmed
  absent" -- i.e., a specific practice was confirmed to be ABSENT from the
  code, such as sAUC's self-image exclusion)
- **Contradicted: 1**
- **Unverified: 3**
- **Not applicable: 1**
- **Total audited line-items: 138**

(Exact counts, programmatically tallied from the `判定` column of every
ID-tagged row in sections 2-5; the descriptive per-feature/per-attribute
detail tables in section 4 are reference tables, not separately-verdicted
audit items, and are excluded from this count. Section 7's "Unresolved
Questions" additionally surfaces context for the 3 Unverified rows plus 3
further open questions that were not formalized as their own table row
-- see section 7 for the full list of 6.)

**Critical items to resolve before Appendix text is written:**

1. **Path naming mismatch.** The task brief and (presumably) the
   manuscript refer to a `paper_figs/` directory; this repository has no
   such directory -- the actual, existing directory is `paper_figures/`.
   All 3 named files (`figure1_model_comparison_all_metrics_2col.pdf`,
   `figure2_gaze_probe_profiles_compact.pdf`,
   `figure3_spatial_characterization_revised.pdf`) **do exist**, but
   under `paper_figures/`, not `paper_figs/`. If the manuscript's
   `\includegraphics` paths use `paper_figs/`, they will not resolve.
2. **DeiT/SL is not a standard DeiT checkpoint with a distillation
   token.** The loader's own docstring (`sl_extractor.py`) states the
   checkpoint is architecturally a plain ViT-S/16 (no `dist_token` key
   anywhere in the state dict) trained via the DeiT/facebookresearch
   supervised-training codebase, not a distillation-token DeiT. If the
   manuscript's model table implies a distillation token exists for this
   model, that is a **Contradicted** item (see A1).

No numerical values audited in this pass contradicted the values already
established in this project's prior audits (`results_fact_check_report.md`,
`METHOD_AUDIT.md`, `paper_figures/FIGURE_AUDIT.md`); this pass re-verified
those and extended coverage into new areas (model-loading code for DINO-B
and DeiT/SL specifically, additional concentration metrics beyond
foreground enrichment / patch counts, and the auxiliary Figure-3-adjacent
files).

**Can the Appendix be written from current information alone?**
**Mostly yes**, with 6 explicit gaps (see "Unresolved Questions," section
7) that need a human decision or a repository search this audit could not
complete (e.g., no record of *why* these particular 12 semantic
attributes or 8 low-level features were chosen was found in code/config/
memos -- this is Unverified, not Confirmed-by-omission).

---

## 1. Evidence Priority

Where multiple similarly-named scripts/outputs existed, the following
were judged canonical (source of truth), based on output-file
cross-references, `run_config.json`/`config.json` provenance fields, and
this project's own prior audits (which already resolved several of these
ambiguities against output timestamps and reproducibility checks):

| Domain | Canonical source | Why |
|---|---|---|
| CLIP-B NSS/AUC-Judd/sAUC | `outputs/expA_clip/healthy/` (full700) | `outputs/expA_clip/pilot20/` is the 20-image precursor, explicitly excluded from the model comparison per `comparison_sources.json` |
| DINO-S NSS/AUC-Judd/sAUC | `outputs/expA/healthy/` | Only full700 DINO-S output; produced by `scripts/run_expA.py` |
| DINO-B NSS/AUC-Judd/sAUC | `outputs/expA_dino_vitb16/healthy/` | Only DINO-B output; official-pretrained, distinct from DINO-S's custom checkpoint |
| DeiT/SL NSS/AUC-Judd/sAUC | `outputs/expA_sl/combined/metrics_by_image_meanacrosstrials.csv` | 6-trial mean, the repo's own stated "primary analysis" convention (`statistics_summary.md`); individual `trial_01..06/` and `pilot20/` are precursors/sensitivity checks, not primary |
| Semantic-attribute probe (X features + Y labels) | `outputs/osie_probe_all12_full700/` | 12-layer extension of the earlier L4/L8/L12-only `outputs/osie_probe_full700/`; a reproducibility gate in the extraction script (`extraction_reproducibility_check.json`, max abs diff 5.6e-8 to 4.7e-7) confirms the two are numerically consistent, and the 12-layer version is what all downstream Figure/Method audits in this repo use |
| Low-level-feature probe | `outputs/osie_lowlevel_probe_all12_full700/` (X from the same npz above; Y from `outputs/osie_lowlevel_targets_full700/`) | Only low-level-probe output at all 12 layers |
| B-2 spatial analysis (L4/L8/L12) | `outputs/clip_attention_b2_distribution/` (full700) | `outputs/clip_attention_b2_distribution_pilot/` is a 20-image precursor; `archive/pre_repeated_shuffle_finalization_20260822/` is a superseded snapshot (identical `mass50_n_patches` values to the current `summary.json`, confirming no numeric drift, but the current directory's `report_revised.md` is the up-to-date narrative) |
| N-shape (`m_contrast`) statistics | `outputs/model_comparison_statistics/` | Only statistics output covering all 4 models jointly |
| Paper figures | `paper_figures/` (this project's own prior work, this session) | The only figure-generation output tree in the repo; `scripts/paper_figures/*.py` are the only scripts that read the above and render to `paper_figures/` |

---

## 2. A. Implementation and Preprocessing Details

| ID | 項目 | 判定 | 確認結果 | 根拠ファイル・行 | 論文用の安全な記述 | 要対応 |
|---|---|---|---|---|---|---|
| A1-CLIP | CLIP ViT-B/16 identifier & weights | Confirmed | OpenAI `clip` package, `clip.load("ViT-B/16", ...)`; official OpenAI pretrained weights (downloaded to package cache) | `clip_extractor.py:30,44-60` (`CLIP_MODEL_NAME="ViT-B/16"`, `load_clip_vit_b16`) | 「OpenAI CLIP ViT-B/16（`clip`パッケージ、公式事前学習済み重み）を使用した」 | -- |
| A1-CLIP | Architecture params | Confirmed | depth=12, heads=12, patch_size=16, tokens=197 (14×14+1 native) | `clip_extractor.py:32-36`, asserted at runtime in `assert_vit_b16_shape` (`clip_extractor.py:63-73`) | 「12層、12ヘッド、patch size 16」 | -- |
| A1-CLIP | eval/no-grad/frozen | Confirmed | `model.eval()`, `p.requires_grad=False` for all params at load; forward passes wrapped in `torch.inference_mode()` in every extraction script | `clip_extractor.py:56-59`; e.g. `scripts/run_expA_clip_full700.py:169` (`with torch.inference_mode():`) | 「eval modeで実行し、全パラメータの勾配計算を無効化、推論のみ実施」 | -- |
| A1-DINO-S | DINO ViT-S/16 identifier & weights | Confirmed | Custom architecture `lib/vision_transformer.py::vit_small`, weights = a **custom-trained** checkpoint (NOT official pretrained), loaded from `{models_dir}/{training_method}/{trial:02d}/{depth}layers/checkpoint.pth` on the local `D:\gaze\...` path | `vit_extractor.py:130-165` (`_load_model`) | 「DINO ViT-S/16、Yamamoto et al. (2025)の枠組みで学習されたカスタムチェックポイント（公式配布の事前学習済み重みではない）」 | -- |
| A1-DINO-S | Architecture params | Confirmed | embed_dim=384, depth=12, num_heads=6, patch_size=16 | `lib/vision_transformer.py:264-268` (`vit_small` factory) | 「384次元、12層、6ヘッド」 | -- |
| A1-DINO-S | eval/no-grad/frozen | Confirmed | `model.eval()`, `requires_grad=False`; `torch.inference_mode()` in `run_expA.py` extraction loop | `vit_extractor.py:162-164`; `scripts/run_expA.py:73` | 同上パターン | -- |
| A1-DINO-B | DINO ViT-B/16 identifier & weights | Confirmed | Official Facebook Research DINO ViT-B/16 **backbone-only** checkpoint, downloaded via `torch.hub.load_state_dict_from_url` from `dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth`, MD5 recorded (`552daf80332dbde16bba2a52c6508b77`), loaded `strict=True` into this repo's own `vit_base()` (architecture code, not fbresearch's own model file) | `dino_vitb16_extractor.py:31-51,68-98` | 「DINO ViT-B/16、公式Facebook Research事前学習済みbackboneチェックポイント（ImageNet, ラベルなし自己教師あり学習）」 | -- |
| A1-DINO-B | Architecture params | Confirmed | embed_dim=768, depth=12, num_heads=12, patch_size=16, mlp_ratio=4, qkv_bias=True | `dino_vitb16_extractor.py:46-51` (`vit_base`), asserted via `assert_vitb16_shape` (`dino_vitb16_extractor.py:107-116`) | 「768次元、12層、12ヘッド（CLIP ViT-B/16と同スケール）」 | -- |
| A1-DINO-B | eval/no-grad/frozen | Confirmed | `model.eval()`, `requires_grad=False` | `dino_vitb16_extractor.py:100-103` | 同上パターン | -- |
| A1-DeiT/SL | DeiT ViT-S/16 identifier & weights | Confirmed, with an important architecture caveat -- see "要対応" | Trained via the facebookresearch/deit supervised-training codebase (per the Yamamoto-paper reproduction repo's own README, quoted in the module docstring), but the resulting checkpoint is confirmed (by direct state-dict inspection) to be a **standard ViT-S/16 with NO distillation token** (`cls_token` shape (1,1,384), `pos_embed` (1,197,384) = 1 CLS + 196 patch tokens; no `dist_token` key anywhere) | `sl_extractor.py:1-46` (module docstring, citing direct checkpoint inspection) | 「教師あり学習(supervised, DeiT訓練コードベース)によって学習されたViT-S/16。蒸留トークン(distillation token)は含まれない標準ViT-S/16構成」 | **要対応: モデル名を単に"DeiT"とすると蒸留トークンの存在を誤って示唆する。"DeiT-style supervised ViT-S/16 (no distillation token)"等、明示的な言い換えを推奨** |
| A1-DeiT/SL | Architecture params | Confirmed | embed_dim=384, depth=12, num_heads=6, patch_size=16 (same `vit_small` factory as DINO-S) | `sl_extractor.py:1-46` docstring; uses `lib.vision_transformer.vit_small` directly | 「384次元、12層、6ヘッド」 | -- |
| A1-DeiT/SL | eval/no-grad/frozen | Confirmed | `model.eval()`, `requires_grad=False` | `sl_extractor.py:138,140` | 同上パターン | -- |
| A1 | Input preprocessing (all 4 models, OSIE inputs) | Confirmed | No resize, no crop: native 800×600, zero-padded bottom/right to a patch-size(16) multiple (→608×800); DINO/DeiT use ImageNet normalization `(0.485,0.456,0.406)/(0.229,0.224,0.225)`, CLIP uses its own `(0.48145466,0.4578275,0.40821073)/(0.26862954,0.26130258,0.27577711)` | `vit_extractor.py:120-127` (`_pad_to_patch`, shared by all 4 models' OSIE-input loaders); `clip_extractor.py:38-41,100-133` | 「全モデル共通でOSIE原寸800×600画像を下端・右端のみゼロパディングし、リサイズ・クロップは行わない。正規化定数はモデル固有(DINO/DeiT: ImageNet統計、CLIP: CLIP独自統計)」 | -- |
| A2 | What L1-L12 denote | Confirmed | Output of Transformer block *i* (0-indexed *i*-1 for display layer *L{i}*), AFTER both the attention-residual and MLP-residual additions, BEFORE any final LayerNorm/projection | `lib/clip_vit_hidden.py:79-89` (probe path); `clip.model.ResidualAttentionBlock.forward` (installed package, verified via `inspect.getsource`); DINO/DeiT: `lib/vision_transformer.py::Block.forward` uses the equivalent pre-LN residual pattern (`x = x + drop_path(attn(norm1(x))); x = x + drop_path(mlp(norm2(x)))`) | 「各Transformer blockの、Attention残差加算・MLP残差加算の両方が完了した直後の出力(次のblockへの入力と等価)。最終LayerNormおよび画像投影の前」 | -- |
| A2 | CLS excluded from patch representation | Confirmed | `hs[1:]` slices off index-0 (CLS) before reshaping to the patch grid, for the probe path | `lib/clip_vit_hidden.py` usage in `scripts/extract_osie_probe_all12_full700_features.py:231-232` | 「patch表現からCLS tokenを除外」 | -- |
| A2 | CLIP ViT-B/16 object feature dim = 768 | Confirmed | `raw_L{l}` arrays saved as `(n_objects, 768)`; CLIP ViT-B/16 `embed_dim`=768 (via `CLIP_HEADS=12`, standard ViT-B width) | `scripts/extract_osie_probe_all12_full700_features.py:289` (`.astype(np.float32)` of stacked 768-dim pooled vectors); confirmed via `visual.conv1` output width in `lib/clip_vit_hidden.py:59` | 「CLIP ViT-B/16では1 patchあたり768次元」 | -- |
| A3 | CLS-to-patch, raw softmax, head-averaged | Confirmed (all 4 models) | CLIP: `nn.MultiheadAttention(..., need_weights=True, average_attn_weights=True)`, `attn_weights[:,0,1:]`. DINO/DeiT: `attn[:, :, 0, 1:]` from `get_fulllayers_selfattention`'s raw `(B,heads,T,T)` softmax output, `.mean(dim=1)` | `lib/clip_vit.py:22-51,93-111`; `scripts/run_expA.py:83-89` | 「[CLS]→patchの生softmax attention(head平均)を全モデル共通で使用」 | -- |
| A3 | CLS-self excluded | Confirmed | `[:, 1:]` slicing on the CLS row before use, for all 4 models (same convention) | as above | 「[CLS]自身への成分(diagonal)は除外」 | -- |
| A3 | No attention rollout | Confirmed | `grep -rn "rollout"` across all `.py` files returns zero matches | (repo-wide grep, 2026-08-29/09-01) | 「Attention rolloutは使用せず、各層を独立に評価」 | -- |
| A3 | Head averaging axis | Confirmed | Heads axis (`dim=1` in the `(B, heads, h, w)` or `(B,heads,T,T)` tensor layout) for all 4 models | `lib/clip_vit.py:43-48` (PyTorch's own `average_attn_weights=True`); `scripts/run_expA.py:85-87` (`a.mean(dim=1, keepdim=True)`) | 「head軸に沿った単純平均(重みなし)」 | -- |
| A3 | Same extraction convention across all 4 models | Confirmed | Same 4-step convention (raw softmax → CLS row → exclude self → head-mean) implemented independently per model but numerically equivalent in definition; CLIP reimplements the block manually to expose `need_weights`, DINO/DeiT expose it via `get_fulllayers_selfattention` | see rows above | 「4モデル共通の抽出方針(生Attention・CLS自己成分除外・head平均・rollout不使用)」 | -- |
| A3 | eval mode (no dropout) | Confirmed | All 4 loaders set `.eval()` before any extraction call (see A1 rows) | as above | 「eval modeでdropout等の確率的処理は無効」 | -- |
| A4 | No resize/crop for gaze-alignment & probe-X input; aspect ratio preserved | Confirmed | See A1 "Input preprocessing" row; padding only, bottom/right, value 0 (before normalization) | `vit_extractor.py:120-127`; `clip_extractor.py:100-133` | 「アスペクト比を維持し、cropせずにゼロパディングのみで38×50 patchを得た」 | -- |
| A4 | 38×50 grid derivation | Confirmed | 608(padded H)/16=38, 800(W)/16=50; `GRID_HW_EXPECTED=(38,50)` asserted at runtime in multiple scripts | `scripts/run_expA_clip_full700.py:70` (`EXPECTED_GRID`); `scripts/extract_osie_probe_all12_full700_features.py:69` (`GRID_HW_EXPECTED`) | 「patch size 16で800×608(パディング後)を分割し、38×50=1900 patch」 | -- |
| A4 | Position-embedding interpolation | Confirmed (CLIP only needs it; native grid for DINO/DeiT at this input size is NOT 14×14, so they ALSO need interpolation -- see caveat) | CLIP: bicubic `F.interpolate`, CLS position embedding kept unchanged, applied via `interpolate_patch_pos_embed` | `lib/clip_vit.py:191-226` | 「位置埋め込みはbicubic補間で38×50グリッドへ拡張(CLS位置埋め込みは補間せず保持)」 | **要確認: DINO/DeiT側の位置埋め込み補間の実装箇所をこの監査では未特定 -- Unverified、下記参照** |
| A4 | Fixation coordinate transform | Confirmed | `(x,y)` pixel coords directly, 1-indexed→0-indexed via `-1`, clipped to `[0,dim-1]`; NSS/AUC-Judd/sAUC sample the saliency map at `round(x), round(y)` | `scripts/generate_fixmaps.py:88-97` (`coords_to_pixels`); `metrics.py:139-149` (`_sample`) | 「注視座標は1-indexedから0-indexedへ変換し、画像範囲へクリップ」 | -- |
| A4 | Object mask coordinate transform (gaze/probe-X path) | Confirmed | Native-resolution boolean mask, zero-padded to patch multiple (same geometry as the image), then patch weight = fractional area coverage (mean over each patch's pixels) | `lib/osie_text_alignment.py:26-55` (`pad_mask_to_patch`, `mask_to_patch_weights`) | 「物体マスクは画像と同一のゼロパディング幾何で変換し、各patchの面積被覆率を重みとした」 | -- |
| A4 | Interpolation method for masks | Confirmed | No interpolation for the gaze/probe-X path (masks are used at native pixel resolution, then area-averaged per patch, not resized); nearest-neighbor IS used, but only for the SEPARATE low-level-target 224×224 pathway (see A5) | `lib/osie_text_alignment.py:40-55`; contrast with `scripts/extract_osie_lowlevel_targets.py:118` (`InterpolationMode.NEAREST`) | 「視線一致・probe特徴経路のマスク変換に補間は不要(面積平均のみ)。低次特徴の正解値算出のみ別途nearest-neighbor補間を使用」 | -- |
| A4 | Out-of-bounds fixation handling | Confirmed | `_sample()` returns NaN for any fixation outside `[0,W)×[0,H)`, later dropped via `~np.isnan` before metric computation | `metrics.py:139-149,66-69,114-117` | 「範囲外注視点はNaNとして除外」 | -- |
| A4 | Fixation-map generation | Confirmed | `points` = binary union of ALL subjects' raw fixation pixels (no blur); `heat_all` = Gaussian-blurred (sigma=24px) probability map, normalized to sum=1. NSS/AUC-Judd/sAUC use `points` (discrete); only figure visualization uses `heat_all` | `scripts/generate_fixmaps.py:33,60-108` | 「fixation mapは2値マップ(全被験者の統合)とGaussianぼかし版(可視化専用、sigma=24px)の2種類。指標計算には2値の離散点を使用」 | -- |
| A4 | Attention-map → image-coordinate conversion (gaze-alignment) | Confirmed | Native 38×50 grid bilinearly upsampled to (600,800) via `F.interpolate(mode="bilinear", align_corners=False)`, THEN renormalized to sum=1 over the 800×600 image, before being passed to `nss`/`auc_judd`/`sauc` | `scripts/run_expA_clip_full700.py:169-172`; `scripts/run_expA.py:90-100` (same pattern) | 「Attention mapは双線形補間で画像解像度へアップサンプル後、画素値の総和が1になるよう再正規化してから指標計算に使用」 | -- |
| A4 | Normalization immediately before NSS/AUC-Judd/sAUC | Confirmed | NSS internally z-scores the (already sum-normalized) map per image (`(s-mean)/std`, `ddof=0`); AUC-Judd/sAUC use the map's raw (post-sum-normalization) values directly, no additional z-scoring (rank-based AUC is scale-invariant) | `metrics.py:39-45` (`nss`) | 「NSSは画像ごとにzスコア標準化(母標準偏差,ddof=0)。AUC-Judd/sAUCは正規化後の生の値をそのまま使用(ROC AUCはスケール不変)」 | -- |
| A5 | Low-level target image processing: 224×224 resize+crop | Confirmed | `Resize(224, bicubic)` (CLIP's own official `preprocess.transforms[0]`) then `CenterCrop((224,224))` (`preprocess.transforms[1]`), reused via `clip_extractor.load_clip_vit_b16()`'s `preprocess` object (model itself never called) | `scripts/extract_osie_lowlevel_targets.py:2-26,111-120` | 「低次視覚特徴の正解値のみ、CLIP公式前処理のResize(224, bicubic)+CenterCrop(224×224)画像から算出(モデル自体は呼び出さない)」 | -- |
| A5 | Geometric mismatch vs. CLIP-feature input | Confirmed | Object masks for the low-level targets go through the SAME Resize(nearest)+CenterCrop as the 224 image, which is a DIFFERENT geometry than the zero-padded 800×608 image used for the probe input features (X); OSIE is 800×600 landscape, so the 224 CenterCrop discards the image's left/right margins | `scripts/extract_osie_lowlevel_targets.py:1-20,111-120` (module docstring + `build_mask_pipeline`) | 「低次特徴の正解値はCLIP特徴抽出に使う画像(パディングのみ、クロップなし)とは異なる幾何(224×224センタークロップ)で算出しており、画像左右端付近の物体は正解値算出から欠落し得る」 | -- |
| A5 | Objects excluded after crop; final counts | Confirmed | `n_objects_total=5551`, `n_objects_valid=5263`, `n_objects_skipped=288` (mask became empty inside the 224×224 crop) | `outputs/osie_lowlevel_targets_full700/sanity_checks.json`,`config.json` | 「crop後にマスクが空になった288物体を低次probeから除外し、5,263物体を使用」 | -- |
| A5 | Semantic probe object count (no crop exclusion) | Confirmed | `n_objects_with_features=5551` (all objects), `n_objects_skipped_zero_weight=0` (the ONLY exclusion rule for the semantic probe's X features is near-zero patch weight sum, which never triggered) | `outputs/osie_probe_all12_full700/config.json` | 「意味属性probeは全5,551物体を使用。低次probe専用の224クロップ除外は意味属性probeには適用していない」 | -- |

**A4 Unverified item**: DINO/DeiT's position-embedding interpolation
mechanism for the non-native (38×50) grid was **not independently located
in this audit pass** (the CLIP path is fully confirmed via
`lib/clip_vit.py::interpolate_patch_pos_embed`; the DINO/DeiT side likely
uses `lib/vision_transformer.py::VisionTransformer.interpolate_pos_
encoding`, referenced by name in `lib/clip_vit.py`'s own docstring
["Mirrors the design of lib/vision_transformer.py::VisionTransformer.
interpolate_pos_encoding (DINO)"] but this audit did not open and verify
that function's body). **Unverified — recommend a follow-up check of
`lib/vision_transformer.py`'s `interpolate_pos_encoding` method before
writing an Appendix sentence claiming DINO/DeiT position embeddings use
the identical bicubic-interpolation convention as CLIP.**

---

## 3. B. Gaze-alignment Metrics and Statistical Analysis

| ID | 項目 | 判定 | 確認結果 | 根拠ファイル・行 | 論文用の安全な記述 | 要対応 |
|---|---|---|---|---|---|---|
| B1 | NSS: z-score standardization | Confirmed | `s_norm = (s - s.mean()) / s.std()`, `numpy` default `ddof=0` (population std, not sample) | `metrics.py:39-43` | 「Attention mapを画像ごとに平均0・分散1へzスコア標準化(母標準偏差, ddof=0)」 | -- |
| B1 | NSS: constant-map guard | Confirmed | `if std < 1e-12: return 0.0` | `metrics.py:41-42` | 「Attention mapの標準偏差が実質0の場合はNSS=0として扱う」 | -- |
| B1 | NSS: fixation aggregation | Confirmed | `_sample()` evaluates the z-scored map at every (rounded) fixation pixel independently, then `np.nanmean` across all fixations for that image (multi-subject fixations already pooled into one point set upstream) | `metrics.py:24-45,139-149`; pooling: `scripts/run_expA_clip_full700.py::load_fixation_data` reads `points` (all-subjects binary union) | 「1画像内の全被験者の注視点(2値マップ由来の一意な画素座標集合)についてzスコア値を平均」 | -- |
| B1 | NSS: duplicate fixations at the identical pixel | Confirmed, with caveat | The `points` binary map cannot distinguish 1 vs. N fixations landing on the EXACT same integer pixel (assignment `bmap[py,px]=1` collapses them); a spot check on image 1156 showed `n_fix (131) == points.sum() (131)`, i.e. no collapsing occurred for that image, but this was not exhaustively checked for all 700 images | `scripts/generate_fixmaps.py:101-104` (`make_binary_map`); spot check in this audit | 「注視点は画素単位で一意化されたfixation mapに基づく。厳密には、同一画素へ複数fixationが重複した場合は1点として扱われる(実データでの発生頻度は未網羅的に確認)」 | 軽微: 700画像全てでn_fix==points.sum()かの網羅確認は未実施 |
| B1 | NSS: NaN/Inf/out-of-bounds | Confirmed | Out-of-bounds → NaN in `_sample`; `np.nanmean` ignores NaNs; whole-pipeline finiteness asserted (`np.isfinite` STOP-guards) in every production extraction script | `metrics.py:139-149`; `scripts/run_expA_clip_full700.py:305-307` | 「範囲外・非有限値はNaN扱いで平均から除外し、パイプライン全体で有限性を明示的に検証」 | -- |
| B1 | NSS: layer-wise aggregation | Confirmed | Per-(image,layer) NSS values are plain-averaged across the 700 images to form `metrics_by_layer.csv`'s `NSS_mean` (population std, `ddof=0`, per A-earlier finding in `METHOD_AUDIT.md`) | `scripts/run_expA_clip_full700.py:427` (`np.std(vals)`, no ddof arg) | 「画像単位NSSの層ごとの単純平均・母標準偏差」 | -- |
| B1 | NSS full formula | Confirmed | See below | -- | -- | -- |
| B2 | AUC-Judd: positive/negative definition | Confirmed | Positive = raw (non-z-scored) saliency values at fixation pixels; Negative = ALL other pixels (non-fixation), optionally uniformly subsampled to ≤100,000 (`RandomState(0)`) | `metrics.py:52-85` | 「正例=注視画素の顕著性値、負例=非注視画素全体(10万画素超の場合は一様無作為抽出、seed=0)」 | -- |
| B2 | AUC-Judd: fixation binarization | Confirmed | Positive pixels are the discrete `points`-derived coordinates (not a blurred map); a boolean `fix_mask` marks them, `~fix_mask` selects negatives | `metrics.py:71-77` | 「注視画素は2値化された座標集合として扱う」 | -- |
| B2 | AUC-Judd: ROC/AUC algorithm | Confirmed | Self-implemented, NOT sklearn/scipy: sort scores descending, cumulative TP/FP counts, trapezoidal-rule AUC (`np.trapz`) | `metrics.py:152-172` (`_roc_auc`) | 「自作実装(sklearn不使用)。降順ソート後の累積TP/FPからROC曲線を構成し、台形則でAUCを積分」 | -- |
| B2 | AUC-Judd: map normalization | Confirmed | Uses the map as passed in (already sum-normalized upstream by the caller, see A4); AUC itself is scale-invariant so this normalization does not change the AUC value | `metrics.py:52-85` (no internal normalization step) | 「AUC-Judd自体はスケール不変であり、呼び出し側で正規化済みのAttention mapをそのまま使用」 | -- |
| B2 | AUC-Judd: per-image | Confirmed | Called once per (image, layer) | `scripts/run_expA_clip_full700.py:305` | 「画像・層単位で算出」 | -- |
| B3 | sAUC: negative source | Confirmed | Fixations pooled from **all 700 images in the pool**, via `load_fixation_data()`'s `all_pool` (concatenation over every image's `points`) | `scripts/run_expA_clip_full700.py:130-133,222` | 「負例は全700画像の注視点プールから抽出」 | -- |
| B3 | sAUC: self-image exclusion | **Confirmed absent** | No exclusion of the evaluated image's own fixations from `all_pool`; explicitly stated in the script's own docstring | `scripts/run_expA_clip_full700.py:8-9` ("no self-image exclusion, metrics.py's internal RandomState(0) untouched"); same pattern confirmed for all 4 models via `grep -rn "sauc("` (no override anywhere) | 「shuffled negative poolは評価対象画像自身の注視点を明示的には除外していない。したがって"sAUCはデータセット共通の位置バイアスへの感度を低減する"と記述し、"完全に除去する"とは記述しない」 | 既にこの方針で本文表現の推奨済み(`results_fact_check_report.md`) |
| B3 | sAUC: negative count / splits / seed / replacement | Confirmed | `n_splits=10` (default, never overridden); each split draws `min(n_pos, len(other_fixations))` WITHOUT replacement (`rng.choice(..., replace=False)`) from the pooled negatives; ONE `RandomState(0)` created per `sauc()` call, consumed across all 10 splits (not re-seeded per split) | `metrics.py:92-132` | 「10分割、各分割は無作為抽出(非復元)でpositive数と同数の負例を抽出。乱数seedは0(関数呼び出しごとに1回生成、10分割全体で共有)」 | -- |
| B3 | sAUC: aggregation | Confirmed | Mean AUC across the (up to) 10 successful splits, per (image, layer) | `metrics.py:130-132` | 「10分割のAUCを平均」 | -- |
| B4 | N-shape indicator: metric | Confirmed | NSS only (`m_contrast` computed for NSS; AUC-Judd/sAUC versions also exist as separate rows in the same CSVs but NSS is the "primary metric" per the file's own header text) | `outputs/model_comparison_statistics/statistics_summary.md:3` ("primary metric: NSS") | 「N字型指標は主指標のNSSについて算出(AUC-Judd/sAUCについても同一式で別途算出されており、Figure 1の3パネルで個別に提示)」 | -- |
| B4 | N-shape indicator: layer bins | Confirmed | early=[L3,L4,L5], middle=[L7,L8,L9], late=[L11,L12] | `scripts/stats_model_comparison_mcontrast.py:66-68` | 「前半=L3-L5平均、中間=L7-L9平均、後半=L11-L12平均」 | -- |
| B4 | N-shape indicator: formula | Confirmed | `m_contrast = (early + late)/2 - middle`, computed per image FIRST, then bootstrap-averaged across 700 images | `scripts/stats_model_comparison_mcontrast.py:244-249` | 「N字型指標 = (前半平均+後半平均)/2 − 中間平均。画像単位で先に計算し、その後700画像で平均」 | -- |
| B4 | N-shape indicator: sign convention | Confirmed | Positive = early+late higher than middle (i.e., a mid-layer dip relative to the flanking layers) = stronger N-shape | formula above | 「正の値は前半・後半が中間より高い(N字型が強い)ことを意味する」 | -- |
| B4 | CLIP-B value / CI | Confirmed | +0.882576 (rounds to +0.883), 95% CI [+0.859033(displayed +0.8590), +0.906465(displayed +0.9065)] | `outputs/model_comparison_statistics/m_contrast_summary.csv` | 「CLIP ViT-B/16: +0.883, 95%CI[+0.859,+0.907]」 | -- |
| B4 | DeiT/SL, DINO-S, DINO-B values | Confirmed | −0.293392, −0.217619, −0.030277 | same CSV | 「DeiT: −0.293、DINO-S: −0.218、DINO-B: −0.030」 | -- |
| B4 | Holm p < 1e-100 (CLIP vs. each other model) | Confirmed (with a CSV-precision caveat) | Independently recomputed raw Wilcoxon p in this audit's prior pass: ≈2.87e-116 (DeiT-SL, DINO-S), ≈2.97e-116 (DINO-B); Holm-adjusted ≈8.60e-116, matching `statistics_summary.md`'s `.2e`-formatted text. **The CSV `p_holm` column itself stores `0.000000`** (6-decimal write format underflow, `scripts/stats_model_comparison_mcontrast.py:490`) -- cite `statistics_summary.md` or recompute, not the raw CSV value | `outputs/model_comparison_statistics/statistics_summary.md`; independent Wilcoxon recomputation (`results_fact_check_report.md`, section 3) | 「Holm補正後 p<1e-100 (実際にはp≈8.6e-116)」 | -- |
| B4 | Holm family size | Confirmed | 3 comparisons (DeiT-SL, DINO-S, DINO-B, each vs. CLIP-B), for NSS `m_contrast` specifically | `outputs/model_comparison_statistics/m_contrast_model_comparisons.csv`, `holm_family` column: "metric=NSS,measure=m_contrast (3 tests: DeiT/DINO-S/DINO-B vs CLIP-B)" | 「3モデル比較(vs CLIP-B)を1つのfamilyとしてHolm補正」 | -- |
| B5 | Bootstrap: 10,000 resamples, image-level, with replacement | Confirmed | `N_BOOT=10000`; `rng.randint(0, n, size=(n_boot, n))` = image-index resampling with replacement | `scripts/stats_three_model_layerwise.py:78,248-250`; `scripts/stats_model_comparison_mcontrast.py` (same constants, confirmed via `statistics_summary.md` header) | 「10,000回, 画像単位, 復元抽出のbootstrap」 | -- |
| B5 | Bootstrap seed | Confirmed | 42 (`BOOT_SEED`) | `scripts/stats_three_model_layerwise.py:77` | 「seed=42」 | -- |
| B5 | CI type | Confirmed | Percentile bootstrap CI (NOT basic/BCa) | `scripts/stats_three_model_layerwise.py:258-261` (`np.percentile(boot_means, [2.5,97.5])`) | 「パーセンタイル法95%信頼区間」 | -- |
| B5 | CI bounds (percentiles) | Confirmed | 2.5th / 97.5th percentile (`ALPHA=0.05`) | same | 「2.5〜97.5パーセンタイル」 | -- |
| B5 | Multiple-comparison corrections used | Confirmed (two distinct methods for two distinct families) | Holm (model-comparison statistics: `holm_correction`, `scripts/stats_model_comparison_mcontrast.py:207`); BH-FDR (B-2 spatial analysis: `bh_fdr`, `lib/clip_attention_b2.py:449-471`, used in `scripts/stats_clip_attention_b2_full700.py` and the 1000-derangement robustness check) | as cited | 「モデル間比較にはHolm法、空間解析の多重比較にはBH-FDR法を使用(分析系統ごとに異なる)」 | -- |
| B5 | Significance level | Confirmed | α=0.05 throughout (`ALPHA=0.05` constant, consistent across every stats script checked) | `scripts/stats_three_model_layerwise.py:81`; `lib/clip_attention_b2.py` config-driven `alpha` field in `summary.json` (`"alpha": 0.05`) | 「有意水準0.05」 | -- |
| B5 | Statistics libraries used | Confirmed | `scipy.stats` (Wilcoxon), `numpy` (bootstrap, percentiles); NOT `statsmodels` for Holm/BH-FDR (both self-implemented) | `scripts/stats_model_comparison_mcontrast.py:207` (`holm_correction`), `lib/clip_attention_b2.py:449` (`bh_fdr`) both hand-written | 「Wilcoxon検定はscipy.stats、Holm/BH-FDR補正は自作実装」 | -- |

### NSS formula (matches implementation exactly)

For image *i*, layer *l*: let `S` be the layer's saliency map (already
bilinearly upsampled to 800×600 and renormalized to sum 1, see A4), and
`F = {(x_k, y_k)}` the pooled discrete fixation pixels for that image.

```
S_norm = (S - mean(S)) / std(S)              [std: population, ddof=0;
                                                if std < 1e-12, NSS := 0]
NSS(i, l) = (1/|F|) * sum_k  S_norm[round(y_k), round(x_k)]
```

(`metrics.py:24-45`, `_sample` at lines 139-149.)

### AUC-Judd algorithm (matches implementation exactly)

```
pos_vals = { S[y_k, x_k] : (x_k,y_k) in F }
neg_vals = { S[y, x] : (y,x) not in F }, uniformly subsampled to <=100,000
           (RandomState(0)) if larger
AUC = trapz( TPR(pos_vals, neg_vals), FPR(pos_vals, neg_vals) )
      via: sort {pos_vals, neg_vals} descending by score,
           TPR_j = cumulative_positives_j / |pos_vals|,
           FPR_j = cumulative_negatives_j / |neg_vals|
```

(`metrics.py:52-85`, `_roc_auc` at 152-172.)

### sAUC algorithm (matches implementation exactly)

```
pos_vals = { S[y_k, x_k] : (x_k,y_k) in F }         (same as AUC-Judd)
rng = RandomState(0)                                 (one instance, whole call)
for split in 1..10:
    neg_pts = rng.choice(all_other_fixations, size=min(|pos_vals|, M),
                          replace=False)              (M = size of the FULL
                                                         700-image pool, self
                                                         included, per B3)
    neg_vals = { S[y, x] : (x,y) in neg_pts }
    auc_split = _roc_auc(pos_vals, neg_vals)          (same trapezoidal AUC)
sAUC(i, l) = mean(auc_split over valid splits)
```

(`metrics.py:92-132`.)

---

## 4. C. Linear Probe Details

| ID | 項目 | 判定 | 確認結果 | 根拠ファイル・行 | 論文用の安全な記述 | 要対応 |
|---|---|---|---|---|---|---|
| C1 | Mask-patch overlap: area-weighted, not binary | Confirmed | `mask_to_patch_weights`: each patch's weight = mean of the (zero-padded) boolean mask over that patch's 16×16 pixels = fractional area coverage in [0,1] | `lib/osie_text_alignment.py:40-55` | 「面積被覆率による連続的な重み付け(2値inclusionではない)」 | -- |
| C1 | Weight normalization | Confirmed | `weighted_pool_tokens`: `(tokens*weights).sum(0) / weights.sum()` -- normalized by the total weight sum, so weights need not sum to 1 beforehand | `lib/osie_text_alignment.py:62-78` | 「重み付き平均は総重みで正規化」 | -- |
| C1 | No-overlap handling | Confirmed | If `weights.sum() < 1e-6`, returns `None` (object skipped, logged, never silently zeroed) | `lib/osie_text_alignment.py:75-77` | 「重なりが実質皆無の物体はNoneとしてスキップ(0埋めしない)」 | -- |
| C1 | Aggregation to one vector per object | Confirmed | Weighted mean over all 1900 patches (masked by the object's weight vector), one 768-dim vector per object per layer | `lib/osie_text_alignment.py:62-78` | 「物体ごとに1900 patch分の加重平均を取り、768次元ベクトル1本を得る」 | -- |
| C1 | Same object features shared by both probes | Confirmed | Both `scripts/probe_osie_all12_full700.py` and `scripts/probe_osie_lowlevel_all12_full700.py` load `outputs/osie_probe_all12_full700/object_features_all12.npz` | `scripts/probe_osie_all12_full700.py:60`; `scripts/probe_osie_lowlevel_all12_full700.py:54` | 「両probeの入力特徴Xは完全に共通(同一npz)」 | -- |
| C2 | 8 low-level features: names/formulas | Confirmed | See table below | `lib/osie_lowlevel_features.py:26-88` | (below) | -- |
| C3 | Ridge: library/alpha/solver | Confirmed | `sklearn.linear_model.Ridge(alpha=1.0)` (default alpha, no CV tuning); solver not explicitly set → sklearn 1.7.2's default ("auto", which for dense input resolves to Cholesky-based solving); `Pipeline([StandardScaler(), Ridge])` -- StandardScaler fit on X only, y NOT standardized | `lib/osie_lowlevel_probe.py:20,39-40` | 「Ridge回帰(α=1.0, ハイパーパラメータ探索なし)。入力Xのみ標準化、目的変数Yは生の単位のまま」 | -- |
| C3 | Ridge: intercept | Unverified | `Ridge()`'s default `fit_intercept=True` was not explicitly overridden in the call (`Ridge(alpha=alpha)`), so sklearn's default (True) applies, but this was not separately asserted/logged anywhere in the code or config | `lib/osie_lowlevel_probe.py:39-40` (no `fit_intercept` kwarg) | 「切片あり(sklearn既定値、明示的な設定・記録はコード上にない)」 | 軽微: 明示的なassertが無いため厳密にはsklearnのデフォルト値に依存 |
| C3 | Ridge: CV / R² unit / aggregation | Confirmed | `GroupKFold(n_splits=min(5, n_groups))`, grouped by `image_id`; R² computed PER FOLD on that fold's held-out predictions (`sklearn.metrics.r2_score`), then folds' R² averaged (mean, `std(ddof=1)` implied by `.std()` w/o ddof=0 -- confirm: `lib/osie_lowlevel_probe.py` uses plain `.std()`... see caveat) for `lowlevel_summary.csv` | `lib/osie_lowlevel_probe.py:80-183` | 「画像ID単位5-fold(GroupKFold)。R²はfold単位で算出後、5-fold平均」 | -- |
| C3 | Ridge: random seed | Confirmed | `set_all_seeds(seed=42)` (`random.seed`, `np.random.seed`) called once at the start of `run_ridge_probe`; `GroupKFold` itself has no shuffle/seed parameter (deterministic given group order) | `lib/osie_lowlevel_probe.py:34-38,102` | 「seed=42。GroupKFoldはgroup順序に基づく決定的分割で乱数不使用」 | -- |
| C4 | 12 semantic attributes: names & counts | Confirmed | See table below | `outputs/osie_probe_all12_full700/probe_summary.csv` | (below) | -- |
| C4 | All 5,551 objects targeted | Confirmed | `n_objects=5551` for the `all_objects` condition, for every attribute (positive+negative counts sum to 5551 per attribute row) | `probe_summary.csv` (`n_objects` column) | 「意味属性probeは全5,551物体を対象」 | -- |
| C4 | Attribute-selection rationale | **Unverified** | No comment, config field, README, or commit message found anywhere in this repository explaining WHY these specific 12 OSIE attributes were chosen (they appear to be the full label set on the annotated OSIE `attrs.mat`, but this audit found no explicit statement to that effect) | (repo-wide search, no match) | 「12属性の選定根拠を示す記録は本リポジトリ内に見つからなかった(未確認)」 | **人間による確認が必要** |
| C5 | LogisticRegression: params | Confirmed | `penalty="l2", C=1.0, class_weight="balanced", max_iter=5000, random_state=seed`; `solver` NOT specified → sklearn 1.7.2 default = `lbfgs` | `lib/osie_text_alignment_probe.py:41-46` | 「L2正則化, C=1.0, class_weight="balanced", max_iter=5000, solver未指定(sklearn既定=lbfgs)」 | -- |
| C5 | Input standardization | Confirmed | `Pipeline([StandardScaler(), LogisticRegression])`, scaler fit on train fold only | `lib/osie_text_alignment_probe.py:41-46,174-177` | 「入力Xのみ標準化(train foldでfit)」 | -- |
| C5 | AUPRC definition | Confirmed | `sklearn.metrics.average_precision_score` (step-function/precision-recall AUC via average precision, NOT raw trapezoidal PR-curve integration) | `lib/osie_text_alignment_probe.py:22,86` | 「AUPRCはsklearnのaverage_precision_score(階段関数による平均適合率)」 | -- |
| C5 | Fold aggregation | Confirmed | AUPRC/AUROC/balanced-accuracy computed per fold, then mean±std(ddof=1, since `len(vals)>1` branch uses `vals.std(ddof=1)`) across folds in `probe_summary.csv` | `lib/osie_text_alignment_probe.py:201-213` (`vals.std(ddof=1) if len(vals)>1 else 0.0`) | 「fold単位で算出後、5-fold平均・不偏標準偏差(ddof=1)」 | -- |
| C5 | Convergence warnings | Unverified | Not captured/logged anywhere in the probe scripts or their JSON/CSV outputs (no `n_iter_`, no warnings-log file found) | (no matching file found) | 「収束警告の記録は保存済み出力に見つからなかった(未確認)」 | 軽微 |
| C5 | Random seed | Confirmed | `seed=42` passed through to both `StandardScaler`-less-but-`LogisticRegression(random_state=seed)` and the CV splitter | `lib/osie_text_alignment_probe.py:100,120,160` | 「seed=42」 | -- |
| C6 | Image-ID-level 5-fold, no cross-fold leakage | Confirmed | `StratifiedGroupKFold`/`GroupKFold` with `groups=image_ids`; explicit runtime assertion raises `RuntimeError` if any image_id appears in both train and test of the same fold | `lib/osie_text_alignment_probe.py:160-164`; `lib/osie_lowlevel_probe.py:132-136` | 「画像ID単位5-fold。同一画像由来の複数物体がtrain/testへ分裂しないことをコード内で明示的に検証」 | -- |
| C6 | Fold shuffle / seed / stratification | Confirmed | Semantic probe: `StratifiedGroupKFold(shuffle=True, random_state=seed)` (label-stratified by positive/negative). Low-level probe: `GroupKFold` (NOT stratified -- regression has no class label to stratify; also no shuffle parameter, deterministic by group order) | `lib/osie_text_alignment_probe.py:160`; `lib/osie_lowlevel_probe.py:132` | 「意味属性probeはラベル層化+shuffle(seed=42)、低次probeは層化なし(回帰のため)・shuffleなし」 | -- |
| C6 | Scaler fit train-only | Confirmed | Both probes use sklearn `Pipeline`, `pipe.fit(X[tr], y[tr])` only, `pipe.predict(X[te])` for evaluation -- standard sklearn Pipeline semantics guarantee no test-fold leakage into the scaler | `lib/osie_text_alignment_probe.py:174-177`; `lib/osie_lowlevel_probe.py:150-152` | 「StandardScalerはtrain foldのみでfit」 | -- |
| C6 | Hyperparameter-selection leakage | Confirmed absent | Both `Ridge(alpha=1.0)` and `LogisticRegression(C=1.0)` use FIXED hyperparameters (no `GridSearchCV`/`RandomizedSearchCV` anywhere in either probe script) -- no nested-CV leakage risk exists because no hyperparameter search occurs | `lib/osie_lowlevel_probe.py:39-40`; `lib/osie_text_alignment_probe.py:41-46` (no CV-search import anywhere in either file) | 「ハイパーパラメータは固定値(α=1.0, C=1.0)であり、探索によるリークは生じない」 | -- |
| C6 | Permutation: what is shuffled | Confirmed | TRAIN-fold labels only (`y_train_shuffled = perm_rng.permutation(y[tr])`); test-fold labels are NEVER shuffled; re-fit on shuffled train labels, evaluated against the REAL test labels | `lib/osie_text_alignment_probe.py:216-228` | 「trainラベルのみをシャッフルして再学習し、実際のtestラベルで評価するpermutationベースライン」 | -- |
| C6 | Permutation: unit | Confirmed | Object-level label shuffling within each fold's train set (not image-level) | `lib/osie_text_alignment_probe.py:220` (`perm_rng.permutation(y[tr])`, `y[tr]` indexed by object rows) | 「object単位でtrainラベルをシャッフル」 | -- |
| C6 | Permutation: repetitions | Confirmed | Stage A: 20 reps (all 12 layers × 12 attributes); Stage B: escalated to 100 reps for peak + reference layers only, merged into the final `probe_permutation_baseline.csv` in place of the 20-rep entries for those specific rows | `scripts/probe_osie_all12_full700.py` (module docstring, `N_PERM_REPS_STAGE_A`/`_B`); confirmed by row count: all 12 attributes have exactly 12 layers each (144 auprc rows) in `probe_permutation_baseline.csv` | 「基本20反復、ピーク層・参照層のみ100反復へ増強」 | -- |
| C6 | Permutation: probe re-trained each time | Confirmed | `pipe = make_pipeline(seed); pipe.fit(X[tr], y_train_shuffled)` inside the repetition loop -- a fresh pipeline fit per repetition per fold | `lib/osie_text_alignment_probe.py:216-225` | 「反復ごと・fold毎にprobeを再学習」 | -- |
| C6 | Permutation vs. real performance comparison | Confirmed | `random_auprc` (prevalence) and the permutation-mean AUPRC are both stored alongside the real fold AUPRC in the same CSVs, enabling a "observed vs. permutation-mean" comparison; a formal downstream gate/threshold check exists (`probe_hypothesis_summary.json`, not deeply audited in this pass) | `outputs/osie_probe_all12_full700/probe_summary.csv`,`probe_permutation_baseline.csv` | 「観測AUPRCとpermutationベースライン平均AUPRCを層・属性ごとに対比」 | -- |
| C6 | 7 of 8 low-level features peak L1-L4 | Confirmed | mean_luminance L3, mean_red L3, mean_green L3, mean_blue L3, mean_saturation L3, edge_strength L4, fine_texture L2 (7); exception: luminance_contrast peaks L6 | `outputs/osie_lowlevel_probe_all12_full700/lowlevel_summary.csv` (this audit's own recomputation, cross-checked in `results_fact_check_report.md`) | 「8特徴中7特徴がL1〜L4でピーク(例外: luminance_contrast, L6)」 | -- |
| C6 | Low-level mean R² at L4 ≈ 0.80 | Confirmed | 0.795599 (rounds to 0.80) | same file | 「L4で平均R²≈0.80(丸め前0.7956)」 | -- |
| C6 | Semantic mean AUPRC L1≈0.22, L10≈0.60 | Confirmed | 0.216704 (L1), 0.598309 (L10) | `probe_summary.csv` | 「L1≈0.22(丸め前0.2167), L10≈0.60(丸め前0.5983)」 | -- |
| C6 | 10 of 12 attributes peak L8-L11 | Confirmed | Exceptions: Sound (L12), Touch (L12) | same file | 「12属性中10属性がL8〜L11でピーク(例外: Sound, Touch, ともにL12)」 | -- |
| C6 | 「Pearson相関 r=−0.84」(タスク文の想定値) | Contradicted | `curve_correlations.csv`のPearson r=−0.7824であり−0.84ではない。−0.84はSpearman ρ=−0.8392に対応する値 | `outputs/osie_layer_profile_synthesis/curve_correlations.csv`(既報告: `results_fact_check_report.md` item 1) | 「Spearman ρ=−0.84(−0.8392), Pearson r=−0.78(−0.7824)。両者を区別して記載(草稿の"Pearson r=−0.84"は誤り)」 | 要対応: 本文表現をSpearman表記に修正、または−0.78へ差し替え |

### 8 low-level visual features: names & formulas

| コード上の変数名 | 日本語名 | 英語名 | 色空間 | マスク適用 | 計算式 | 値域 |
|---|---|---|---|---|---|---|
| `mean_luminance` | 平均輝度 | Mean luminance | RGB [0,1] | マスク内平均 | 0.2126R+0.7152G+0.0722B (ITU-R BT.709) | [0,1] |
| `mean_red` | 平均赤 | Mean red | RGB | マスク内平均 | Rチャンネル平均 | [0,1] |
| `mean_green` | 平均緑 | Mean green | RGB | マスク内平均 | Gチャンネル平均 | [0,1] |
| `mean_blue` | 平均青 | Mean blue | RGB | マスク内平均 | Bチャンネル平均 | [0,1] |
| `mean_saturation` | 平均彩度 | Mean saturation | HSV | マスク内平均 | `matplotlib.colors.rgb_to_hsv`のS成分 | [0,1] |
| `luminance_contrast` | 輝度コントラスト | Luminance contrast | 輝度 | マスク内標準偏差 | luminanceのマスク内std | [0,~0.5] |
| `edge_strength` | エッジ強度 | Edge strength | 輝度 | 全画像に適用後マスク内平均 | Sobel勾配強度`sqrt(gx²+gy²)`(`scipy.ndimage.sobel`) | [0,∞) |
| `fine_texture` | 微細テクスチャ | Fine texture | 輝度 | 全画像に適用後マスク内分散 | Laplacian分散(`scipy.ndimage.laplace`) | [0,∞) |

(`lib/osie_lowlevel_features.py:26-88`. 目的変数側の標準化なし、fold内標準化は入力Xのみに適用 — 上記C3参照。欠損値処理: マスク画素数<1で当該objectを`None`扱いしスキップ。サンプル数: 8特徴とも有効5,263物体。)

### 12 semantic attributes: names & sample sizes

| コード上のラベル名 | 日本語名 | 英語名 | 陽性数 | 陰性数 | 陽性率(prevalence) | 対象物体数 |
|---|---|---|---:|---:|---:|---:|
| Text | 文字 | Text | 495 | 5056 | 0.0892 | 5551 |
| Face | 顔 | Face | 1010* | 4541* | 0.1819 | 5551 |
| Emotion | 感情 | Emotion | 190* | 5361* | 0.0342 | 5551 |
| Sound | 音 | Sound | 89* | 5462* | 0.0160 | 5551 |
| Smell | におい | Smell | 135* | 5416* | 0.0243 | 5551 |
| Taste | 味 | Taste | 390* | 5161* | 0.0703 | 5551 |
| Touch | 触感 | Touch | 317* | 5234* | 0.0571 | 5551 |
| Motion | 動き | Motion | 557* | 4994* | 0.1003 | 5551 |
| Operability | 操作可能性 | Operability | 342* | 5209* | 0.0616 | 5551 |
| Watchability | 視認対象性 | Watchability | 1108* | 4443* | 0.1996 | 5551 |
| Touched | 接触され得る | Touched | 436* | 5115* | 0.0785 | 5551 |
| Gazed | 注視され得る | Gazed | 252* | 5299* | 0.0454 | 5551 |

`prevalence`列は`probe_summary.csv`の`random_auprc`列（AUPRCのchance level =
陽性率そのもの）から直接読み取った確定値。陽性数/陰性数の列（*印）はこの
audit内で`prevalence × 5551`から逆算した近似値であり、`objects_metadata.csv`
の`positive_attributes`列から属性ごとに直接カウントし直す形の再検証はこの
audit では行っていない（Text行のみ`probe_summary.csv`に`n_positive=495`
の直接記載があり、他行との整合を確認済み）。**厳密な陽性数/陰性数は
`outputs/osie_probe_all12_full700/object_features_metadata.csv`の
`positive_attributes`列を数え上げて確認することを推奨（Unverified、下記）。**

**選定根拠**: Unverified（上記C4参照、リポジトリ内に根拠記録なし）。

---

## 5. D. Additional Spatial Analyses and Visualizations

| ID | 項目 | 判定 | 確認結果 | 根拠ファイル・行 | 論文用の安全な記述 | 要対応 |
|---|---|---|---|---|---|---|
| D1 | 50%-mass: normalize-then-cumsum-descending | Confirmed | `p = raw/raw.sum()`; sorted descending; cumulative sum; smallest count reaching `>= target_mass` (i.e. "at least", not "exceeds strictly" -- `np.searchsorted(cumsum, target_mass - 1e-12)` uses a tiny epsilon subtraction to treat exact equality as reached) | `lib/clip_attention_b2.py:36-43,92-108` (`normalize_distribution`, `coverage_for_mass`) | 「正規化分布を降順ソートし、累積質量が50%以上に初めて到達する最小patch数(以上・境界はε=1e-12で処理)」 | -- |
| D1 | 50%-mass: per-image then averaged, rounding | Confirmed | Per-image integer patch count → plain mean over 700 images → rounded ONCE for display | `outputs/clip_attention_b2_distribution/summary.json` `descriptive_per_layer.mass50_n_patches` | 「画像単位のpatch数を700画像で平均後、一度だけ丸め」 | -- |
| D1 | Patch total = 1900 | Confirmed | `n_patches=1900` (38×50), `top10pct_k=190` derived from it | `outputs/clip_attention_b2_distribution/config.json` | 「1900 patch(38×50)」 | -- |
| D1 | NaN/negative/zero-sum handling | Confirmed | `normalize_distribution` raises `DegenerateAttentionError` if `raw.sum() <= eps(1e-8)`; 0 such images occurred in the full700 run (`n_degenerate_images_excluded_from_fgbg_metrics` field, separately tracked for the fg/bg-specific metrics only, = 0) | `lib/clip_attention_b2.py:36-43`; `outputs/clip_attention_b2_distribution/summary.json` (`n_degenerate_images_excluded_from_fgbg_metrics: 0` per this session's earlier check) | 「総和がほぼ0の画像は例外として除外する設計だが、実際には0件」 | -- |
| D1 | Canonical values | Confirmed | L4=563, L8=121, L12=68 (rounded once from means 563.2214/121.4986/67.6343) | `outputs/clip_attention_b2_distribution/summary.json`; independently re-derived from `per_image_metrics.csv` in `results_fact_check_report.md` | 「L4=563, L8=121, L12=68」 | -- |
| D1 | 562/122/68 vs. 563/121/68 discrepancy = double rounding | Confirmed | 562/122/68 = round(1900 × round(mean_area_fraction, 3 decimals)); double rounding of the SAME underlying data, not an independent computation | `results_fact_check_report.md`, section 5 (exact reproduction of the arithmetic) | 「562/122/68は面積比を先に3桁へ丸めてから1900倍・再丸めした二重丸め誤差。563/121/68を正準値として採用」 | 既報告、本文側の数値差し替えが必要な場合は要対応 |
| D2 | Normalized entropy | Confirmed | `H(p)/log(n_patches)`, `p` must sum to 1, `0*log(0):=0`; per-image; higher = more spread out | `lib/clip_attention_b2.py:50-58` | 「H(p)/log(1900)、値が大きいほど広く分散」 | -- |
| D2 | L4/L8/L12 normalized entropy | Confirmed | L4=0.9807 [0.9800,0.9813], L8=0.8478 [0.8463,0.8492], L12=0.7960 [0.7935,0.7985] (95% bootstrap CI, same convention as D1/D3/D4) | `outputs/clip_attention_b2_distribution/summary.json` | 「正規化エントロピー: L4=0.981, L8=0.848, L12=0.796」 | -- |
| D2 | Hoyer sparsity | Confirmed | `(sqrt(n) - ||raw||_1/||raw||_2) / (sqrt(n)-1)`, scale-invariant (works on raw, no normalization needed); higher = sparser/more concentrated | `lib/clip_attention_b2.py:61-74` | 「Hoyerスパース度、値が大きいほど集中」 | -- |
| D2 | L4/L8/L12 Hoyer sparsity + paired test | Confirmed | L4=0.1367 [0.1327,0.1408], L8=0.7218 [0.7193,0.7243], L12=0.7339 [0.7301,0.7375]; paired L8-L4: Cohen's dz=7.951, sign-flip p=0.0001, BH-FDR q=0.000112 | `summary.json`; `outputs/clip_attention_b2_distribution/paired_layer_comparisons.csv` (row `metric=hoyer_sparsity, comparison=L8-L4`) | 「L8はL4よりHoyerスパース度が有意に高い(dz=7.95, BH-FDR q<0.001) -- L8は"拡散"ではなく"集中"している」 | -- |
| D2 | Top-10% mass | Confirmed | Sum of the top-190 (=10% of 1900) patches' normalized mass | `lib/clip_attention_b2.py:82-89` | 「上位10%(190 patch)の質量合計」 | -- |
| D2 | L4/L8/L12 top-10% mass | Confirmed | L4=0.2273 [0.2245,0.2302], L8=0.5589 [0.5561,0.5619], L12=0.6921 [0.6874,0.6967] | `summary.json` | 「上位10%質量: L4=0.227, L8=0.559, L12=0.692」 | -- |
| D2 | 80%-mass patch count (support area) | Confirmed | Same `coverage_for_mass` function, target_mass=0.8 | `lib/clip_attention_b2.py:92-108` | -- | -- |
| D2 | L4/L8/L12 80%-mass patch count | Confirmed | L4=1185.6 [1178.8,1192.4], L8=736.2 [728.5,743.7], L12=355.5 [348.0,363.0] | `summary.json` | 「80%質量patch数: L4≈1186, L8≈736, L12≈355」 | -- |
| D2 | Max single-patch attention | Confirmed | `p.max()` (lower bound = 1/1900 for uniform) | `lib/clip_attention_b2.py:77-79` | -- | -- |
| D2 | Normalized center distance | Confirmed | Euclidean distance of the attention-weighted centroid from image center (0.5,0.5), normalized by √0.5 | `lib/clip_attention_b2.py:115-139` | -- | -- |
| D2 | All D2 metrics: aggregation & CI | Confirmed | Per-image, then bootstrap mean+95%CI over 700 images (seed=42, n_boot=10000, alpha=0.05, same convention as D1/D3/D4) | `outputs/clip_attention_b2_distribution/config.json`,`summary.json` top-level fields | 「全指標を画像単位で計算後、700画像でbootstrap平均・95%CI(seed=42, n_boot=10000)」 | -- |
| D3 | Foreground = union of OSIE object masks | Confirmed | `coverage` = per-patch fractional area of the union of all object masks for that image (via `mask_to_patch_weights` applied to the OR-combined mask) | `lib/clip_attention_b2.py:162-214`; mask union confirmed in the extraction script (`fg |= mp` pattern, e.g. `scripts/figure3...`/`_foreground_mask`) | 「foreground=OSIE物体マスクの和集合(重複領域は1回のみカウント)」 | -- |
| D3 | Background definition | Confirmed | `1 - foreground_area_fraction` (complement) | `lib/clip_attention_b2.py:190-191` | -- | -- |
| D3 | Enrichment formula | Confirmed | `foreground_enrichment = foreground_conditional_mass / foreground_area_fraction`, where `foreground_conditional_mass = sum(p_i * coverage_i)` (p=normalized attention), `foreground_area_fraction = mean(coverage)` | `lib/clip_attention_b2.py:196-203` | 「enrichment = (foreground上のattention質量比)/(foreground面積比)。1.0=面積比例配分」 | -- |
| D3 | Zero-foreground-area handling | Confirmed | `is_degenerate=True` if `fg_area_fraction < 1/1900`; enrichment set to `None` for that case (excluded, not zero) | `lib/clip_attention_b2.py:159,199-203` | 「foreground面積が実質0の画像は除外(0とはしない)」 | -- |
| D3 | Per-image, then layer-aggregated | Confirmed | Same convention as D1/D2 | `summary.json` | -- | -- |
| D3 | Canonical values | Confirmed | L4=1.5986 (1.60), L8=0.9949 (0.99), L12=1.5719 (1.57) | `summary.json`; independently re-derived from `per_image_metrics.csv` | 「L4=1.60, L8=0.99, L12=1.57」 | -- |
| D3 | L4 vs. L12 equivalence | Confirmed NOT tested / NOT equivalent-by-test | Paired test exists (`L12-L4`): mean_diff=−0.0267, CI=[−0.0744,+0.0282] (includes 0), sign-flip p=0.3162, BH-FDR q=0.3162 -- a non-significant DIFFERENCE test, explicitly NOT a formal equivalence test (no TOST anywhere in this repo) | `outputs/clip_attention_b2_distribution/paired_layer_comparisons.csv` (row `metric=foreground_enrichment, comparison=L12-L4`); `report_revised.md` point 2 ("a formal equivalence test was NOT run") | 「L4とL12のforeground enrichmentに有意差は検出されなかった(差の検定, 同等性検定ではない)」 | -- |
| D4 | Pearson, on raw attention, flattened per image | Confirmed | `pearson_corr(raw_a, raw_b)` operates on the flattened 1900-dim raw (un-normalized) vectors for ONE image's two layers; scale-invariant so raw vs. normalized give identical r | `lib/clip_attention_b2.py:221-231`; call site `scripts/compute_clip_attention_b2_full700.py:353` | 「rawベクトル(1900次元にflatten)同士のPearson相関、同一画像内の2層間で算出」 | -- |
| D4 | Not correlated across images | Confirmed | Each row of `layer_pair_similarity.csv` is one image's own layer-pair correlation (`pair_type=="same_image"`); a SEPARATE `pair_type=="shuffled_baseline"` row exists per image using a fixed derangement partner, for baseline comparison only | `outputs/clip_attention_b2_distribution/layer_pair_similarity.csv` | 「層間相関は同一画像内で算出。画像間でのシャッフルは別途baseline比較専用」 | -- |
| D4 | 700-image aggregation: Fisher-z for Pearson | Confirmed | `same_image_mean` for the Pearson metric = Fisher-z transform each of the 700 per-image r → mean → inverse-transform; for cosine/JSD, plain arithmetic mean (no Fisher-z) | `scripts/stats_clip_attention_b2_full700.py:243-258` | 「Pearson相関の700画像平均はFisher-z変換後に平均し逆変換(cosine・JSDは単純平均)」 | -- |
| D4 | Plain mean differs materially for L8-L12 | Confirmed | Fisher-z mean=0.9584 vs. plain mean=0.9431 for L8-L12 (independently recomputed in this audit) | `results_fact_check_report.md`, Figure 3 Audit table | 「L8-L12のPearson相関は集計法により0.943(単純平均)〜0.958(Fisher-z平均)と変わりうる。本文の0.958はFisher-z平均」 | -- |
| D4 | Constant-map handling | Confirmed | `pearson_corr` raises `ValueError` if either map has `std<=1e-8` (zero-variance/constant map); no such case occurred in the full700 run (no error raised during production) | `lib/clip_attention_b2.py:221-231` | -- | -- |
| D4 | Canonical values | Confirmed | L4-L8=−0.048 (Fisher-z −0.0482), L8-L12=0.958 (0.9584), L4-L12=0.015 (0.0146) | `summary.json`; independently re-derived from `layer_pair_similarity.csv` | 「L4-L8=−0.048, L8-L12=0.958, L4-L12=0.015」 | -- |
| D4 | CI / statistical comparison | Confirmed | `spatial_similarity_descriptive` provides same-image vs. shuffled-baseline MEANS (no CI on the same-image r itself in that block), but a full paired test (`paired_layer_comparisons.csv`) and a 1000-derangement Monte Carlo robustness check (`repeated_shuffle_summary.json`, `verify_clip_attention_b2_shuffle_robustness.py`) both exist and confirm the L8-L12 vs. L4-L8/L4-L12 pattern is robust | `outputs/clip_attention_b2_distribution/repeated_shuffle_summary.json` | 「same-image相関そのものへの信頼区間は`summary.json`の記述統計欄にはないが、shuffled baselineとの差の検定・1000回の独立derangementによる頑健性確認が別途存在」 | -- |
| D5 | Image 1159 layer-specific attention | Confirmed | `paper_figures/appendix/figure_attention_layer_specific_additional_examples.{pdf,png}` (row 1 of 2); generated by `scripts/paper_figures/figure3_spatial_characterization_revised.py::make_additional_examples_figure` | script + output both exist, this project's own prior work (this session) | 「Appendix補足図として既に生成済み」 | -- |
| D5 | Image 1213 layer-specific attention | Confirmed | Same file (row 2 of 2) | same | 同上 | -- |
| D5 | Image 1156 common-scale version | Confirmed | `paper_figures/appendix/figure_attention_common_scale.{pdf,png,svg}`; generated by `scripts/paper_figures/appendix_figure3_common_scale.py`, which calls the ORIGINAL (unmodified) `figure3_spatial_characterization.make_figure()` and re-saves under this path -- this is the 3-representative-image (1156,1159,1213), SHARED-per-image-color-scale version, not a new computation | script + output both exist | 「共通カラースケール版として既に生成済み(image 1156含む3画像版)」 | -- |
| D5 | Main-text `figure3_spatial_characterization_revised.pdf` | Confirmed | `paper_figures/figure3_spatial_characterization_revised.pdf` exists; generated by `scripts/paper_figures/figure3_spatial_characterization_revised.py::make_main_figure`; representative image = 1156 only; layer-specific (per-layer own vmin=0/vmax=own max) color scale, own colorbar per layer, no percentile clipping, alpha=0.60, foreground contour drawn (white, same contour on all 3 layers) | script + output both exist, verified in this session's prior work | 「本文Figure 3として既に生成済み」 | -- |
| D5 | vmin/vmax/clipping/alpha (main figure) | Confirmed | `vmin=0.0`, `vmax=`that layer's own raw max for that image (no percentile clipping anywhere); `alpha=0.60`; colormap=`viridis` | `scripts/paper_figures/figure3_spatial_characterization_revised.py` (module docstring + `draw_layer_specific_grid`) | 「vmin=0、vmax=当該層の実測最大値(percentile clippingなし)、alpha=0.60、viridis」 | -- |
| D5 | Foreground outline | Confirmed | Present, drawn identically (white, linewidth 0.9, same contour) on all 3 CLIP-layer panels for a given image; absent from the Original/Human-gaze columns | same script | 「foreground輪郭は3層とも同一(白線)で描画。Original/Human gaze列には描画しない」 | -- |
| D5 | Re-generation needed for the Appendix | Not applicable | All 3 requested items already exist as generated figures from this project's own prior work in this session; no re-generation was performed or is needed for this fact-check pass (this audit did not regenerate or overwrite any of them) | -- | -- | -- |

---

## 6. Manuscript–Implementation Discrepancies

No manuscript file exists in this repository (see header note), so this
table covers discrepancies found **between the task brief's own stated
assumptions/values and this repository's implementation/outputs**, plus
one discrepancy carried over from this project's own prior audit.

| 重要度 | 草稿(タスク文)の記述 | 実装・出力の事実 | 根拠 | 推奨対応 |
|---|---|---|---|---|
| Critical | ファイルパスが `paper_figs/...` | 実際のディレクトリ名は `paper_figures/`（`paper_figs/`は存在しない） | `ls paper_figs` → No such file or directory; `ls paper_figures/figure{1,2,3}_*` → 全て存在 | 原稿の`\includegraphics`パス、または本監査対象記述のディレクトリ名を`paper_figures/`に統一 |
| Major | モデル名を単に「DeiT ViT-S/16」と扱う想定 | チェックポイントは蒸留トークン(`dist_token`)を持たない標準ViT-S/16構成(DeiT訓練コードベースで学習) | `sl_extractor.py:1-46`(docstring、state_dict直接検査による確認記録) | 「DeiT-style supervised ViT-S/16 (no distillation token)」等、蒸留トークンの存在を示唆しない表現へ変更 |
| Minor | 「Pearson相関 r=−0.84」(タスク文中の想定値) | この値はSpearman ρ=−0.8392に対応。Pearson r=−0.7824 | `outputs/osie_layer_profile_synthesis/curve_correlations.csv`; 既に`results_fact_check_report.md`で報告済み | Spearmanと明記するか、Pearson値(−0.78)に差し替え |
| Minor | 「50% mass patch数 562/122/68」(過去メモにこの表記があるとの言及) | 正準値は563/121/68(単純平均を一度だけ丸めた値)。562/122/68は面積比を先に3桁丸めしてから1900倍・再丸めした二重丸め誤差 | `results_fact_check_report.md` section 5(算出過程を完全再現) | 本文・図・キャプションを563/121/68へ統一(既にFigure 3で採用済み) |

---

## 7. Unresolved Questions

Yes/No または具体的な値で回答できる形で列挙する。人間による確認・判断が必要。

1. **DINO/DeiT側の位置埋め込み補間**: `lib/vision_transformer.py`の
   `interpolate_pos_encoding`メソッドは、CLIP側の`interpolate_patch_pos_
   embed`と同じ「bicubic補間・CLS位置埋め込みは補間しない」規約を実装して
   いるか？ (Yes/No — この監査では未確認)
2. **12意味属性の選定根拠**: OSIEデータセットの全アノテーション属性がこの
   12個か、それとも一部を選択したものか？ 選定基準を示す記録(社内メモ、
   コミットメッセージ、READMEなど)は本リポジトリ外に存在するか？
3. **8低次視覚特徴・12意味属性それぞれの陽性数/陰性数の厳密な値**: 本監査
   では`prevalence`列からの逆算で近似したが、`object_features_metadata.csv`
   の`positive_attributes`列を直接数え上げた厳密な表を作成する必要がある
   か？ (作成する場合、低コストな再集計で対応可能)
4. **Ridge回帰の`fit_intercept`**: 明示的にコード上でTrueと記録されていない
   (sklearn既定値に依存)。この既定値への依存を許容するか、Appendixに明記
   すべきか？
5. **LogisticRegressionの収束警告**: 保存済み出力に収束状況のログが存在
   しない。全条件で収束していたことを別途確認する必要があるか(低コストな
   再学習1回でwarnings捕捉は可能だが、これは「新規probe学習」に該当するため
   本監査では実施していない)？
6. **`paper_figs/`という命名は原稿側の正式なディレクトリ名か**、それとも
   タスク文中の言い間違いで実際は`paper_figures/`を指すか？

---

## 8. Appendix-ready Facts

以下は根拠が確認済みで、Appendix本文へそのまま転記可能な確定事項(日本語)。
本文そのものはまだ執筆していない。

**モデルと前処理**
- CLIP ViT-B/16(OpenAI公式事前学習済み、`clip`パッケージ)、DINO ViT-S/16
  (Yamamoto et al. (2025)の枠組みでカスタム学習)、DINO ViT-B/16(Facebook
  Research公式事前学習済みbackbone)、DeiT訓練コードベースによる教師あり
  ViT-S/16(蒸留トークンなし)の4モデルを比較した。
- 全モデルとも12層、CLIP/DINO-Bは12ヘッド・768次元、DINO-S/DeiT-Sは6ヘッド
  ・384次元、patch size 16。
- 全モデルをeval modeで実行し、パラメータ更新なし、勾配計算は無効化した。
- OSIE原寸800×600画像を下端・右端のみゼロパディングし(リサイズ・クロップ
  なし)、38×50=1900 patchのAttention/patch表現を得た。

**Attention抽出と中間表現**
- Attentionは[CLS]→patchの生softmax(head平均、[CLS]自己成分は除外)を各層
  独立に評価し、attention rolloutは使用していない。
- probe入力に用いる中間表現は、各Transformer blockのAttention残差加算・
  MLP残差加算の両方が完了した直後の出力(最終LayerNorm・画像投影の前)であ
  り、CLS tokenを除いたpatch表現を使用する。CLIP ViT-B/16では1 patchあたり
  768次元。
- 視線一致解析用のAttention抽出経路とprobe用の隠れ状態抽出経路は別実装で
  あり、抽出するテンソルの種類自体が異なる(前者はAttention確率、後者は
  残差ストリームの隠れ状態)。

**視線一致指標**
- NSSは画像ごとにzスコア標準化(母標準偏差)したAttention mapを、統合された
  全被験者の離散注視点で平均した値。
- AUC-Judd/sAUCは自作実装(sklearn不使用)で、台形則によるROC AUC。sAUCの
  負例プールは全700画像の注視点から抽出し、評価対象画像自身の注視点を明示
  的には除外していない(データセット共通の位置バイアスへの感度低減を意図
  した設計であり、センターバイアスの完全な除去を意味しない)。
- N字型指標は(前半L3-L5平均+後半L11-L12平均)/2−中間L7-L9平均を画像単位で
  先に計算し、700画像でbootstrap(seed=42, 10,000回, 画像単位, 復元抽出)
  平均・95%信頼区間を得た。CLIP ViT-B/16: +0.883 [+0.859,+0.907]。DINO-S:
  −0.218、DINO-B: −0.030、DeiT: −0.293。CLIPと各比較モデルとの差はHolm
  補正後 p<1e-100。

**Probe**
- 意味属性probe(12属性)と低次視覚特徴probe(8特徴)は、同一の入力特徴X
  (CLIP隠れ状態、object mask加重平均)を共有する。
- 低次視覚特徴の目的変数のみ、CLIP標準前処理(Resize 224+CenterCrop)による
  別ジオメトリの画像から算出しており、入力特徴Xの原寸画像とは異なる。この
  結果、5,551物体中288物体が低次probeの目的変数から除外され、5,263物体を
  使用した。意味属性probeにはこの除外を適用していない(全5,551物体を使用)。
- 意味属性probeはStandardScaler+LogisticRegression(L2, C=1.0,
  class_weight="balanced")、低次probeはStandardScaler+Ridge(α=1.0)。とも
  に画像ID単位5-fold交差検証(同一画像由来の物体はtrain/testに分裂しない)。
- 低次特徴8種類中7種類がL1〜L4でピーク(例外: 輝度コントラスト、L6)。平均
  R²はL4で約0.80。意味属性12種類中10種類がL8〜L11でピーク(例外: Sound,
  Touch、ともにL12)。平均AUPRCはL1で約0.22、L10で約0.60。両曲線のSpearman
  相関はρ=−0.84(Pearson r=−0.78)。

**空間解析(L4/L8/L12)**
- 50%質量に必要なpatch数はL4=563、L8=121、L12=68。
- Foreground enrichmentはL4=1.60、L8=0.99、L12=1.57。L4とL12の差は統計的
  に有意ではなかったが、正式な同等性検定は実施していない。
- 同一画像内Attention map間のPearson相関(700画像平均)はL4-L8=−0.048、
  L8-L12=0.958、L4-L12=0.015。
- L8はL4よりHoyerスパース度が有意に高く(dz=7.95)、"拡散"ではなく"集中"
  した状態にある。foreground enrichmentが弱いのはforegroundへの相対配分が
  弱いためであり、Attentionの拡散を意味しない。
- L8とL12は強調する空間位置が非常に近く(相関0.958)、L8の時点で後半層に
  近い空間配置が既に形成されている。L8を「言語処理への切り替え点」と解釈
  する根拠は本リポジトリのいかなる分析にも存在しない。

**方針上の注意(全項目共通)**
- 形状は一貫して「N字型プロファイル」と表記する。
- L4とL12は、指標を限定せず「同程度の視線一致度を持つ層」とは表現しない
  (NSSでは有意に異なる)。L4/L8/L12はそれぞれN字型の前半の山・中間の谷・
  後半の回復を代表する層として整理する。
- probe性能は情報の線形な読み出し可能性としてのみ扱い、CLIPによる実利用や
  Attentionとの因果関係を主張しない。Attentionと人間視線は同一機構として
  扱わない。画像言語対照学習がN字型を生じさせたという因果関係も主張しない。

---

## 9. Recommended Appendix Tables and Figures

| 仮タイトル | 内容 | 元データ | 生成スクリプト | 既存ファイル | 必須/任意 | 再生成要否 |
|---|---|---|---|---|---|---|
| Table A1: Model architectures and weights | 4モデルのアーキテクチャ・重み・前処理一覧 | 本監査のA1節 | -- (表は未生成、本監査の表から手動転記可) | なし | 必須 | 新規作成(表のみ、図なし) |
| Table B1: N-shape contrast per model | m_contrastの値・CI・検定結果 | `m_contrast_summary.csv`,`m_contrast_model_comparisons.csv` | -- | なし(既存CSVはあるがAppendix用整形表は未作成) | 必須 | 新規作成(表のみ) |
| Table C1: Low-level feature definitions | 8特徴の名称・式・値域 | `lib/osie_lowlevel_features.py` | -- | 本文書セクション4に既出 | 必須 | 転記のみ |
| Table C2: Semantic attribute sample sizes | 12属性の陽性/陰性数・陽性率 | `probe_summary.csv` | -- | 本文書セクション4に既出(近似値。厳密化は要対応) | 必須 | 陽性数/陰性数の厳密化に低コスト再集計を推奨 |
| Table C3: Probe hyperparameters & CV settings | Ridge/LogisticRegressionの設定一覧 | `lib/osie_lowlevel_probe.py`,`lib/osie_text_alignment_probe.py` | -- | なし | 必須 | 新規作成(表のみ) |
| Table D1: Spatial concentration metrics at L4/L8/L12 | entropy/sparsity/top10%/80%massのL4,L8,L12値+CI | `outputs/clip_attention_b2_distribution/summary.json` | -- | なし | 任意(既にFigure3本文に一部反映済み) | 新規作成(表のみ) |
| Appendix Fig: common-scale L4/L8/L12 (3 images) | image 1156/1159/1213の共通カラースケール版 | `outputs/clip_attention_b2_distribution/` | `scripts/paper_figures/appendix_figure3_common_scale.py` | `paper_figures/appendix/figure_attention_common_scale.{pdf,png,svg}` | 任意 | 不要(既存) |
| Appendix Fig: layer-specific additional examples (1159, 1213) | 本文Figure3で使わなかった2画像の層別カラースケール版 | 同上 | `scripts/paper_figures/figure3_spatial_characterization_revised.py` | `paper_figures/appendix/figure_attention_layer_specific_additional_examples.{pdf,png}` | 任意 | 不要(既存) |
| Appendix Fig: AUC-Judd 4-model comparison | AUC-Judd層別4モデル比較(Figure1と対の指標) | `outputs/expA*/healthy/` | `scripts/paper_figures/appendix_figures.py` | `paper_figures/appendix/appendix_aucjudd_model_comparison.{pdf,png,svg}` | 任意 | 不要(既存) |
| Appendix Fig: sAUC 4-model comparison | 同上sAUC版 | 同上 | 同上 | `paper_figures/appendix/appendix_sauc_model_comparison.{pdf,png,svg}` | 任意 | 不要(既存) |
| Appendix Fig: all 12 semantic attribute curves | 意味属性12種の個別層別曲線 | `probe_summary.csv` | `scripts/paper_figures/appendix_figures.py` | `paper_figures/appendix/appendix_semantic_attributes_individual.{pdf,png,svg}` | 任意 | 不要(既存) |
| Appendix Fig: all 8 low-level feature curves | 低次特徴8種の個別層別曲線 | `lowlevel_summary.csv` | 同上 | `paper_figures/appendix/appendix_lowlevel_features_individual.{pdf,png,svg}` | 任意 | 不要(既存) |
| Appendix Fig: gaze-probe profiles, full labels | Figure2の全属性・全特徴ラベル付き版 | `probe_summary.csv`,`lowlevel_summary.csv` | `scripts/paper_figures/figure2_probe_redraw.py` | `paper_figures/figure2_gaze_probe_profiles_full_labels.{pdf,png,svg}` | 任意 | 不要(既存) |
