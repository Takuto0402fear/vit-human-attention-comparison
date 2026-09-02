# Method実装監査 (METHOD_AUDIT.md)

本監査は既存コード・既存出力ファイルの読解のみに基づく。新規のモデル推論・
再学習・再抽出は一切行っていない。すべての結論は「コード上の根拠」欄に示す
ファイル・関数・行番号（2026-08-29時点のリポジトリ状態）で検証可能である。
推測が必要だった箇所は明示的に「確認できなかった点」に記載し、断定していない。

**前提の確認**: リポジトリ内に既存のMethod草稿ファイル（.tex / .md / .docx等）
は見つからなかった（`find . -iname "*.tex" -o -iname "*method*.md" -o
-iname "*draft*" -o -iname "*manuscript*"` が空を返す）。したがって「現在の
Method暫定稿から修正すべき文章」は、リポジトリ外（Overleaf等）に存在する草稿
との照合ではなく、**一般的にMethod節が誤りやすい/曖昧になりやすい記述**を
本監査で確定した事実と対比する形で提示する。実際の草稿がある場合は、本書の
「確定事項」と「英語/日本語確定表現」を直接その節と突き合わせて修正されたい。

---

## 最優先事項: 意味属性probe・低次視覚特徴probeの入力特徴テンソル

### 1. 特徴抽出のファイル名・関数名・行番号

**確定事項**: 両probeとも同一の抽出パスを使用する。

- 抽出関数: `lib/clip_vit_hidden.py:27` `visual_forward_with_hidden_states()`
- 呼び出し元（semantic probe用の生特徴を作る抽出スクリプト）:
  `scripts/extract_osie_probe_all12_full700_features.py:224-226`
  ```python
  hidden_states, _ = visual_forward_with_hidden_states(
      model.visual, tensor.type(model.dtype), pos_embed=pos_interp,
      layers_zero_based=LAYERS_ZERO_BASED)
  ```
- 低次視覚特徴probeは、**この抽出スクリプトが生成した同一npzファイルを再利用**
  しており、独自の特徴抽出は行っていない（後述、根拠は項目9）。

**使用した出力ファイル**: `outputs/osie_probe_all12_full700/object_features_all12.npz`
（`raw_L1`〜`raw_L12`キー、object毎にpooling済み768次元ベクトル）。

### 2. hookまたは保存対象となる具体的なテンソル

**確定事項**: PyTorchのforward hookは使用していない。CLIPの`ResidualAttention
Block.forward`をそのまま呼び出し（`block(x)`、再実装なし）、そのブロックの
**戻り値そのもの**（= 12層それぞれのtransformer blockの出力トークン列、
LayerNorm前・projection前の生の残差ストリーム）を保存している。

**コード上の根拠**:
- `lib/clip_vit_hidden.py:79-82`
  ```python
  for i, block in enumerate(visual.transformer.resblocks):
      x = block(x)                                       # official forward, unmodified
      if i in wanted:
          hidden_states[i] = x.permute(1, 0, 2).clone()   # LND -> NLD, (B, seq, width)
  ```
- `block`はインストール済み`clip`パッケージ（`clip.model.ResidualAttentionBlock`、
  `site-packages/clip/model.py`）のインスタンスで、そのforwardは:
  ```python
  def forward(self, x):
      x = x + self.attention(self.ln_1(x))
      x = x + self.mlp(self.ln_2(x))
      return x
  ```
  （`python -c "import inspect,clip.model as m; print(inspect.getsource(m.ResidualAttentionBlock))"`
  で2026-08-29時点のインストール済みパッケージから直接確認済み。）

### 3. Transformer block 入力／出力のどちらか

**確定事項**: **出力**（block `i`のforward計算が完了した直後の状態 = block
`i+1`への入力と等価）。「L{i}」という表示層番号は、この0-indexedブロック
`i-1`の出力に対応する（`extract_osie_probe_all12_full700_features.py:63-65`
で `LAYERS_DISPLAY = 1..12`, `LAYERS_ZERO_BASED = 0..11`, および
`assert [d-1 for d in LAYERS_DISPLAY] == LAYERS_ZERO_BASED` により、表示層と
0-indexedブロックの対応が明示的に検証されている）。

### 4. LayerNormの前後

**確定事項**: 保存されるテンソル自体には、そのブロックの出力として追加の
LayerNormは適用されていない（`ln_post`は最終層の[CLS]トークンにのみ、公式
forwardの最後で1回だけ適用され、`visual_forward_with_hidden_states`の戻り値
`hidden_states`には一切含まれない）。ただし、ブロック内部では標準のPre-LN
Transformer構成として `ln_1`が attention の直前、`ln_2`が mlp の直前に適用
されている（上記forwardのソース参照）。保存テンソル = 「`ln_1`/`ln_2`適用後の
attention/mlp出力を残差加算した後の状態」であり、それ自体に対する追加の
LayerNormは無い。

**コード上の根拠**: `lib/clip_vit_hidden.py`の関数docstring(50-52行目)
「the RAW residual-stream output after that block (before ln_post/proj)」、
および `extract_osie_probe_all12_full700_features.py`が書き出す
`config.json`の`"note"`フィールド（後述、項目10で引用）。

### 5. Attention残差加算の前後

**確定事項**: **後**（`x = x + self.attention(self.ln_1(x))` が実行済みの
状態）。抽出関数はこの1行を含む`block(x)`全体を1回だけ呼ぶので、attention
残差加算の前の状態を個別に保存することはしていない。

### 6. MLP残差加算の前後

**確定事項**: **後**（`x = x + self.mlp(self.ln_2(x))` まで実行済みの状態、
= ブロックの最終出力）。

### 7. 保存時のテンソルshape

**確定事項**:
- 抽出直後（CLSトークン込み）: `(batch=1, seq_len, width=768)`,
  `seq_len = 1 + grid_h*grid_w`。
- CLS除外・patchグリッドへのreshape後（object pooling前の中間形状）:
  `(38, 50, 768)` float64
  （`extract_osie_probe_all12_full700_features.py:232`
  `patch_tokens = hs[1:].detach().cpu().numpy().reshape(gh, gw, 768)`,
  `gh, gw = grid_hw = (38, 50)`）。
- object単位でmask加重平均した後、npzに保存される最終shape:
  `raw_L{l}` は `(n_objects=5551, 768)` float32
  （`extract_osie_probe_all12_full700_features.py:289`
  `save_dict[f"raw_L{l_disp}"] = np.stack(raw_feats[l_disp]).astype(np.float32)`）。

### 8. CLS tokenを除外するタイミング

**確定事項**: 抽出直後、patchグリッドへreshapeする**前**に除外している。
`hs = hidden_states[l_zero][0]` （`(seq_len, 768)`, CLSはindex 0）に対し、
`hs[1:]`でCLSを除いてから `(gh, gw, 768)` にreshapeする
（`extract_osie_probe_all12_full700_features.py:231-232`）。CLSトークン自体
の値はpooled object特徴には一切混入しない。

### 9. 意味属性probeと低次特徴probeで同じ抽出経路を使っているか

**確定事項**: **YES、完全に同一の特徴ファイルを共有している**（入力Xに関して）。

**コード上の根拠**（両スクリプトが同一パスを指す）:
- `scripts/probe_osie_all12_full700.py:60`
  `OBJECT_FEATURES_NPZ = os.path.join(OUT_DIR, "object_features_all12.npz")`
  （`OUT_DIR = outputs/osie_probe_all12_full700`）
- `scripts/probe_osie_lowlevel_all12_full700.py:54`
  `OBJECT_FEATURES_NPZ = r"C:\Users\user\gaze\outputs\osie_probe_all12_full700\object_features_all12.npz"`

**ただし、目的変数（ラベルY）の生成経路は全く別である**（低次特徴probeのみ、
項目「低次視覚特徴の入力解像度」を参照）。意味属性probeのラベルは
OSIE `attrs.mat`の人手アノテーションをそのまま使い、画像処理は一切介在しない。
低次視覚特徴probeのラベルは、CLIPの標準前処理（Resize+CenterCrop 224x224）
で作った別画像から独自に計算されている。**特徴(X)は共通、ラベル(Y)の生成
画像ジオメトリは異なる**、という点を正確に区別してMethodに記載する必要がある。

### 10. 最終LayerNormやCLIP画像投影を適用しているか

**確定事項**: **適用していない**。両probeとも「RAW（un-projected）」特徴を使用。

**コード上の根拠**: `outputs/osie_probe_all12_full700/config.json`の
`"note"`フィールド:
> "RAW hidden states only (no ln_post/proj), pooled per object per layer;
> full patch grids are never written to disk -- discarded after pooling,
> per image."

なお `lib/clip_vit_hidden.py:92-111` に `project_and_normalize_tokens()`
という、任意層のトークンに`ln_post`+`proj`を適用する関数が別途存在するが、
これは同モジュールのdocstringで明記されている通り「text-alignment analysis
専用のlogit-lens的分析」のためのものであり、**両probeのX特徴には一切使われ
ていない**（`lib/clip_vit_hidden.py:104-105`「Never used for the linear-probe
features (see scripts/extract_osie_text_alignment_features.py)」）。

### 11. 物体マスクとの重なり率による重み付き平均の実装

**確定事項**:
1. `lib/osie_text_alignment.py:40-55` `mask_to_patch_weights(mask, patch_size)`
   — objectマスク(H,W)をpatch_size(16)倍数へゼロパディング後、各patchに
   reshapeし、`weights = reshaped.mean(axis=(1,3))` でpatch内マスク被覆率
   （0〜1の連続値、面積比）を計算。
2. `lib/osie_text_alignment.py:62-78` `weighted_pool_tokens(tokens, weights,
   min_weight_sum=1e-6)` — `(t * w[:,None]).sum(axis=0) / w.sum()` の加重平均。
   `w.sum() < min_weight_sum` の場合は`None`を返し、呼び出し側でスキップ
   （0埋めしない）。

**使用した出力ファイル**: `outputs/osie_probe_all12_full700/config.json`
（`n_objects_skipped_zero_weight: 0` — 全700画像・5551 objectでこの閾値に
かかったものは無かったことを確認済み）。

### 12. 視線一致解析用Attentionの抽出経路との違い

**確定事項**: 両probeのX抽出（hidden states, `lib/clip_vit_hidden.py`）と、
NSS/AUC-Judd/sAUC用のAttention抽出（`lib/clip_vit.py`）は、**全く別の関数
であり、保存するテンソルの種類そのものが異なる**。

| | probe (X特徴) | 視線一致 (NSS/AUC-Judd/sAUC) |
|---|---|---|
| モジュール | `lib/clip_vit_hidden.py` | `lib/clip_vit.py` |
| 関数 | `visual_forward_with_hidden_states` | `visual_forward_with_cls_patch_attention` / `resblock_forward_with_attention` |
| block呼び出し | `block(x)`（公式forward、無改変） | `ln_1`→`attn(need_weights=True)`→残差→`ln_2`→`mlp`→残差、を手動で再実装（attention重み取得のため） |
| 保存対象 | 残差ストリームの隠れ状態（(seq,768)） | softmax後のattention確率行列の[CLS]行（(grid_h,grid_w)） |
| 何を測るか | 「何が線形に読み出せるか」 | 「どこを見ているか」 |

**コード上の根拠**: `lib/clip_vit.py:22-51` `resblock_forward_with_attention`
のdocstring「Same computation order as clip.model.ResidualAttentionBlock.
forward... but requests need_weights=True instead of False」。

---

## Methodその他監査項目

### A. 各分析の入力解像度・resize・crop・padding・位置埋め込み補間

**確定事項（2系統に分かれる。取り違えるとMethod記述が誤りになる最重要点）**:

**系統1: OSIE原寸（800x600）ベース** — 視線一致(NSS/AUC-Judd/sAUC)、B-2空間
解析、両probeの入力特徴(X)すべてがこちら。
- resize・cropは**一切行わない**。
- 下端・右端のみゼロパディングでpatch_size(16)の倍数化: 600→608 (+8px),
  800→800 (変更なし)。結果のpatchグリッドは`(38, 50)`。
  根拠: `clip_extractor.py:100-133` `load_osie_image_tensor()`,
  `vit_extractor.py::_pad_to_patch`を読み取り専用で再利用（同一ジオメトリ）。
- 位置埋め込み: CLIPネイティブな14x14グリッド用の位置埋め込みを、
  `torch.nn.functional.interpolate(..., mode="bicubic", align_corners=False)`
  で38x50グリッドへ補間。CLSトークンの位置埋め込み（1行目）は補間せずそのまま
  保持。根拠: `lib/clip_vit.py:191-226` `interpolate_patch_pos_embed()`。
- 正規化: CLIP公式のmean/std
  `(0.48145466,0.4578275,0.40821073)` / `(0.26862954,0.26130258,0.27577711)`
  （`clip_extractor.py:40-41`）。

**系統2: CLIP標準前処理（224x224、Resize+CenterCrop）** — **低次視覚特徴
probeの目的変数(Y)の計算にのみ**使用。
- `clip.load()`が返す公式`preprocess`（`Compose([Resize(224, bicubic),
  CenterCrop(224,224), ...])`、2026-08-29時点でインストール済みのclipパッケージ
  から直接確認: `Resize(size=224, interpolation=bicubic, antialias=True)` →
  `CenterCrop(size=(224,224))`。
- objectマスクも同じ`Resize`（**nearest**補間に変更）+`CenterCrop`で224x224へ
  変換。根拠: `scripts/extract_osie_lowlevel_targets.py:111-120`
  `build_mask_pipeline()`。
- **含意（要Method記載）**: OSIE画像は800x600（4:3横長）。CLIPのResize(224)
  は短辺(600→224)基準で拡大縮小するため、幅は600x(800/600)=約224*(800/600)
  ≈299pxに、そこから中央224x224をCenterCropする。つまり**画像の左右の縁が
  切り落とされる**。この結果、マスクが224x224の中央領域に一切かからない
  objectは、低次視覚特徴の目的変数からは欠測となる
  （`outputs/osie_lowlevel_targets_full700/sanity_checks.json`:
  `"n_objects_total": 5551, "n_objects_valid": 5263, "n_objects_skipped": 288`
  — 全objectの約5.2%が、この224x224クロップ幾何学的制約のみを理由に除外
  されている）。**入力特徴(X)は系統1（クロップなし）で計算されるため、
  同じobjectでもXとYで「見えている画像領域」が異なりうる**。これは意味属性
  probeには影響しない（意味属性ラベルはOSIE人手アノテーションであり、画像
  処理を経由しない）。

**使用した出力ファイル**: `outputs/osie_lowlevel_targets_full700/config.json`,
`sanity_checks.json`。

### B. 実際に使用した画像数・物体数・除外条件

**確定事項**:
- 画像数: 全分析で700枚固定（`outputs/osie_attribute_grounding_full700/
  full700_image_list.csv`由来、全スクリプトが「参照するだけで再選定しない」）。
- object総数: 5551（OSIE `attrs.mat`由来）。
- 意味属性probe: 5551/5551 使用（0件除外、`n_objects_skipped_zero_weight: 0`）。
- 低次視覚特徴probe: 5263/5551 使用（288件除外、上記の224x224クロップ幾何学
  的理由。加えてtarget単位で`len(idx) < 10`なら学習自体をスキップする閾値も
  存在するが、8ターゲットいずれも該当しなかった — `lowlevel_summary.csv`に
  全8ターゲット×12層の行が揃っていることから確認）。

### C. Attention headの集約方法

**確定事項**: PyTorch `nn.MultiheadAttention`の`average_attn_weights=True`
引数により、12ヘッドの単純平均（重みなし一様平均）。
根拠: `lib/clip_vit.py:43-48`
`block.attn(ln1, ln1, ln1, need_weights=True, average_attn_weights=True, ...)`。
ヘッド数=12は`clip_extractor.py:33` `CLIP_HEADS = 12`。

### D. raw attentionかattention rolloutか

**確定事項**: **raw attention**（単一層のsoftmax出力そのもの）。Attention
rolloutは実装されていない。リポジトリ全体を`grep -rn "rollout"`で検索し、
一致なしを確認済み（2026-08-29時点）。

### E. Attention mapの補間・平滑化・正規化

**確定事項（視線一致メトリクス用と、B-2/Figure3の空間解析用で扱いが異なる）**:

- **視線一致(NSS/AUC-Judd/sAUC)用**: 生の(38,50)attentionを
  `F.interpolate(..., mode="bilinear", align_corners=False)` で(600,800)へ
  アップサンプル後、`resized[l] /= resized[l].sum()`で合計1に再正規化してから
  メトリクス計算に渡す（`scripts/run_expA_clip_full700.py:169-172`）。
  ガウシアン平滑化などの追加ぼかしは適用しない。
- **B-2空間解析(50%/80%質量、foreground enrichment、層間相関)用**: 生の
  (38,50)ネイティブグリッド上で全指標を計算し、アップサンプルは一切行わない
  （`lib/clip_attention_b2.py`の全関数がnative gridを直接受け取る設計、
  `scripts/compute_clip_attention_b2_full700.py`のdocstring・実装より確認）。
  Figure3等の可視化で行うアップサンプルは表示専用であり、指標計算には使われ
  ない。
- **人間視線側**: `heat_all`（Gaussianブラー sigma=24px、`scripts/
  generate_fixmaps.py:33,101-108`）は可視化専用。NSS/AUC-Judd/sAUCの計算では
  ブラーされていない離散fixation点（`points`キー、`np.where(d["points"])`）
  を使う。この区別（メトリクス=離散点、可視化=ブラー済みヒートマップ）は
  Methodに明記すべき。

### F. NSS、AUC-Judd、sAUCの実装と集約単位

**確定事項**:
- 実装: `metrics.py`（root）。`nss()`(24-45行), `auc_judd()`(52-85行),
  `sauc()`(92-132行)。
- 集約単位: **(image, layer)ごとに1スカラー**。fixationは同一image内の
  healthy群「全被験者」を**プールした1点群**として扱う（被験者ごとの計算・
  平均ではない）。根拠: `scripts/run_expA_clip_full700.py`の
  `load_fixation_data()`が`points`キー（全被験者の2値マップ）から座標を
  一括抽出し、被験者IDを一切追跡していない。
- `metrics_by_layer.csv`の`*_mean`/`*_std`は、この(image,layer)単位のスコア
  を**700画像にわたって**平均・SD（母標準偏差ではなく`ddof`指定なしのnumpy
  既定=ddof=0であることに注意——`aggregation.py`の`summarise_group_layer`は
  `ddof=1`を使うが、これは別パイプライン(旧README記載の健常/統合失調症比較
  用)であり、CLIP/DINO/DeiTのfull700本番スクリプトは独自にnumpy集計している
  ため、**両者のSD定義が同一かどうかは本監査で個別に確認していない**——
  「確認できなかった点」参照）。

### G. sAUCの負例生成方法・反復回数・seed

**確定事項**:
- 負例プール: **同じ700画像プール内の全fixation点**（自画像を除外しない）。
  `scripts/run_expA_clip_full700.py`冒頭docstring 8-9行目
  「identical sAUC negative-pool convention (one shared 700-image pool, no
  self-image exclusion, metrics.py's internal RandomState(0) untouched)」。
  DINO-S/DINO-B/DeiT-SLの全スクリプトも同一パターン（`sauc(sal, pts,
  all_pool)`、`all_pool`に自画像を除外するコードなし、`scripts/run_expA*.py`
  一式で`sauc(`呼び出しをgrepし全10箇所で確認）。
- 反復回数: `n_splits=10`（`metrics.py:96`のデフォルト値、全呼び出しで
  override無し、`grep -rn "sauc("`で確認）。
- seed: **0固定**（`metrics.py:119` `rng = np.random.RandomState(0)`、
  関数内で1度だけ生成し10分割全てに使い回す。リポジトリの他の統計処理で
  広く使われる`seed=42`とは異なる値であることに注意——これは意図的な設計か
  単なる既定値放置かは、コードのコメントからは判断できない。「確認できな
  かった点」参照）。

### H. N字contrast（M_contrast）の対象層・数式・bootstrap・検定

**確定事項**:
- 層ビン: `EARLY_LAYERS=[3,4,5]`, `MIDDLE_LAYERS=[7,8,9]`,
  `LATE_LAYERS=[11,12]`（`scripts/stats_model_comparison_mcontrast.py:66-68`）。
- 数式（`scripts/stats_model_comparison_mcontrast.py:244-249`）:
  ```
  early = mean(値, layers=[3,4,5])   # 画像ごとに計算（per-image）
  middle = mean(値, layers=[7,8,9])
  late = mean(値, layers=[11,12])
  m_contrast = (early + late) / 2 - middle
  early_drop = early - middle
  late_recovery = late - middle
  ```
  すなわちm_contrastは**画像単位のスカラー**として先に計算され、その後
  700画像にわたってbootstrap平均・95%CIが取られる（`outputs/
  model_comparison_statistics/m_contrast_per_image.csv`が画像単位の値を
  保持）。
- Bootstrap: seed=42, n_boot=10000, パーセンタイル法95%CI（`scripts/
  stats_three_model_layerwise.py:77-81` `BOOT_SEED=42, N_BOOT=10000,
  ALPHA=0.05`、同一定数が`stats_model_comparison_mcontrast.py`でも踏襲されて
  いることを`statistics_summary.md`冒頭の記述
  「Bootstrap: seed=42, n_boot=10000. Permutation: seed=42, n_perm=10000.」
  で確認）。
- 検定: Wilcoxon符号順位検定 + 効果量として signed rank-biserial r
  （`statistics_summary.md`列見出し「r (signed)」）、複数比較補正は**Holm法**
  （`scripts/stats_model_comparison_mcontrast.py:207` `holm_correction()`
  関数、familyの単位は比較の種類ごとに異なる——例:「3モデル×3指標=9検定を
  1つのfamily」等、`holm_families`として`config.json`的な出力に明記されて
  いる）。B-2空間解析側は別に**BH-FDR**（Benjamini-Hochberg）を使用しており
  （`lib/clip_attention_b2.py:449-471` `bh_fdr()`、`scripts/
  stats_clip_attention_b2_full700.py`のFamily1/Family2）、**補正手法が分析
  系統によって異なる（モデル間比較=Holm、B-2空間分析=BH-FDR）**点は正確に
  書き分ける必要がある。有意水準はいずれも`alpha=0.05`。

**確定値**: CLIP-B の m_contrast（NSS） mean=+0.8826, 95%CI=[+0.8590,
+0.9065]（`outputs/model_comparison_statistics/m_contrast_summary.csv`）。

### I. 低次8特徴の名称・計算式・値域

**確定事項**（すべて`lib/osie_lowlevel_features.py:26-88`で code-verified）:

| 特徴名 | 計算式 | 値域(理論上) |
|---|---|---|
| mean_luminance | 0.2126R+0.7152G+0.0722B（ITU-R BT.709）のmask内平均 | [0,1] |
| mean_red / mean_green / mean_blue | 各chの mask内平均 | [0,1] |
| mean_saturation | `matplotlib.colors.rgb_to_hsv`のS成分、mask内平均 | [0,1] |
| luminance_contrast | luminanceのmask内標準偏差 | [0, ~0.5] |
| edge_strength | Sobel勾配強度`sqrt(gx^2+gy^2)`（`scipy.ndimage.sobel`, 全画像に適用後mask内平均） | [0, ∞) 実用上小さい正値 |
| fine_texture | Laplacian（`scipy.ndimage.laplace`, 全画像に適用後）のmask内**分散** | [0, ∞) |

いずれもSobel/Laplacianは**マスクでクロップする前の全画像**に適用してから
mask内で集約する（マスク境界の人工的エッジを避けるため、
`lib/osie_lowlevel_features.py`冒頭docstring明記）。除外した3つの幾何学的
コントロール変数（mask_area_fraction, centroid_x, centroid_y）は低次視覚
特徴には含めていない。

### J. Ridgeのalpha、入力・正解値の標準化

**確定事項**:
- `Ridge(alpha=1.0)`（デフォルト値、CVによるハイパーパラメータ探索は行って
  いない）。`lib/osie_lowlevel_probe.py:39-40` `make_ridge_pipeline`。
- 標準化: `Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha))])`。
  **入力特徴Xのみ標準化**し、目的変数Y（低次視覚特徴の生値）は標準化しない
  （Ridgeにy側の標準化オプションは無く、コード上も渡していない）。R2は生の
  Y単位で計算される。

### K. 意味属性probeのC、class_weight、solver

**確定事項**: `LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
max_iter=5000, random_state=seed)`（`lib/osie_text_alignment_probe.py:44-45`）。
`solver`は明示的に指定されていない → 使用中のsklearn (1.7.2, 2026-08-29時点
のインストール環境で確認) の既定値である`lbfgs`が実際には使われている。
Method記載時は「solverは指定なし（sklearn既定=lbfgs）」と明記するのが正確。

### L. 画像ID単位5-fold分割と、StandardScalerをtrain foldだけでfitしているか

**確定事項**: 両方YES。
- 意味属性probe: `StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
  random_state=seed)`、`groups=image_ids`（`lib/osie_text_alignment_probe.py:
  133,160-161`）。同一image_idが同一foldのtrain/testに分裂しないことを
  明示的にassertで検証（163-164行目、違反時はRuntimeErrorでSTOP）。
  `max_splits=5`がデフォルト（98行目）で、これが`probe_summary.csv`の
  `n_folds=5`に対応。
- 低次視覚特徴probe: `GroupKFold(n_splits=n_splits)`、`groups=image_ids`
  （`lib/osie_lowlevel_probe.py:132-136`、同様のリーク検証assert付き）。
- StandardScalerのtrain-onlyフィット: 両者ともsklearn `Pipeline`を用いて
  `pipe.fit(X[tr], y[tr])`のみを呼び、テストfoldへは`pipe.predict(X[te])`
  （内部で学習済みscalerのtransformのみ適用）。したがってスケーラの平均・
  分散はtrain foldからのみ推定され、test foldの情報は漏れない
  （`lib/osie_text_alignment_probe.py:174-177`,
  `lib/osie_lowlevel_probe.py:150-152`）。

### M. permutationの反復回数とseed

**確定事項（意味属性probeのみ；低次視覚特徴probeにpermutationベースラインは
実装されていない——`lib/osie_lowlevel_probe.py`に`permutation`という語は
出現しない、確認済み）**:
- Stage A: 全12層×13(attribute,condition)組で **20反復**
  （`scripts/probe_osie_all12_full700.py`冒頭docstring、
  `N_PERM_REPS_STAGE_A`）。
- Stage B: ピーク層+参照層のみ **100反復**へ増強（`N_PERM_REPS_STAGE_B`）。
- 各反復・各foldで「train側のラベルのみ」をシャッフルして再学習し、実際の
  test側ラベルで評価（`lib/osie_text_alignment_probe.py:216-228`
  `y_train_shuffled = perm_rng.permutation(y[tr])`）。test側は一切シャッフル
  しない。
- seed: `perm_rng = np.random.RandomState(seed)`（seed=42）を**1回だけ生成**
  し、全反復・全foldで使い回す（`lib/osie_text_alignment_probe.py:216`）。

### N. 50% mass、foreground enrichment、spatial correlationの正確な定義

**確定事項**（すべて`lib/clip_attention_b2.py`でcode-verified、既に
`paper_figures/FIGURE_AUDIT.md`にも記載済みだが、数式の完全な形をここに再掲）:

- **50%/80% mass patch数**: `coverage_for_mass(p, target_mass)`
  （92-108行目）。`p=raw/raw.sum()`を降順ソートし、累積和が`target_mass`
  以上に達する最小のpatch数。
- **foreground enrichment**: `foreground_conditional_mass / foreground_area_
  fraction`（162-214行目）。
  `foreground_conditional_mass = sum(p_i * coverage_i)`
  （`p=raw/raw.sum()`, `coverage`=objectマスクのpatch内被覆率）。
  `foreground_area_fraction = mean(coverage)`（画像全体に対するforeground
  面積比）。**1.0が「面積比に比例した一様配分」の基準線**であり、「chance」
  という表現より「面積比例配分（area-proportional allocation）」と呼ぶ方が
  正確（この言い換えは既にFigure3改訂で反映済み）。
  `foreground_area_fraction < 1/1900`の場合は`None`扱い（自明に縮退した画像
  は除外、`is_degenerate`フラグ）。
- **層間空間相関(Pearson)**: `pearson_corr(a, b) = np.corrcoef(...)`
  （221-231行目）を**各画像ごとに個別計算**（rawベクトル同士、1900次元）。
  `summary.json`の`same_image_mean`は、**PearsonのみFisher-z変換した上で
  700画像平均し、逆変換して報告**（`scripts/stats_clip_attention_b2_full700.
  py:247-252`）。cosine類似度とnormalized JSDは単純な算術平均（同253-257行
  目）。**中央値(`same_image_median`)はどの指標でも常に単純な中央値**
  （Fisher-z変換なし）。

### O. Pearson／Spearman、補間前後、画像単位の集約方法

**確定事項**:
- B-2空間解析のPearson: 上記N参照。**アップサンプル前のネイティブ(38,50)
  グリッド**上で計算（アップサンプルは表示専用、指標計算には一切使わない）。
- 低次視覚特徴probeのPearson/Spearman: `lib/osie_lowlevel_probe.py`の
  `compute_metrics()`（本ファイル冒頭70行目付近で言及、fold内のy_true/
  y_predから計算）——**fold単位のスカラー**として計算され、`lowlevel_
  fold_scores.csv`に1行/fold、`lowlevel_summary.csv`で12層×8特徴について
  5-fold平均・SDに集約。
- 意味属性probeにはPearson/Spearmanは使われていない（AUPRC/AUROC/balanced
  accuracyのみ）。

### P. 信頼区間・検定・多重比較補正・有意水準

**確定事項（分析系統ごとに整理、既出内容の一覧化）**:

| 分析 | CI/検定 | seed | n_boot/n_perm | 補正 | alpha |
|---|---|---|---|---|---|
| Figure1-2/appendixの4モデル比較（本監査対象の描画作業側、`scripts/paper_figures/`で新規に再集計） | 画像単位percentile bootstrap | 42 | n_boot=10000 | なし（単一モデルの記述統計として提示、モデル間差の検定はしていない） | 0.05 |
| モデル間 m_contrast 比較 (`stats_model_comparison_mcontrast.py`) | 画像単位bootstrap CI + Wilcoxon符号順位検定 | 42 | n_boot=10000, n_perm=10000 | Holm法（family定義は比較ごとに個別） | 0.05 |
| 意味属性/低次probeの層間差 (`image_level_bootstrap_auprc_diff` / `_r2_diff`) | OOF予測に対する画像単位bootstrap | 42 | n_boot=10000 | BH-FDR（呼び出し元スクリプトで後付け適用） | 0.05 |
| B-2空間解析 (`stats_clip_attention_b2_full700.py`) | 画像単位bootstrap CI + sign-flip検定 | 42 | n_boot=10000, n_signflip=10000 | BH-FDR（family=同一analysis内の全比較） | 0.05 |
| B-2 repeated-shuffle頑健性チェック | Monte Carlo p値（1000 derangement） | 42（`derangement_seed`で各回別seed導出） | n_deran=1000 | BH-FDR（9検定1family） | 0.05 |

---

## 確認できなかった点（推測せず、未解決として明記）

1. **sAUCのseed=0が意図的か既定値放置か**: `metrics.py`の`sauc()`/
   `auc_judd()`内部の`np.random.RandomState(0)`は、他の統計処理全般で使わ
   れる`seed=42`規約と異なる。コード・コミット履歴・コメントのいずれからも
   「なぜ0なのか」の明示的な理由は見つからなかった。Method執筆時は「0」と
   正確に書くべきで、42と混同しないこと。
2. ~~`aggregation.py`のddof=1平均/SD計算とfull700本番スクリプトの一致性~~
   **解決済み**: `scripts/run_expA_clip_full700.py:427`
   `row[f"{m}_std"] = float(np.std(vals))` は`ddof`未指定 =
   **母標準偏差(ddof=0)**。一方`aggregation.py::summarise_group_layer`は
   `np.std(..., ddof=1)`（不偏標準偏差）。**両者は異なるddof規約**であり、
   Figure1等の描画で`comparison_values.csv`の`NSS_std`列（=`metrics_by_
   layer.csv`由来）を直接使う場合はddof=0由来であることに注意
   （本タスクの`paper_figures/`側は既存の`std`列を使わず700画像の生データ
   から自前でbootstrap CIを再計算しているため、この差異はFigure1-2の数値
   には影響していない）。
3. ~~意味属性/低次probeの層間差bootstrapに多重比較補正が適用されているか~~
   **解決済み**: `image_level_bootstrap_auprc_diff`自体は生のp値のみを返す
   が、呼び出し元`scripts/probe_osie_all12_full700.py:175-177`が
   `benjamini_hochberg_fdr()`（`lib/osie_text_alignment_probe.py:389`）で
   その分析内の全p値に対しBH-FDR補正を後付けしている
   （`p_fdr_bh`列として`probe_layer_differences.csv`に保存）。よって、
   ここもBH-FDR系統（モデル間比較のHolm系統とは異なる）に分類される。
4. **Method暫定稿そのものが見つからない**: 上述の通り、比較対象となる既存
   のMethod草稿ファイルはリポジトリ内に存在しない。Overleaf等リポジトリ外
   に草稿がある場合は、本書の確定事項と直接突き合わせる必要がある。
5. **`outputs/osie_probe_full700/`（12層拡張前の旧L4/L8/L12版）とAppendix
   図で使われている全12層版の間の関係**は、`extraction_reproducibility_
   check.json`で数値一致が検証済み（最大絶対誤差 5.6e-8〜4.7e-7）ではある
   が、どちらの版が最終的にpaper内の数値として採用されるべきかは、
   `paper_figures/FIGURE_AUDIT.md`側の合意（全12層版=`osie_probe_all12_
   full700`を採用）と重複確認しておくことを推奨する。

---

## Method本文へ転記できる確定表現

### 日本語

> 意味属性および低次視覚特徴の線形probeは、いずれもOpenAI CLIP ViT-B/16
> （`clip`パッケージ, ViT-B/16）の各Transformer blockの**出力**（Attention
> 残差加算・MLP残差加算の両方が完了した後の残差ストリーム、Pre-LN構成の
> 内部LayerNormはブロック内部でのみ使用され、ブロック出力自体には追加の
> LayerNormや最終画像投影（`ln_post`/`proj`）は適用しない）を、CLSトークン
> を除いた38×50パッチトークンについて、各objectマスクのパッチ内被覆率を
> 重みとした加重平均でpoolingしたものを入力特徴として用いた。入力画像は
> OSIE原寸（800×600）を下端・右端のみゼロパディングしてpatchサイズ(16)の
> 倍数（608×800）に揃えたものであり、リサイズやセンタークロップは行って
> いない。パッチ位置埋め込みは元の14×14グリッド用の埋め込みを双三次
> (bicubic)補間して38×50グリッドへ拡張した（CLSトークンの位置埋め込みは
> 補間せずそのまま使用）。人間視線とのAttention一致度解析（NSS/AUC-Judd/
> sAUC）で用いる[CLS]→patch Attentionは、これとは別に、各Transformer
> blockのマルチヘッド自己注意をhead平均した生のsoftmax確率（Attention
> rolloutではなく単層の生Attention）を直接使用しており、probeの入力特徴
> （残差ストリームの隠れ状態）とは異なるテンソルである。低次視覚特徴probe
> の目的変数（8種の連続値ターゲット）のみ、CLIP公式前処理（Resize(224,
> bicubic)+CenterCrop(224×224)）で作成した別ジオメトリの画像から計算して
> おり、これは入力特徴の抽出に使う原寸画像とは異なる（そのため一部の
> objectがこのクロップにより目的変数から除外されている: 5551件中288件）。

### 英語

> Both the semantic-attribute and low-level visual-feature linear probes
> use, as their input features, the **output of each Transformer block**
> of OpenAI's CLIP ViT-B/16 (the `clip` package's official, unmodified
> `ResidualAttentionBlock.forward`) -- i.e., the residual stream after
> both the attention-residual and the MLP-residual additions have been
> applied. The Pre-LN block's own internal LayerNorms (`ln_1`, `ln_2`)
> are used only inside the block's self-attention/MLP sublayers; no
> additional LayerNorm or the final image projection (`ln_post`/`proj`)
> is applied to the saved block output. The CLS token is dropped before
> the remaining 38x50 patch tokens are mask-weighted-averaged (each
> patch weighted by its fractional area coverage under the object's
> mask) to produce one pooled feature vector per object per layer.
> Input images are at OSIE's native resolution (800x600), zero-padded
> on the bottom/right only to the nearest multiple of the patch size
> (16), yielding a 608x800 padded image (38x50 patches); no resize or
> center-crop is applied for this pathway. Patch position embeddings
> are bicubic-interpolated from CLIP's native 14x14 grid to the 38x50
> grid (the CLS position embedding is kept unchanged). The [CLS]-to-
> patch attention used for the human-gaze-alignment analysis (NSS/
> AUC-Judd/sAUC) is extracted through a separate code path: it is the
> raw, head-averaged softmax attention probability from each individual
> layer's own multi-head self-attention (not attention rollout), which
> is a different tensor from the probes' residual-stream input
> features. Only the low-level visual-feature probe's regression
> targets (the 8 continuous targets) are computed on a differently
> pre-processed image -- CLIP's official Resize(224, bicubic) +
> CenterCrop(224x224) -- distinct from the native-resolution image used
> to extract the input features; as a consequence, 288 of the 5551
> OSIE objects are excluded from the low-level targets because their
> mask falls entirely outside this 224x224 center crop.
