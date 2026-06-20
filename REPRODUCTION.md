# Reproduction Notes

Reproduction of **Fig. 2c** from Yamamoto et al., *"Emergence of Human-Like Attention in Self-Supervised Vision Transformers"* on a consumer GPU.

## Environment

| Component | Version / Details |
|---|---|
| OS | Windows 11 Pro (10.0.26200) |
| GPU | NVIDIA GeForce GTX 1060 6 GB (Pascal, SM 6.1) |
| Python | 3.11 (conda) |
| PyTorch | 2.8.0+cu126 |
| torchvision | 0.23.0+cu126 |
| CUDA toolkit | cu126 (last build supporting sm_61) |
| Key packages | kornia 0.8.3, numpy 2.4.4, scipy 1.17.1 |

Full package list: `requirements_frozen.txt` (pip freeze) and `environment.yml`.

### Environment setup

```bash
conda create -p D:/conda_envs/vit-repro python=3.11 -y
conda activate D:/conda_envs/vit-repro
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Data sources

| Data | OSF project | Download |
|---|---|---|
| Trained model weights (36 checkpoints) | [c3snp](https://osf.io/c3snp/) | `trained_model_weights/{dino,supervised}/{01..06}/{4,8,12}layers/checkpoint.pth` |
| Nakano 2010 dataset (frames, gaze, keypoints) | [x6f8t](https://osf.io/x6f8t/) | `dataset/Nakano_etal_2010/` |
| Demo ensemble models | [c3snp](https://osf.io/c3snp/) | `demo/demo/model_data/*.pth` |

### Weight file reorganization

OSF distributes weights as flat files (`dino_01_04layers.pth`, etc.). These were renamed to match the code's expected directory structure:

```
trained_model_weights/
  dino/
    01/4layers/checkpoint.pth    # from dino_01_04layers.pth
    01/8layers/checkpoint.pth    # from dino_01_08layers.pth
    ...
  supervised/
    01/4layers/checkpoint.pth    # from supervised_01_04layers.pth
    ...
```

## Code changes

### 1. `analysis_python/utils_analysis.py` — Checkpoint format compatibility

**Lines 76-90**: Added fallback for OSF direct state_dict format. The original code expected training checkpoint format with `checkpoint["teacher"]` key, but OSF weights are bare state_dicts.

```python
if "teacher" in checkpoint:
    # Full training checkpoint format (original)
    ...
else:
    # Direct state_dict format (e.g. from OSF release)
    model_state_dict = checkpoint
```

Same pattern applied for supervised models (line 96): `checkpoint["model"] if "model" in checkpoint else checkpoint`.

### 2. `analysis_python/get_model_gaze_pos_N2010.ipynb` — Batch size reduction

**Cell `38a6163d`**: `batch_size = 32` → `batch_size = 4`

Frames are 720x480 pixels = 1350 tokens at patch_size=16. With depth=12, attention tensors at batch_size=32 would require ~16.8 GB VRAM, exceeding the GTX 1060's 6 GB. Batch size 4 keeps peak usage at ~2.9 GB.

### 3. `analysis_python/visualize_gaze_points_N2010.ipynb` — OSF data compatibility

Three categories of changes:

**(a) Missing files (commented out):**
- Cell `dc94e5b7`: Commented out `vit_official_gaze_pos.npz` and `gbvs_gaze_pos.npz` loads (files not included in OSF release; not used in active code paths since mds_dist only covers 6-model ensembles)
- Cell `a92bb663`: Commented out `gbvs_gaze_pos["labels"]` check

**(b) OSF key name differences:**
- Cell `1425f77d`: `gaze_data_goodsubj` → `gaze_data`, `gr_goodsubj` → `group_index` (OSF NPZ uses different key names than authors' local version)

**(c) Group index scheme:**
- Cell `2be11f61`: `gr_indices = [0, 1, 4, 3]` → `[0, 1, 2, 3]`
  - Authors' local data used a 5-group scheme with 1-based MATLAB indices (groups 1,2,3,4,5 → 0-indexed: 0,1,2,3,4), where ASD adults = index 4
  - OSF data uses 4-group 0-based sequential indices (0=TD adults, 1=TD children, 2=ASD adults, 3=ASD children)
  - With the old `[0,1,4,3]` on OSF data, `gr_idx=4` matches 0 subjects → ASD adults were missing

**(d) Frame source and output format:**
- Cell `7e2045a7`: Changed from `masked_frames/` (not available) to original `frames/`
- Cell `95fb9030`: Added PNG output (`gaze_points_examples.png`) alongside SVG

### 4. `demo/demo/demo_group_attention_maps.ipynb` — torchvision API update

**Cell `86a1f448`**: `transforms.ToImageTensor()` → `transforms.ToImage()`

`ToImageTensor` was removed in torchvision 0.23.0 (part of the v2 transforms migration).

## Execution steps

### STEP 4: Demo verification

```bash
cd demo/demo
jupyter nbconvert --to notebook --execute demo_group_attention_maps.ipynb
```

Output: `mean_attention_maps.png` — attention maps for G1/G2/G3 groups on test images.

### STEP 5: Fig. 2c reproduction

Run visualization using the author's pre-computed `vit_gaze_pos.npz` from OSF:

```bash
cd analysis_python
jupyter nbconvert --to notebook --execute visualize_gaze_points_N2010.ipynb
```

Output: `figures/gaze_points_examples.png` and `.svg` — Fig. 2c showing ViT gaze positions overlaid with human eye-tracking data.

### Verification: self-computed vs author data

To verify the computation pipeline is correct:

```bash
cd analysis_python
python -u verify_single_model.py
```

Computes gaze positions for DINO/depth=12/trial=1 (1 model, 2327 frames) and compares against the author's pre-computed data.

Output: `figures/verify_self_vs_author_dino_d12_t1.png` — scatter plot and per-layer comparison.

### Full 36-model computation (optional)

```bash
cd analysis_python
python -u compute_all_gaze_pos.py          # first run
python -u compute_all_gaze_pos.py --resume  # resume after interruption
```

Saves per-model results to `dataset/.../preprocessed_data/partial/` for crash resilience, then assembles final `vit_gaze_pos.npz`. Estimated ~18 hours on GTX 1060 (thermal throttled at 91 C), ~12 hours with better cooling.

## Verification results

**Condition:** DINO / depth=12 / trial=1 — self-computed vs author's pre-computed data

| Metric | Value |
|---|---|
| Exact match | 384,078 / 390,936 (**98.2%**) |
| Within 1 pixel | 390,935 / 390,936 (**100.0%**) |
| Max absolute difference | 2.0 px |
| Mean absolute difference | 0.018 px |

The 1.8% of non-exact matches are caused by `argmax_2d()` in `utils_analysis.py` (line 24-35): when multiple pixels share the maximum blurred attention value, one is chosen randomly (`random_choice=True`), introducing non-deterministic 1-2 pixel differences between runs. This is expected behavior, not a computation error.

## Output files

| File | Description |
|---|---|
| `figures/gaze_points_examples.png` | Fig. 2c reproduction (876 KB) |
| `figures/gaze_points_examples.svg` | Fig. 2c vector version (1.3 MB) |
| `figures/verify_self_vs_author_dino_d12_t1.png` | Verification scatter plot |
| `demo/demo/mean_attention_maps.png` | Demo output: G1/G2/G3 attention maps |
