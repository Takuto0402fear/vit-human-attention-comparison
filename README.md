# Gaze x ViT Attention: Static Image Group Comparison Pipeline

Compare human gaze patterns (healthy vs schizophrenia) with DINO ViT-S/16
self-attention across transformer layers.

## Quick Start (PowerShell)

```powershell
# Dummy end-to-end test (no GPU / model weights needed)
python run_pipeline.py --dummy

# With reduced layers and image limit (faster)
python run_pipeline.py --dummy --depth 4 --limit 2
```

## Requirements

- Python 3.11+
- numpy, scipy, torch, torchvision
- kornia (for real ViT extraction only)
- pytest (for tests)

```powershell
pip install numpy scipy torch torchvision kornia pytest
```

## Pipeline Overview

```
load_gaze()          fixations per (subject, image)
     |
     v
vit_extractor        attention heatmaps per (image, layer)
     |
     v
metrics              NSS / AUC-Judd / sAUC per (subject, image, layer)
     |
     v
aggregation          group x layer summary table -> CSV
```

## Module Input/Output

| Module | Input | Output |
|--------|-------|--------|
| `schema.py` | - | `GazeRecord` dataclass definition |
| `data_loader.py` | source name + Config | `List[GazeRecord]` |
| `vit_extractor.py` | `{image_id: path}` + Config | `{image_id: ndarray(L, H, W)}` |
| `gaze_heatmap.py` | `List[GazeRecord]` + Config | `{(subj, img): ndarray(H, W)}` |
| `metrics.py` | saliency map + fixation points | float (NSS / AUC / sAUC) |
| `aggregation.py` | records + attention maps | CSV summary table |

## Output Files

- `results/group_layer_summary.csv` -- group x layer means and SDs
- `results/detail_results.csv` -- per (subject, image, layer) metric values

## Configuration

All parameters are in `config.py` with defaults for GTX 1060 6GB:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed |
| `depth` | 12 | ViT layers |
| `batch_size` | 2 | Batch size for GPU |
| `gaussian_sigma` | 30.0 | Fixation heatmap blur (px) |
| `heatmap_width` | 720 | Output map width |
| `heatmap_height` | 480 | Output map height |
| `common_images_only` | True | Only compare on shared images |

## CLI Options

```
--dummy           Use synthetic data (no GPU needed)
--image-dir DIR   Stimulus image directory (real mode)
--limit N         Cap image count (0 = all)
--batch N         ViT batch size
--depth N         Number of ViT layers
--sigma F         Gaussian sigma for fixation heatmap
--seed N          Random seed
```

## Running Tests

```powershell
# Unit tests (no GPU needed)
python -m pytest tests/test_pipeline.py -v

# ViT extraction regression test (requires GPU + D:\gaze model weights)
python scripts/verify_extractor.py --n-frames 50     # quick (50 frames)
python scripts/verify_extractor.py --n-frames 2327   # full
python scripts/verify_extractor.py --skip-map         # skip Part 2 (C: vs D: model)
```

## Regression Test Results (ViT Extraction)

Verified against `vit_gaze_pos_author.npz` from Yamamoto et al. (2025).
Model: DINO ViT-S/16, depth=12, trial=1, 50 Nakano frames.

**Part 1 -- Gaze position comparison (C: output vs author reference):**

| Metric | Result | Threshold |
|--------|--------|-----------|
| Exact match | 98.1% | >= 95% |
| Within 1 pixel | 100.0% | >= 99.9% |
| Max difference | 1.0 px | |

Non-exact matches occur only in shallow layers (1-4, 6) due to random
tie-breaking in `argmax_2d` when multiple pixels share the same maximum
blurred-attention value. Deep layers (5, 7-12) show 100% exact match.

**Part 2 -- Raw attention-map comparison (C: model vs D: model):**

All 12 layers: max_abs_diff = 0.00, cosine_similarity = 1.0000000000.
The two implementations produce **byte-identical** attention tensors.

**Verdict: PASS -- extraction code is validated for downstream use.**

## Adding Real Data

1. Write a new loader function in `data_loader.py` that converts your data
   format into `List[GazeRecord]`
2. Register it in the `loaders` dict inside `load_gaze()`
3. Update `run_pipeline.py` to accept the new source name

The `GazeRecord` schema is documented in `schema.py`.

## Pending / Not Yet Implemented

- Real data loader (waiting for data format from Yoshida-sensei)
- d_ij redefinition for MDS (cosine similarity of gaze maps; note: not
  directly comparable to Yamamoto et al. MDS values due to different scale)
- Statistical tests for group differences (t-test / permutation test)
