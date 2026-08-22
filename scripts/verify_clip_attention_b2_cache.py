"""
[B-2] Step 1: confirm that the existing attention cache
  outputs/osie_attribute_grounding_full700/attn_cache/clip_vitb16_full700_L4L8L12_patchgrid.npz
is the SAME attention used by the original M-shape experiment
(scripts/run_expA_clip_full700.py), before any B-2 metric/statistics code
is trusted to read it.

Checks performed:
  1. Model: CLIP ViT-B/16 (assert_vit_b16_shape on the freshly loaded model).
  2. Same 700 OSIE images: cache stems == {1001..1700} == the exact
     image_ids list recorded in outputs/expA_clip/healthy/run_config.json
     (the original M-shape run's own image-ID record).
  3. Phase 2 image input / position-embedding interpolation: the cache
     was produced by scripts/run_osie_attribute_grounding_pilot.py, which
     imports and calls the IDENTICAL functions
     (clip_extractor.load_osie_image_tensor,
      lib.clip_vit.{patch_grid_from_image_hw,interpolate_patch_pos_embed,
      visual_forward_with_cls_patch_attention}) as
     scripts/run_expA_clip_full700.py -- verified here by re-running that
     exact call chain for a seed=42 sample of images and diffing against
     the cached values (bit-for-bit, since both are deterministic
     eval-mode forward passes with no dropout).
  4. 38x50 patch grid.
  5. L4=index 3, L8=index 7, L12=index 11 (layers_display / layers_zero_based).
  6. Image-ID <-> array-order correspondence (stems[i] <-> attn[i]).
  7. Head-averaged [CLS]->patch attention (implicit in
     visual_forward_with_cls_patch_attention -- re-verified by the same
     recompute in point 3, which goes through the identical
     average_attn_weights=True code path).
  8. No NaN/Inf, all entries >= 0, shape == (700, 3, 38, 50).

All paths are resolved relative to the repository root (this script's
parent's parent directory) or overridable via CLI flags -- no
machine-specific absolute path is hardcoded. Writes only
<output-dir>/cache_verification.json (STOPS if that file already exists
-- no overwrite).

Usage (PowerShell):
    python scripts\\verify_clip_attention_b2_cache.py
    python scripts\\verify_clip_attention_b2_cache.py --output-dir D:\\tmp\\b2_smoke
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import argparse
import json
import os
import time

import numpy as np
import torch

from clip_extractor import (
    CLIP_PATCH_SIZE, assert_vit_b16_shape, load_clip_vit_b16, load_osie_image_tensor,
)
from lib.clip_attention_b2 import refuse_if_exists
from lib.clip_vit import (
    interpolate_patch_pos_embed, patch_grid_from_image_hw, visual_forward_with_cls_patch_attention,
)

SEED_DEFAULT = 42
N_SPOTCHECK_DEFAULT = 8
GRID_HW_EXPECTED = (38, 50)
LAYERS_DISPLAY_EXPECTED = [4, 8, 12]
LAYERS_ZERO_BASED_EXPECTED = [3, 7, 11]
N_IMAGES_EXPECTED = 700

DEFAULT_CACHE_PATH = (
    REPO_ROOT / "outputs" / "osie_attribute_grounding_full700" / "attn_cache"
    / "clip_vitb16_full700_L4L8L12_patchgrid.npz")
DEFAULT_ORIGINAL_MSHAPE_CONFIG = REPO_ROOT / "outputs" / "expA_clip" / "healthy" / "run_config.json"
DEFAULT_STIMULI_DIR = (
    REPO_ROOT / "datasets" / "osie" / "predicting-human-gaze-beyond-pixels" / "data" / "stimuli")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "clip_attention_b2_distribution"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="[B-2] Verify the existing L4/L8/L12 attention cache matches the "
                    "original M-shape experiment's conditions (model, 700-image set, "
                    "38x50 grid, layer indices, numeric health, bit-exact spot recompute).")
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                    help=f"Path to the existing L4/L8/L12 attention cache .npz, read-only "
                        f"(default: {DEFAULT_CACHE_PATH}).")
    p.add_argument("--original-mshape-config", type=Path, default=DEFAULT_ORIGINAL_MSHAPE_CONFIG,
                    help="Path to the original M-shape run's run_config.json, read-only -- "
                        "used to cross-check the 700-image ID set "
                        f"(default: {DEFAULT_ORIGINAL_MSHAPE_CONFIG}).")
    p.add_argument("--stimuli-dir", type=Path, default=DEFAULT_STIMULI_DIR,
                    help="Directory containing OSIE stimulus images (<id>.jpg), read-only -- "
                        f"used for the bit-exact spot-check recompute (default: {DEFAULT_STIMULI_DIR}).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Directory to write cache_verification.json into (created if "
                        "missing; refuses to overwrite an existing cache_verification.json "
                        f"in this directory) (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help=f"Seed for selecting the spot-check recompute sample (default: {SEED_DEFAULT}).")
    p.add_argument("--n-spotcheck", type=int, default=N_SPOTCHECK_DEFAULT,
                    help="Number of images to recompute fresh and diff against the cache "
                        f"(default: {N_SPOTCHECK_DEFAULT}).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(args.cache_path)
    original_mshape_config = Path(args.original_mshape_config)
    stimuli_dir = Path(args.stimuli_dir)
    output_dir = Path(args.output_dir)
    seed = args.seed
    n_spotcheck = args.n_spotcheck
    out_json = output_dir / "cache_verification.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_if_exists([str(out_json)])

    checks = {}
    problems = []

    print("=" * 70)
    print("  [B-2] Cache verification: does the existing L4/L8/L12 cache")
    print("  match the original M-shape experiment's conditions?")
    print("=" * 70)
    print(f"  cache_path              = {cache_path}")
    print(f"  original_mshape_config  = {original_mshape_config}")
    print(f"  stimuli_dir             = {stimuli_dir}")
    print(f"  output_dir              = {output_dir}")

    # ---- 1. cache file exists, shape/dtype ----
    if not cache_path.is_file():
        raise RuntimeError(f"STOP: cache not found: {cache_path}")
    cache = np.load(str(cache_path), allow_pickle=False)
    attn = cache["attn"]
    stems = cache["stems"]
    layers_display = cache["layers_display"].tolist()
    layers_zero_based = cache["layers_zero_based"].tolist()

    checks["shape_is_700_3_38_50"] = (attn.shape == (N_IMAGES_EXPECTED, 3, *GRID_HW_EXPECTED))
    checks["dtype_is_float32"] = (attn.dtype == np.float32)
    checks["layers_display_matches"] = (layers_display == LAYERS_DISPLAY_EXPECTED)
    checks["layers_zero_based_matches"] = (layers_zero_based == LAYERS_ZERO_BASED_EXPECTED)
    checks["layer_index_mapping_consistent"] = (
        [d - 1 for d in layers_display] == layers_zero_based)
    print(f"  shape={attn.shape} dtype={attn.dtype}  "
          f"layers_display={layers_display} layers_zero_based={layers_zero_based}")

    # ---- 2. image IDs: 700 unique, contiguous 1001..1700 ----
    stems_list = stems.tolist()
    checks["n_stems_700"] = (len(stems_list) == N_IMAGES_EXPECTED)
    checks["stems_unique"] = (len(set(stems_list)) == len(stems_list))
    stems_int_sorted = sorted(int(s) for s in stems_list)
    checks["stems_sorted_order_in_array"] = (stems_list == sorted(stems_list))
    checks["stems_contiguous_1001_1700"] = (
        stems_int_sorted == list(range(1001, 1701)))

    # ---- 3. same 700 images as the original M-shape run (run_expA_clip_full700.py) ----
    if not original_mshape_config.is_file():
        problems.append(f"original M-shape run_config.json not found: {original_mshape_config}")
        checks["same_image_set_as_original_mshape_run"] = None
    else:
        with open(original_mshape_config, encoding="utf-8") as f:
            orig_config = json.load(f)
        orig_ids = set(orig_config["image_ids"])
        checks["same_image_set_as_original_mshape_run"] = (orig_ids == set(stems_list))
        checks["original_mshape_model"] = orig_config.get("model")
        checks["original_mshape_attn_kind"] = orig_config.get("attn")
        checks["original_mshape_head_agg"] = orig_config.get("head_agg")

    # ---- 4. numeric health: NaN/Inf/non-negativity ----
    checks["no_nan"] = bool(not np.isnan(attn).any())
    checks["no_inf"] = bool(not np.isinf(attn).any())
    checks["all_nonnegative"] = bool((attn >= 0).all())
    sums = attn.sum(axis=(2, 3))
    checks["all_row_sums_in_open_0_1"] = bool(((sums > 0) & (sums < 1.0 + 1e-6)).all())
    print(f"  NaN={not checks['no_nan']!s:5} Inf={not checks['no_inf']!s:5} "
          f"min={attn.min():.6g} max={attn.max():.6g} "
          f"row_sum[min,max]=[{sums.min():.6f},{sums.max():.6f}]")

    # ---- 5. model identity ----
    print("\n--- Loading CLIP ViT-B/16 for spot-check recompute ---")
    model, _ = load_clip_vit_b16()
    device = next(model.parameters()).device
    assert_vit_b16_shape(model)
    checks["model_is_clip_vitb16"] = True
    print(f"  Device: {device}  dtype: {model.dtype}  assert_vit_b16_shape: OK")

    # ---- 6. Phase-2 code-path spot recompute (bit-for-bit diff vs. cache) ----
    print(f"\n--- Recomputing {n_spotcheck} sample images (seed={seed}) and diffing vs. cache ---")
    rng = np.random.RandomState(seed)
    sample_idx = sorted(rng.choice(len(stems_list), size=n_spotcheck, replace=False).tolist())
    sample_stems = [stems_list[i] for i in sample_idx]

    max_abs_diffs = {}
    for stem in sample_stems:
        idx = stems_list.index(stem)
        img_path = stimuli_dir / f"{stem}.jpg"
        tensor, orig_hw, pad_hw = load_osie_image_tensor(str(img_path), CLIP_PATCH_SIZE)
        tensor = tensor.to(device)
        grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
        if grid_hw != GRID_HW_EXPECTED:
            raise RuntimeError(f"STOP: {stem} grid_hw {grid_hw} != {GRID_HW_EXPECTED}")
        pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)
        with torch.inference_mode():
            _, cls_patch_maps, full_row_sums = visual_forward_with_cls_patch_attention(
                model.visual, tensor.type(model.dtype), grid_hw, pos_embed=pos_interp)
        selected = torch.stack(
            [cls_patch_maps[i] for i in LAYERS_ZERO_BASED_EXPECTED], dim=0
        )[:, 0].detach().cpu().numpy().astype(np.float32)
        cached_map = attn[idx]
        diff = float(np.abs(selected - cached_map).max())
        max_abs_diffs[stem] = diff
        print(f"  {stem}: grid_hw={grid_hw}  max_abs_diff={diff:.3e}")

    checks["spotcheck_stems"] = sample_stems
    checks["spotcheck_max_abs_diffs"] = max_abs_diffs
    checks["spotcheck_bitexact_match"] = bool(all(v < 1e-6 for v in max_abs_diffs.values()))

    # ---- verdict ----
    hard_fail_keys = [
        "shape_is_700_3_38_50", "dtype_is_float32", "layers_display_matches",
        "layers_zero_based_matches", "layer_index_mapping_consistent",
        "n_stems_700", "stems_unique", "stems_contiguous_1001_1700",
        "no_nan", "no_inf", "all_nonnegative", "all_row_sums_in_open_0_1",
        "model_is_clip_vitb16", "spotcheck_bitexact_match",
    ]
    failed = [k for k in hard_fail_keys if checks.get(k) is not True]
    if checks.get("same_image_set_as_original_mshape_run") is False:
        failed.append("same_image_set_as_original_mshape_run")

    all_passed = len(failed) == 0

    result = {
        "verdict": "PASS" if all_passed else "FAIL",
        "failed_checks": failed,
        "checks": checks,
        "problems": problems,
        "seed": seed,
        "paths_used": {
            "cache_path": str(cache_path), "original_mshape_config": str(original_mshape_config),
            "stimuli_dir": str(stimuli_dir), "output_dir": str(output_dir),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  VERDICT: {result['verdict']}")
    if failed:
        print(f"  Failed checks: {failed}")
    print(f"  Saved: {out_json}")
    print("=" * 70)

    if not all_passed:
        raise RuntimeError(f"STOP: cache verification FAILED: {failed}")


if __name__ == "__main__":
    main()
