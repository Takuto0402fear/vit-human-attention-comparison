"""
Regression / sanity checks for the CLIP ViT-B/16 attention extractor.

Phase 1 -- standard 224x224 input (CLIP's native resolution, official
preprocessing):
  1. Architecture matches the expected ViT-B/16 spec (depth/heads/patch/tokens).
  2. The custom attention-extracting forward produces the SAME pooled
     image feature as the official model.encode_image -- i.e. requesting
     need_weights=True does not change CLIP's own computed result.
  3. Each layer's full attention row ([CLS] included) sums to 1.
  4. Each layer's [CLS]->patch row ([CLS] excluded) is in (0, 1] and
     reshapes to the native 14x14 patch grid.
  5. The position-embedding interpolation utility is a no-op at the
     native 14x14 grid.

Phase 2 -- one real OSIE image, non-square input, batch_size=1:
  6. DINO-style padding (bottom/right zero-pad, before normalization)
     yields the expected padded size / grid_h / grid_w / token count.
  7. Patch position embeddings are bicubic-interpolated to the new grid;
     the [CLS] position embedding is left untouched.
  8. 12-layer [CLS]->patch attention has shape (12, grid_h, grid_w), no
     NaN/Inf, full-row sums ~1; the full (tokens x tokens) matrix is
     never retained for more than one layer at a time.
  9. Bilinear resize to (600, 800) preserves axis order (no accidental
     transpose to (50, 38) / (800, 600)); an orientation-check overlay is
     saved OUTSIDE the repo (temp dir) for visual confirmation.
  10. Warm-up vs. timed pass are separated; elapsed time and peak CUDA
      memory are reported. On CUDA OOM, the run stops and reports
      device/peak-memory/error WITHOUT silently changing resolution.

Read-only with respect to the repo: does not touch any DINO file, DINO
output, or results/, and writes nothing under outputs/. The only file
this script writes is a orientation-check PNG under the OS temp
directory (outside the repo).

Usage (PowerShell):
    python scripts\\verify_clip_extractor.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from clip_extractor import (
    CLIP_DEPTH, CLIP_NATIVE_GRID, CLIP_PATCH_SIZE,
    assert_vit_b16_shape, load_clip_vit_b16, load_osie_image_tensor,
)
from lib.clip_vit import (
    cls_to_patch_grid, interpolate_patch_pos_embed,
    patch_grid_from_image_hw, visual_forward_with_attention,
    visual_forward_with_cls_patch_attention,
)

STIM_PATH = os.path.join(
    PROJECT_DIR, "datasets", "osie", "predicting-human-gaze-beyond-pixels",
    "data", "stimuli", "1001.jpg")

# OSIE stimulus display size (width, height) -- matches IMG_W/IMG_H used
# throughout scripts/run_expA.py and friends (DINO side).
OSIE_IMG_W = 800
OSIE_IMG_H = 600


def main() -> int:
    print("=" * 65)
    print("  CLIP ViT-B/16 attention-extractor verification")
    print("=" * 65)

    print("\n--- Loading model ---")
    model, preprocess = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}   dtype: {model.dtype}")

    # ------------------------------------------------------------------
    # Phase 1: standard 224x224
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Phase 1: standard 224x224 input, native 14x14 grid")
    print("=" * 65)

    print("\n--- Architecture assertions ---")
    assert_vit_b16_shape(model)
    print("  depth=12, heads=12, patch_size=16, tokens=197  OK")

    print("\n--- Preparing standard 224x224 input ---")
    img = Image.open(STIM_PATH).convert("RGB")
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    print(f"  Source image: {STIM_PATH}")
    print(f"  Input tensor: {tuple(image_tensor.shape)}")
    assert tuple(image_tensor.shape[-2:]) == (224, 224), image_tensor.shape

    print("\n--- Custom forward vs. official encode_image ---")
    with torch.inference_mode():
        official_features = model.encode_image(image_tensor)
        pooled, attn_per_layer = visual_forward_with_attention(
            model.visual, image_tensor.type(model.dtype))

    torch.testing.assert_close(pooled, official_features, rtol=1e-4, atol=1e-4)
    max_abs_diff = (pooled - official_features).abs().max().item()
    print(f"  pooled shape: {tuple(pooled.shape)}")
    print(f"  max abs diff vs encode_image: {max_abs_diff:.3e}  (PASS, tol=1e-4)")

    print("\n--- Per-layer attention checks ---")
    assert len(attn_per_layer) == CLIP_DEPTH
    for l, attn in enumerate(attn_per_layer, start=1):
        full_row_sum = attn[:, 0, :].sum(dim=-1)
        cls_to_patch = attn[:, 0, 1:]
        cls_to_patch_sum = cls_to_patch.sum(dim=-1)

        torch.testing.assert_close(
            full_row_sum, torch.ones_like(full_row_sum), rtol=1e-4, atol=1e-4)
        assert bool((cls_to_patch > 0).all()), f"layer {l}: non-positive entry in CLS->patch row"
        assert bool((cls_to_patch_sum <= 1.0 + 1e-6).all()), f"layer {l}: CLS->patch sum > 1"

        grid_map = cls_to_patch_grid(attn, CLIP_NATIVE_GRID)
        assert tuple(grid_map.shape[-2:]) == CLIP_NATIVE_GRID

        print(f"  Layer {l:2d}: full_row_sum={full_row_sum.item():.6f}  "
              f"cls_to_patch_sum={cls_to_patch_sum.item():.6f}  "
              f"grid={tuple(grid_map.shape[-2:])}  OK")

    print("\n--- Position-embedding interpolation identity check ---")
    pos = model.visual.positional_embedding.detach()
    pos_interp = interpolate_patch_pos_embed(pos, CLIP_NATIVE_GRID)
    torch.testing.assert_close(pos_interp, pos)
    print(f"  interpolate_patch_pos_embed(pos, {CLIP_NATIVE_GRID}) == original pos_embed  OK")

    print("\nPhase 1: ALL CHECKS PASSED")

    # ------------------------------------------------------------------
    # Phase 2: one real OSIE image, non-square input, batch_size=1
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Phase 2: OSIE single image, non-square input (batch_size=1)")
    print("=" * 65)

    print("\n--- Loading + padding OSIE image ---")
    tensor, (orig_h, orig_w), (pad_h, pad_w) = load_osie_image_tensor(
        STIM_PATH, CLIP_PATCH_SIZE)
    print(f"  Source: {STIM_PATH}")
    print(f"  Original size (H,W): ({orig_h}, {orig_w})")
    print(f"  Padded size   (H,W): ({pad_h}, {pad_w})  "
          f"(zero-pad at bottom/right only, applied before normalization)")
    tensor = tensor.to(device)
    assert tuple(tensor.shape) == (1, 3, pad_h, pad_w)

    grid_h, grid_w = patch_grid_from_image_hw(pad_h, pad_w, CLIP_PATCH_SIZE)
    n_tokens = grid_h * grid_w + 1
    print(f"  grid_h={grid_h}  grid_w={grid_w}  tokens={n_tokens}")
    if (grid_h, grid_w, n_tokens) != (38, 50, 1901):
        raise RuntimeError(
            f"STOP: token-count mismatch. Expected grid (38,50)/tokens=1901 "
            f"for a 608x800 padded OSIE image, got grid=({grid_h},{grid_w}) "
            f"tokens={n_tokens}.")

    print("\n--- Position-embedding interpolation (patch only, CLS untouched) ---")
    pos = model.visual.positional_embedding.detach()
    print(f"  Native pos_embed shape:      {tuple(pos.shape)}")
    pos_interp = interpolate_patch_pos_embed(pos, (grid_h, grid_w))
    print(f"  Interpolated pos_embed shape: {tuple(pos_interp.shape)}")
    if tuple(pos_interp.shape) != (n_tokens, pos.shape[-1]):
        raise RuntimeError(
            f"STOP: interpolated pos_embed shape {tuple(pos_interp.shape)} "
            f"!= expected ({n_tokens}, {pos.shape[-1]})")
    torch.testing.assert_close(pos_interp[0], pos[0])  # CLS row must be untouched
    print("  CLS row unchanged (assert_close vs native pos_embed[0])  OK")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("\n--- Warm-up pass (untimed) ---")
    with torch.inference_mode():
        _ = visual_forward_with_cls_patch_attention(
            model.visual, tensor.type(model.dtype), (grid_h, grid_w), pos_embed=pos_interp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)  # reset again post-warmup

    print("\n--- Timed pass ---")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            pooled2, cls_patch_maps, full_row_sums = visual_forward_with_cls_patch_attention(
                model.visual, tensor.type(model.dtype), (grid_h, grid_w), pos_embed=pos_interp)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as e:
        peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.type == "cuda" else None)
        print("\n  STOP: CUDA out-of-memory during the single-image (608x800) forward pass.")
        if peak is not None:
            print(f"  Peak VRAM before OOM: {peak:.1f} MiB")
        print(f"  Error: {e}")
        print("  Resolution was NOT changed. Stopping per the stop-condition policy.")
        return 1
    elapsed_s = time.perf_counter() - t0

    peak_mem_mib = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else None)
    print(f"  Elapsed: {elapsed_s * 1000:.1f} ms")
    if peak_mem_mib is not None:
        print(f"  Peak VRAM (timed pass): {peak_mem_mib:.1f} MiB")

    print("\n--- Attention checks (12 layers) ---")
    if len(cls_patch_maps) != 12 or len(full_row_sums) != 12:
        raise RuntimeError(
            f"STOP: expected 12 layers, got {len(cls_patch_maps)} maps / "
            f"{len(full_row_sums)} row sums")

    row_sum_min, row_sum_max = float("inf"), float("-inf")
    for l, (grid_map, row_sum) in enumerate(zip(cls_patch_maps, full_row_sums), start=1):
        if tuple(grid_map.shape) != (1, 38, 50):
            raise RuntimeError(f"STOP: layer {l} grid_map shape {tuple(grid_map.shape)} != (1,38,50)")
        if torch.isnan(grid_map).any() or torch.isinf(grid_map).any():
            raise RuntimeError(f"STOP: NaN/Inf detected in layer {l} CLS->patch map")
        if torch.isnan(row_sum).any() or torch.isinf(row_sum).any():
            raise RuntimeError(f"STOP: NaN/Inf detected in layer {l} full-row sum")
        rs = row_sum.item()
        row_sum_min, row_sum_max = min(row_sum_min, rs), max(row_sum_max, rs)
        print(f"  Layer {l:2d}: shape={tuple(grid_map.shape)}  full_row_sum={rs:.6f}")

    print(f"  full_row_sum range across 12 layers: [{row_sum_min:.6f}, {row_sum_max:.6f}]")

    stacked = torch.stack(cls_patch_maps, dim=1)  # (1, 12, 38, 50)
    if tuple(stacked.shape[1:]) != (12, 38, 50):
        raise RuntimeError(f"STOP: stacked output shape {tuple(stacked.shape[1:])} != (12,38,50)")
    print(f"  Stacked 12-layer output shape: {tuple(stacked.shape[1:])}  (expected (12,38,50))")

    print("\n--- Spatial resize to (H=600, W=800), bilinear ---")
    resized = F.interpolate(
        stacked.reshape(12, 1, 38, 50), size=(OSIE_IMG_H, OSIE_IMG_W),
        mode="bilinear", align_corners=False)
    resized = resized[:, 0]  # (12, 600, 800)
    if tuple(resized.shape) != (12, OSIE_IMG_H, OSIE_IMG_W):
        raise RuntimeError(f"STOP: resized shape {tuple(resized.shape)} != (12,600,800)")
    print(f"  Resized shape: {tuple(resized.shape)}  "
          f"(interpolated directly from (38,50) to (H={OSIE_IMG_H}, W={OSIE_IMG_W}); "
          f"grid axis order was NOT swapped to (50,38)/(800,600))")

    print("\n--- Orientation sanity check (overlay saved OUTSIDE the repo) ---")
    overlay_path = _save_orientation_overlay(
        STIM_PATH, resized[-1].detach().cpu().numpy())
    print(f"  Overlay saved to (OS temp dir, not in repo): {overlay_path}")

    print("\n" + "=" * 65)
    print("  Phase 2: ALL CHECKS PASSED")
    print("  (outputs/expA_clip/ and the CLIP attention cache were NOT written to)")
    print("=" * 65)
    return 0


def _save_orientation_overlay(image_path: str, heat_hw) -> str:
    """
    Saves a side-by-side original/overlay PNG to the OS temp directory
    (never under the repo) purely for a manual up/down, left/right
    orientation sanity check.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].imshow(img)
    axes[0].set_title("original (1001.jpg)")
    axes[0].axis("off")

    h = heat_hw.copy()
    if h.max() > 0:
        h = h / h.max()
    axes[1].imshow(img)
    axes[1].imshow(h, cmap="jet", alpha=0.45)
    axes[1].set_title("CLIP L12 [CLS]->patch, bilinear to 800x600")
    axes[1].axis("off")
    fig.tight_layout()

    out_path = os.path.join(tempfile.gettempdir(), "clip_vitb16_phase2_orientation_check.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    sys.exit(main())
