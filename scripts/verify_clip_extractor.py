"""
Phase-1 regression check for the CLIP ViT-B/16 attention extractor.

On a standard 224x224 input (CLIP's native resolution, official
preprocessing, no non-square handling yet), confirms:
  1. Architecture matches the expected ViT-B/16 spec (depth/heads/patch/tokens).
  2. The custom attention-extracting forward produces the SAME pooled
     image feature as the official model.encode_image -- i.e. requesting
     need_weights=True does not change CLIP's own computed result.
  3. Each layer's full attention row ([CLS] included) sums to 1.
  4. Each layer's [CLS]->patch row ([CLS] excluded) is in (0, 1] and
     reshapes to the native 14x14 patch grid.
  5. The position-embedding interpolation utility is a no-op at the
     native 14x14 grid.

Read-only: does not touch any DINO file, DINO output, or results/, and
writes nothing under outputs/.

Usage (PowerShell):
    python scripts\\verify_clip_extractor.py
"""
from __future__ import annotations

import os
import sys

import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from clip_extractor import (
    CLIP_DEPTH, CLIP_NATIVE_GRID, assert_vit_b16_shape, load_clip_vit_b16,
)
from lib.clip_vit import cls_to_patch_grid, interpolate_patch_pos_embed, visual_forward_with_attention

STIM_PATH = os.path.join(
    PROJECT_DIR, "datasets", "osie", "predicting-human-gaze-beyond-pixels",
    "data", "stimuli", "1001.jpg")


def main() -> int:
    print("=" * 65)
    print("  CLIP ViT-B/16 attention-extractor verification (224x224, native grid)")
    print("=" * 65)

    print("\n--- Loading model ---")
    model, preprocess = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}   dtype: {model.dtype}")

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

    print("\n" + "=" * 65)
    print("  ALL CHECKS PASSED (standard 224x224 / native 14x14 grid)")
    print("  Next (not run here): non-square position-embedding interpolation")
    print("  and a single OSIE image, per the phased plan.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
