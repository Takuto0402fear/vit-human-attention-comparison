"""
CLIP ViT-B/16 loading + layerwise [CLS] -> patch attention extraction.

Mirrors vit_extractor.py's role for the DINO side of the comparison, but
is fully independent: no shared state, no shared output paths, no shared
Config fields. Does not import or modify vit_extractor.py, config.py, or
lib/vision_transformer.py (DINO).

Phase 1 (current): standard 224x224 CLIP input only, using the model's
own official preprocessing. Non-square OSIE-sized input and the
associated position-embedding interpolation are Phase 2 (not wired up
here yet).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import clip
import torch

from lib.clip_vit import cls_to_patch_grid, visual_forward_with_attention

CLIP_MODEL_NAME = "ViT-B/16"
CLIP_PATCH_SIZE = 16
CLIP_DEPTH = 12
CLIP_HEADS = 12
CLIP_NATIVE_RESOLUTION = 224
CLIP_NATIVE_GRID: Tuple[int, int] = (14, 14)   # 224 / 16
CLIP_NUM_TOKENS = 197                          # 14*14 + 1


def load_clip_vit_b16(
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Callable]:
    """
    Load OpenAI CLIP ViT-B/16 in float32 (numerically stable for the
    attention-extraction regression tests). Downloads to the package's
    default cache (~/.cache/clip) on first use.

    Returns (model, official_preprocess_transform).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
    model = model.float()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess


def assert_vit_b16_shape(model: torch.nn.Module) -> None:
    """Raise AssertionError if the loaded model does not match the expected spec."""
    visual = model.visual
    depth = len(visual.transformer.resblocks)
    heads = visual.transformer.resblocks[0].attn.num_heads
    patch_size = visual.conv1.kernel_size[0]
    n_tokens = visual.positional_embedding.shape[0]
    assert depth == CLIP_DEPTH, f"expected depth={CLIP_DEPTH}, got {depth}"
    assert heads == CLIP_HEADS, f"expected heads={CLIP_HEADS}, got {heads}"
    assert patch_size == CLIP_PATCH_SIZE, f"expected patch_size={CLIP_PATCH_SIZE}, got {patch_size}"
    assert n_tokens == CLIP_NUM_TOKENS, f"expected tokens={CLIP_NUM_TOKENS}, got {n_tokens}"


def extract_layerwise_cls_attention(
    model: torch.nn.Module,
    images: torch.Tensor,
    grid_hw: Tuple[int, int] = CLIP_NATIVE_GRID,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    images : (B, 3, H, W), already preprocessed (CLIP normalization applied).

    Returns
    -------
    pooled_features : (B, output_dim) -- equivalent to model.encode_image(images).
    layer_maps : list of length 12, each (B, grid_h, grid_w) -- raw
        [CLS] -> patch attention (mean over heads), not renormalized.
    """
    images = images.type(model.dtype)
    pooled, attn_per_layer = visual_forward_with_attention(model.visual, images)
    layer_maps = [cls_to_patch_grid(a, grid_hw) for a in attn_per_layer]
    return pooled, layer_maps
