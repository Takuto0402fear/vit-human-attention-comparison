"""
CLIP ViT-B/16 loading + layerwise [CLS] -> patch attention extraction.

Mirrors vit_extractor.py's role for the DINO side of the comparison, but
is fully independent: no shared state, no shared output paths, no shared
Config fields. Does not import or modify vit_extractor.py, config.py, or
lib/vision_transformer.py (DINO).

Phase 1: standard 224x224 CLIP input, using the model's own official
preprocessing (extract_layerwise_cls_attention).
Phase 2: non-square OSIE-sized input, DINO-style bottom/right padding,
CLIP's own mean/std, no CenterCrop / resize, bicubic-interpolated patch
position embeddings (load_osie_image_tensor / extract_single_image_layerwise_attention).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import clip
import torch

from lib.clip_vit import (
    cls_to_patch_grid,
    interpolate_patch_pos_embed,
    patch_grid_from_image_hw,
    visual_forward_with_attention,
    visual_forward_with_cls_patch_attention,
)

CLIP_MODEL_NAME = "ViT-B/16"
CLIP_PATCH_SIZE = 16
CLIP_DEPTH = 12
CLIP_HEADS = 12
CLIP_NATIVE_RESOLUTION = 224
CLIP_NATIVE_GRID: Tuple[int, int] = (14, 14)   # 224 / 16
CLIP_NUM_TOKENS = 197                          # 14*14 + 1

# Official CLIP normalization constants (site-packages/clip/clip.py::_transform).
# Deliberately distinct from DINO's ImageNet stats used in vit_extractor.py.
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


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


# ======================================================================
# Phase 2: non-square OSIE-sized input (no CenterCrop, no resize)
# ======================================================================

def load_osie_image_tensor(
    image_path: str, patch_size: int = CLIP_PATCH_SIZE,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Same geometric preprocessing as vit_extractor.py's DINO path: pad H/W
    up to a patch_size multiple (zero-pad at the bottom/right only,
    applied BEFORE normalization, exactly mirroring
    vit_extractor.py::_pad_to_patch / _ImageListDataset) -- but with
    CLIP's own mean/std, and no CenterCrop / resize.

    Returns
    -------
    tensor  : (1, 3, H_pad, W_pad) float32, CLIP-normalized.
    orig_hw : (H, W) before padding.
    pad_hw  : (H, W) after padding (multiples of patch_size).
    """
    from PIL import Image
    import numpy as np
    from torchvision import transforms as pth_transforms

    from vit_extractor import _pad_to_patch  # read-only reuse; DINO file untouched

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)
    img_np = _pad_to_patch(img_np, patch_size)
    pad_h, pad_w = img_np.shape[:2]

    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    tensor = transform(Image.fromarray(img_np)).unsqueeze(0)
    return tensor, (orig_h, orig_w), (pad_h, pad_w)


def extract_single_image_layerwise_attention(
    model: torch.nn.Module,
    image_path: str,
    patch_size: int = CLIP_PATCH_SIZE,
):
    """
    Phase-2 single-image, non-square extraction. batch_size=1 only
    (loads one image; callers are expected to wrap this in
    torch.inference_mode()).

    Returns
    -------
    pooled          : (1, output_dim)
    cls_patch_maps  : list of 12, each (1, grid_h, grid_w)
    full_row_sums   : list of 12, each (1,)
    grid_hw         : (grid_h, grid_w) derived from the padded image size
    orig_hw, pad_hw : (H, W) before/after padding
    """
    device = next(model.parameters()).device
    tensor, orig_hw, pad_hw = load_osie_image_tensor(image_path, patch_size)
    tensor = tensor.to(device)

    grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], patch_size)
    pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)

    pooled, cls_patch_maps, full_row_sums = visual_forward_with_cls_patch_attention(
        model.visual, tensor.type(model.dtype), grid_hw, pos_embed=pos_interp)
    return pooled, cls_patch_maps, full_row_sums, grid_hw, orig_hw, pad_hw
