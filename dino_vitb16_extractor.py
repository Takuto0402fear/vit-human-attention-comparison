"""
DINO ViT-B/16 loading + layerwise [CLS] -> patch attention extraction.

Purpose: isolate whether the layer-profile shape difference already
observed between DINO ViT-S/16 (outputs/expA/healthy/) and CLIP ViT-B/16
(outputs/expA_clip/healthy/) comes from training method (DINO vs CLIP)
or from model scale (Small vs Base) -- by adding DINO at the SAME Base
scale as the existing CLIP model (depth=12, patch_size=16, embed_dim=768,
heads=12).

Architecture: reuses lib/vision_transformer.py::VisionTransformer
UNMODIFIED (imported, not copied) -- that module has no `vit_base`
factory (only vit_small/vit_tiny), so one is defined here, mirroring
vit_small's exact pattern with Base-sized dimensions. The class's own
defaults already happen to be embed_dim=768/depth=12/num_heads=12, but
qkv_bias and norm_layer are passed explicitly to match vit_small/
vit_tiny's convention (and the official DINO training config).
get_fulllayers_selfattention is architecture-generic (only touches
self.blocks / self.patch_embed / self.pos_embed) and needs no changes to
work with this Base-sized instance.

Weights: the official Facebook Research DINO ViT-B/16 backbone-only
checkpoint (NOT the student/teacher/optimizer "full checkpoint" variant),
downloaded via torch.hub.load_state_dict_from_url (caches under
~/.cache/torch/hub/checkpoints/, entirely outside this repo and
independent of the existing custom-trained DINO-S checkpoint at
D:\\gaze\\...\\trained_model_weights\\dino\\01\\12layers\\checkpoint.pth,
which is a different training run in a different checkpoint format --
untouched, read-only, not used here). The model is constructed via our
own lib/vision_transformer.py (not torch.hub's model entrypoint / not
facebookresearch's own vision_transformer.py) specifically so
get_fulllayers_selfattention and the rest of this project's extraction
convention keep working unchanged -- only the WEIGHTS are official, the
architecture code path is this repo's own.

Preprocessing: reuses vit_extractor.py's _pad_to_patch and
_ImageListDataset AS-IS (read-only import) -- DINO's ImageNet
normalization is identical across backbone sizes, so nothing new is
needed there, unlike CLIP which required its own normalization constants.
"""
from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple

import torch
from torch import nn

import lib.vision_transformer as vits
from vit_extractor import _ImageListDataset, _pad_to_patch  # read-only reuse; DINO file untouched

DINO_VITB16_URL = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
# From the object store's own metadata (x-amz-meta-s3cmd-attrs on the URL above), not computed by us.
DINO_VITB16_EXPECTED_MD5 = "552daf80332dbde16bba2a52c6508b77"

DINO_PATCH_SIZE = 16
DINO_DEPTH = 12
DINO_EMBED_DIM = 768
DINO_HEADS = 12
DINO_NATIVE_RESOLUTION = 224
DINO_NATIVE_GRID: Tuple[int, int] = (14, 14)
DINO_NUM_TOKENS = 197


def vit_base(patch_size: int = DINO_PATCH_SIZE, depth: int = DINO_DEPTH, **kwargs) -> nn.Module:
    """
    Not present in lib/vision_transformer.py (only vit_small/vit_tiny are)
    -- defined here, mirroring vit_small's exact factory pattern with
    Base-sized dimensions, WITHOUT modifying lib/vision_transformer.py.
    """
    return vits.VisionTransformer(
        patch_size=patch_size, embed_dim=DINO_EMBED_DIM, depth=depth, num_heads=DINO_HEADS,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def load_dino_vitb16(device: Optional[str] = None) -> Tuple[nn.Module, str]:
    """
    Downloads (if not already cached) and loads the official DINO
    ViT-B/16 backbone-only checkpoint into a fresh vit_base() instance.

    Raises RuntimeError if load_state_dict(strict=False) reports ANY
    missing or unexpected key -- a clean backbone-only checkpoint against
    a head=Identity model should produce empty missing/unexpected lists;
    anything else means the checkpoint or the architecture doesn't match
    and this function stops rather than silently proceeding.

    Returns (model, device_str).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.hub.load_state_dict_from_url(
        DINO_VITB16_URL, map_location="cpu", progress=True)

    model = vit_base()
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "STOP: unexpected missing/unexpected keys loading the official "
            f"DINO ViT-B/16 backbone checkpoint.\n  missing={missing}\n  unexpected={unexpected}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model, device


def assert_vitb16_shape(model: nn.Module) -> None:
    depth = len(model.blocks)
    heads = model.num_heads
    embed_dim = model.embed_dim
    patch_size = model.patch_embed.patch_size
    assert depth == DINO_DEPTH, f"expected depth={DINO_DEPTH}, got {depth}"
    assert heads == DINO_HEADS, f"expected heads={DINO_HEADS}, got {heads}"
    assert embed_dim == DINO_EMBED_DIM, f"expected embed_dim={DINO_EMBED_DIM}, got {embed_dim}"
    assert patch_size == DINO_PATCH_SIZE, f"expected patch_size={DINO_PATCH_SIZE}, got {patch_size}"


def get_last_selfattention_reference(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Independent reference implementation of the well-known upstream
    facebookresearch/dino VisionTransformer.get_last_selfattention() --
    NOT present in this repo's lib/vision_transformer.py (which only
    has get_fulllayers_selfattention), reimplemented here purely to
    cross-check get_fulllayers_selfattention(x)[-1] against a
    faithful-to-upstream independent code path (same math, written
    separately) rather than trusting a single implementation.

    Returns (B, heads, N, N) raw per-head attention for the last block only.
    """
    x = model.prepare_tokens(x)
    n_blocks = len(model.blocks)
    for i, blk in enumerate(model.blocks):
        if i < n_blocks - 1:
            x = blk(x)
        else:
            return blk(x, return_attention=True)
    raise RuntimeError("STOP: model has no blocks")


def cls_to_patch_grid_mean_heads(attn_raw: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
    """
    attn_raw : (B, heads, N, N) raw per-head softmax attention.
    Returns (B, grid_h, grid_w) -- [CLS] -> patch row, averaged over
    heads, NOT renormalized after dropping the CLS column (matches
    vit_extractor.py's DINO extraction convention: `attn[:, :, 0, 1:]`
    used as-is).
    """
    grid_h, grid_w = grid_hw
    B, heads = attn_raw.shape[0], attn_raw.shape[1]
    cls_row = attn_raw[:, :, 0, 1:]  # (B, heads, grid_h*grid_w)
    expected = grid_h * grid_w
    if cls_row.shape[-1] != expected:
        raise ValueError(
            f"CLS row has {cls_row.shape[-1]} patch entries, expected {expected} for grid {grid_hw}")
    cls_row = cls_row.reshape(B, heads, grid_h, grid_w)
    return cls_row.mean(dim=1)


def extract_layerwise_cls_attention(
    model: nn.Module, images: torch.Tensor, grid_hw: Tuple[int, int] = DINO_NATIVE_GRID,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    images : (B, 3, H, W), already preprocessed (ImageNet normalization,
        patch-aligned H/W -- e.g. via vit_extractor.py's _pad_to_patch).

    Returns
    -------
    layer_maps : list of 12, each (B, grid_h, grid_w) -- head-averaged
        [CLS] -> patch attention per layer, not renormalized.
    attn_per_layer_raw : list of 12, each (B, heads, N, N) -- the raw
        per-head attention returned by get_fulllayers_selfattention,
        kept only for verification (e.g. comparing layer 12 against
        get_last_selfattention_reference); not used for evaluation.
    """
    with torch.inference_mode():
        attn_per_layer_raw = model.get_fulllayers_selfattention(images)
    layer_maps = [cls_to_patch_grid_mean_heads(a, grid_hw) for a in attn_per_layer_raw]
    return layer_maps, attn_per_layer_raw
