"""
Custom forward utilities for OpenAI CLIP ViT-B/16 that mirror
clip.model.VisionTransformer / ResidualAttentionBlock exactly, while
also exposing per-layer [CLS] -> patch attention (averaged over heads).

This is a read-only wrapper around the loaded model's own submodules --
it does not modify the installed `clip` package, and it does not import
or touch lib/vision_transformer.py or any other DINO code.

Reference (installed package, commit d05afc4):
  site-packages/clip/model.py :: ResidualAttentionBlock, VisionTransformer
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def resblock_forward_with_attention(
    block: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same computation order as clip.model.ResidualAttentionBlock.forward
    (ln_1 -> attn -> residual -> ln_2 -> mlp -> residual), but requests
    need_weights=True instead of False so the softmax attention matrix
    is returned instead of discarded.

    x : (seq_len, batch, width) -- LND, matching clip.model's convention.

    Returns
    -------
    x_out        : (seq_len, batch, width)
    attn_weights : (batch, seq_len, seq_len) -- head-averaged softmax output.
    """
    attn_mask = (
        block.attn_mask.to(dtype=x.dtype, device=x.device)
        if block.attn_mask is not None else None
    )
    ln1 = block.ln_1(x)
    attn_out, attn_weights = block.attn(
        ln1, ln1, ln1,
        need_weights=True,
        average_attn_weights=True,
        attn_mask=attn_mask,
    )
    x = x + attn_out
    x = x + block.mlp(block.ln_2(x))
    return x, attn_weights


def visual_forward_with_attention(
    visual: nn.Module,
    x: torch.Tensor,
    pos_embed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Same computation order as clip.model.VisionTransformer.forward, but
    returns the per-layer [CLS]-row attention alongside the pooled image
    feature (numerically equivalent to model.encode_image's output).

    pos_embed : optional override for visual.positional_embedding, used
        for non-square patch grids (Phase 2). None => use the model's own
        parameter unchanged.
    """
    x = visual.conv1(x)                                   # (B, width, gh, gw)
    x = x.reshape(x.shape[0], x.shape[1], -1)              # (B, width, gh*gw)
    x = x.permute(0, 2, 1)                                 # (B, gh*gw, width)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)                         # (B, gh*gw+1, width)

    pe = visual.positional_embedding if pos_embed is None else pos_embed
    x = x + pe.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)                                 # NLD -> LND
    attn_per_layer: List[torch.Tensor] = []
    for block in visual.transformer.resblocks:
        x, attn_weights = resblock_forward_with_attention(block, x)
        attn_per_layer.append(attn_weights)
    x = x.permute(1, 0, 2)                                 # LND -> NLD

    pooled = visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        pooled = pooled @ visual.proj

    return pooled, attn_per_layer


def cls_to_patch_grid(
    attn_weights: torch.Tensor, grid_hw: Tuple[int, int]
) -> torch.Tensor:
    """
    attn_weights : (batch, seq_len, seq_len), [CLS] at index 0.

    Returns (batch, grid_h, grid_w) -- the [CLS] -> patch row, taken as-is
    (NOT renormalized after dropping the CLS column), matching the DINO
    extractor's convention in vit_extractor.py (`attn[:, :, 0, 1:]` is used
    directly, with no re-softmax / re-sum-to-1 step).
    """
    grid_h, grid_w = grid_hw
    cls_row = attn_weights[:, 0, 1:]  # (batch, grid_h*grid_w)
    expected = grid_h * grid_w
    if cls_row.shape[-1] != expected:
        raise ValueError(
            f"CLS row has {cls_row.shape[-1]} patch entries, "
            f"expected {expected} for grid {grid_hw}")
    return cls_row.reshape(cls_row.shape[0], grid_h, grid_w)


def patch_grid_from_image_hw(h: int, w: int, patch_size: int) -> Tuple[int, int]:
    """
    (grid_h, grid_w) implied by conv1 (kernel=stride=patch_size, no conv
    padding) for an already patch-aligned (H, W). Raises if H or W is not
    an exact multiple of patch_size -- callers must pad first (see
    vit_extractor.py::_pad_to_patch, reused as-is for the identical
    bottom/right zero-padding geometry).
    """
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"image size (H={h}, W={w}) is not a multiple of patch_size={patch_size}")
    return h // patch_size, w // patch_size


def visual_forward_with_cls_patch_attention(
    visual: nn.Module,
    x: torch.Tensor,
    grid_hw: Tuple[int, int],
    pos_embed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Memory-conscious variant of visual_forward_with_attention for large
    (non-square, high patch-count) inputs: only the [CLS] -> patch row
    (reshaped to grid_hw) and the full-row sum (for sanity checks) are
    kept per layer. The full (tokens x tokens) attention matrix from a
    given layer is dropped as soon as it has been reduced, so at most one
    such matrix exists at a time -- never all 12 simultaneously.

    Raises ValueError if pos_embed's token count or conv1's actual output
    token count disagrees with grid_hw (stop condition: shape mismatch).

    Returns
    -------
    pooled : (B, output_dim) -- equivalent to model.encode_image's output.
    cls_patch_maps : list of 12, each (B, grid_h, grid_w) -- [CLS]->patch
        row, NOT renormalized after dropping the CLS column (same
        convention as cls_to_patch_grid / the DINO extractor).
    full_row_sums : list of 12, each (B,) -- sum of the FULL [CLS] row
        (CLS included), captured before that layer's full matrix is
        discarded, so callers can verify softmax normalization (~1)
        without ever holding a full matrix beyond its own layer.
    """
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)

    pe = visual.positional_embedding if pos_embed is None else pos_embed
    expected_tokens = grid_hw[0] * grid_hw[1] + 1
    if pe.shape[0] != expected_tokens or x.shape[1] != expected_tokens:
        raise ValueError(
            f"token count mismatch: pos_embed has {pe.shape[0]} tokens, "
            f"conv1 output implies {x.shape[1]} tokens, "
            f"grid_hw={grid_hw} implies {expected_tokens} tokens")

    x = x + pe.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    cls_patch_maps: List[torch.Tensor] = []
    full_row_sums: List[torch.Tensor] = []
    for block in visual.transformer.resblocks:
        x, attn_weights = resblock_forward_with_attention(block, x)
        full_row_sums.append(attn_weights[:, 0, :].sum(dim=-1).detach())
        cls_patch_maps.append(cls_to_patch_grid(attn_weights, grid_hw))
        del attn_weights
    x = x.permute(1, 0, 2)  # LND -> NLD

    pooled = visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        pooled = pooled @ visual.proj

    return pooled, cls_patch_maps, full_row_sums


def interpolate_patch_pos_embed(
    pos_embed: torch.Tensor, patch_grid_hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Bicubic-interpolate CLIP's patch position embeddings to a new
    (grid_h, grid_w) patch grid. The [CLS] position embedding (row 0) is
    kept unchanged. Mirrors the design of
    lib/vision_transformer.py::VisionTransformer.interpolate_pos_encoding
    (DINO) -- adapted to CLIP's flat (num_tokens, width) parameter layout
    (DINO's pos_embed carries an extra leading batch dim of size 1).

    At the native grid size, returns the input unchanged (same shortcut
    DINO takes: `if npatch == N and w == h: return self.pos_embed`).

    pos_embed : (1 + old_grid*old_grid, width)
    Returns   : (1 + grid_h*grid_w, width)
    """
    grid_h, grid_w = patch_grid_hw
    num_patches = pos_embed.shape[0] - 1
    old_grid = int(round(num_patches ** 0.5))
    if old_grid * old_grid != num_patches:
        raise ValueError(
            f"pos_embed has {num_patches} patch tokens, not a perfect square")

    if (grid_h, grid_w) == (old_grid, old_grid):
        return pos_embed

    width = pos_embed.shape[-1]
    cls_pos = pos_embed[:1]
    patch_pos = pos_embed[1:].reshape(1, old_grid, old_grid, width).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(), size=(grid_h, grid_w), mode="bicubic", align_corners=False,
    ).to(pos_embed.dtype)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, width)

    return torch.cat([cls_pos, patch_pos], dim=0)
