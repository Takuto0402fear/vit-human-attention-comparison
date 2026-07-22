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
from lib.clip_vit import patch_grid_from_image_hw  # read-only reuse; generic (H,W,patch_size) -> grid arithmetic

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

    Uses load_state_dict(strict=True): Phase 1 already confirmed (via a
    strict=False inspection) that missing_keys and unexpected_keys are
    both empty for this checkpoint against this architecture, so the
    production loader uses strict=True directly -- if that ever stops
    holding (e.g. a different checkpoint file), torch itself raises a
    RuntimeError here rather than silently degrading to a partial load.

    Returns (model, device_str).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.hub.load_state_dict_from_url(
        DINO_VITB16_URL, map_location="cpu", progress=True)

    model = vit_base()
    model.load_state_dict(state_dict, strict=True)

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


def audit_official_dino_vitb16_config(model: nn.Module) -> dict:
    """
    Compares this instance's actual, runtime-inspected configuration
    against facebookresearch/dino's official vision_transformer.py
    vit_base():

        def vit_base(patch_size=16, **kwargs):
            model = VisionTransformer(
                patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
            return model

    LayerNorm eps is NOT stored in a checkpoint's state_dict (it's a
    module hyperparameter, not a parameter/buffer), so it cannot be
    checked by inspecting loaded weights -- every nn.LayerNorm submodule
    actually present in the constructed model is inspected directly here.

    Returns a dict of {field: (expected, observed, matches)} plus raises
    AssertionError immediately on any mismatch (fields are still checked
    exhaustively before that point so the raised message is informative).
    """
    results = {}

    def check(field, expected, observed):
        results[field] = {"expected": expected, "observed": observed, "matches": expected == observed}

    check("embed_dim", DINO_EMBED_DIM, model.embed_dim)
    check("depth", DINO_DEPTH, len(model.blocks))
    check("num_heads", DINO_HEADS, model.num_heads)
    check("patch_size", DINO_PATCH_SIZE, model.patch_embed.patch_size)

    blk0 = model.blocks[0]
    mlp_ratio_observed = blk0.mlp.fc1.out_features / model.embed_dim
    check("mlp_ratio", 4.0, mlp_ratio_observed)
    check("qkv_bias", True, blk0.attn.qkv.bias is not None)
    check("act_layer", "GELU", type(blk0.mlp.act).__name__)
    check("cls_token_count", 1, model.cls_token.shape[1])
    check("has_distillation_token", False, hasattr(model, "dist_token"))
    check("head_is_identity", True, isinstance(model.head, nn.Identity))
    check("drop_rate", 0.0, model.pos_drop.p)
    check("attn_drop_rate", 0.0, blk0.attn.attn_drop.p)
    check("mlp_drop_rate", 0.0, blk0.mlp.drop.p)
    check("eval_mode", True, not model.training)
    check("drop_path_disabled_at_eval", True, not blk0.drop_path.training if hasattr(blk0.drop_path, "training") else True)

    norm_modules = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    eps_values = sorted(set(round(m.eps, 12) for m in norm_modules))
    check("layernorm_count", 2 * DINO_DEPTH + 1, len(norm_modules))  # norm1+norm2 per block, plus final norm
    check("layernorm_eps", [1e-6], eps_values)
    check("norm_layer_type", "LayerNorm", type(norm_modules[0]).__name__ if norm_modules else None)

    mismatches = [f for f, r in results.items() if not r["matches"]]
    if mismatches:
        lines = [f"  {f}: expected={results[f]['expected']!r} observed={results[f]['observed']!r}"
                 for f in mismatches]
        raise AssertionError("STOP: official-config audit mismatch(es):\n" + "\n".join(lines))

    return results


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


def extract_layerwise_cls_attention_memory_conscious(
    model: nn.Module, images: torch.Tensor, grid_hw: Tuple[int, int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Memory-conscious variant of extract_layerwise_cls_attention for
    large (non-square, high patch-count) inputs such as the 608x800
    padded OSIE stimuli (1901 tokens): a full (B, heads, 1901, 1901)
    attention matrix is ~174 MB (fp32, heads=12) per layer, so holding
    all 12 layers' full matrices simultaneously (as
    get_fulllayers_selfattention's returned list would, if kept whole)
    risks ~2+ GB just for attention tensors on a 6 GB GPU.

    Computation order deliberately mirrors
    VisionTransformer.get_fulllayers_selfattention EXACTLY (same
    prepare_tokens call, same per-block `blk(x, return_attention=True)`
    to obtain that layer's attention, same separate `blk(x)` call to
    advance the residual stream to the next layer's input) -- the only
    difference is that each layer's full attention matrix is reduced
    (head-averaged, CLS row extracted, reshaped to grid_hw) and
    discarded immediately, so at most one full matrix exists in memory
    at a time, never all 12 at once.

    Returns
    -------
    layer_maps    : list of 12, each (B, grid_h, grid_w) -- head-averaged
        [CLS]->patch attention, NOT renormalized (same convention as
        cls_to_patch_grid_mean_heads / vit_extractor.py's DINO path).
    full_row_sums : list of 12, each (B,) -- sum of the FULL [CLS] row
        (CLS included), captured before that layer's full matrix is
        discarded, so callers can verify softmax normalization (~1)
        without ever holding a full matrix beyond its own layer.
    """
    grid_h, grid_w = grid_hw
    expected_patches = grid_h * grid_w

    with torch.inference_mode():
        x = model.prepare_tokens(images)
        n_tokens = x.shape[1]
        if n_tokens != expected_patches + 1:
            raise ValueError(
                f"STOP: prepare_tokens produced {n_tokens} tokens, "
                f"expected {expected_patches + 1} for grid {grid_hw}")

        layer_maps: List[torch.Tensor] = []
        full_row_sums: List[torch.Tensor] = []
        n_blocks = len(model.blocks)

        for i, blk in enumerate(model.blocks):
            attn = blk(x, return_attention=True)          # (B, heads, N, N)
            head_mean = attn.mean(dim=1)                    # (B, N, N)
            full_row_sums.append(head_mean[:, 0, :].sum(dim=-1).detach())

            cls_row = head_mean[:, 0, 1:]                    # (B, N-1)
            if cls_row.shape[-1] != expected_patches:
                raise ValueError(
                    f"STOP: layer {i + 1} CLS row has {cls_row.shape[-1]} "
                    f"patch entries, expected {expected_patches} for grid {grid_hw}")
            layer_maps.append(cls_row.reshape(cls_row.shape[0], grid_h, grid_w))

            del attn, head_mean, cls_row

            if i < n_blocks - 1:
                x = blk(x)

    return layer_maps, full_row_sums


def check_native_pos_embed_untouched_at_grid(
    model: nn.Module, images: torch.Tensor, grid_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Directly exercises the EXISTING, unmodified
    VisionTransformer.interpolate_pos_encoding on `images`' actual
    (possibly non-square) size, mirroring prepare_tokens' own
    construction of the CLS-prepended token sequence, so the resulting
    interpolated position embedding can be inspected from outside
    (interpolate_pos_encoding is not otherwise exposed by
    get_fulllayers_selfattention / prepare_tokens' return value).

    Returns the interpolated pos_embed, shape (1, grid_h*grid_w + 1, embed_dim).
    Caller is expected to assert its shape and that row 0 (CLS) is
    unchanged from model.pos_embed[:, 0].
    """
    with torch.inference_mode():
        B, _, h_px, w_px = images.shape
        patch_tokens = model.patch_embed(images)
        cls_tokens = model.cls_token.expand(patch_tokens.shape[0], -1, -1)
        x_with_cls = torch.cat((cls_tokens, patch_tokens), dim=1)
        pos_interp = model.interpolate_pos_encoding(x_with_cls, h_px, w_px)
    return pos_interp
