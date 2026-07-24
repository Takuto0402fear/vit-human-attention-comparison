"""
SL (supervised-learning) ViT loading + layerwise [CLS] -> patch attention
extraction -- the third model condition (alongside DINO ViT-S/16 and
CLIP ViT-B/16) needed to test whether CLIP's M-shaped layerwise profile
is specific to CLIP's training objective or general to non-DINO ViTs.

Provenance (verified directly, not assumed) -- see
D:\\gaze\\vit-human-attention-comparison, Yamamoto et al. (2025)'s own
reproduction repo:
  - README.md ("Model training codes"): "Supervised learning: DeiT
    (https://github.com/facebookresearch/deit/)" -- this IS the official
    Yamamoto-paper SL condition.
  - analysis_python/utils_analysis.py::model_load's "supervised" branch
    builds the SAME generic VisionTransformer class used for DINO
    (vits.vit_small/vit_base/vit_tiny -- no distillation-token variant
    exists anywhere in that codebase) and attaches a plain nn.Linear
    classification head, loaded from
    trained_model_weights/supervised/{trial:02d}/{depth}layers/
    checkpoint.pth.
  - Confirmed directly from the checkpoint's own state_dict (trial 01,
    12layers): cls_token (1,1,384), pos_embed (1,197,384) = 1 CLS +
    14x14=196 patch tokens, NO dist_token key anywhere,
    patch_embed.proj.weight (384,3,16,16), 12 `blocks.N.*` groups
    (N=0..11), qkv weight (1152,384)=3x384, head.weight (1000,384) a
    plain nn.Linear (NOT the DINOHead used for the DINO checkpoints).
    I.e. STANDARD ViT-S/16 (embed_dim=384, depth=12, heads=6, patch=16)
    -- matching DeiT-Small's dimensions but trained WITHOUT a
    distillation token. There is no distillation-token ambiguity to
    resolve: the checkpoint has none. 4layers/8layers checkpoints were
    also inspected and differ only in block count (confirmed same
    embed_dim=384/pos_embed token count).
  - These are the SAME checkpoints already present locally at
    D:\\gaze\\...\\trained_model_weights\\supervised\\ (OSF release
    c3snp, downloaded during the earlier Yamamoto-paper reproduction
    project) -- read here read-only, never re-downloaded, never
    modified.
  - The earlier reproduction's own compute_all_gaze_pos.py /
    verify_single_model.py never actually ran the supervised model:
    dataset/Nakano_etal_2010/preprocessed_data/vit_gaze_pos.npz and
    vit_gaze_pos_author.npz are byte-identical (md5
    ab6cdcf8a83c79f6517e258cff5d3fda), i.e. the "self-computed" file is
    just a copy of the author's precomputed Nakano-video gaze
    positions -- the SL checkpoint itself was never loaded and run
    there, and never against OSIE stimuli. This module is the first
    place the SL backbone is actually executed on this project's OSIE
    static images.
  - Official attention convention (utils_analysis.py::get_gaze_pos_model
    / get_gaze_pos_model_dataset, used identically for BOTH "dino" and
    "supervised" models loaded via model_load): `attentions[:, :, 0, 1:]`
    -- [CLS] token (index 0) attending to patch tokens (indices 1:).
    Same convention already used by vit_extractor.py / clip_extractor.py
    / dino_vitb16_extractor.py in this repo.

Architecture: reuses lib/vision_transformer.py::VisionTransformer /
vit_small UNMODIFIED (imported, not copied) -- vit_small already has the
exact dims confirmed above (embed_dim=384, num_heads=6), so unlike
dino_vitb16_extractor.py's vit_base(), no new factory function is
needed here.

Extraction: reuses dino_vitb16_extractor.py's
extract_layerwise_cls_attention_memory_conscious /
cls_to_patch_grid_mean_heads / get_last_selfattention_reference AS-IS
(read-only import) -- those functions only touch model.prepare_tokens /
model.blocks / attention-tensor shapes, nothing DINO-specific, so they
apply unchanged to this SL vit_small instance. No duplicate
implementation.

Preprocessing: reuses vit_extractor.py's _pad_to_patch and
_ImageListDataset AS-IS (read-only import) -- the DeiT/SL training
pipeline uses the same ImageNet normalization constants
(0.485,0.456,0.406)/(0.229,0.224,0.225) as DINO, so nothing new is
needed there.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import nn

import lib.vision_transformer as vits
from config import Config
from dino_vitb16_extractor import (  # read-only reuse; architecture-generic despite the module name
    cls_to_patch_grid_mean_heads,
    extract_layerwise_cls_attention_memory_conscious,
    get_last_selfattention_reference,
)

SL_TRAINING_METHOD = "supervised"
SL_ARCH = "vit_small"
SL_PATCH_SIZE = 16
SL_EMBED_DIM = 384
SL_HEADS = 6
SL_NATIVE_RESOLUTION = 224
SL_NATIVE_GRID: Tuple[int, int] = (14, 14)
SL_NUM_TOKENS = 197
SL_NUM_CLASSES = 1000  # ImageNet-1k; confirmed from checkpoint head.weight shape (1000, 384)
SL_DEFAULT_TRIAL = 1
SL_DEFAULT_DEPTH = 12  # matches the existing DINO-S/16 expA depth for a like-for-like comparison


def checkpoint_path(trial_num: int = SL_DEFAULT_TRIAL, depth: int = SL_DEFAULT_DEPTH) -> str:
    """
    D:\\gaze\\...\\trained_model_weights\\supervised\\{trial:02d}\\{depth}layers\\checkpoint.pth
    -- same directory convention as vit_extractor.py::_load_model, just
    training_method="supervised" instead of "dino". Read-only; this repo
    never writes to models_dir.
    """
    models_dir = Config().models_dir
    return os.path.join(str(models_dir), SL_TRAINING_METHOD, f"{trial_num:02d}", f"{depth}layers", "checkpoint.pth")


def load_sl_vit(
    trial_num: int = SL_DEFAULT_TRIAL, depth: int = SL_DEFAULT_DEPTH,
    device: Optional[str] = None,
) -> Tuple[nn.Module, str, str]:
    """
    Loads the official Yamamoto-paper SL (DeiT-trained, standard, no
    distillation token) ViT-S/16 checkpoint. Mirrors D:\\gaze\\...\\
    analysis_python\\utils_analysis.py::model_load's "supervised" branch:
    vits.vit_small(...) + plain nn.Linear head, checkpoint["model"] if
    present else the bare state_dict (this repo's local checkpoints are
    the bare-state_dict OSF format, per D:\\gaze\\...\\REPRODUCTION.md),
    load_state_dict(strict=True).

    Returns (model, device_str, checkpoint_path).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoint_path(trial_num, depth)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model = vits.vit_small(patch_size=SL_PATCH_SIZE, depth=depth, drop_path_rate=0.1)
    model.head = nn.Linear(model.embed_dim, SL_NUM_CLASSES)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model, device, ckpt_path


def assert_sl_vit_shape(model: nn.Module, expected_depth: int = SL_DEFAULT_DEPTH) -> None:
    depth = len(model.blocks)
    heads = model.num_heads
    embed_dim = model.embed_dim
    patch_size = model.patch_embed.patch_size
    assert depth == expected_depth, f"expected depth={expected_depth}, got {depth}"
    assert heads == SL_HEADS, f"expected heads={SL_HEADS}, got {heads}"
    assert embed_dim == SL_EMBED_DIM, f"expected embed_dim={SL_EMBED_DIM}, got {embed_dim}"
    assert patch_size == SL_PATCH_SIZE, f"expected patch_size={SL_PATCH_SIZE}, got {patch_size}"


def audit_official_sl_config(model: nn.Module, expected_depth: int = SL_DEFAULT_DEPTH) -> dict:
    """
    Compares this instance's actual, runtime-inspected configuration
    against D:\\gaze\\...\\analysis_python\\utils_analysis.py::model_load's
    "supervised" branch + vision_transformer.py::vit_small (both read
    directly, not from memory): vit_small(patch_size=16, depth=N,
    drop_path_rate=0.1), i.e. embed_dim=384/num_heads=6/mlp_ratio=4/
    qkv_bias=True/norm_layer=LayerNorm(eps=1e-6), plus
    model.head=nn.Linear(embed_dim, 1000) (NOT DINOHead, NOT Identity),
    a single cls_token, and NO distillation token anywhere.

    Returns a dict of {field: (expected, observed, matches)} and raises
    AssertionError on any mismatch (fields are still checked exhaustively
    before that point so the raised message is informative).
    """
    results = {}

    def check(field, expected, observed):
        results[field] = {"expected": expected, "observed": observed, "matches": expected == observed}

    check("embed_dim", SL_EMBED_DIM, model.embed_dim)
    check("depth", expected_depth, len(model.blocks))
    check("num_heads", SL_HEADS, model.num_heads)
    check("patch_size", SL_PATCH_SIZE, model.patch_embed.patch_size)

    blk0 = model.blocks[0]
    mlp_ratio_observed = blk0.mlp.fc1.out_features / model.embed_dim
    check("mlp_ratio", 4.0, mlp_ratio_observed)
    check("qkv_bias", True, blk0.attn.qkv.bias is not None)
    check("act_layer", "GELU", type(blk0.mlp.act).__name__)
    check("cls_token_count", 1, model.cls_token.shape[1])
    check("has_distillation_token", False, hasattr(model, "dist_token"))
    check("head_type", "Linear", type(model.head).__name__)
    check("head_out_features", SL_NUM_CLASSES, getattr(model.head, "out_features", None))
    check("drop_rate", 0.0, model.pos_drop.p)
    check("attn_drop_rate", 0.0, blk0.attn.attn_drop.p)
    check("mlp_drop_rate", 0.0, blk0.mlp.drop.p)
    check("eval_mode", True, not model.training)

    norm_modules = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    eps_values = sorted(set(round(m.eps, 12) for m in norm_modules))
    check("layernorm_count", 2 * expected_depth + 1, len(norm_modules))
    check("layernorm_eps", [1e-6], eps_values)
    check("norm_layer_type", "LayerNorm", type(norm_modules[0]).__name__ if norm_modules else None)

    mismatches = [f for f, r in results.items() if not r["matches"]]
    if mismatches:
        lines = [f"  {f}: expected={results[f]['expected']!r} observed={results[f]['observed']!r}"
                 for f in mismatches]
        raise AssertionError("STOP: official-config audit mismatch(es):\n" + "\n".join(lines))

    return results


# Re-exported for convenience so callers only need to import this module
# for the full SL extraction pipeline; these are dino_vitb16_extractor.py's
# own generic functions, not copies.
__all__ = [
    "SL_TRAINING_METHOD", "SL_ARCH", "SL_PATCH_SIZE", "SL_EMBED_DIM", "SL_HEADS",
    "SL_NATIVE_RESOLUTION", "SL_NATIVE_GRID", "SL_NUM_TOKENS", "SL_NUM_CLASSES",
    "SL_DEFAULT_TRIAL", "SL_DEFAULT_DEPTH",
    "checkpoint_path", "load_sl_vit", "assert_sl_vit_shape", "audit_official_sl_config",
    "cls_to_patch_grid_mean_heads", "extract_layerwise_cls_attention_memory_conscious",
    "get_last_selfattention_reference",
]
