"""
Hidden-state (residual-stream) extraction for OpenAI CLIP ViT-B/16, for the
OSIE text-alignment / linear-probe pilot (outputs/osie_text_alignment_pilot/).

Distinct from lib/clip_vit.py (which extracts [CLS]->patch ATTENTION
weights for the earlier attribute-grounding pilot): this module extracts
the actual per-token HIDDEN STATES after each transformer block (the
residual stream itself), for both the [CLS] token and every patch token.
Does not import or modify lib/clip_vit.py or clip_extractor.py.

Unlike lib/clip_vit.py's resblock_forward_with_attention (which reimplements
the block's internals to request attention weights), hidden-state extraction
needs no attention weights, so this module calls the block's own official
forward (`block(x)`) UNCHANGED -- no reimplementation, so there is no room
for the extraction to silently diverge from clip.model.VisionTransformer's
actual computation. This is the same design property that
scripts/sanity_check_clip_hidden_states.py's exact-match test relies on.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn


def visual_forward_with_hidden_states(
    visual: nn.Module,
    x: torch.Tensor,
    pos_embed: Optional[torch.Tensor] = None,
    layers_zero_based: Optional[Iterable[int]] = None,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Same input-side computation as clip.model.VisionTransformer.forward /
    lib.clip_vit.visual_forward_with_attention (conv1 -> flatten -> [CLS]
    prepend -> + pos_embed -> ln_pre), but each transformer block is called
    via its own unmodified forward (`block(x)`) since no attention weights
    are needed here.

    x : (B, 3, H, W), already CLIP-normalized (and padded to a patch_size
        multiple by the caller, for non-square OSIE-sized input).
    pos_embed : optional override for visual.positional_embedding (bicubic-
        interpolated for non-native grids -- see
        lib.clip_vit.interpolate_patch_pos_embed, reused unchanged by callers).
    layers_zero_based : which block outputs to keep (0-indexed into
        visual.transformer.resblocks). None => keep all.

    Returns
    -------
    hidden_states : {zero_based_layer_idx: (B, seq_len, width) float},
        the RAW residual-stream output after that block (before ln_post/proj).
        seq_len = 1 (CLS) + grid_h*grid_w.
    pooled : (B, output_dim) -- identical computation to
        clip.model.VisionTransformer.forward's final pooled output
        (ln_post(x[:,0,:]) @ proj), i.e. equivalent to model.encode_image
        WHEN the full 12 blocks are run (verified in
        scripts/sanity_check_clip_hidden_states.py).
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

    n_layers = len(visual.transformer.resblocks)
    wanted = set(layers_zero_based) if layers_zero_based is not None else set(range(n_layers))
    for w in wanted:
        if not (0 <= w < n_layers):
            raise ValueError(f"layer index {w} out of range [0, {n_layers})")

    hidden_states: Dict[int, torch.Tensor] = {}
    for i, block in enumerate(visual.transformer.resblocks):
        x = block(x)                                       # official forward, unmodified
        if i in wanted:
            hidden_states[i] = x.permute(1, 0, 2).clone()   # LND -> NLD, (B, seq, width)

    x = x.permute(1, 0, 2)                                 # LND -> NLD (mirrors visual_forward_with_attention)
    pooled = visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        pooled = pooled @ visual.proj

    return hidden_states, pooled


def project_and_normalize_tokens(visual: nn.Module, hidden_tokens: torch.Tensor) -> torch.Tensor:
    """
    Applies the model's own final LayerNorm and visual projection to
    arbitrary tokens (CLS or patch, from any layer), then L2-normalizes:
        normalize(visual_projection(final_layer_norm(hidden_token)))

    hidden_tokens : (..., width) float, e.g. (B, seq_len, width).
    Returns        : (..., output_dim) float, L2-normalized over the last dim.

    This is a deliberate "logit-lens"-style probe: ln_post/proj are only
    used by the OFFICIAL forward pass on the last layer's [CLS] token; here
    they are reused (unmodified, same nn.Parameter tensors) on earlier
    layers' tokens too, for the text-alignment analysis only. Never used
    for the linear-probe features (see scripts/extract_osie_text_alignment_features.py).
    """
    x = visual.ln_post(hidden_tokens)
    if visual.proj is not None:
        x = x @ visual.proj
    norm = x.norm(dim=-1, keepdim=True)
    return x / norm.clamp_min(1e-12)
