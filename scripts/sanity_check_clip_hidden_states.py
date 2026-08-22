"""
Sanity-check gate for lib/clip_vit_hidden.py, required to PASS before any
L4/L8 (or L12) result from the OSIE text-alignment pilot is trusted.

Two checks, both on the 50 pilot images
(outputs/osie_attribute_grounding_pilot/pilot_image_list.csv, reused
read-only):

  1. NATIVE 224x224 (official CLIP preprocessing, native 14x14 pos_embed,
     no interpolation): our new hidden-state extraction's L12 pooled
     output (ln_post + proj applied to the block-12 [CLS] hidden state)
     vs. the REAL, unmodified model.encode_image() on the exact same
     preprocessed tensor. This is the strictest possible test -- official
     architecture, zero approximation, exactly mirrors
     tests/test_clip_extractor.py::TestCustomForwardMatchesOfficial.

  2. OSIE 800x600 (non-square, patch-size-padded, bicubic-interpolated
     pos_embed -- required since model.encode_image cannot accept a
     token count different from its native 197): our new hidden-state
     extraction's L12 pooled output vs. lib.clip_vit.visual_forward_with_attention's
     pooled output (the EXISTING, already-tested extractor from the prior
     attribute-grounding pilot) on the same input and the same
     interpolated pos_embed. There is no way to compare against the
     official encode_image directly at this resolution (it does not
     support non-native token counts), so this is the strongest available
     check: two independently-coded paths through the identical official
     submodules (conv1/cls/ln_pre/blocks/ln_post/proj) must agree exactly.

Threshold (per spec): cosine similarity >= 0.9999 on BOTH checks, for
every one of the 50 images. If violated for any image, this script exits
non-zero and writes the failure detail to sanity_checks.json --
scripts/extract_osie_text_alignment_features.py refuses to run without a
passing sanity_checks.json.

Does not modify clip_extractor.py, lib/clip_vit.py, or lib/clip_vit_hidden.py.
Writes only outputs/osie_text_alignment_pilot/sanity_checks.json.

Usage (PowerShell):
    python scripts\\sanity_check_clip_hidden_states.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from clip_extractor import CLIP_PATCH_SIZE, load_clip_vit_b16, load_osie_image_tensor
from lib.clip_vit import (
    interpolate_patch_pos_embed, patch_grid_from_image_hw,
    visual_forward_with_attention,
)
from lib.clip_vit_hidden import visual_forward_with_hidden_states

# ======================= CONFIG =======================
SEED = 42
COSINE_THRESHOLD = 0.9999
L12_ZERO_BASED = 11

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")

PILOT_IMAGE_LIST = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_pilot\pilot_image_list.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot"
OUT_JSON = os.path.join(OUT_DIR, "sanity_checks.json")
# ======================================================


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pilot_image_ids():
    if not os.path.isfile(PILOT_IMAGE_LIST):
        raise RuntimeError(
            f"STOP: prior pilot manifest not found at {PILOT_IMAGE_LIST}. "
            "Cannot reuse the seed=42/50-image set as instructed.")
    with open(PILOT_IMAGE_LIST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["image_id"] for r in rows]
    if len(ids) != 50:
        raise RuntimeError(f"STOP: expected 50 images in {PILOT_IMAGE_LIST}, got {len(ids)}")
    return ids


def check_native_224(model, preprocess, image_ids):
    """Check 1: native 224x224, our L12 pooled vs. real model.encode_image."""
    device = next(model.parameters()).device
    results = []
    for stem in image_ids:
        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            official = model.encode_image(tensor)
            hidden_states, pooled = visual_forward_with_hidden_states(
                model.visual, tensor.type(model.dtype), pos_embed=None,
                layers_zero_based=[L12_ZERO_BASED])

        official_n = official / official.norm(dim=-1, keepdim=True)
        pooled_n = pooled / pooled.norm(dim=-1, keepdim=True)
        cos = float((official_n * pooled_n).sum(dim=-1).item())
        max_abs_diff = float((official - pooled).abs().max().item())

        results.append({
            "image": stem, "cosine_similarity": cos, "max_abs_diff": max_abs_diff,
            "official_shape": list(official.shape), "official_dtype": str(official.dtype),
            "pooled_shape": list(pooled.shape), "pooled_dtype": str(pooled.dtype),
            "has_nan": bool(torch.isnan(official).any() or torch.isnan(pooled).any()),
            "has_inf": bool(torch.isinf(official).any() or torch.isinf(pooled).any()),
        })
    return results


def check_osie_nonsquare(model, image_ids):
    """Check 2: OSIE 800x600, our L12 pooled vs. lib.clip_vit's existing extractor."""
    device = next(model.parameters()).device
    results = []
    for stem in image_ids:
        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        tensor, orig_hw, pad_hw = load_osie_image_tensor(img_path, CLIP_PATCH_SIZE)
        tensor = tensor.to(device)
        grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
        pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)

        with torch.inference_mode():
            existing_pooled, _ = visual_forward_with_attention(
                model.visual, tensor.type(model.dtype), pos_embed=pos_interp)
            hidden_states, new_pooled = visual_forward_with_hidden_states(
                model.visual, tensor.type(model.dtype), pos_embed=pos_interp,
                layers_zero_based=[L12_ZERO_BASED])

        existing_n = existing_pooled / existing_pooled.norm(dim=-1, keepdim=True)
        new_n = new_pooled / new_pooled.norm(dim=-1, keepdim=True)
        cos = float((existing_n * new_n).sum(dim=-1).item())
        max_abs_diff = float((existing_pooled - new_pooled).abs().max().item())

        hs = hidden_states[L12_ZERO_BASED]
        expected_tokens = grid_hw[0] * grid_hw[1] + 1

        results.append({
            "image": stem, "cosine_similarity": cos, "max_abs_diff": max_abs_diff,
            "grid_hw": list(grid_hw), "hidden_state_shape": list(hs.shape),
            "expected_tokens": expected_tokens,
            "token_count_matches": bool(hs.shape[1] == expected_tokens),
            "has_nan": bool(torch.isnan(existing_pooled).any() or torch.isnan(new_pooled).any()
                            or torch.isnan(hs).any()),
            "has_inf": bool(torch.isinf(existing_pooled).any() or torch.isinf(new_pooled).any()
                            or torch.isinf(hs).any()),
        })
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(SEED)

    print("=" * 65)
    print("  Sanity check: lib/clip_vit_hidden.py vs. official CLIP")
    print("=" * 65)

    image_ids = load_pilot_image_ids()
    print(f"  Reusing {len(image_ids)} pilot images from {PILOT_IMAGE_LIST}")

    print("\n--- Loading CLIP ViT-B/16 ---")
    model, preprocess = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}  dtype: {model.dtype}")

    t0 = time.perf_counter()
    print("\n--- Check 1: native 224x224 vs. official model.encode_image ---")
    check1 = check_native_224(model, preprocess, image_ids)
    cos1 = [r["cosine_similarity"] for r in check1]
    fail1 = [r for r in check1 if r["cosine_similarity"] < COSINE_THRESHOLD
             or r["has_nan"] or r["has_inf"]]
    print(f"  min cosine: {min(cos1):.8f}  max abs diff: {max(r['max_abs_diff'] for r in check1):.3e}")
    print(f"  images below threshold ({COSINE_THRESHOLD}) or with NaN/Inf: {len(fail1)}")

    print("\n--- Check 2: OSIE 800x600 (interpolated pos_embed) internal consistency ---")
    check2 = check_osie_nonsquare(model, image_ids)
    cos2 = [r["cosine_similarity"] for r in check2]
    fail2 = [r for r in check2 if r["cosine_similarity"] < COSINE_THRESHOLD
             or r["has_nan"] or r["has_inf"] or not r["token_count_matches"]]
    print(f"  min cosine: {min(cos2):.8f}  max abs diff: {max(r['max_abs_diff'] for r in check2):.3e}")
    print(f"  images below threshold ({COSINE_THRESHOLD}) or with NaN/Inf/shape issues: {len(fail2)}")

    elapsed = time.perf_counter() - t0
    passed = (len(fail1) == 0) and (len(fail2) == 0)

    result = {
        "passed": passed,
        "cosine_threshold": COSINE_THRESHOLD,
        "n_images": len(image_ids),
        "check1_native_224": {
            "description": "our L12 pooled (ln_post+proj on block-12 CLS hidden state) "
                            "vs. real model.encode_image, native 224x224, no interpolation",
            "min_cosine": min(cos1), "mean_cosine": float(np.mean(cos1)),
            "max_abs_diff": max(r["max_abs_diff"] for r in check1),
            "n_failed": len(fail1), "failed_detail": fail1[:10],
            "per_image": check1,
        },
        "check2_osie_nonsquare": {
            "description": "our L12 pooled vs. lib.clip_vit.visual_forward_with_attention's "
                            "pooled output, OSIE 800x600 padded input, interpolated pos_embed",
            "min_cosine": min(cos2), "mean_cosine": float(np.mean(cos2)),
            "max_abs_diff": max(r["max_abs_diff"] for r in check2),
            "n_failed": len(fail2), "failed_detail": fail2[:10],
            "per_image": check2,
        },
        "elapsed_s": elapsed,
        "device": str(device),
        "seed": SEED,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n  Saved: {OUT_JSON}")

    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise RuntimeError(
            "STOP: sanity check FAILED -- refusing to proceed to L4/L8/L12 extraction. "
            f"See {OUT_JSON} for per-image detail.")

    print("\n" + "=" * 65)
    print("  Sanity check: PASS")
    print("=" * 65)


if __name__ == "__main__":
    main()
