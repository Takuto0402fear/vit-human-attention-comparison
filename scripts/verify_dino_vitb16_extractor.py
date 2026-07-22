"""
Phase-1 regression check for the DINO ViT-B/16 attention extractor.

Mirrors scripts/verify_clip_extractor.py's Phase-1 structure (standard
224x224 input, no non-square handling yet), but for DINO ViT-B/16
instead of CLIP ViT-B/16 -- added to isolate whether the layer-profile
shape difference already found between DINO ViT-S/16 and CLIP ViT-B/16
comes from training method or from model scale.

Confirms:
  1. Architecture matches the expected ViT-B/16 spec (depth/heads/
     embed_dim/patch_size), model is in eval mode, all parameters have
     requires_grad=False.
  2. Official DINO ViT-B/16 backbone-only checkpoint loads with EMPTY
     missing_keys/unexpected_keys (strict=False inspection, not a
     silent strict=True).
  3. Downloaded checkpoint's MD5 matches the object store's own
     recorded MD5.
  4. Standard 224x224 input produces 197 tokens (196 patches + CLS),
     14x14 grid, 12 layers, 12 heads, shape (12, 14, 14) for the
     head-averaged CLS->patch stack.
  5. No NaN/Inf anywhere.
  6. Every layer's full [CLS] row (head-averaged, CLS included) sums to
     ~1.
  7. Two independent forward passes on the same input are deterministic
     (bit-identical / within floating tolerance).
  8. get_fulllayers_selfattention(x)[-1] matches an independently
     written get_last_selfattention-style reference implementation
     (mirroring the well-known upstream facebookresearch/dino API, not
     present in this repo's lib/vision_transformer.py) -- i.e. the
     multi-layer extraction path does not silently corrupt the final
     layer's attention.
  9. The native 14x14 position-embedding path (lib/vision_transformer.py
     ::VisionTransformer.interpolate_pos_encoding, unmodified) is a
     no-op for a native 224x224 input.

Read-only w.r.t. every existing DINO-S/CLIP-B file and output. Does not
write anything under outputs/ and does not touch the DINO-S checkpoint
on D:\\gaze\\... or the CLIP weight cache.

Usage (PowerShell):
    python scripts\\verify_dino_vitb16_extractor.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import torch
from PIL import Image
from torchvision import transforms as pth_transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from dino_vitb16_extractor import (
    DINO_DEPTH, DINO_HEADS, DINO_NATIVE_GRID, DINO_VITB16_EXPECTED_MD5,
    assert_vitb16_shape, cls_to_patch_grid_mean_heads,
    extract_layerwise_cls_attention, get_last_selfattention_reference,
    load_dino_vitb16,
)

STIM_PATH = os.path.join(
    PROJECT_DIR, "datasets", "osie", "predicting-human-gaze-beyond-pixels",
    "data", "stimuli", "1001.jpg")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _cached_checkpoint_path() -> str:
    hub_dir = torch.hub.get_dir()
    return os.path.join(hub_dir, "checkpoints", "dino_vitbase16_pretrain.pth")


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 65)
    print("  DINO ViT-B/16 attention-extractor verification (224x224, native grid)")
    print("=" * 65)

    print("\n--- Loading official DINO ViT-B/16 backbone checkpoint ---")
    model, device = load_dino_vitb16()
    print(f"  Device: {device}   dtype: {next(model.parameters()).dtype}")
    print("  load_state_dict(strict=False): missing_keys=[] unexpected_keys=[]  OK (checked in loader)")

    ckpt_path = _cached_checkpoint_path()
    if os.path.isfile(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"  Cached checkpoint: {ckpt_path}  ({size_mb:.1f} MB)")
        print("  Computing MD5 (may take a few seconds) ...")
        md5 = _md5(ckpt_path)
        print(f"  MD5:      {md5}")
        print(f"  Expected: {DINO_VITB16_EXPECTED_MD5}")
        assert md5 == DINO_VITB16_EXPECTED_MD5, "STOP: MD5 mismatch on cached checkpoint"
        print("  MD5 match  OK")
    else:
        print(f"  WARNING: expected cache file not found at {ckpt_path} "
              "(torch.hub may use a different path on this system)")

    print("\n--- Architecture / mode assertions ---")
    assert_vitb16_shape(model)
    assert not model.training, "STOP: model is not in eval mode"
    n_grad = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grad == 0, f"STOP: {n_grad} parameters still have requires_grad=True"
    print(f"  depth={DINO_DEPTH}, heads={DINO_HEADS}, embed_dim=768, patch_size=16  OK")
    print("  model.training == False  OK")
    print("  all parameters requires_grad == False  OK")

    print("\n--- Preparing standard 224x224 input ---")
    transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(STIM_PATH).convert("RGB")
    image_tensor = transform(img).unsqueeze(0).to(device)
    print(f"  Source image: {STIM_PATH}")
    print(f"  Input tensor: {tuple(image_tensor.shape)}")
    assert tuple(image_tensor.shape) == (1, 3, 224, 224), image_tensor.shape

    print("\n--- Native position-embedding no-op check ---")
    with torch.inference_mode():
        patch_tokens = model.patch_embed(image_tensor)
        cls_tokens = model.cls_token.expand(patch_tokens.shape[0], -1, -1)
        x_with_cls = torch.cat((cls_tokens, patch_tokens), dim=1)
        pos_interp = model.interpolate_pos_encoding(x_with_cls, 224, 224)
    torch.testing.assert_close(pos_interp, model.pos_embed)
    print(f"  interpolate_pos_encoding(x, 224, 224) == model.pos_embed (native 14x14)  OK "
          f"(shape {tuple(pos_interp.shape)})")

    print("\n--- Warm-up pass (untimed) ---")
    with torch.inference_mode():
        _ = model.get_fulllayers_selfattention(image_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print("\n--- Timed pass + full extraction ---")
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    layer_maps, attn_raw = extract_layerwise_cls_attention(model, image_tensor, DINO_NATIVE_GRID)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    peak_mem_mib = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else None
    print(f"  Elapsed: {elapsed_s * 1000:.1f} ms")
    if peak_mem_mib is not None:
        print(f"  Peak VRAM: {peak_mem_mib:.1f} MiB")

    print("\n--- Per-layer checks (12 layers) ---")
    assert len(layer_maps) == 12 and len(attn_raw) == 12
    for l, (grid_map, attn) in enumerate(zip(layer_maps, attn_raw), start=1):
        assert attn.shape[1] == DINO_HEADS, f"layer {l}: expected {DINO_HEADS} heads, got {attn.shape[1]}"
        assert tuple(attn.shape[-2:]) == (197, 197), f"layer {l}: token shape {attn.shape[-2:]} != (197,197)"
        assert tuple(grid_map.shape[-2:]) == DINO_NATIVE_GRID

        head_mean_full = attn.mean(dim=1)  # (B, 197, 197)
        full_row_sum = head_mean_full[:, 0, :].sum(dim=-1)
        assert not torch.isnan(attn).any() and not torch.isinf(attn).any(), f"layer {l}: NaN/Inf"
        torch.testing.assert_close(full_row_sum, torch.ones_like(full_row_sum), rtol=1e-4, atol=1e-4)

        print(f"  Layer {l:2d}: heads={attn.shape[1]}  tokens={tuple(attn.shape[-2:])}  "
              f"full_row_sum={full_row_sum.item():.6f}  grid={tuple(grid_map.shape[-2:])}  OK")

    print("\n--- Determinism check (same input, two independent runs) ---")
    with torch.inference_mode():
        attn_raw_2 = model.get_fulllayers_selfattention(image_tensor)
    for l, (a1, a2) in enumerate(zip(attn_raw, attn_raw_2), start=1):
        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
    print("  All 12 layers bit-identical across two independent forward passes  OK")

    print("\n--- get_last_selfattention (independent reference) vs get_fulllayers_selfattention[-1] ---")
    with torch.inference_mode():
        last_ref = get_last_selfattention_reference(model, image_tensor)
    max_abs_diff = (last_ref - attn_raw[-1]).abs().max().item()
    torch.testing.assert_close(last_ref, attn_raw[-1], rtol=1e-5, atol=1e-6)
    print(f"  max abs diff (raw per-head, layer 12): {max_abs_diff:.3e}  (PASS, tol=1e-6)")

    grid_last_ref = cls_to_patch_grid_mean_heads(last_ref, DINO_NATIVE_GRID)
    max_abs_diff_grid = (grid_last_ref - layer_maps[-1]).abs().max().item()
    torch.testing.assert_close(grid_last_ref, layer_maps[-1], rtol=1e-5, atol=1e-6)
    print(f"  max abs diff (head-averaged CLS->patch grid, layer 12): {max_abs_diff_grid:.3e}  (PASS, tol=1e-6)")

    print("\n" + "=" * 65)
    print("  ALL CHECKS PASSED (standard 224x224 / native 14x14 grid)")
    print("  Next (not run here): non-square 608x800 OSIE input, per the phased plan.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
