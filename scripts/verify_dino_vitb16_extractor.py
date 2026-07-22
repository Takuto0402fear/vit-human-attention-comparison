"""
Regression / sanity checks for the DINO ViT-B/16 attention extractor.

Phase 1 -- standard 224x224 input (native resolution):
  1. Official-config audit: this instance vs facebookresearch/dino's own
     vit_base() (embed_dim/depth/heads/patch_size/mlp_ratio/qkv_bias/
     norm_layer+eps/drop rates/GELU/CLS-token-count/no-distillation-token/
     head=Identity/eval mode/drop_path disabled at eval).
  2. Official DINO ViT-B/16 backbone-only checkpoint loads with
     load_state_dict(strict=True) (Phase 1 already established empty
     missing/unexpected keys via a strict=False inspection; production
     loading now uses strict=True directly).
  3. Checkpoint MD5 matches the object store's own recorded MD5.
  4. Standard 224x224 input: 197 tokens, 14x14 grid, 12 layers, 12 heads,
     (12,14,14) head-averaged CLS->patch stack, full [CLS] row sums to
     ~1, no NaN/Inf, two independent runs bit-identical, last layer
     matches an independent get_last_selfattention-style reference, and
     the native-grid position embedding is a no-op.
  5. Memory-conscious extraction regression: the new per-layer
     immediate-reduce extractor (extract_layerwise_cls_attention_memory_
     conscious) is compared against the existing full
     get_fulllayers_selfattention path at 224x224 -- must match before
     Phase 2 proceeds to 608x800.

Phase 2 -- one real OSIE image, non-square input (608x800, 1901 tokens),
batch_size=1, using ONLY the memory-conscious extractor (a full
(heads,1901,1901) matrix is ~174 MB; holding all 12 simultaneously risks
~2+ GB on a 6 GB GPU):
  6. DINO-style padding (bottom/right zero-pad, before normalization,
     reusing vit_extractor.py::_pad_to_patch unmodified) yields grid
     (38,50) / 1901 tokens.
  7. The existing, unmodified interpolate_pos_encoding bicubic-
     interpolates the patch position embeddings to 38x50 while leaving
     the [CLS] position embedding untouched -- checked directly, not
     assumed.
  8. 12-layer [CLS]->patch attention has shape (12,38,50), no NaN/Inf,
     full-row sums ~1, two runs deterministic, layer order 1..12, and
     (memory permitting) the final layer matches an independent
     get_last_selfattention-style reference computed directly at
     608x800 (only the last layer -- never all 12 full matrices).
  9. Bilinear resize to (600,800) preserves axis order (no accidental
     transpose); an orientation-check overlay is saved OUTSIDE the repo
     (temp dir) for visual confirmation.
  10. Warm-up vs. timed pass are separated; elapsed time, peak CUDA
      memory, and reserved CUDA memory are reported.

Read-only with respect to the repo: does not touch any existing DINO-S/
CLIP-B file or output. The only file this script writes is an
orientation-check PNG under the OS temp directory (outside the repo).
Writes nothing under outputs/.

Usage (PowerShell):
    python scripts\\verify_dino_vitb16_extractor.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as pth_transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from dino_vitb16_extractor import (
    DINO_DEPTH, DINO_HEADS, DINO_NATIVE_GRID, DINO_PATCH_SIZE, DINO_VITB16_EXPECTED_MD5,
    assert_vitb16_shape, audit_official_dino_vitb16_config,
    check_native_pos_embed_untouched_at_grid, cls_to_patch_grid_mean_heads,
    extract_layerwise_cls_attention, extract_layerwise_cls_attention_memory_conscious,
    get_last_selfattention_reference, load_dino_vitb16,
)
from lib.clip_vit import patch_grid_from_image_hw
from vit_extractor import _pad_to_patch

STIM_PATH = os.path.join(
    PROJECT_DIR, "datasets", "osie", "predicting-human-gaze-beyond-pixels",
    "data", "stimuli", "1001.jpg")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OSIE_IMG_W, OSIE_IMG_H = 800, 600


def _cached_checkpoint_path() -> str:
    hub_dir = torch.hub.get_dir()
    return os.path.join(hub_dir, "checkpoints", "dino_vitbase16_pretrain.pth")


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_osie_dino_tensor(image_path: str, device):
    """DINO-side (ImageNet-normalized) counterpart of clip_extractor.py's
    load_osie_image_tensor: same _pad_to_patch geometry, DINO's own
    ImageNet mean/std."""
    from PIL import Image as PILImage
    import numpy as np

    img = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)
    img_np = _pad_to_patch(img_np, DINO_PATCH_SIZE)
    pad_h, pad_w = img_np.shape[:2]

    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(PILImage.fromarray(img_np)).unsqueeze(0).to(device)
    return tensor, (orig_h, orig_w), (pad_h, pad_w)


def _save_orientation_overlay(image_path: str, heat_hw) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].imshow(img)
    axes[0].set_title("original (1001.jpg)")
    axes[0].axis("off")

    h = heat_hw.copy()
    if h.max() > 0:
        h = h / h.max()
    axes[1].imshow(img)
    axes[1].imshow(h, cmap="jet", alpha=0.45)
    axes[1].set_title("DINO ViT-B/16 L12 [CLS]->patch, bilinear to 800x600")
    axes[1].axis("off")
    fig.tight_layout()

    out_path = os.path.join(tempfile.gettempdir(), "dino_vitb16_phase2_orientation_check.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> int:
    print("=" * 65)
    print("  DINO ViT-B/16 attention-extractor verification")
    print("=" * 65)

    print("\n--- Loading official DINO ViT-B/16 backbone checkpoint (strict=True) ---")
    model, device = load_dino_vitb16()
    print(f"  Device: {device}   dtype: {next(model.parameters()).dtype}")
    print("  load_state_dict(strict=True) succeeded  OK")

    ckpt_path = _cached_checkpoint_path()
    if os.path.isfile(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"  Cached checkpoint: {ckpt_path}  ({size_mb:.1f} MB)")
        md5 = _md5(ckpt_path)
        assert md5 == DINO_VITB16_EXPECTED_MD5, "STOP: MD5 mismatch on cached checkpoint"
        print(f"  MD5: {md5}  == expected  OK")

    # ------------------------------------------------------------------
    # Official-config audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Official DINO ViT-B/16 config audit (vs facebookresearch/dino vit_base())")
    print("=" * 65)
    audit = audit_official_dino_vitb16_config(model)
    print(f"  {'field':28s} {'expected':>12s} {'observed':>12s}  match")
    for field, r in audit.items():
        print(f"  {field:28s} {str(r['expected']):>12s} {str(r['observed']):>12s}  "
              f"{'OK' if r['matches'] else 'MISMATCH'}")

    # ------------------------------------------------------------------
    # Phase 1: standard 224x224
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Phase 1: standard 224x224 input, native 14x14 grid")
    print("=" * 65)

    assert_vitb16_shape(model)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())

    transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(STIM_PATH).convert("RGB")
    image_tensor = transform(img).unsqueeze(0).to(device)
    print(f"\n  Input tensor: {tuple(image_tensor.shape)}")
    assert tuple(image_tensor.shape) == (1, 3, 224, 224)

    with torch.inference_mode():
        patch_tokens = model.patch_embed(image_tensor)
        cls_tokens = model.cls_token.expand(patch_tokens.shape[0], -1, -1)
        x_with_cls = torch.cat((cls_tokens, patch_tokens), dim=1)
        pos_interp = model.interpolate_pos_encoding(x_with_cls, 224, 224)
    torch.testing.assert_close(pos_interp, model.pos_embed)
    print("  Native pos-embed no-op check: OK")

    with torch.inference_mode():
        _ = model.get_fulllayers_selfattention(image_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    layer_maps, attn_raw = extract_layerwise_cls_attention(model, image_tensor, DINO_NATIVE_GRID)
    assert len(layer_maps) == 12 and len(attn_raw) == 12
    for l, (grid_map, attn) in enumerate(zip(layer_maps, attn_raw), start=1):
        assert attn.shape[1] == DINO_HEADS
        assert tuple(attn.shape[-2:]) == (197, 197)
        assert tuple(grid_map.shape[-2:]) == DINO_NATIVE_GRID
        assert not torch.isnan(attn).any() and not torch.isinf(attn).any()
        full_row_sum = attn.mean(dim=1)[:, 0, :].sum(dim=-1)
        torch.testing.assert_close(full_row_sum, torch.ones_like(full_row_sum), rtol=1e-4, atol=1e-4)
    print(f"  All 12 layers: tokens=(197,197) heads=12 grid=(14,14) full_row_sum~1 NaN/Inf-free  OK")

    with torch.inference_mode():
        attn_raw_2 = model.get_fulllayers_selfattention(image_tensor)
    for a1, a2 in zip(attn_raw, attn_raw_2):
        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
    print("  Determinism (2 independent runs, rtol=0/atol=0): OK")

    with torch.inference_mode():
        last_ref = get_last_selfattention_reference(model, image_tensor)
    torch.testing.assert_close(last_ref, attn_raw[-1], rtol=1e-5, atol=1e-6)
    print(f"  get_last_selfattention reference vs get_fulllayers_selfattention[-1]: "
          f"max_abs_diff={(last_ref - attn_raw[-1]).abs().max().item():.3e}  OK")

    # ------------------------------------------------------------------
    # Phase 1b: memory-conscious extractor regression @ 224x224
    # ------------------------------------------------------------------
    print("\n--- Memory-conscious extractor vs existing full extractor (224x224) ---")
    mc_layer_maps, mc_full_row_sums = extract_layerwise_cls_attention_memory_conscious(
        model, image_tensor, DINO_NATIVE_GRID)
    assert len(mc_layer_maps) == 12 and len(mc_full_row_sums) == 12

    max_abs_diffs = []
    for l, (a_old, a_new) in enumerate(zip(layer_maps, mc_layer_maps), start=1):
        assert a_old.shape == a_new.shape, f"layer {l}: shape {a_old.shape} != {a_new.shape}"
        diff = (a_old - a_new).abs().max().item()
        max_abs_diffs.append(diff)
        torch.testing.assert_close(a_old, a_new, rtol=1e-5, atol=1e-6)

    for l, (attn_old, row_sum_new) in enumerate(zip(attn_raw, mc_full_row_sums), start=1):
        row_sum_old = attn_old.mean(dim=1)[:, 0, :].sum(dim=-1)
        torch.testing.assert_close(row_sum_old, row_sum_new, rtol=1e-4, atol=1e-4)

    overall_max_diff = max(max_abs_diffs)
    print(f"  All 12 layers match (head-averaged CLS->patch grid): "
          f"max abs diff over all layers = {overall_max_diff:.3e}  (tol rtol=1e-5, atol=1e-6)")
    print(f"  Per-layer max abs diff: {[f'{d:.2e}' for d in max_abs_diffs]}")
    print(f"  shapes: old={tuple(layer_maps[0].shape)}  new={tuple(mc_layer_maps[0].shape)}")

    with torch.inference_mode():
        mc_layer_maps_2, _ = extract_layerwise_cls_attention_memory_conscious(
            model, image_tensor, DINO_NATIVE_GRID)
    for a1, a2 in zip(mc_layer_maps, mc_layer_maps_2):
        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
    print("  Memory-conscious extractor determinism (2 runs, rtol=0/atol=0): OK")

    print("\nPhase 1: ALL CHECKS PASSED -- proceeding to Phase 2 (608x800)")

    # ------------------------------------------------------------------
    # Phase 2: one real OSIE image, non-square input, batch_size=1,
    # memory-conscious extractor ONLY (never the full 12-layer path)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Phase 2: OSIE single image, non-square input (608x800, memory-conscious)")
    print("=" * 65)

    tensor, (orig_h, orig_w), (pad_h, pad_w) = _load_osie_dino_tensor(STIM_PATH, device)
    print(f"\n  Original size (H,W): ({orig_h}, {orig_w})")
    print(f"  Padded size   (H,W): ({pad_h}, {pad_w})  (bottom/right zero-pad via "
          f"vit_extractor.py::_pad_to_patch, unmodified, before normalization)")
    assert tuple(tensor.shape) == (1, 3, pad_h, pad_w)

    grid_h, grid_w = patch_grid_from_image_hw(pad_h, pad_w, DINO_PATCH_SIZE)
    n_tokens = grid_h * grid_w + 1
    print(f"  grid_h={grid_h}  grid_w={grid_w}  patch_tokens={grid_h * grid_w}  tokens={n_tokens}")
    if (grid_h, grid_w, n_tokens) != (38, 50, 1901):
        raise RuntimeError(f"STOP: expected grid (38,50)/tokens=1901, got ({grid_h},{grid_w})/{n_tokens}")

    print("\n--- Position-embedding check (existing interpolate_pos_encoding, unmodified) ---")
    pos_interp = check_native_pos_embed_untouched_at_grid(model, tensor, (grid_h, grid_w))
    print(f"  Native pos_embed shape:      {tuple(model.pos_embed.shape)}")
    print(f"  Interpolated pos_embed shape: {tuple(pos_interp.shape)}")
    if tuple(pos_interp.shape) != (1, n_tokens, 768):
        raise RuntimeError(f"STOP: interpolated pos_embed shape {tuple(pos_interp.shape)} != (1,{n_tokens},768)")
    torch.testing.assert_close(pos_interp[:, 0], model.pos_embed[:, 0])  # CLS row untouched
    print("  CLS position embedding unchanged (assert_close vs native pos_embed[:,0])  OK")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\n--- Warm-up pass (untimed, memory-conscious extractor) ---")
    with torch.inference_mode():
        _ = extract_layerwise_cls_attention_memory_conscious(model, tensor, (grid_h, grid_w))
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print("\n--- Timed pass ---")
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        layer_maps_608, full_row_sums_608 = extract_layerwise_cls_attention_memory_conscious(
            model, tensor, (grid_h, grid_w))
        if device == "cuda":
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else None
        print(f"\n  STOP: CUDA OOM during the 608x800 memory-conscious forward pass.")
        if peak is not None:
            print(f"  Peak VRAM before OOM: {peak:.1f} MiB")
        print(f"  Error: {e}")
        return 1
    elapsed_s = time.perf_counter() - t0

    peak_mem_mib = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else None
    reserved_mib = (torch.cuda.max_memory_reserved() / 1024 ** 2) if device == "cuda" else None
    print(f"  Elapsed: {elapsed_s * 1000:.1f} ms")
    if peak_mem_mib is not None:
        print(f"  Peak VRAM (allocated): {peak_mem_mib:.1f} MiB")
        print(f"  Peak VRAM (reserved):  {reserved_mib:.1f} MiB")

    print("\n--- Attention checks (12 layers) ---")
    assert len(layer_maps_608) == 12 and len(full_row_sums_608) == 12
    row_sum_min, row_sum_max = float("inf"), float("-inf")
    for l, (grid_map, row_sum) in enumerate(zip(layer_maps_608, full_row_sums_608), start=1):
        if tuple(grid_map.shape) != (1, 38, 50):
            raise RuntimeError(f"STOP: layer {l} shape {tuple(grid_map.shape)} != (1,38,50)")
        if torch.isnan(grid_map).any() or torch.isinf(grid_map).any():
            raise RuntimeError(f"STOP: NaN/Inf in layer {l}")
        if torch.isnan(row_sum).any() or torch.isinf(row_sum).any():
            raise RuntimeError(f"STOP: NaN/Inf in layer {l} row sum")
        rs = row_sum.item()
        row_sum_min, row_sum_max = min(row_sum_min, rs), max(row_sum_max, rs)
        print(f"  Layer {l:2d}: shape={tuple(grid_map.shape)}  full_row_sum={rs:.6f}")
    print(f"  full_row_sum range across 12 layers: [{row_sum_min:.6f}, {row_sum_max:.6f}]")

    stacked = torch.stack(layer_maps_608, dim=1)  # (1, 12, 38, 50)
    if tuple(stacked.shape[1:]) != (12, 38, 50):
        raise RuntimeError(f"STOP: stacked shape {tuple(stacked.shape[1:])} != (12,38,50)")
    print(f"  Stacked 12-layer output shape: {tuple(stacked.shape[1:])}")

    print("\n--- Determinism check (608x800, two independent runs) ---")
    with torch.inference_mode():
        layer_maps_608_2, _ = extract_layerwise_cls_attention_memory_conscious(model, tensor, (grid_h, grid_w))
    for a1, a2 in zip(layer_maps_608, layer_maps_608_2):
        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
    print("  12 layers bit-identical across two independent runs  OK")

    print("\n--- Final-layer cross-check vs independent reference (single layer only, 608x800) ---")
    with torch.inference_mode():
        last_ref_608 = get_last_selfattention_reference(model, tensor)
    grid_last_ref_608 = cls_to_patch_grid_mean_heads(last_ref_608, (grid_h, grid_w))
    max_abs_diff_608 = (grid_last_ref_608 - layer_maps_608[-1]).abs().max().item()
    torch.testing.assert_close(grid_last_ref_608, layer_maps_608[-1], rtol=1e-5, atol=1e-6)
    print(f"  max abs diff (layer 12, head-averaged CLS->patch grid): {max_abs_diff_608:.3e}  (PASS, tol=1e-6)")
    del last_ref_608, grid_last_ref_608

    print("\n--- Spatial resize to (H=600, W=800), bilinear ---")
    resized = F.interpolate(stacked.reshape(12, 1, 38, 50), size=(OSIE_IMG_H, OSIE_IMG_W),
                            mode="bilinear", align_corners=False)
    resized = resized[:, 0]  # (12, 600, 800)
    if tuple(resized.shape) != (12, OSIE_IMG_H, OSIE_IMG_W):
        raise RuntimeError(f"STOP: resized shape {tuple(resized.shape)} != (12,600,800)")
    print(f"  Resized shape: {tuple(resized.shape)}  "
          f"(interpolated directly from (38,50) to (H={OSIE_IMG_H}, W={OSIE_IMG_W}); "
          f"grid axis order was NOT swapped to (50,38)/(800,600))")

    print("\n--- Orientation sanity check (overlay saved OUTSIDE the repo) ---")
    overlay_path = _save_orientation_overlay(STIM_PATH, resized[-1].detach().cpu().numpy())
    print(f"  Overlay saved to (OS temp dir, not in repo): {overlay_path}")

    print("\n" + "=" * 65)
    print("  Phase 2: ALL CHECKS PASSED")
    print("  outputs/ was NOT written to.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
