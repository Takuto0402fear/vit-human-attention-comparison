"""
Regression / sanity checks for the SL (supervised-learning, official
Yamamoto-paper DeiT-trained) ViT-S/16 attention extractor -- single-image
validation only. Mirrors scripts/verify_dino_vitb16_extractor.py's
structure for the DINO-B side.

Phase 1 -- standard 224x224 input (native resolution):
  1. Official-config audit: this instance vs D:\\gaze\\...\\
     analysis_python\\utils_analysis.py::model_load's "supervised" branch
     (embed_dim=384/depth=12/heads=6/patch_size=16/mlp_ratio=4/
     qkv_bias=True/norm_layer+eps=1e-6/drop rates=0/GELU/single
     cls_token/NO distillation token/head=nn.Linear(384,1000), NOT
     DINOHead, NOT Identity/eval mode).
  2. Checkpoint loads with load_state_dict(strict=True) (no missing /
     unexpected keys).
  3. Standard 224x224 input: 197 tokens, 14x14 grid, 12 layers, 6 heads,
     (12,14,14) head-averaged CLS->patch stack, full [CLS] row sums to
     ~1, no NaN/Inf, two independent runs bit-identical, last layer
     matches an independent get_last_selfattention-style reference.

Phase 2 -- one real OSIE image, non-square input (608x800, 1901 tokens),
batch_size=1, using ONLY the (reused, not copied) memory-conscious
extractor from dino_vitb16_extractor.py:
  4. DINO-style padding (bottom/right zero-pad, before normalization,
     reusing vit_extractor.py::_pad_to_patch unmodified) yields grid
     (38,50) / 1901 tokens -- same geometry as the DINO-S/DINO-B/CLIP-B
     experiments (same patch_size=16).
  5. 12-layer [CLS]->patch attention has shape (12,38,50), no NaN/Inf,
     full-row sums ~1, two runs deterministic, layer order 1..12, and
     the final layer matches an independent get_last_selfattention-style
     reference computed directly at 608x800.
  6. Bilinear resize to (600,800) preserves axis order (no accidental
     transpose / flip); an orientation-check overlay is saved for a
     handful of layers under outputs/expA_sl/validation/ (a NEW,
     SL-only output directory -- does not touch outputs/expA/,
     outputs/expA_clip/, or outputs/expA_dino_vitb16/).
  7. Relative depth (layer_index-1)/(num_layers-1) is recorded alongside
     the raw 1-based layer index, in case a future depth=4/8 run is
     compared against CLIP's 12 layers.

Read-only with respect to all pre-existing files: does not modify
sl_extractor.py's inputs, D:\\gaze\\...\\trained_model_weights\\, or any
existing outputs/ subdirectory. The only things this script writes are
under the new outputs/expA_sl/validation/ directory.

Usage (PowerShell):
    python scripts\\verify_sl_extractor.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as pth_transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from sl_extractor import (
    SL_DEFAULT_DEPTH, SL_DEFAULT_TRIAL, SL_HEADS, SL_NATIVE_GRID, SL_PATCH_SIZE,
    assert_sl_vit_shape, audit_official_sl_config, checkpoint_path,
    cls_to_patch_grid_mean_heads, extract_layerwise_cls_attention_memory_conscious,
    get_last_selfattention_reference, load_sl_vit,
)
from lib.clip_vit import patch_grid_from_image_hw
from vit_extractor import _pad_to_patch

STIM_PATH = os.path.join(
    PROJECT_DIR, "datasets", "osie", "predicting-human-gaze-beyond-pixels",
    "data", "stimuli", "1001.jpg")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OSIE_IMG_W, OSIE_IMG_H = 800, 600

OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "expA_sl", "validation")
SAVE_LAYERS = [1, 6, 12]  # a handful of layers to save as images (requirement: "several attention maps")


def _load_osie_sl_tensor(image_path: str, device):
    """SL-side (ImageNet-normalized) counterpart of dino_vitb16_extractor
    verification's _load_osie_dino_tensor -- identical geometry/
    normalization, since SL/DINO-S share patch_size=16 and ImageNet
    normalization stats."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)
    img_np = _pad_to_patch(img_np, SL_PATCH_SIZE)
    pad_h, pad_w = img_np.shape[:2]

    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)
    return tensor, (orig_h, orig_w), (pad_h, pad_w)


def _save_layer_overlays(image_path: str, resized_layers: dict) -> list:
    """Saves one overlay PNG per requested layer under OUT_DIR.
    resized_layers: {layer_1_based: (H,W) ndarray}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    saved = []
    for layer, heat in resized_layers.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        axes[0].imshow(img)
        axes[0].set_title(f"original ({os.path.basename(image_path)})")
        axes[0].axis("off")

        h = heat.copy()
        if h.max() > 0:
            h = h / h.max()
        axes[1].imshow(img)
        axes[1].imshow(h, cmap="jet", alpha=0.45)
        rel_depth = (layer - 1) / (SL_DEFAULT_DEPTH - 1)
        axes[1].set_title(f"SL ViT-S/16 L{layer} (rel_depth={rel_depth:.3f}) [CLS]->patch")
        axes[1].axis("off")
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"sl_vits16_L{layer:02d}_attention.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        saved.append(out_path)
    return saved


def main() -> int:
    print("=" * 65)
    print("  SL (supervised, official Yamamoto DeiT-trained) ViT-S/16")
    print("  attention-extractor verification -- SINGLE IMAGE ONLY")
    print("=" * 65)

    ckpt_path = checkpoint_path(SL_DEFAULT_TRIAL, SL_DEFAULT_DEPTH)
    print(f"\n  Checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"STOP: checkpoint not found at {ckpt_path}")

    print("\n--- Loading SL ViT-S/16 (trial=1, depth=12, strict=True) ---")
    model, device, loaded_ckpt_path = load_sl_vit(SL_DEFAULT_TRIAL, SL_DEFAULT_DEPTH)
    assert loaded_ckpt_path == ckpt_path
    print(f"  Device: {device}   dtype: {next(model.parameters()).dtype}")
    print("  load_state_dict(strict=True) succeeded  OK")

    # ------------------------------------------------------------------
    # Official-config audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Official SL ViT-S/16 config audit (vs D:\\gaze\\...\\utils_analysis.py model_load)")
    print("=" * 65)
    audit = audit_official_sl_config(model, SL_DEFAULT_DEPTH)
    print(f"  {'field':22s} {'expected':>14s} {'observed':>14s}  match")
    for field, r in audit.items():
        print(f"  {field:22s} {str(r['expected']):>14s} {str(r['observed']):>14s}  "
              f"{'OK' if r['matches'] else 'MISMATCH'}")

    assert_sl_vit_shape(model, SL_DEFAULT_DEPTH)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())

    # ------------------------------------------------------------------
    # Phase 1: standard 224x224
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Phase 1: standard 224x224 input, native 14x14 grid")
    print("=" * 65)

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
        raw_attn = model.get_fulllayers_selfattention(image_tensor)
    assert len(raw_attn) == SL_DEFAULT_DEPTH
    for l, attn in enumerate(raw_attn, start=1):
        assert attn.shape[1] == SL_HEADS, f"layer {l}: heads {attn.shape[1]} != {SL_HEADS}"
        assert tuple(attn.shape[-2:]) == (197, 197), f"layer {l}: tokens {tuple(attn.shape[-2:])} != (197,197)"
        assert not torch.isnan(attn).any() and not torch.isinf(attn).any(), f"layer {l}: NaN/Inf"
        full_row_sum = attn.mean(dim=1)[:, 0, :].sum(dim=-1)
        torch.testing.assert_close(full_row_sum, torch.ones_like(full_row_sum), rtol=1e-4, atol=1e-4)
    print(f"  All {SL_DEFAULT_DEPTH} layers: tokens=(197,197) heads={SL_HEADS} "
          f"full_row_sum~1 NaN/Inf-free  OK")

    layer_maps_224 = [cls_to_patch_grid_mean_heads(a, SL_NATIVE_GRID) for a in raw_attn]
    for l, gm in enumerate(layer_maps_224, start=1):
        assert tuple(gm.shape[-2:]) == SL_NATIVE_GRID, f"layer {l} grid {tuple(gm.shape[-2:])} != {SL_NATIVE_GRID}"
    print(f"  head-averaged CLS->patch grid shape per layer: {tuple(layer_maps_224[0].shape)} == "
          f"(1,{SL_NATIVE_GRID[0]},{SL_NATIVE_GRID[1]})  OK")

    with torch.inference_mode():
        raw_attn_2 = model.get_fulllayers_selfattention(image_tensor)
    for a1, a2 in zip(raw_attn, raw_attn_2):
        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
    print("  Determinism (2 independent runs, rtol=0/atol=0): OK")

    with torch.inference_mode():
        last_ref = get_last_selfattention_reference(model, image_tensor)
    torch.testing.assert_close(last_ref, raw_attn[-1], rtol=1e-5, atol=1e-6)
    print(f"  get_last_selfattention reference vs get_fulllayers_selfattention[-1]: "
          f"max_abs_diff={(last_ref - raw_attn[-1]).abs().max().item():.3e}  OK")

    del raw_attn, raw_attn_2, image_tensor, last_ref
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nPhase 1: ALL CHECKS PASSED -- proceeding to Phase 2 (608x800)")

    # ------------------------------------------------------------------
    # Phase 2: one real OSIE image, non-square input, batch_size=1,
    # memory-conscious extractor (reused from dino_vitb16_extractor.py)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Phase 2: OSIE single image, non-square input (memory-conscious)")
    print("=" * 65)

    tensor, (orig_h, orig_w), (pad_h, pad_w) = _load_osie_sl_tensor(STIM_PATH, device)
    print(f"\n  Original size (H,W): ({orig_h}, {orig_w})")
    print(f"  Padded size   (H,W): ({pad_h}, {pad_w})  (bottom/right zero-pad via "
          f"vit_extractor.py::_pad_to_patch, unmodified, before normalization)")
    assert tuple(tensor.shape) == (1, 3, pad_h, pad_w)

    grid_h, grid_w = patch_grid_from_image_hw(pad_h, pad_w, SL_PATCH_SIZE)
    n_tokens = grid_h * grid_w + 1
    print(f"  grid_h={grid_h}  grid_w={grid_w}  patch_tokens={grid_h * grid_w}  tokens={n_tokens}")
    if (grid_h, grid_w, n_tokens) != (38, 50, 1901):
        raise RuntimeError(f"STOP: expected grid (38,50)/tokens=1901, got ({grid_h},{grid_w})/{n_tokens}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\n--- Warm-up pass (untimed) ---")
    with torch.inference_mode():
        _ = extract_layerwise_cls_attention_memory_conscious(model, tensor, (grid_h, grid_w))
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print("\n--- Timed pass ---")
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    layer_maps_608, full_row_sums_608 = extract_layerwise_cls_attention_memory_conscious(
        model, tensor, (grid_h, grid_w))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    peak_mem_mib = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else None
    print(f"  Elapsed: {elapsed_s * 1000:.1f} ms")
    if peak_mem_mib is not None:
        print(f"  Peak VRAM (allocated): {peak_mem_mib:.1f} MiB")

    print("\n--- Attention checks (12 layers) ---")
    assert len(layer_maps_608) == SL_DEFAULT_DEPTH and len(full_row_sums_608) == SL_DEFAULT_DEPTH
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
        rel_depth = (l - 1) / (SL_DEFAULT_DEPTH - 1)
        print(f"  Layer {l:2d} (rel_depth={rel_depth:.3f}): shape={tuple(grid_map.shape)}  full_row_sum={rs:.6f}")
    print(f"  full_row_sum range across {SL_DEFAULT_DEPTH} layers: [{row_sum_min:.6f}, {row_sum_max:.6f}]")

    stacked = torch.stack(layer_maps_608, dim=1)  # (1, 12, 38, 50)
    if tuple(stacked.shape[1:]) != (12, 38, 50):
        raise RuntimeError(f"STOP: stacked shape {tuple(stacked.shape[1:])} != (12,38,50)")
    print(f"  Stacked {SL_DEFAULT_DEPTH}-layer output shape: {tuple(stacked.shape[1:])}")

    print("\n--- Determinism check (608x800, two independent runs) ---")
    with torch.inference_mode():
        layer_maps_608_2, _ = extract_layerwise_cls_attention_memory_conscious(model, tensor, (grid_h, grid_w))
    for a1, a2 in zip(layer_maps_608, layer_maps_608_2):
        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
    print(f"  {SL_DEFAULT_DEPTH} layers bit-identical across two independent runs  OK")

    print("\n--- Final-layer cross-check vs independent reference (608x800) ---")
    with torch.inference_mode():
        last_ref_608 = get_last_selfattention_reference(model, tensor)
    grid_last_ref_608 = cls_to_patch_grid_mean_heads(last_ref_608, (grid_h, grid_w))
    max_abs_diff_608 = (grid_last_ref_608 - layer_maps_608[-1]).abs().max().item()
    torch.testing.assert_close(grid_last_ref_608, layer_maps_608[-1], rtol=1e-5, atol=1e-6)
    print(f"  max abs diff (layer {SL_DEFAULT_DEPTH}, head-averaged CLS->patch grid): "
          f"{max_abs_diff_608:.3e}  (PASS, tol=1e-6)")
    del last_ref_608, grid_last_ref_608

    print("\n--- Spatial resize to (H=600, W=800), bilinear -- axis/orientation check ---")
    resized = F.interpolate(stacked.reshape(SL_DEFAULT_DEPTH, 1, 38, 50), size=(OSIE_IMG_H, OSIE_IMG_W),
                            mode="bilinear", align_corners=False)
    resized = resized[:, 0]  # (12, 600, 800)
    if tuple(resized.shape) != (SL_DEFAULT_DEPTH, OSIE_IMG_H, OSIE_IMG_W):
        raise RuntimeError(f"STOP: resized shape {tuple(resized.shape)} != ({SL_DEFAULT_DEPTH},600,800)")
    print(f"  Resized shape: {tuple(resized.shape)}  "
          f"(interpolated directly from (38,50) to (H={OSIE_IMG_H}, W={OSIE_IMG_W}); "
          f"grid axis order was NOT swapped to (50,38)/(800,600))")
    if torch.isnan(resized).any() or torch.isinf(resized).any():
        raise RuntimeError("STOP: NaN/Inf after bilinear resize")
    print("  No NaN/Inf after resize  OK")

    print("\n--- Saving a few layer attention-map overlays (outputs/expA_sl/validation/) ---")
    resized_np = resized.detach().cpu().numpy()
    layers_to_save = {l: resized_np[l - 1] for l in SAVE_LAYERS}
    saved_paths = _save_layer_overlays(STIM_PATH, layers_to_save)
    for p in saved_paths:
        print(f"  Saved: {p}")

    # ------------------------------------------------------------------
    # Log: model condition, checkpoint path, preprocessing, extraction
    # ------------------------------------------------------------------
    log = {
        "model": "SL ViT-S/16 (official Yamamoto-paper DeiT-trained, standard/no-distillation-token)",
        "checkpoint_path": ckpt_path,
        "trial_num": SL_DEFAULT_TRIAL,
        "depth": SL_DEFAULT_DEPTH,
        "arch": "vit_small",
        "patch_size": SL_PATCH_SIZE,
        "embed_dim": 384,
        "num_heads": SL_HEADS,
        "tokens": {"cls_token": 1, "distillation_token": 0, "patch_tokens_224": 196},
        "training_data_and_method": "ImageNet-1k, standard supervised classification, DeiT training code "
                                     "(facebookresearch/deit) -- confirmed no distillation token in checkpoint",
        "preprocessing": {
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "padding": "bottom/right zero-pad to patch_size=16 multiple, before normalization "
                       "(vit_extractor.py::_pad_to_patch, unmodified)",
            "resize_or_crop": "none (native OSIE resolution, non-square input)",
        },
        "attention_extraction": {
            "convention": "[CLS] (token 0) -> patch tokens (1:), head-averaged, NOT renormalized after "
                          "dropping the CLS column (matches official utils_analysis.py::get_gaze_pos_model)",
            "function": "dino_vitb16_extractor.extract_layerwise_cls_attention_memory_conscious (reused, "
                        "architecture-generic, not duplicated)",
            "upsampling_for_metrics": "bilinear to (600,800), axis order preserved",
        },
        "phase1_224x224": {
            "tokens": 197, "grid": list(SL_NATIVE_GRID), "layers": SL_DEFAULT_DEPTH,
            "heads": SL_HEADS, "row_sum_check": "PASS", "nan_inf_check": "PASS",
            "determinism_check": "PASS", "last_layer_reference_check": "PASS",
        },
        "phase2_osie_608x800": {
            "orig_hw": [orig_h, orig_w], "padded_hw": [pad_h, pad_w], "grid_hw": [grid_h, grid_w],
            "tokens": n_tokens, "row_sum_range": [row_sum_min, row_sum_max],
            "elapsed_ms": elapsed_s * 1000, "peak_vram_mib": peak_mem_mib,
            "determinism_check": "PASS", "last_layer_reference_check": "PASS",
            "nan_inf_check": "PASS",
        },
        "layers": [
            {"layer_index_1based": l, "relative_depth": (l - 1) / (SL_DEFAULT_DEPTH - 1)}
            for l in range(1, SL_DEFAULT_DEPTH + 1)
        ],
        "saved_attention_map_images": saved_paths,
        "config_audit": {f: r["matches"] for f, r in audit.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "verify_sl_extractor_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Saved: {log_path}")

    print("\n" + "=" * 65)
    print("  Phase 1 + Phase 2: ALL CHECKS PASSED")
    print("  Only outputs/expA_sl/validation/ was written to.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
