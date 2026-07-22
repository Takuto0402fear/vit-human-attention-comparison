"""
DINO ViT-S/16 vs CLIP ViT-B/16 -- Attention comparison figures for the 3
representative images selected by scripts/plot_dino_clip_layerwise.py
(outputs/expA_clip/comparison/representative_image_selection.csv).

For each image: [original] [human fixation overlay] on top, then a
2 (model) x 4 (layer: 4, 8, 10, 12) grid of attention overlays below.

DINO source: outputs/attn_cache/dino_vits16_patchgrid.npz, read-only.
This cache stores the RAW per-layer [CLS]->patch mean-head attention at
patch-grid resolution (700, 12, 38, 50), BEFORE the bilinear upsample to
800x600 and the sum-to-1 renormalization -- confirmed by reading
scripts/run_expB.py::get_attention_cache(), which reconstructs the
official per-layer saliency map from this cache with exactly
`F.interpolate(size=(600,800), mode="bilinear", align_corners=False)`
then per-layer sum-to-1 normalize. That exact reconstruction is
duplicated here (read-only w.r.t. the cache and w.r.t. run_expB.py / the
DINO model -- no DINO file is imported or modified, and the DINO model
is never reloaded for this script).

CLIP source: recomputed for just these 3 images via the validated
clip_extractor.py / lib/clip_vit.py extractor (same convention as
scripts/run_expA_clip_full700.py: batch_size=1, fp32, inference_mode,
bilinear upsample to 800x600, per-layer sum-to-1 normalize). Neither
clip_extractor.py, lib/clip_vit.py, nor run_expA_clip_full700.py is
modified.

Human fixation overlay: reuses the exact overlay convention from
scripts/generate_fixmaps.py::save_overlay_png (imshow(img) then
imshow(heat/heat.max(), cmap="jet", alpha=0.45)) -- the only existing
"half-transparent overlay on the stimulus" tone-and-manner in this repo;
no dedicated DINO *attention* visualization script existed previously,
so this is the closest applicable precedent, and it is matched exactly
for both the fixation panel and the attention panels.

Display-only processing (documented here and in visualization_config.json,
never fed back into any CSV or metric):
  - The per-layer sum-to-1 normalization is NOT display-only -- it is the
    same step scripts/run_expA.py and scripts/run_expA_clip_full700.py
    already apply before computing NSS/AUC-Judd/sAUC, reused unchanged.
  - What IS display-only: within one image, all 8 attention panels
    (DINO L4/L8/L10/L12 + CLIP L4/L8/L10/L12) share a single vmin=0 /
    vmax = max-over-the-8-maps color scale, so that raw magnitude
    differences between models/layers remain visible (independent
    per-panel min-max normalization would flatten them away).

Reads (read-only): representative_image_selection.csv, the DINO
attention cache, OSIE stimuli, outputs/fixmaps/healthy/*.npz.
Writes only under outputs/expA_clip/comparison/representative_attention/
and a montage in outputs/expA_clip/comparison/.

Usage (PowerShell):
    python scripts\\render_dino_clip_attention_comparison.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from clip_extractor import CLIP_PATCH_SIZE, load_clip_vit_b16, load_osie_image_tensor
from lib.clip_vit import (
    interpolate_patch_pos_embed, patch_grid_from_image_hw,
    visual_forward_with_cls_patch_attention,
)

# ======================= CONFIG =======================
IMG_W, IMG_H = 800, 600
LAYERS_SHOWN = [4, 8, 10, 12]
CMAP = "jet"
OVERLAY_ALPHA = 0.45

DATA_BASE  = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR   = os.path.join(DATA_BASE, "data", "stimuli")
FIXMAP_DIR = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
DINO_ATTN_CACHE = r"C:\Users\user\gaze\outputs\attn_cache\dino_vits16_patchgrid.npz"

COMPARISON_DIR = r"C:\Users\user\gaze\outputs\expA_clip\comparison"
SELECTION_CSV = os.path.join(COMPARISON_DIR, "representative_image_selection.csv")
ATTN_OUT_DIR = os.path.join(COMPARISON_DIR, "representative_attention")
MONTAGE_PNG = os.path.join(COMPARISON_DIR, "representative_3images_montage.png")
MONTAGE_PDF = os.path.join(COMPARISON_DIR, "representative_3images_montage.pdf")
VIZ_CONFIG_JSON = os.path.join(COMPARISON_DIR, "visualization_config.json")
# ======================================================


def load_selection():
    with open(SELECTION_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_dino_layers(image_id, layers):
    """
    Read-only: reconstruct the official per-layer DINO saliency maps for
    one image from the patch-grid cache (see module docstring for the
    exact reconstruction, duplicated from run_expB.py::get_attention_cache).
    Returns {layer: (IMG_H, IMG_W) float32, sum=1}.
    """
    cache = np.load(DINO_ATTN_CACHE)
    stems = list(cache["stems"])
    if image_id not in stems:
        raise RuntimeError(f"STOP: {image_id} not found in DINO attention cache")
    idx = stems.index(image_id)
    pg = cache["attn"][idx]  # (12, 38, 50)

    t = torch.from_numpy(pg).unsqueeze(1).float()  # (12,1,38,50)
    t_up = F.interpolate(t, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
    m = t_up[:, 0].numpy()  # (12, IMG_H, IMG_W)
    for l in range(12):
        s = m[l].sum()
        if s > 0:
            m[l] /= s

    return {l: m[l - 1] for l in layers}


def get_clip_layers(model, image_id, layers):
    """
    Recompute CLIP attention for one image (batch_size=1, fp32,
    inference_mode), same convention as run_expA_clip_full700.py.
    Returns {layer: (IMG_H, IMG_W) float32, sum=1}.
    """
    device = next(model.parameters()).device
    image_path = os.path.join(STIM_DIR, f"{image_id}.jpg")
    tensor, orig_hw, pad_hw = load_osie_image_tensor(image_path, CLIP_PATCH_SIZE)
    tensor = tensor.to(device)
    grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
    pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)

    with torch.inference_mode():
        _, cls_patch_maps, _ = visual_forward_with_cls_patch_attention(
            model.visual, tensor.type(model.dtype), grid_hw, pos_embed=pos_interp)

    stacked = torch.stack(cls_patch_maps, dim=1)  # (1, 12, gh, gw)
    resized = F.interpolate(stacked.reshape(12, 1, *grid_hw), size=(IMG_H, IMG_W),
                            mode="bilinear", align_corners=False)
    resized = resized[:, 0].detach().cpu().numpy().astype(np.float32)  # (12, IMG_H, IMG_W)
    for l in range(resized.shape[0]):
        s = resized[l].sum()
        if s > 0:
            resized[l] /= s

    return {l: resized[l - 1] for l in layers}


def load_fixation_heatmap(image_id):
    d = np.load(os.path.join(FIXMAP_DIR, f"{image_id}.npz"))
    return d["heat_all"].astype(np.float32)


def render_one_image(image_id, selection_type, difference, model, out_dir):
    stim_path = os.path.join(STIM_DIR, f"{image_id}.jpg")
    img = Image.open(stim_path).convert("RGB")

    dino_maps = get_dino_layers(image_id, LAYERS_SHOWN)
    clip_maps = get_clip_layers(model, image_id, LAYERS_SHOWN)
    fix_heat = load_fixation_heatmap(image_id)

    all_vals = np.concatenate(
        [dino_maps[l].ravel() for l in LAYERS_SHOWN] + [clip_maps[l].ravel() for l in LAYERS_SHOWN])
    vmin, vmax = 0.0, float(all_vals.max())

    for l in LAYERS_SHOWN:
        if np.isnan(dino_maps[l]).any() or np.isinf(dino_maps[l]).any():
            raise RuntimeError(f"STOP: NaN/Inf in DINO {image_id} layer {l}")
        if np.isnan(clip_maps[l]).any() or np.isinf(clip_maps[l]).any():
            raise RuntimeError(f"STOP: NaN/Inf in CLIP {image_id} layer {l}")
        if dino_maps[l].shape != (IMG_H, IMG_W) or clip_maps[l].shape != (IMG_H, IMG_W):
            raise RuntimeError(f"STOP: shape mismatch for {image_id} layer {l}")

    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1], hspace=0.35, wspace=0.15)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img)
    ax_orig.set_title(f"{image_id}.jpg (original)", fontsize=10)
    ax_orig.axis("off")

    ax_fix = fig.add_subplot(gs[0, 1])
    ax_fix.imshow(img)
    h_disp = fix_heat.copy()
    if h_disp.max() > 0:
        h_disp = h_disp / h_disp.max()
    ax_fix.imshow(h_disp, cmap=CMAP, alpha=OVERLAY_ALPHA)
    ax_fix.set_title("Human fixation heatmap", fontsize=10)
    ax_fix.axis("off")

    for c in range(2, 4):
        fig.add_subplot(gs[0, c]).axis("off")

    last_im = None
    for row, (model_name, maps) in enumerate([("DINO ViT-S/16", dino_maps), ("CLIP ViT-B/16", clip_maps)]):
        for col, l in enumerate(LAYERS_SHOWN):
            ax = fig.add_subplot(gs[row + 1, col])
            ax.imshow(img)
            last_im = ax.imshow(maps[l], cmap=CMAP, alpha=OVERLAY_ALPHA, vmin=vmin, vmax=vmax)
            ax.set_title(f"{model_name}  L{l}", fontsize=9)
            ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.55])
    fig.colorbar(last_im, cax=cbar_ax, label="Attention (display scale, shared per image)")

    fig.suptitle(
        f"{image_id}  --  {selection_type}  (DINO L10 NSS - CLIP L4 NSS = {float(difference):+.4f})",
        fontsize=13)

    png_path = os.path.join(out_dir, f"{image_id}_dino_clip_attention.png")
    pdf_path = os.path.join(out_dir, f"{image_id}_dino_clip_attention.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_montage(selection_rows, png_paths, out_png, out_pdf):
    order = ["dino_favored_p90", "typical_p50", "clip_favored_p10"]
    by_type = {r["selection_type"]: r for r in selection_rows}
    ordered = [(t, by_type[t], png_paths[by_type[t]["image_id"]]) for t in order]

    fig, axes = plt.subplots(3, 1, figsize=(18, 30))
    for ax, (sel_type, row, png_path) in zip(axes, ordered):
        # Each per-image PNG already carries its own suptitle (image id,
        # selection type, difference) -- no extra ax.set_title() here to
        # avoid a redundant duplicate header in the montage.
        im = plt.imread(png_path)
        ax.imshow(im)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(ATTN_OUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  DINO vs CLIP attention comparison -- representative images")
    print("=" * 65)

    selection_rows = load_selection()
    if len(selection_rows) != 3:
        raise RuntimeError(f"STOP: expected 3 selected images, got {len(selection_rows)}")
    print(f"\nSelected images: {[(r['image_id'], r['selection_type']) for r in selection_rows]}")

    print("\n--- Loading CLIP ViT-B/16 ---")
    model, _ = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}  dtype: {model.dtype}")

    png_paths = {}
    for row in selection_rows:
        img_id = row["image_id"]
        print(f"\n--- Rendering {img_id} ({row['selection_type']}) ---")
        png_path, pdf_path = render_one_image(
            img_id, row["selection_type"], row["difference"], model, ATTN_OUT_DIR)
        png_paths[img_id] = png_path
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")

    print("\n--- Building montage ---")
    make_montage(selection_rows, png_paths, MONTAGE_PNG, MONTAGE_PDF)
    print(f"  Saved: {MONTAGE_PNG}")
    print(f"  Saved: {MONTAGE_PDF}")

    print("\n--- Saving visualization_config.json ---")
    viz_config = {
        "colormap": CMAP,
        "overlay_alpha": OVERLAY_ALPHA,
        "coordinate_space": {"img_w": IMG_W, "img_h": IMG_H, "note": "shared by DINO and CLIP, matches OSIE stimulus size"},
        "interpolation": "bilinear (torch.nn.functional.interpolate, align_corners=False) for BOTH models' patch-grid -> 800x600 upsample",
        "layers_shown": LAYERS_SHOWN,
        "layer_rationale": {
            "4": "CLIP's early-layer NSS peak",
            "8": "CLIP's mid-layer dip",
            "10": "DINO's NSS peak",
            "12": "final layer for both models, near the sAUC peak",
        },
        "sum_to_one_normalization": {
            "is_display_only": False,
            "note": ("Applied identically to the official evaluation pipeline in "
                     "scripts/run_expA.py and scripts/run_expA_clip_full700.py "
                     "(per-layer sum=1 after the bilinear upsample, before metric "
                     "computation) -- reused unchanged here, NOT a display-only step."),
        },
        "shared_color_scale": {
            "is_display_only": True,
            "vmin": 0.0,
            "vmax": "max over all 8 panels (DINO L4/L8/L10/L12 + CLIP L4/L8/L10/L12) for that image, computed per image",
            "note": ("Chosen so raw magnitude differences between models/layers remain "
                     "visible; independent per-panel min-max normalization would flatten "
                     "them away. This scale is used ONLY for imshow() rendering here and "
                     "is never fed back into any CSV or metric computation."),
        },
        "dino_attention_source": {
            "file": DINO_ATTN_CACHE, "access": "read-only",
            "reconstruction": "F.interpolate(size=(600,800), mode=bilinear, align_corners=False) then per-layer sum-to-1 normalize, duplicated from scripts/run_expB.py::get_attention_cache (not imported, not modified)",
        },
        "clip_attention_source": {
            "extractor": "clip_extractor.py / lib/clip_vit.py (validated, unmodified)",
            "batch_size": 1, "dtype": "float32", "mode": "torch.inference_mode()",
        },
        "human_fixation_overlay": {
            "source": "outputs/fixmaps/healthy/<id>.npz['heat_all']",
            "style_reference": "scripts/generate_fixmaps.py::save_overlay_png (imshow(img) then imshow(heat/heat.max(), cmap=jet, alpha=0.45)) -- matched exactly; no dedicated DINO attention-visualization tone-and-manner existed previously",
        },
        "orientation_check": "visually confirmed no up/down or left/right flip and no transpose (see report)",
        "outputs": {
            "per_image": [os.path.basename(p) for p in png_paths.values()],
            "montage": [os.path.basename(MONTAGE_PNG), os.path.basename(MONTAGE_PDF)],
        },
    }
    with open(VIZ_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(viz_config, f, indent=2)
    print(f"  Saved: {VIZ_CONFIG_JSON}")

    print("\n" + "=" * 65)
    print("  Done. Only outputs/expA_clip/comparison/ was written to.")
    print("  outputs/expA_clip/healthy/, pilot20/, outputs/expA/, "
          "outputs/attn_cache/ (read-only) were not modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
