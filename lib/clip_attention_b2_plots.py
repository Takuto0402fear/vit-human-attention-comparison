"""
Shared rendering for [B-2] layer-scale comparison figures (foreground mask
+ L4/L8/L12 RAW [CLS]->patch attention overlays). Used by both
scripts/plot_clip_attention_b2_pilot_layer_scale_figures.py (adds the
corrected two-version figures for the pilot's 3 existing images, without
touching the pilot's original *_overlay.png) and
scripts/report_clip_attention_b2_full700.py (the main 700-image run's
figures/).

Two variants per image, both using the SAME raw (un-normalized)
[CLS]->patch attention, colormap, alpha, upsample method (bilinear,
display-only -- metrics are always computed on the native 38x50 grid,
never on this upsampled copy), and foreground-mask contour:

  - per_layer_scale : each panel's own vmin=0/vmax=that layer's own max.
    Shows each layer's spatial pattern in isolation -- colors are NOT
    comparable in intensity across layers (stated in the figure title).
  - shared_scale    : vmin=0, vmax = max over L4/L8/L12 for that image,
    ONE shared colorbar. Colors ARE comparable in absolute attention
    magnitude across layers.
"""
from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CMAP = "jet"
ALPHA = 0.55
UPSAMPLE_RESAMPLE = Image.BILINEAR
MASK_CONTOUR_LEVEL = 0.5


def _upsample(arr_2d: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(arr_2d.astype(np.float32)).resize(size_wh, UPSAMPLE_RESAMPLE))


def render_per_layer_and_shared_scale_figures(
    stem: str,
    img_arr: np.ndarray,
    coverage: np.ndarray,
    layer_maps: Dict[int, np.ndarray],
    layers_display: Sequence[int],
    out_dir: str,
    img_wh: Tuple[int, int],
) -> Tuple[str, str]:
    """
    img_arr    : (H, W, 3) uint8 original image.
    coverage   : (gh, gw) foreground patch-coverage fraction, native grid.
    layer_maps : {layer_display: (gh, gw) raw [CLS]->patch attention}.

    Returns (per_layer_scale_path, shared_scale_path). Refuses to
    overwrite either path if it already exists (caller is expected to
    have already checked its own filenames, but this is a second,
    cheap guard specific to this shared renderer).
    """
    os.makedirs(out_dir, exist_ok=True)
    img_w, img_h = img_wh
    coverage_up = _upsample(coverage, (img_w, img_h))
    shared_vmax = max(float(m.max()) for m in layer_maps.values())

    out_paths = {}
    for mode in ("per_layer_scale", "shared_scale"):
        path = os.path.join(out_dir, f"{stem}_{mode}.png")
        if os.path.isfile(path):
            raise RuntimeError(f"STOP: output file already exists, refusing to overwrite: {path}")
        out_paths[mode] = path

    for mode, path in out_paths.items():
        fig, axes = plt.subplots(
            1, len(layers_display) + 1, figsize=(5 * (len(layers_display) + 1), 5.2))
        axes[0].imshow(img_arr)
        axes[0].imshow(
            np.where(coverage_up > MASK_CONTOUR_LEVEL, 1.0, np.nan),
            cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        axes[0].set_title(f"{stem}: image + foreground mask\n(patch coverage > 0.5)", fontsize=9)
        axes[0].axis("off")

        ims = []
        for ax, layer_display in zip(axes[1:], layers_display):
            raw = layer_maps[layer_display]
            attn_up = _upsample(raw, (img_w, img_h))
            vmin = 0.0
            vmax = float(raw.max()) if mode == "per_layer_scale" else shared_vmax
            ax.imshow(img_arr)
            im = ax.imshow(attn_up, cmap=CMAP, alpha=ALPHA, vmin=vmin, vmax=vmax)
            ax.contour(coverage_up, levels=[MASK_CONTOUR_LEVEL], colors="white", linewidths=1.2)
            ax.set_title(f"L{layer_display} raw attention", fontsize=9)
            ax.axis("off")
            ims.append(im)
            if mode == "per_layer_scale":
                plt.colorbar(im, ax=ax, fraction=0.046)

        if mode == "per_layer_scale":
            fig.suptitle(
                f"[B-2] {stem}: PER-LAYER color scale (each panel's own vmin=0/vmax=own max) -- "
                f"spatial pattern only; colors are NOT comparable across layers",
                fontsize=11)
            fig.tight_layout()
        else:
            fig.colorbar(
                ims[-1], ax=axes[1:].tolist(), fraction=0.025, pad=0.02,
                label="raw [CLS]->patch attention (shared scale)")
            fig.suptitle(
                f"[B-2] {stem}: SHARED color scale (vmin=0, vmax={shared_vmax:.4g} = "
                f"max over L4/L8/L12) -- colors ARE comparable across layers",
                fontsize=11)

        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    return out_paths["per_layer_scale"], out_paths["shared_scale"]
