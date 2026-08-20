"""
OSIE attribute-grounding pilot: does CLIP ViT-B/16's M-shaped layerwise
attention profile (L4 rise, L8 trough, L12 recovery -- see
outputs/expA_clip/statistics/) correspond to attention landing on
different kinds of image content at each of those three layers?

For a seed-fixed 50-image OSIE subset (default), computes -- per image, per
layer (L4/L8/L12 by default), per semantic category (foreground / background
/ the 12 attrs.mat attributes) -- how much of that layer's [CLS]->patch
attention mass falls inside the category's object mask, relative to the
mask's area fraction (area_normalized_enrichment = attention_mass /
area_fraction).

CLI (all optional; defaults reproduce the original 50-image pilot exactly):
    --all-images     process all 700 attrs.mat/stimuli images instead of a
                      seeded 50-image sample (used for the full700 extension)
    --output-dir DIR output directory (default: outputs/osie_attribute_grounding_pilot)
    --seed N         random seed for image sampling and bootstrap CI (default: 42)
    --layers L,L,L   comma-separated 1-indexed display layers (default: 4,8,12)

Scope (explicitly NOT this script):
  - Does not touch outputs/expA_clip/, outputs/attn_cache/, or any DINO file.
  - Does not compute NSS/AUC/sAUC or use human fixation data at all -- this
    is about attention-vs-object-mask overlap, independent of outputs/fixmaps/.
  - Does not draw causal conclusions; see report.md.

Reuses, unmodified:
  - clip_extractor.load_clip_vit_b16 / load_osie_image_tensor
  - lib.clip_vit.patch_grid_from_image_hw / interpolate_patch_pos_embed /
    visual_forward_with_cls_patch_attention
  - The exact bilinear-upsample-to-(600,800)-then-sum-to-1-per-layer
    convention already used by scripts/run_expA_clip_full700.py and
    scripts/render_dino_clip_attention_comparison.py to reconstruct the
    M-shape in the first place (approved as the primary analysis convention;
    the ~1.3% vertical stretch from interpolating a 608px-tall padded grid
    directly onto 600px is a known pre-existing property of that convention,
    not introduced here -- see README.md).

New (this script only, since no existing code in this repo reads attrs.mat):
an h5py-based loader for data/attrs.mat (MATLAB v7.3/HDF5; scipy.io.loadmat
cannot read it). Structure and the required `.T` mask transpose were
verified against real data and a visual overlay in
scripts/verify_osie_attribute_coords.py, which MUST be run first.

Does not modify clip_extractor.py, vit_extractor.py, dino_vitb16_extractor.py,
sl_extractor.py, config.py, lib/, outputs/expA_clip/, outputs/attn_cache/,
or any OSIE source file. Writes only under the resolved --output-dir
(default: outputs/osie_attribute_grounding_pilot/; the full700 extension
uses --output-dir outputs/osie_attribute_grounding_full700 and never touches
the pilot directory).

Usage (PowerShell):
    python scripts\\run_osie_attribute_grounding_pilot.py
    python scripts\\run_osie_attribute_grounding_pilot.py --all-images --output-dir outputs\\osie_attribute_grounding_full700
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import argparse
import csv
import gc
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from clip_extractor import CLIP_PATCH_SIZE, load_clip_vit_b16, load_osie_image_tensor
from lib.clip_vit import (
    interpolate_patch_pos_embed, patch_grid_from_image_hw,
    visual_forward_with_cls_patch_attention,
)

# ======================= CONFIG (defaults; overridable via CLI) =======================
DEFAULT_SEED = 42
DEFAULT_N_PILOT = 50
DEFAULT_LAYERS_DISPLAY = [4, 8, 12]
IMG_W, IMG_H = 800, 600
GRID_HW_EXPECTED = (38, 50)

ATTR_NAMES = [
    "text", "face", "emotion", "sound", "smell", "taste", "touch",
    "motion", "operability", "watchability", "touched", "gazed",
]
LOW_SAMPLE_ATTRS = {"sound", "smell"}  # flagged, not excluded -- see report.md
CATEGORIES = ["foreground", "background"] + ATTR_NAMES

N_BOOT = 10000
ALPHA = 0.05

DATA_BASE  = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR   = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

DEFAULT_OUT_DIR = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_pilot"
# ======================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="OSIE attribute-grounding pilot / full700 extension "
                    "(CLIP ViT-B/16 attention vs. object/attribute masks).")
    p.add_argument("--all-images", action="store_true",
                    help="Process all 700 attrs.mat/stimuli images (sorted, "
                        "no sampling) instead of the default seeded 50-image "
                        "sample. Nothing else about the pipeline changes.")
    p.add_argument("--output-dir", type=str, default=None,
                    help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for image sampling (ignored when "
                        "--all-images is set) and for bootstrap CI (default: 42).")
    p.add_argument("--layers", type=str,
                    default=",".join(str(x) for x in DEFAULT_LAYERS_DISPLAY),
                    help="Comma-separated 1-indexed display layers, e.g. "
                        "'4,8,12' (default).")
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# attrs.mat loading (new; no existing code in this repo reads attrs.mat).
# Mirrors scripts/verify_osie_attribute_coords.py exactly (repo
# convention: self-contained per-script duplication rather than shared
# imports between scripts/*.py -- see run_expA_clip.py's own docstring).
# ----------------------------------------------------------------------

def h5_char_str(f, ref):
    arr = f[ref][()]
    return "".join(chr(int(c)) for c in arr.flatten())


def load_attrs_index(f):
    """Returns (attr_names list[12], {img_stem: h5py.Group}) -- read-only."""
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        img_arr = g["img"][()]
        raw_name = "".join(chr(int(c)) for c in img_arr.flatten())
        stem = os.path.splitext(raw_name)[0]
        if stem in index:
            raise RuntimeError(f"STOP: duplicate attrs.mat image id {stem}")
        index[stem] = g
    return attr_names, index


def object_masks_for_image(f, group, attr_names, expected_hw):
    """
    Returns (fg_mask (H,W) bool, per_attr_masks {name: (H,W) bool}, n_objects,
    per_attr_object_count {name: int}).

    group['objs'][0, j]['map'] is read as-is from h5py (shape (800,600))
    and TRANSPOSED (.T) to match the image's native (H,W)=(600,800)
    layout -- verified in scripts/verify_osie_attribute_coords.py.
    """
    objs_ds = group["objs"]
    n_objs = objs_ds.shape[1]
    if n_objs == 0:
        raise RuntimeError("STOP: image with zero objects")

    h, w = expected_hw
    fg = np.zeros((h, w), dtype=bool)
    per_attr = {name: np.zeros((h, w), dtype=bool) for name in attr_names}
    per_attr_obj_count = {name: 0 for name in attr_names}
    for j in range(n_objs):
        obj_g = f[objs_ds[0, j]]
        mp = obj_g["map"][()].T.astype(bool)
        if mp.shape != (h, w):
            raise RuntimeError(f"STOP: object {j} mask shape {mp.shape} != {(h, w)}")
        fg |= mp
        feat = obj_g["features"][()].flatten()
        if feat.shape[0] != len(attr_names):
            raise RuntimeError(
                f"STOP: feature vector length {feat.shape[0]} != "
                f"{len(attr_names)} attrNames")
        for k, name in enumerate(attr_names):
            if feat[k] > 0:
                per_attr[name] |= mp
                per_attr_obj_count[name] += 1
    return fg, per_attr, n_objs, per_attr_obj_count


# ----------------------------------------------------------------------
# CLIP attention extraction (reused unmodified from clip_extractor.py /
# lib/clip_vit.py; same convention as run_expA_clip_full700.py).
# ----------------------------------------------------------------------

def extract_selected_layers_grid_attention(model, image_path, layers_zero_based):
    """
    Returns (grid_maps (n_layers, gh, gw) float32 in layers_zero_based order,
    grid_hw, orig_hw, pad_hw, row_sums for the selected layers, t_io, t_infer).
    """
    device = next(model.parameters()).device

    t0 = time.perf_counter()
    tensor, orig_hw, pad_hw = load_osie_image_tensor(image_path, CLIP_PATCH_SIZE)
    tensor = tensor.to(device)
    grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
    pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_io = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.inference_mode():
        _, cls_patch_maps, full_row_sums = visual_forward_with_cls_patch_attention(
            model.visual, tensor.type(model.dtype), grid_hw, pos_embed=pos_interp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_infer = time.perf_counter() - t1

    selected = [cls_patch_maps[i] for i in layers_zero_based]
    selected_row_sums = [full_row_sums[i] for i in layers_zero_based]
    grid_maps = torch.stack(selected, dim=0)[:, 0].detach().cpu().numpy().astype(np.float32)
    row_sums = torch.stack(selected_row_sums).squeeze(-1).detach().cpu().numpy()
    return grid_maps, grid_hw, orig_hw, pad_hw, row_sums, t_io, t_infer


def upsample_and_normalize(grid_maps):
    """
    grid_maps: (n_layers, gh, gw) float32. Returns (n_layers, IMG_H, IMG_W)
    float32, each layer bilinear-upsampled to (600,800) then renormalized to
    sum=1 -- IDENTICAL convention to run_expA_clip_full700.py /
    render_dino_clip_attention_comparison.py (the same reconstruction that
    produced the M-shape). Not changed for this pilot/extension (see
    README.md for the ~1.3% vertical stretch caveat, inherited unmodified).
    """
    n_layers = grid_maps.shape[0]
    t = torch.from_numpy(grid_maps).unsqueeze(1).float()  # (n,1,gh,gw)
    up = F.interpolate(t, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
    up = up[:, 0].numpy().astype(np.float32)  # (n, IMG_H, IMG_W)
    for l in range(n_layers):
        s = up[l].sum()
        if s > 0:
            up[l] /= s
    return up


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------

def bootstrap_ci_mean(values, seed, n_boot=N_BOOT, alpha=ALPHA):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()

    seed = args.seed
    all_images = args.all_images
    layers_display = [int(x) for x in args.layers.split(",")]
    if len(layers_display) != len(set(layers_display)):
        raise RuntimeError(f"STOP: duplicate layers in --layers: {layers_display}")
    for d in layers_display:
        if not (1 <= d <= 12):
            raise RuntimeError(f"STOP: layer {d} out of range [1,12]: --layers={args.layers}")
    layers_zero_based = [d - 1 for d in layers_display]
    assert [d - 1 for d in layers_display] == layers_zero_based, (
        "STOP: display-layer <-> zero-based-index mapping is inconsistent")

    out_dir = args.output_dir or DEFAULT_OUT_DIR
    n_images_target = None if all_images else DEFAULT_N_PILOT
    mode_tag = "full700" if all_images else "pilot"
    layers_tag = "".join(f"L{d}" for d in layers_display)

    os.makedirs(out_dir, exist_ok=True)
    attn_cache_dir = os.path.join(out_dir, "attn_cache")
    os.makedirs(attn_cache_dir, exist_ok=True)

    attn_cache_path = os.path.join(
        attn_cache_dir, f"clip_vitb16_{mode_tag}_{layers_tag}_patchgrid.npz")
    image_list_csv_name = "full700_image_list.csv" if all_images else "pilot_image_list.csv"
    image_list_csv = os.path.join(out_dir, image_list_csv_name)
    long_csv = os.path.join(out_dir, "per_image_long.csv")
    summary_csv = os.path.join(out_dir, "attribute_layer_summary.csv")
    run_config_json = os.path.join(out_dir, "run_config.json")
    timing_json = os.path.join(out_dir, "timing.json")
    exec_log_json = os.path.join(out_dir, "execution_log.json")

    print("=" * 70)
    print(f"  OSIE Attribute-Grounding {'Full700' if all_images else 'Pilot'}: "
          f"CLIP ViT-B/16 {'/'.join(f'L{d}' for d in layers_display)}")
    print("=" * 70)
    print(f"  output_dir={out_dir}  seed={seed}  layers={layers_display}  "
          f"all_images={all_images}")

    warnings_list = []

    # ------------------------------------------------------------------
    # Load attrs.mat
    # ------------------------------------------------------------------
    print("\n--- Loading attrs.mat ---")
    f = h5py.File(ATTRS_PATH, "r")
    attr_names, attrs_index = load_attrs_index(f)
    if attr_names != ATTR_NAMES:
        raise RuntimeError(f"STOP: attrNames mismatch, got {attr_names}")
    if len(attrs_index) != 700:
        raise RuntimeError(f"STOP: expected 700 attrs.mat images, got {len(attrs_index)}")
    print(f"  attrNames: {attr_names}")
    print(f"  attrs.mat images: {len(attrs_index)}")

    stim_ids = set(
        os.path.splitext(fn)[0] for fn in os.listdir(STIM_DIR) if fn.endswith(".jpg"))
    if stim_ids != set(attrs_index.keys()):
        raise RuntimeError(
            "STOP: attrs.mat image ids do not exactly match data/stimuli/*.jpg "
            "(re-run scripts/verify_osie_attribute_coords.py)")
    print(f"  stimuli images: {len(stim_ids)}  (exact match with attrs.mat: OK)")

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------
    universe = sorted(attrs_index.keys())
    if all_images:
        print(f"\n--- Image selection: ALL {len(universe)} images (--all-images) ---")
        image_ids = universe
        selection_method = (
            "sorted(attrs.mat image ids) [700, == data/stimuli ids exactly]; "
            "ALL 700 processed, no sampling (--all-images)")
    else:
        print(f"\n--- Image selection (seed={seed}, n={n_images_target}) ---")
        rng = np.random.RandomState(seed)
        chosen_idx = rng.choice(len(universe), size=n_images_target, replace=False)
        image_ids = sorted(universe[i] for i in chosen_idx)
        selection_method = (
            f"sorted(attrs.mat image ids) [700, == data/stimuli ids exactly]; "
            f"np.random.RandomState({seed}).choice(700, {n_images_target}, replace=False); "
            f"result re-sorted for stable output ordering")
        print(f"  Selected {len(image_ids)} images: {image_ids}")

    # ------------------------------------------------------------------
    # Load CLIP
    # ------------------------------------------------------------------
    print("\n--- Loading CLIP ViT-B/16 ---")
    model, _ = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}  dtype: {model.dtype}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    # ------------------------------------------------------------------
    # Single pass: masks + attention extraction + metrics, per image.
    # (Masks are built and discarded per-image, not pre-computed for all
    # images at once, to keep peak RAM bounded at full-700 scale.)
    # ------------------------------------------------------------------
    print(f"\n--- Extracting attention + masks ({len(image_ids)} images) ---")
    long_rows = []
    timings = []
    cache_stems = []
    cache_attn = []
    image_list_rows = []

    n_images = len(image_ids)
    for idx, stem in enumerate(image_ids):
        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        t_total0 = time.perf_counter()

        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        if (W, H) != (IMG_W, IMG_H):
            raise RuntimeError(f"STOP: {stem} has unexpected size {(W,H)}, expected {(IMG_W,IMG_H)}")

        fg, per_attr, n_objs, per_attr_obj_count = object_masks_for_image(
            f, attrs_index[stem], attr_names, expected_hw=(H, W))
        bg = ~fg
        cat_masks = {"foreground": fg, "background": bg}
        cat_masks.update(per_attr)
        cat_obj_counts = {"foreground": n_objs, "background": 0}
        cat_obj_counts.update(per_attr_obj_count)

        image_list_rows.append({
            "image_id": stem,
            "n_objects": n_objs,
            "fg_area_fraction": float(fg.mean()),
            **{f"has_{name}": int(per_attr[name].any()) for name in attr_names},
        })

        grid_maps, grid_hw, orig_hw, pad_hw, row_sums, t_io, t_infer = \
            extract_selected_layers_grid_attention(model, img_path, layers_zero_based)

        if grid_hw != GRID_HW_EXPECTED:
            raise RuntimeError(f"STOP: {stem} grid {grid_hw} != {GRID_HW_EXPECTED}")
        if orig_hw != (IMG_H, IMG_W):
            raise RuntimeError(f"STOP: {stem} orig_hw {orig_hw} != {(IMG_H, IMG_W)}")
        if np.isnan(grid_maps).any() or np.isinf(grid_maps).any():
            raise RuntimeError(f"STOP: NaN/Inf in {stem} raw grid attention")
        if np.isnan(row_sums).any() or np.isinf(row_sums).any() or np.any(np.abs(row_sums - 1.0) > 1e-2):
            raise RuntimeError(f"STOP: {stem} CLS-row-sum out of range for {layers_display}: {row_sums}")

        cache_stems.append(stem)
        cache_attn.append(grid_maps)

        upsampled = upsample_and_normalize(grid_maps)
        if np.isnan(upsampled).any() or np.isinf(upsampled).any():
            raise RuntimeError(f"STOP: NaN/Inf in {stem} upsampled attention")
        for l in range(len(layers_display)):
            s = upsampled[l].sum()
            if abs(s - 1.0) > 1e-3:
                raise RuntimeError(
                    f"STOP: {stem} layer L{layers_display[l]} upsampled attention "
                    f"does not sum to 1 (got {s})")

        for l_idx, layer_display in enumerate(layers_display):
            attn_map = upsampled[l_idx]
            for cat in CATEGORIES:
                mask = cat_masks[cat]
                pixel_count = int(mask.sum())
                if pixel_count == 0:
                    continue  # attribute absent in this image -- excluded as missing, not zero
                area_fraction = pixel_count / (IMG_H * IMG_W)
                attention_mass = float(attn_map[mask].sum())
                if area_fraction <= 0 or not np.isfinite(area_fraction):
                    raise RuntimeError(f"STOP: invalid area_fraction for {stem}/{cat}: {area_fraction}")
                enrichment = attention_mass / area_fraction
                if not np.isfinite(attention_mass) or not np.isfinite(enrichment):
                    raise RuntimeError(
                        f"STOP: non-finite metric for {stem}/L{layer_display}/{cat}: "
                        f"mass={attention_mass} enrichment={enrichment}")
                if not (0.0 <= attention_mass <= 1.0 + 1e-6):
                    raise RuntimeError(
                        f"STOP: attention_mass out of [0,1] for {stem}/L{layer_display}/{cat}: {attention_mass}")
                long_rows.append({
                    "image_name": f"{stem}.jpg",
                    "layer": layer_display,
                    "attribute": cat,
                    "attention_mass": attention_mass,
                    "area_fraction": area_fraction,
                    "area_normalized_enrichment": enrichment,
                    "mask_object_count": cat_obj_counts[cat],
                    "mask_pixel_count": pixel_count,
                })

        t_total = time.perf_counter() - t_total0
        timings.append({
            "image": stem, "is_warmup": idx == 0,
            "t_io_s": t_io, "t_infer_s": t_infer, "t_total_s": t_total,
        })
        tag = "WARMUP" if idx == 0 else f"{idx + 1:4d}/{n_images}"
        if idx == 0 or (idx + 1) % 50 == 0 or idx == n_images - 1:
            print(f"  [{tag}] {stem}: io={t_io*1000:6.1f}ms infer={t_infer*1000:6.1f}ms total={t_total*1000:6.1f}ms")

        del grid_maps, upsampled, cat_masks, per_attr, fg, bg
        gc.collect()

    peak_mem_mib = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else None)

    attr_valid_counts = {
        name: sum(1 for r in image_list_rows if r[f"has_{name}"]) for name in attr_names}
    print("\n--- Valid-image counts per attribute ---")
    for name in attr_names:
        flag = "  (LOW SAMPLE)" if name in LOW_SAMPLE_ATTRS else ""
        print(f"    {name:15s}: {attr_valid_counts[name]:4d} / {n_images}{flag}")

    with open(image_list_csv, "w", newline="", encoding="utf-8") as fh:
        fieldnames = list(image_list_rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(image_list_rows)
    print(f"  Saved: {image_list_csv}")

    # ------------------------------------------------------------------
    # Save patch-grid attention cache (new cache file; does not touch
    # outputs/attn_cache/dino_vits16_patchgrid.npz or the pilot's cache).
    # ------------------------------------------------------------------
    np.savez_compressed(
        attn_cache_path,
        stems=np.array(cache_stems),
        attn=np.stack(cache_attn, axis=0),  # (n_images, n_layers, 38, 50)
        layers_display=np.array(layers_display),
        layers_zero_based=np.array(layers_zero_based),
    )
    print(f"\n  Saved attention cache: {attn_cache_path}")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n--- Verification ---")
    n_expected_max = n_images * len(layers_display) * len(CATEGORIES)
    if len(long_rows) == 0 or len(long_rows) > n_expected_max:
        raise RuntimeError(f"STOP: unexpected row count {len(long_rows)} (max possible {n_expected_max})")
    seen = set()
    for r in long_rows:
        key = (r["image_name"], r["layer"], r["attribute"])
        if key in seen:
            raise RuntimeError(f"STOP: duplicate row {key}")
        seen.add(key)
    n_missing_attr_rows = n_expected_max - len(long_rows)
    print(f"  rows written: {len(long_rows)} (max possible {n_expected_max}, "
          f"{n_missing_attr_rows} excluded as attribute-absent-in-image)")
    print("  no duplicate (image,layer,attribute) rows: OK")
    print("  no NaN/Inf survived any row (checked inline, fail-fast): OK")

    # ------------------------------------------------------------------
    # Save per_image_long.csv
    # ------------------------------------------------------------------
    with open(long_csv, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["image_name", "layer", "attribute", "attention_mass",
                      "area_fraction", "area_normalized_enrichment",
                      "mask_object_count", "mask_pixel_count"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(long_rows)
    print(f"  Saved: {long_csv}")

    # ------------------------------------------------------------------
    # Attribute x layer summary (with bootstrap 95% CI)
    # ------------------------------------------------------------------
    print("\n--- Building attribute x layer summary ---")
    summary_rows = []
    for cat in CATEGORIES:
        for layer_display in layers_display:
            vals = np.array([
                r["area_normalized_enrichment"] for r in long_rows
                if r["attribute"] == cat and r["layer"] == layer_display
            ], dtype=float)
            n_valid = len(vals)
            if n_valid == 0:
                mean_v = median_v = std_v = ci_lo = ci_hi = float("nan")
            else:
                mean_v = float(np.mean(vals))
                median_v = float(np.median(vals))
                std_v = float(np.std(vals, ddof=1)) if n_valid > 1 else 0.0
                ci_lo, ci_hi = bootstrap_ci_mean(vals, seed=seed)
            summary_rows.append({
                "attribute": cat,
                "layer": layer_display,
                "valid_image_count": n_valid,
                "mean": mean_v,
                "median": median_v,
                "standard_deviation": std_v,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "low_sample_flag": int(cat in LOW_SAMPLE_ATTRS),
            })

    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["attribute", "layer", "valid_image_count", "mean", "median",
                      "standard_deviation", "ci95_lo", "ci95_hi", "low_sample_flag"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {summary_csv}")

    print(f"\n=== area_normalized_enrichment: mean (valid_image_count) ===")
    header = f"{'attribute':15s}" + "".join(f"  L{l:<9d}" for l in layers_display)
    print(header)
    for cat in CATEGORIES:
        line = f"{cat:15s}"
        for layer_display in layers_display:
            row = next(r for r in summary_rows if r["attribute"] == cat and r["layer"] == layer_display)
            line += f"  {row['mean']:6.3f}({row['valid_image_count']:2d})"
        print(line)

    # ------------------------------------------------------------------
    # Timing / run config / execution log
    # ------------------------------------------------------------------
    steady = timings[1:]
    n_steady = len(steady)
    sum_total = sum(t["t_total_s"] for t in steady)
    avg_total = sum_total / n_steady if n_steady else float("nan")

    timing_summary = {
        "mode": mode_tag,
        "n_images": n_images,
        "n_images_steady_state": n_steady,
        "warmup_image": timings[0]["image"],
        "warmup_total_s": timings[0]["t_total_s"],
        "steady_state": {
            "sum_total_s": sum_total,
            "avg_total_s": avg_total,
            "avg_io_s": sum(t["t_io_s"] for t in steady) / n_steady if n_steady else float("nan"),
            "avg_infer_s": sum(t["t_infer_s"] for t in steady) / n_steady if n_steady else float("nan"),
        },
        "peak_vram_mib": peak_mem_mib,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "extrapolated_700_images_s": avg_total * 700 if n_steady else None,
        "wall_clock_total_s": time.time() - t_start,
        "per_image_timings": timings,
    }
    with open(timing_json, "w", encoding="utf-8") as fh:
        json.dump(timing_summary, fh, indent=2)
    print(f"\n  Saved: {timing_json}")
    if peak_mem_mib is not None:
        print(f"  Steady-state avg/image: {avg_total*1000:.1f}ms  Peak VRAM: {peak_mem_mib:.1f} MiB")
    else:
        print(f"  Steady-state avg/image: {avg_total*1000:.1f}ms  Peak VRAM: n/a")

    run_config = {
        "model": "clip_vitb16",
        "mode": mode_tag,
        "all_images": all_images,
        "layers_display": layers_display,
        "layers_zero_based": layers_zero_based,
        "categories": CATEGORIES,
        "low_sample_attrs": sorted(LOW_SAMPLE_ATTRS),
        "img_w": IMG_W, "img_h": IMG_H,
        "grid_hw": list(GRID_HW_EXPECTED),
        "upsample": "bilinear (F.interpolate, align_corners=False), then per-layer sum-to-1 "
                    "-- identical convention to run_expA_clip_full700.py / "
                    "render_dino_clip_attention_comparison.py (unchanged from the pilot)",
        "seed": seed,
        "n_images": n_images,
        "image_ids": image_ids,
        "image_selection_method": selection_method,
        "attr_valid_image_counts": attr_valid_counts,
        "n_boot": N_BOOT,
        "alpha": ALPHA,
        "n_rows_long_csv": len(long_rows),
        "n_rows_excluded_attribute_absent": n_missing_attr_rows,
        "coordinate_convention": (
            "attrs.mat obj.map read via h5py has shape (800,600); .T recovers "
            "native (H,W)=(600,800) matching the image array and "
            "clip_extractor.load_osie_image_tensor's orig_hw. No flip. "
            "Verified in scripts/verify_osie_attribute_coords.py."
        ),
        "known_limitation": (
            "CLIP attention grid (38,50) is derived from the padded 608x800 "
            "input (height padded from 600 to 608 to reach a multiple of "
            "patch_size=16) and is bilinear-upsampled directly to 600x800 "
            "without first cropping the padding fraction out of the grid -- "
            "an approx. 1.3% (8/608px) vertical stretch. This is the existing "
            "production convention (used to produce the M-shape results in "
            "outputs/expA_clip/) and is intentionally NOT changed here."
        ),
        "cli_args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(run_config_json, "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)
    print(f"  Saved: {run_config_json}")

    exec_log = {
        "mode": mode_tag,
        "image_ids": image_ids,
        "n_images_requested": n_images_target if not all_images else 700,
        "n_images_processed": len(image_ids),
        "n_images_missing": (n_images_target if not all_images else 700) - len(image_ids),
        "attr_valid_image_counts": attr_valid_counts,
        "n_attributes_total": len(CATEGORIES),
        "n_attributes_with_zero_valid_images": sum(
            1 for name in attr_names if attr_valid_counts[name] == 0),
        "nan_or_inf_detected": False,
        "warnings": warnings_list,
        "input_paths": {
            "stimuli_dir": STIM_DIR, "attrs_mat": ATTRS_PATH,
        },
        "output_paths": {
            "image_list_csv": image_list_csv,
            "per_image_long_csv": long_csv,
            "attribute_layer_summary_csv": summary_csv,
            "attn_cache": attn_cache_path,
            "run_config_json": run_config_json,
            "timing_json": timing_json,
        },
        "seed": seed,
        "wall_clock_total_s": time.time() - t_start,
    }
    with open(exec_log_json, "w", encoding="utf-8") as fh:
        json.dump(exec_log, fh, indent=2)
    print(f"  Saved: {exec_log_json}")

    print("\n" + "=" * 70)
    print(f"  OSIE Attribute-Grounding {'Full700' if all_images else 'Pilot'}: DONE")
    print("  outputs/expA_clip/, outputs/attn_cache/, and all OSIE source "
          "files were NOT touched.")
    if all_images:
        print(f"  {DEFAULT_OUT_DIR} (pilot output) was NOT touched.")
    print("=" * 70)


if __name__ == "__main__":
    main()
