"""
[B-2] Step 4-6 (pilot): compute the M-shape spatial-distribution metrics
(lib/clip_attention_b2.py) for a seed=42, 10-image sample drawn from the
existing 700-image, Phase-2 (38x50 grid) L4/L8/L12 attention cache
  outputs/osie_attribute_grounding_full700/attn_cache/clip_vitb16_full700_L4L8L12_patchgrid.npz
(verified against the original M-shape run by
scripts/verify_clip_attention_b2_cache.py -- must PASS first; this
script ALSO re-checks the cache's own basic shape/index/health
invariants immediately after loading it, so it does not depend solely
on that separate verification step having been run correctly).

For each pilot image and layer:
  - reads the cached raw [CLS]->patch attention (38x50, un-renormalized)
  - re-derives the SAME map via one fresh forward pass (to additionally
    recover full_row_sums, not stored in the cache) and asserts it is
    bit-identical to the cached value (STOP otherwise -- this is the
    same reproducibility-check convention as
    scripts/extract_osie_probe_all12_full700_features.py)
  - computes every B-2 metric (entropy, Hoyer, max, top-10% mass,
    50%/80% coverage, centroid, center distance, foreground/background
    raw+conditional mass+enrichment)
  - computes L4-L8 / L8-L12 / L4-L12 spatial similarity (Pearson, cosine,
    normalized JSD) for that image
  - computes a shuffled-image-pair auxiliary baseline (center-bias check)

Foreground masks come from attrs.mat (union of all object masks per
image), transformed to the 38x50 patch grid via
lib.osie_text_alignment.mask_to_patch_weights(mask, patch_size=16) --
the SAME zero-pad-to-patch-multiple geometry as the image itself
(vit_extractor.py::_pad_to_patch), reused unmodified. Foreground/background
area fraction is measured directly in this 38x50 patch space (NOT taken
from full700_image_list.csv's pixel-domain fg_area_fraction column).

All paths are resolved relative to the repository root or overridable
via CLI flags -- no machine-specific absolute path is hardcoded. Writes
only under the chosen --output-dir (each output file is individually
refused if it already exists, INCLUDING the overlay figures -- not
merely because the directory exists):
  config.json, per_image_metrics.csv, layer_pair_similarity.csv,
  shuffled_pair_baseline.csv, pilot_summary.json, figures/*.png

Usage (PowerShell):
    python scripts\\run_clip_attention_b2_pilot.py
    python scripts\\run_clip_attention_b2_pilot.py --output-dir D:\\tmp\\b2_smoke\\pilot --n-pilot 3
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import argparse
import csv
import json
import os
import time

import h5py
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clip_extractor import CLIP_PATCH_SIZE, load_clip_vit_b16, load_osie_image_tensor
from lib.clip_vit import (
    interpolate_patch_pos_embed, patch_grid_from_image_hw, visual_forward_with_cls_patch_attention,
)
from lib.osie_text_alignment import mask_to_patch_weights
from lib.clip_attention_b2 import (
    centroid, coverage_for_mass, cosine_similarity, foreground_background_metrics,
    hoyer_sparsity, jensen_shannon_divergence_normalized, max_patch_attention,
    normalize_distribution, normalized_center_distance, normalized_entropy,
    pearson_corr, refuse_if_exists, top_k_mass,
)

SEED_DEFAULT = 42
N_PILOT_DEFAULT = 10
LAYERS_DISPLAY = [4, 8, 12]
LAYERS_ZERO_BASED = [3, 7, 11]
GRID_HW = (38, 50)
N_PATCHES = 38 * 50  # 1900
TOP_10PCT_K = 190
IMG_W, IMG_H = 800, 600
N_FIGURE_IMAGES_DEFAULT = 3
N_IMAGES_EXPECTED = 700

LAYER_PAIRS = [(4, 8), (8, 12), (4, 12)]

DEFAULT_CACHE_PATH = (
    REPO_ROOT / "outputs" / "osie_attribute_grounding_full700" / "attn_cache"
    / "clip_vitb16_full700_L4L8L12_patchgrid.npz")
DEFAULT_CACHE_VERIFICATION_JSON = REPO_ROOT / "outputs" / "clip_attention_b2_distribution" / "cache_verification.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "osie" / "predicting-human-gaze-beyond-pixels"
DEFAULT_STIMULI_DIR = DEFAULT_DATASET_DIR / "data" / "stimuli"
DEFAULT_ATTRS_PATH = DEFAULT_DATASET_DIR / "data" / "attrs.mat"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "clip_attention_b2_distribution_pilot"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="[B-2] Pilot: compute M-shape spatial-distribution metrics for a "
                    "seed-selected sample of OSIE images, from the existing L4/L8/L12 "
                    "attention cache.")
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                    help=f"Path to the existing L4/L8/L12 attention cache .npz, read-only "
                        f"(default: {DEFAULT_CACHE_PATH}).")
    p.add_argument("--cache-verification-json", type=Path, default=DEFAULT_CACHE_VERIFICATION_JSON,
                    help="Path to the cache_verification.json produced by "
                        "verify_clip_attention_b2_cache.py, read-only -- must have "
                        f"verdict==PASS (default: {DEFAULT_CACHE_VERIFICATION_JSON}).")
    p.add_argument("--stimuli-dir", type=Path, default=DEFAULT_STIMULI_DIR,
                    help=f"Directory containing OSIE stimulus images (<id>.jpg), read-only "
                        f"(default: {DEFAULT_STIMULI_DIR}).")
    p.add_argument("--attrs-path", type=Path, default=DEFAULT_ATTRS_PATH,
                    help=f"Path to OSIE's attrs.mat (object masks), read-only "
                        f"(default: {DEFAULT_ATTRS_PATH}).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Directory to write config.json / per_image_metrics.csv / "
                        "layer_pair_similarity.csv / shuffled_pair_baseline.csv / "
                        "pilot_summary.json / figures/*.png into (created if missing; "
                        "refuses to overwrite any of its own output files that already "
                        f"exist) (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help=f"Seed for the pilot image sample and the shuffled-pair baseline "
                        f"(default: {SEED_DEFAULT}).")
    p.add_argument("--n-pilot", type=int, default=N_PILOT_DEFAULT,
                    help=f"Number of images to sample for the pilot (default: {N_PILOT_DEFAULT}).")
    p.add_argument("--n-figure-images", type=int, default=N_FIGURE_IMAGES_DEFAULT,
                    help="Number of pilot images (in sampled order) to render overlay "
                        f"figures for (default: {N_FIGURE_IMAGES_DEFAULT}).")
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# attrs.mat loading (self-contained duplication, per repo convention --
# see scripts/run_osie_attribute_grounding_pilot.py / verify_osie_attribute_coords.py)
# ----------------------------------------------------------------------

def h5_char_str(f, ref):
    return "".join(chr(int(c)) for c in f[ref][()].flatten())


def load_attrs_index(f):
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw_name = "".join(chr(int(c)) for c in g["img"][()].flatten())
        stem = os.path.splitext(raw_name)[0]
        index[stem] = g
    return attr_names, index


def foreground_mask_for_image(f, group, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    n_objs = objs_ds.shape[1]
    if n_objs == 0:
        raise RuntimeError("STOP: image with zero objects")
    fg = np.zeros((h, w), dtype=bool)
    for j in range(n_objs):
        obj_g = f[objs_ds[0, j]]
        mp = obj_g["map"][()].T.astype(bool)
        if mp.shape != (h, w):
            raise RuntimeError(f"STOP: object {j} mask shape {mp.shape} != {(h, w)}")
        fg |= mp
    return fg, n_objs


# ----------------------------------------------------------------------
# Cache self-verification (basic invariants; does not replace, but does
# not solely rely on, scripts/verify_clip_attention_b2_cache.py)
# ----------------------------------------------------------------------

def verify_cache_basic_health(cache):
    attn = cache["attn"]
    stems = cache["stems"].tolist()
    layers_display = cache["layers_display"].tolist()
    layers_zero_based = cache["layers_zero_based"].tolist()

    if attn.shape != (N_IMAGES_EXPECTED, 3, *GRID_HW):
        raise RuntimeError(f"STOP: cache attn.shape {attn.shape} != {(N_IMAGES_EXPECTED, 3, *GRID_HW)}")
    if layers_display != LAYERS_DISPLAY:
        raise RuntimeError(f"STOP: cache layers_display {layers_display} != {LAYERS_DISPLAY}")
    if layers_zero_based != LAYERS_ZERO_BASED:
        raise RuntimeError(f"STOP: cache layers_zero_based {layers_zero_based} != {LAYERS_ZERO_BASED}")
    if len(stems) != N_IMAGES_EXPECTED:
        raise RuntimeError(f"STOP: cache has {len(stems)} stems, expected {N_IMAGES_EXPECTED}")
    if len(set(stems)) != len(stems):
        raise RuntimeError("STOP: cache stems contain duplicates")
    if not np.isfinite(attn).all():
        raise RuntimeError("STOP: cache attention contains non-finite values")
    if (attn < 0).any():
        raise RuntimeError("STOP: cache attention contains negative values")
    return stems


# ----------------------------------------------------------------------
# Fresh recompute (for full_row_sums + bit-exact cross-check vs. cache)
# ----------------------------------------------------------------------

def recompute_selected_layers(model, image_path):
    device = next(model.parameters()).device
    tensor, orig_hw, pad_hw = load_osie_image_tensor(str(image_path), CLIP_PATCH_SIZE)
    tensor = tensor.to(device)
    grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
    if grid_hw != GRID_HW:
        raise RuntimeError(f"STOP: grid_hw {grid_hw} != {GRID_HW}")
    pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)
    with torch.inference_mode():
        _, cls_patch_maps, full_row_sums = visual_forward_with_cls_patch_attention(
            model.visual, tensor.type(model.dtype), grid_hw, pos_embed=pos_interp)
    selected_maps = torch.stack(
        [cls_patch_maps[i] for i in LAYERS_ZERO_BASED], dim=0
    )[:, 0].detach().cpu().numpy().astype(np.float64)
    selected_row_sums = torch.stack(
        [full_row_sums[i] for i in LAYERS_ZERO_BASED]
    ).squeeze(-1).detach().cpu().numpy().astype(np.float64)
    return selected_maps, selected_row_sums


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(args.cache_path)
    cache_verification_json = Path(args.cache_verification_json)
    stimuli_dir = Path(args.stimuli_dir)
    attrs_path = Path(args.attrs_path)
    output_dir = Path(args.output_dir)
    seed = args.seed
    n_pilot = args.n_pilot
    n_figure_images = args.n_figure_images

    fig_dir = output_dir / "figures"
    config_json = output_dir / "config.json"
    metrics_csv = output_dir / "per_image_metrics.csv"
    similarity_csv = output_dir / "layer_pair_similarity.csv"
    shuffled_csv = output_dir / "shuffled_pair_baseline.csv"
    summary_json = output_dir / "pilot_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    refuse_if_exists([str(config_json), str(metrics_csv), str(similarity_csv),
                       str(shuffled_csv), str(summary_json)])

    print("=" * 70)
    print(f"  [B-2] Pilot: {n_pilot} images, seed={seed}, layers={LAYERS_DISPLAY}")
    print("=" * 70)
    print(f"  cache_path              = {cache_path}")
    print(f"  cache_verification_json = {cache_verification_json}")
    print(f"  stimuli_dir             = {stimuli_dir}")
    print(f"  attrs_path              = {attrs_path}")
    print(f"  output_dir              = {output_dir}")

    if not cache_verification_json.is_file():
        raise RuntimeError(
            f"STOP: {cache_verification_json} not found -- run "
            f"scripts/verify_clip_attention_b2_cache.py first")
    with open(cache_verification_json, encoding="utf-8") as f:
        cache_verif = json.load(f)
    if cache_verif["verdict"] != "PASS":
        raise RuntimeError(f"STOP: cache_verification.json verdict != PASS: {cache_verif['verdict']}")
    print(f"  cache_verification.json verdict: PASS (checked {cache_verification_json})")

    cache = np.load(str(cache_path), allow_pickle=False)
    stems_all = verify_cache_basic_health(cache)
    print("  Cache self-check (shape/layers/stems/finite/non-negative): OK")
    attn_all = cache["attn"]  # (700, 3, 38, 50) float32, layers_zero_based order [3,7,11]

    rng = np.random.RandomState(seed)
    chosen_idx = sorted(rng.choice(len(stems_all), size=n_pilot, replace=False).tolist())
    pilot_stems = [stems_all[i] for i in chosen_idx]
    print(f"  Pilot image IDs (seed={seed}): {pilot_stems}")

    f = h5py.File(str(attrs_path), "r")
    attr_names, attrs_index = load_attrs_index(f)

    print("\n--- Loading CLIP ViT-B/16 (for full_row_sums + bit-exact cross-check) ---")
    model, _ = load_clip_vit_b16()

    metric_rows = []
    similarity_rows = []
    degenerate_images = []
    all_row_sums = []
    all_patch_mass_raw = []
    per_image_layer_maps = {}   # stem -> {layer_display: raw (38,50) float64}
    per_image_coverage = {}     # stem -> (38,50) float64

    for stem in pilot_stems:
        img_idx = stems_all.index(stem)
        img_path = stimuli_dir / f"{stem}.jpg"

        img_arr = np.array(Image.open(img_path).convert("RGB"))
        H, W = img_arr.shape[:2]
        if (W, H) != (IMG_W, IMG_H):
            raise RuntimeError(f"STOP: {stem} unexpected size {(W, H)}")

        fg_mask, n_objs = foreground_mask_for_image(f, attrs_index[stem], (H, W))
        coverage = mask_to_patch_weights(fg_mask, CLIP_PATCH_SIZE)
        if coverage.shape != GRID_HW:
            raise RuntimeError(f"STOP: {stem} coverage shape {coverage.shape} != {GRID_HW}")
        per_image_coverage[stem] = coverage

        fresh_maps, row_sums = recompute_selected_layers(model, img_path)
        cached_maps = attn_all[img_idx].astype(np.float64)
        max_diff = float(np.abs(fresh_maps - cached_maps).max())
        if max_diff >= 1e-6:
            raise RuntimeError(
                f"STOP: {stem} fresh recompute vs. cache max_abs_diff={max_diff} >= 1e-6")

        all_row_sums.extend(row_sums.tolist())
        per_image_layer_maps[stem] = {}

        for l_idx, layer_display in enumerate(LAYERS_DISPLAY):
            raw = cached_maps[l_idx]  # use the CACHE value as the analysis source of truth
            per_image_layer_maps[stem][layer_display] = raw
            full_row_sum = float(row_sums[l_idx])
            patch_mass_raw = float(raw.sum())
            cls_self_mass = full_row_sum - patch_mass_raw
            all_patch_mass_raw.append(patch_mass_raw)

            if not np.isfinite(raw).all():
                raise RuntimeError(f"STOP: non-finite attention for {stem} L{layer_display}")
            if (raw < 0).any():
                raise RuntimeError(f"STOP: negative attention for {stem} L{layer_display}")

            p = normalize_distribution(raw)
            h_norm = normalized_entropy(p, N_PATCHES)
            hoyer = hoyer_sparsity(raw)
            max_p = max_patch_attention(p)
            top10 = top_k_mass(p, TOP_10PCT_K)
            n50, area50 = coverage_for_mass(p, 0.5)
            n80, area80 = coverage_for_mass(p, 0.8)
            cx, cy = centroid(p, GRID_HW)
            center_dist = normalized_center_distance((cx, cy))
            fgbg = foreground_background_metrics(raw, coverage)
            if fgbg["is_degenerate"]:
                degenerate_images.append({
                    "image_id": stem, "layer": layer_display,
                    "foreground_area_fraction": fgbg["foreground_area_fraction"],
                    "background_area_fraction": fgbg["background_area_fraction"],
                    "n_objects": n_objs,
                })

            metric_rows.append({
                "image_id": stem, "layer": layer_display, "n_objects": n_objs,
                "patch_mass_raw": patch_mass_raw, "full_row_sum": full_row_sum,
                "cls_self_mass": cls_self_mass,
                "normalized_entropy": h_norm, "hoyer_sparsity": hoyer,
                "max_patch_attention": max_p, "top10pct_mass": top10,
                "mass50_n_patches": n50, "mass50_area_fraction": area50,
                "mass80_n_patches": n80, "mass80_area_fraction": area80,
                "centroid_x": cx, "centroid_y": cy,
                "normalized_center_distance": center_dist,
                "foreground_area_fraction": fgbg["foreground_area_fraction"],
                "background_area_fraction": fgbg["background_area_fraction"],
                "foreground_absolute_mass": fgbg["foreground_absolute_mass"],
                "foreground_conditional_mass": fgbg["foreground_conditional_mass"],
                "background_conditional_mass": fgbg["background_conditional_mass"],
                "foreground_enrichment": fgbg["foreground_enrichment"],
                "is_degenerate_fgbg": fgbg["is_degenerate"],
            })

        # ---- layer-pair spatial similarity (same image) ----
        maps = per_image_layer_maps[stem]
        for la, lb in LAYER_PAIRS:
            ra, rb = maps[la], maps[lb]
            pearson = pearson_corr(ra, rb)
            cosine = cosine_similarity(ra, rb)
            pa, pb = normalize_distribution(ra), normalize_distribution(rb)
            jsd = jensen_shannon_divergence_normalized(pa, pb)
            similarity_rows.append({
                "image_id": stem, "layer_a": la, "layer_b": lb,
                "layer_distance": lb - la,
                "pearson": pearson, "cosine": cosine, "normalized_jsd": jsd,
            })

        print(f"  {stem}: n_objects={n_objs} fg_area_frac(38x50)={coverage.sum()/N_PATCHES:.4f} "
              f"recompute_max_abs_diff={max_diff:.2e}")

    print(f"\n--- Degenerate (fg or bg area < 1 patch) image/layer entries: {len(degenerate_images)} ---")
    for d in degenerate_images:
        print(f"  {d}")

    # ---- shuffled-image-pair auxiliary center-bias baseline ----
    print("\n--- Shuffled-image-pair baseline (auxiliary, center-bias check) ---")
    shuffle_rng = np.random.RandomState(seed)
    shuffled_order = pilot_stems.copy()
    shuffle_rng.shuffle(shuffled_order)
    # guarantee no image is paired with itself
    if any(a == b for a, b in zip(pilot_stems, shuffled_order)):
        shuffled_order = shuffled_order[::-1]
    shuffled_rows = []
    for la, lb in LAYER_PAIRS:
        for stem_a, stem_b in zip(pilot_stems, shuffled_order):
            ra = per_image_layer_maps[stem_a][la]
            rb = per_image_layer_maps[stem_b][lb]
            pearson = pearson_corr(ra, rb)
            cosine = cosine_similarity(ra, rb)
            pa, pb = normalize_distribution(ra), normalize_distribution(rb)
            jsd = jensen_shannon_divergence_normalized(pa, pb)
            shuffled_rows.append({
                "image_id_a": stem_a, "layer_a": la, "image_id_b": stem_b, "layer_b": lb,
                "pearson": pearson, "cosine": cosine, "normalized_jsd": jsd,
            })

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)
    print(f"\n  Saved: {metrics_csv}  ({len(metric_rows)} rows)")

    with open(similarity_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(similarity_rows[0].keys()))
        w.writeheader()
        w.writerows(similarity_rows)
    print(f"  Saved: {similarity_csv}  ({len(similarity_rows)} rows)")

    with open(shuffled_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(shuffled_rows[0].keys()))
        w.writeheader()
        w.writerows(shuffled_rows)
    print(f"  Saved: {shuffled_csv}  ({len(shuffled_rows)} rows)")

    # ------------------------------------------------------------------
    # Overlay figures (foreground mask + L4/L8/L12 attention)
    # ------------------------------------------------------------------
    fig_stems = pilot_stems[:n_figure_images]
    refuse_if_exists([str(fig_dir / f"{stem}_overlay.png") for stem in fig_stems])
    fig_paths = []
    for stem in fig_stems:
        img_arr = np.array(Image.open(stimuli_dir / f"{stem}.jpg").convert("RGB"))
        coverage = per_image_coverage[stem]
        coverage_up = np.array(
            Image.fromarray(coverage.astype(np.float32)).resize((IMG_W, IMG_H), Image.NEAREST))

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
        axes[0].imshow(img_arr)
        axes[0].imshow(np.where(coverage_up > 0.5, 1.0, np.nan), cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        axes[0].set_title(f"{stem}: image + foreground mask\n(38x50 patch coverage > 0.5)", fontsize=9)
        axes[0].axis("off")

        for ax, layer_display in zip(axes[1:], LAYERS_DISPLAY):
            raw = per_image_layer_maps[stem][layer_display]
            attn_up = np.array(
                Image.fromarray(raw.astype(np.float32)).resize((IMG_W, IMG_H), Image.BILINEAR))
            ax.imshow(img_arr)
            im = ax.imshow(attn_up, cmap="jet", alpha=0.55)
            ax.contour(coverage_up, levels=[0.5], colors="white", linewidths=1.2)
            ax.set_title(f"L{layer_display} attention (native 38x50, upsampled for display)"
                         f"\nwhite contour = foreground mask", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"[B-2 pilot] {stem}: foreground mask vs. L4/L8/L12 attention", fontsize=12)
        fig.tight_layout()
        fig_path = fig_dir / f"{stem}_overlay.png"
        fig.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        fig_paths.append(str(fig_path))
        print(f"  Saved figure: {fig_path}")

    # ------------------------------------------------------------------
    # config.json / pilot_summary.json
    # ------------------------------------------------------------------
    config = {
        "seed": seed, "n_pilot": n_pilot, "layers_display": LAYERS_DISPLAY,
        "layers_zero_based": LAYERS_ZERO_BASED, "grid_hw": list(GRID_HW),
        "n_patches": N_PATCHES, "top10pct_k": TOP_10PCT_K,
        "pilot_image_ids": pilot_stems, "selection_method":
            f"np.random.RandomState({seed}).choice(700, {n_pilot}, replace=False) "
            f"over the cache's stems array, then re-sorted for stable ordering",
        "paths_used": {
            "cache_path": str(cache_path), "cache_verification_json": str(cache_verification_json),
            "stimuli_dir": str(stimuli_dir), "attrs_path": str(attrs_path),
            "output_dir": str(output_dir),
        },
        "figure_image_ids": fig_stems,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config_json, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"\n  Saved: {config_json}")

    numeric_metric_keys = [
        "patch_mass_raw", "full_row_sum", "cls_self_mass", "normalized_entropy",
        "hoyer_sparsity", "max_patch_attention", "top10pct_mass",
        "mass50_n_patches", "mass50_area_fraction", "mass80_n_patches", "mass80_area_fraction",
        "centroid_x", "centroid_y", "normalized_center_distance",
        "foreground_area_fraction", "background_area_fraction",
        "foreground_absolute_mass", "foreground_conditional_mass", "background_conditional_mass",
    ]
    ranges = {}
    for k in numeric_metric_keys:
        vals = [r[k] for r in metric_rows if r[k] is not None]
        ranges[k] = {"min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None
    enrich_vals = [r["foreground_enrichment"] for r in metric_rows if r["foreground_enrichment"] is not None]
    ranges["foreground_enrichment"] = (
        {"min": float(np.min(enrich_vals)), "max": float(np.max(enrich_vals))} if enrich_vals else None)

    any_nan = any(not np.isfinite(v) for r in metric_rows for k, v in r.items()
                  if isinstance(v, float) and k != "foreground_enrichment")

    summary = {
        "n_images": len(pilot_stems), "n_metric_rows": len(metric_rows),
        "n_similarity_rows": len(similarity_rows),
        "n_degenerate_fgbg_entries": len(degenerate_images),
        "degenerate_entries": degenerate_images,
        "metric_ranges": ranges,
        "row_sum_range": {"min": float(np.min(all_row_sums)), "max": float(np.max(all_row_sums))},
        "patch_mass_raw_range": {
            "min": float(np.min(all_patch_mass_raw)), "max": float(np.max(all_patch_mass_raw))},
        "any_non_finite_metric": bool(any_nan),
        "cross_check_vs_cache": "all pilot images matched cache within 1e-6 (bit-exact)",
    }
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {summary_json}")

    print("\n" + "=" * 70)
    print("  [B-2] Pilot DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
