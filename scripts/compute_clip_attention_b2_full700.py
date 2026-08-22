"""
[B-2] Full700 main analysis, step 1/3 (compute): per-image, per-layer
metrics + per-image layer-pair spatial similarity for all 700 OSIE
images, L4/L8/L12 (index 3/7/11), reading ONLY the existing, already
verified Phase-2 (38x50) attention cache -- no CLIP/torch forward pass
in this script (cls_self_mass = 1 - patch_mass_raw, per task spec; no
extra forward pass needed, see lib.clip_attention_b2.cls_self_mass_from_raw).

Requires --cache-verification-json (default: the existing
outputs/clip_attention_b2_distribution/cache_verification.json) to
already record verdict=="PASS" (read, never overwritten, never
recreated here). In addition to trusting that verdict, this script ALSO
directly re-checks the cache's own basic invariants immediately after
loading it (shape, layers_display/layers_zero_based, stem count/
uniqueness, finiteness, non-negativity) -- see verify_cache_basic_health().

Foreground masks: attrs.mat object-mask union (same convention as
scripts/run_osie_attribute_grounding_pilot.py), transformed to the
native 38x50 grid via lib.osie_text_alignment.mask_to_patch_weights
(identical zero-pad-to-patch-multiple geometry as the image itself).
Foreground/background area fractions are measured directly in this
38x50 patch space, NOT taken from any pixel-domain fg_area_fraction
column. A degenerate image (foreground or background area fraction
below 1/1900, i.e. less than one whole patch) has its mask-dependent
metrics (area fractions, absolute/conditional mass, enrichment) marked
missing for ALL THREE layers (the mask does not depend on layer) --
entropy/Hoyer/max/top10%/mass-coverage/centroid metrics are unaffected
and always computed.

All paths are resolved relative to the repository root or overridable
via CLI flags -- no machine-specific absolute path is hardcoded. Writes
only under --output-dir (refuses to overwrite only the specific files
below, not merely because the directory exists):
  config.json, per_image_metrics.csv, layer_pair_similarity.csv

Usage (PowerShell):
    python scripts\\compute_clip_attention_b2_full700.py
    python scripts\\compute_clip_attention_b2_full700.py --output-dir D:\\tmp\\b2_smoke\\full700
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
import time

import h5py
import numpy as np
from PIL import Image

from lib.clip_attention_b2 import (
    centroid, cls_self_mass_from_raw, coverage_for_mass, cosine_similarity,
    foreground_background_metrics, hoyer_sparsity, jensen_shannon_divergence_normalized,
    max_patch_attention, normalize_distribution, normalized_center_distance,
    normalized_entropy, pearson_corr, refuse_if_exists, top_k_mass,
)
from lib.osie_text_alignment import mask_to_patch_weights

SEED_DEFAULT = 42
LAYERS_DISPLAY = [4, 8, 12]
LAYERS_ZERO_BASED = [3, 7, 11]
GRID_HW = (38, 50)
N_PATCHES = 38 * 50  # 1900
TOP_10PCT_K = 190
IMG_W, IMG_H = 800, 600
N_IMAGES_EXPECTED = 700
LAYER_PAIRS = [(4, 8), (8, 12), (4, 12)]
DEGENERATE_AREA_FRACTION_THRESHOLD = 1.0 / N_PATCHES

DEFAULT_CACHE_PATH = (
    REPO_ROOT / "outputs" / "osie_attribute_grounding_full700" / "attn_cache"
    / "clip_vitb16_full700_L4L8L12_patchgrid.npz")
DEFAULT_CACHE_VERIFICATION_JSON = REPO_ROOT / "outputs" / "clip_attention_b2_distribution" / "cache_verification.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "osie" / "predicting-human-gaze-beyond-pixels"
DEFAULT_STIMULI_DIR = DEFAULT_DATASET_DIR / "data" / "stimuli"
DEFAULT_ATTRS_PATH = DEFAULT_DATASET_DIR / "data" / "attrs.mat"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "clip_attention_b2_distribution"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="[B-2] Full700 compute: per-image/per-layer M-shape spatial-"
                    "distribution metrics + layer-pair spatial similarity, from the "
                    "existing L4/L8/L12 attention cache (no CLIP forward pass).")
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                    help=f"Path to the existing L4/L8/L12 attention cache .npz, read-only "
                        f"(default: {DEFAULT_CACHE_PATH}).")
    p.add_argument("--cache-verification-json", type=Path, default=DEFAULT_CACHE_VERIFICATION_JSON,
                    help="Path to cache_verification.json, read-only -- must have "
                        f"verdict==PASS (default: {DEFAULT_CACHE_VERIFICATION_JSON}).")
    p.add_argument("--stimuli-dir", type=Path, default=DEFAULT_STIMULI_DIR,
                    help=f"Directory containing OSIE stimulus images (<id>.jpg), read-only "
                        f"(default: {DEFAULT_STIMULI_DIR}).")
    p.add_argument("--attrs-path", type=Path, default=DEFAULT_ATTRS_PATH,
                    help=f"Path to OSIE's attrs.mat (object masks), read-only "
                        f"(default: {DEFAULT_ATTRS_PATH}).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Directory to write config.json / per_image_metrics.csv / "
                        "layer_pair_similarity.csv into (created if missing; refuses to "
                        "overwrite any of its own output files that already exist) "
                        f"(default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help=f"Seed for the shuffled-image-pair baseline derangement (default: {SEED_DEFAULT}).")
    return p.parse_args(argv)


def load_attrs_index(f):
    def h5_char_str(ref):
        return "".join(chr(int(c)) for c in f[ref][()].flatten())
    names_ds = f["attrNames"]
    _ = [h5_char_str(names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw_name = "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[Path(raw_name).stem] = g
    return index


def foreground_mask_for_image(f, group, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    n_objs = objs_ds.shape[1]
    if n_objs == 0:
        return None, 0
    fg = np.zeros((h, w), dtype=bool)
    for j in range(n_objs):
        obj_g = f[objs_ds[0, j]]
        mp = obj_g["map"][()].T.astype(bool)
        if mp.shape != (h, w):
            raise RuntimeError(f"STOP: object {j} mask shape {mp.shape} != {(h, w)}")
        fg |= mp
    return fg, n_objs


def fixed_derangement(n, seed):
    """A fixed, seeded derangement (permutation with NO fixed points) of
    range(n) -- used to pair each image with a DIFFERENT image for the
    shuffled-pair center-bias baseline. Rejection sampling on a random
    permutation (n=700 makes a fixed-point-free draw essentially certain
    on the first or second try)."""
    rng = np.random.RandomState(seed)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    raise RuntimeError("STOP: could not draw a fixed-point-free permutation")


def verify_cache_basic_health(cache):
    """Directly re-checks the cache's own basic invariants (shape, layer
    indices, stem count/uniqueness, finiteness, non-negativity), rather
    than relying solely on cache_verification.json's PASS verdict."""
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


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(args.cache_path)
    cache_verification_json = Path(args.cache_verification_json)
    stimuli_dir = Path(args.stimuli_dir)
    attrs_path = Path(args.attrs_path)
    output_dir = Path(args.output_dir)
    seed = args.seed

    config_json = output_dir / "config.json"
    metrics_csv = output_dir / "per_image_metrics.csv"
    similarity_csv = output_dir / "layer_pair_similarity.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_if_exists([str(config_json), str(metrics_csv), str(similarity_csv)])

    print("=" * 70)
    print("  [B-2] Full700 compute: per-image metrics + layer-pair similarity")
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
    with open(cache_verification_json, encoding="utf-8") as fh:
        cache_verif = json.load(fh)
    if cache_verif["verdict"] != "PASS":
        raise RuntimeError(f"STOP: cache_verification.json verdict != PASS")
    print(f"  cache_verification.json verdict: PASS")

    cache = np.load(str(cache_path), allow_pickle=False)
    stems_all = verify_cache_basic_health(cache)
    print("  Cache self-check (shape==(700,3,38,50), layers_display==[4,8,12], "
          "layers_zero_based==[3,7,11], 700 unique stems, finite, non-negative): OK")
    attn_all = cache["attn"].astype(np.float64)  # (700, 3, 38, 50), layer order [L4,L8,L12]

    f = h5py.File(str(attrs_path), "r")
    attrs_index = load_attrs_index(f)

    derangement = fixed_derangement(N_IMAGES_EXPECTED, seed)

    metric_rows = []
    similarity_rows = []
    degenerate_stems = []
    missing_mask_stems = []
    layer_maps_by_image = {}  # stem -> {layer_display: raw (38,50) float64}, kept for similarity pass
    seen_image_layer_pairs = set()

    t0 = time.time()
    for idx, stem in enumerate(stems_all):
        img_path = stimuli_dir / f"{stem}.jpg"
        with Image.open(img_path) as im:
            W, H = im.size
        if (W, H) != (IMG_W, IMG_H):
            raise RuntimeError(f"STOP: {stem} unexpected size {(W, H)}")

        if stem not in attrs_index:
            raise RuntimeError(f"STOP: {stem} not found in attrs.mat index")
        fg_mask, n_objs = foreground_mask_for_image(f, attrs_index[stem], (H, W))
        if fg_mask is None:
            missing_mask_stems.append(stem)
            coverage = None
        else:
            coverage = mask_to_patch_weights(fg_mask, 16)
            if coverage.shape != GRID_HW:
                raise RuntimeError(f"STOP: {stem} coverage shape {coverage.shape} != {GRID_HW}")

        raw_maps = {ld: attn_all[idx, l_idx] for l_idx, ld in enumerate(LAYERS_DISPLAY)}
        layer_maps_by_image[stem] = raw_maps

        image_is_degenerate = False
        for layer_display, raw in raw_maps.items():
            key = (stem, layer_display)
            if key in seen_image_layer_pairs:
                raise RuntimeError(f"STOP: duplicate (image_id, layer) pair {key}")
            seen_image_layer_pairs.add(key)

            if not np.isfinite(raw).all():
                raise RuntimeError(f"STOP: non-finite attention {stem} L{layer_display}")
            if (raw < 0).any():
                raise RuntimeError(f"STOP: negative attention {stem} L{layer_display}")

            patch_mass_raw = float(raw.sum())
            cls_self_mass = cls_self_mass_from_raw(raw)
            p = normalize_distribution(raw)

            h_norm = normalized_entropy(p, N_PATCHES)
            hoyer = hoyer_sparsity(raw)
            max_p = max_patch_attention(p)
            top10 = top_k_mass(p, TOP_10PCT_K)
            n50, area50 = coverage_for_mass(p, 0.5)
            n80, area80 = coverage_for_mass(p, 0.8)
            cx, cy = centroid(p, GRID_HW)
            center_dist = normalized_center_distance((cx, cy))

            row = {
                "image_id": stem, "layer": layer_display, "n_objects": n_objs,
                "patch_mass_raw": patch_mass_raw, "cls_self_mass": cls_self_mass,
                "normalized_entropy": h_norm, "hoyer_sparsity": hoyer,
                "max_patch_attention": max_p, "top10pct_mass": top10,
                "mass50_n_patches": n50, "mass50_area_fraction": area50,
                "mass80_n_patches": n80, "mass80_area_fraction": area80,
                "centroid_x": cx, "centroid_y": cy,
                "normalized_center_distance": center_dist,
                "foreground_area_fraction": None, "background_area_fraction": None,
                "foreground_absolute_mass": None, "foreground_conditional_mass": None,
                "background_conditional_mass": None, "foreground_enrichment": None,
                "is_degenerate_fgbg": None, "mask_missing": coverage is None,
            }

            if coverage is not None:
                fgbg = foreground_background_metrics(raw, coverage)
                row.update({
                    "foreground_area_fraction": fgbg["foreground_area_fraction"],
                    "background_area_fraction": fgbg["background_area_fraction"],
                    "foreground_absolute_mass": fgbg["foreground_absolute_mass"],
                    "foreground_conditional_mass": fgbg["foreground_conditional_mass"],
                    "background_conditional_mass": fgbg["background_conditional_mass"],
                    "foreground_enrichment": fgbg["foreground_enrichment"],
                    "is_degenerate_fgbg": fgbg["is_degenerate"],
                })
                if fgbg["is_degenerate"]:
                    image_is_degenerate = True

            metric_rows.append(row)

        if image_is_degenerate:
            degenerate_stems.append(stem)

        if (idx + 1) % 100 == 0 or idx == 0 or idx == N_IMAGES_EXPECTED - 1:
            print(f"  [{idx + 1:4d}/{N_IMAGES_EXPECTED}] {stem}  "
                  f"elapsed={time.time() - t0:6.1f}s")

    if missing_mask_stems:
        raise RuntimeError(
            f"STOP: {len(missing_mask_stems)} images have zero attrs.mat objects "
            f"(no foreground mask at all): {missing_mask_stems[:10]}")

    # ------------------------------------------------------------------
    # Final per_image_metrics.csv verification (before writing anything)
    # ------------------------------------------------------------------
    if len(metric_rows) != N_IMAGES_EXPECTED * len(LAYERS_DISPLAY):
        raise RuntimeError(
            f"STOP: {len(metric_rows)} metric rows, expected "
            f"{N_IMAGES_EXPECTED * len(LAYERS_DISPLAY)} (700 images x 3 layers)")
    row_images = set(r["image_id"] for r in metric_rows)
    if len(row_images) != N_IMAGES_EXPECTED:
        raise RuntimeError(f"STOP: {len(row_images)} unique image_ids in metric_rows, expected {N_IMAGES_EXPECTED}")
    per_image_layers = {}
    for r in metric_rows:
        per_image_layers.setdefault(r["image_id"], set()).add(r["layer"])
    bad = {img: layers for img, layers in per_image_layers.items() if layers != set(LAYERS_DISPLAY)}
    if bad:
        raise RuntimeError(f"STOP: {len(bad)} images do not have exactly layers {LAYERS_DISPLAY}: {list(bad)[:5]}")
    print(f"\n  Verification: {len(metric_rows)} metric rows (700x3), "
          f"{len(row_images)} unique images, every image has exactly layers {LAYERS_DISPLAY}: OK")

    # ------------------------------------------------------------------
    # Layer-pair spatial similarity: same-image + shuffled-image baseline
    # ------------------------------------------------------------------
    print("\n--- Computing layer-pair spatial similarity (same-image + shuffled baseline) ---")
    for la, lb in LAYER_PAIRS:
        for i, stem in enumerate(stems_all):
            ra = layer_maps_by_image[stem][la]
            rb_same = layer_maps_by_image[stem][lb]
            pearson = pearson_corr(ra, rb_same)
            cosine = cosine_similarity(ra, rb_same)
            pa, pb = normalize_distribution(ra), normalize_distribution(rb_same)
            jsd = jensen_shannon_divergence_normalized(pa, pb)
            similarity_rows.append({
                "image_id": stem, "layer_a": la, "layer_b": lb,
                "layer_distance": lb - la, "pair_type": "same_image",
                "partner_image_id": stem,
                "pearson": pearson, "cosine": cosine, "normalized_jsd": jsd,
            })

            partner_stem = stems_all[derangement[i]]
            rb_shuf = layer_maps_by_image[partner_stem][lb]
            pearson_s = pearson_corr(ra, rb_shuf)
            cosine_s = cosine_similarity(ra, rb_shuf)
            pb_s = normalize_distribution(rb_shuf)
            jsd_s = jensen_shannon_divergence_normalized(pa, pb_s)
            similarity_rows.append({
                "image_id": stem, "layer_a": la, "layer_b": lb,
                "layer_distance": lb - la, "pair_type": "shuffled_baseline",
                "partner_image_id": partner_stem,
                "pearson": pearson_s, "cosine": cosine_s, "normalized_jsd": jsd_s,
            })

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)
    print(f"\n  Saved: {metrics_csv}  ({len(metric_rows)} rows, expected {N_IMAGES_EXPECTED * 3})")

    with open(similarity_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(similarity_rows[0].keys()))
        w.writeheader()
        w.writerows(similarity_rows)
    print(f"  Saved: {similarity_csv}  ({len(similarity_rows)} rows, "
          f"expected {N_IMAGES_EXPECTED * len(LAYER_PAIRS) * 2})")

    config = {
        "seed": seed, "n_images": N_IMAGES_EXPECTED, "layers_display": LAYERS_DISPLAY,
        "layers_zero_based": LAYERS_ZERO_BASED, "grid_hw": list(GRID_HW),
        "n_patches": N_PATCHES, "top10pct_k": TOP_10PCT_K, "layer_pairs": LAYER_PAIRS,
        "degenerate_area_fraction_threshold": DEGENERATE_AREA_FRACTION_THRESHOLD,
        "n_degenerate_images": len(degenerate_stems), "degenerate_image_ids": degenerate_stems,
        "paths_used": {
            "cache_path": str(cache_path), "cache_verification_json": str(cache_verification_json),
            "stimuli_dir": str(stimuli_dir), "attrs_path": str(attrs_path),
            "output_dir": str(output_dir),
        },
        "cls_self_mass_method": "1 - patch_mass_raw (no extra forward pass; exact up to "
                                 "float32 softmax-sum rounding, verified against a real "
                                 "full_row_sum recompute for 10 pilot images, max deviation 1.19e-7)",
        "shuffled_baseline_method": f"fixed seeded derangement (no fixed points) of the "
                                     f"700 image indices, np.random.RandomState({seed}).permutation, "
                                     f"rejection-sampled for zero fixed points",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config_json, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Saved: {config_json}")
    print(f"\n  Degenerate images (fg or bg area < 1/1900): {len(degenerate_stems)}")
    if degenerate_stems:
        print(f"    {degenerate_stems}")

    print("\n" + "=" * 70)
    print("  [B-2] Full700 compute DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
