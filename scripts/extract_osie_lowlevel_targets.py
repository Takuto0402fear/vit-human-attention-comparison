"""
Phase 2: automatic (no human labels) low-level visual feature computation
for all 5551 OSIE objects, from the ACTUAL 224x224 image CLIP receives
(pre-normalization RGB) and the OSIE mask transformed through the
IDENTICAL Resize+CenterCrop geometry (nearest-neighbor) -- same
correspondence already verified bug-free in
outputs/osie_text_alignment_spatial_diagnostic/.

For each image (not each object -- the 224x224 image is identical for
every object in that image, so it and its full-image Sobel/Laplacian maps
are computed ONCE per image and reused for every object's mask):
  1. img_224 = CenterCrop(Resize(pil_image))  (bicubic, matches CLIP's own
     preprocessing exactly; Normalize is intentionally NOT applied since
     the targets are computed on raw RGB in [0,1])
  2. luminance / saturation / Sobel-gradient-magnitude / Laplacian are
     computed on the FULL 224x224 image (never mask-cropped first, so no
     spurious edge is introduced at the mask boundary)
  3. each object's mask is transformed through the SAME Resize (nearest-
     neighbor)+CenterCrop, then the 8 continuous low-level targets +
     3 geometry-control values are aggregated inside that mask.

Reuses, unmodified: clip_extractor.load_clip_vit_b16 (only for its
`preprocess` transform's Resize/CenterCrop objects -- the CLIP model
itself is never called, no GPU forward pass in this script), the same
attrs.mat loader and mask geometry as every other OSIE script in this
project, and lib/osie_lowlevel_features.py (new, pure NumPy/SciPy).

Writes only under outputs/osie_lowlevel_targets_full700/. Does not touch
any other existing output.

Usage (PowerShell):
    python scripts\\extract_osie_lowlevel_targets.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os
import random
import time

import h5py
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from clip_extractor import load_clip_vit_b16
from lib.osie_lowlevel_features import compute_object_lowlevel_targets

# ======================= CONFIG =======================
SEED = 42
IMG_W, IMG_H = 800, 600

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

FULL700_IMAGE_LIST = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_full700\full700_image_list.csv"
PRIOR_ATTRIBUTE_MAPPING = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot\attribute_mapping.json"

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_lowlevel_targets_full700"
OBJECT_TARGETS_CSV = os.path.join(OUT_DIR, "lowlevel_object_targets.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "lowlevel_target_summary.csv")
SANITY_JSON = os.path.join(OUT_DIR, "sanity_checks.json")
CONFIG_JSON = os.path.join(OUT_DIR, "config.json")

TARGET_NAMES = [
    "mean_luminance", "mean_red", "mean_green", "mean_blue", "mean_saturation",
    "luminance_contrast", "edge_strength", "fine_texture",
]
GEOMETRY_CONTROL_NAMES = ["mask_area_fraction", "centroid_x", "centroid_y"]
# ======================================================


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def h5_char_str(f, ref):
    return "".join(chr(int(c)) for c in f[ref][()].flatten())


def load_attrs_index(f):
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw = "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[os.path.splitext(raw)[0]] = g
    return attr_names, index


def load_object_masks_for_image(f, group, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    masks = []
    for j in range(objs_ds.shape[1]):
        obj_g = f[objs_ds[0, j]]
        mp = obj_g["map"][()].T.astype(bool)
        if mp.shape != (h, w):
            raise RuntimeError(f"STOP: object {j} mask shape {mp.shape} != {(h, w)}")
        masks.append((j, mp))
    return masks


def build_mask_pipeline(preprocess):
    """Reuses the EXACT Resize target size and CenterCrop box from clip's
    own preprocess (verified bug-free in the Phase 2 spatial diagnostic),
    with the Resize interpolation swapped to nearest-neighbor for masks."""
    resize_t = preprocess.transforms[0]
    crop_t = preprocess.transforms[1]
    assert isinstance(resize_t, T.Resize) and isinstance(crop_t, T.CenterCrop)
    mask_resize = T.Resize(resize_t.size, interpolation=InterpolationMode.NEAREST,
                            max_size=resize_t.max_size, antialias=False)
    return T.Compose([mask_resize, crop_t])


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(SEED)

    print("=" * 70)
    print("  Phase 2: automatic low-level visual feature computation")
    print("=" * 70)

    for path in (FULL700_IMAGE_LIST, PRIOR_ATTRIBUTE_MAPPING):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found")

    with open(FULL700_IMAGE_LIST, encoding="utf-8") as f:
        image_ids = sorted(r["image_id"] for r in csv.DictReader(f))
    if len(image_ids) != 700:
        raise RuntimeError(f"STOP: expected 700 images, got {len(image_ids)}")
    print(f"  Reusing {len(image_ids)} images from {FULL700_IMAGE_LIST}")

    print("\n--- Loading CLIP preprocess (Resize/CenterCrop only -- model never called) ---")
    _, preprocess = load_clip_vit_b16()
    image_pipeline = T.Compose([preprocess.transforms[0], preprocess.transforms[1]])  # Resize (bicubic) + CenterCrop
    mask_pipeline = build_mask_pipeline(preprocess)

    f = h5py.File(ATTRS_PATH, "r")
    _, attrs_index = load_attrs_index(f)
    stim_ids = set(os.path.splitext(fn)[0] for fn in os.listdir(STIM_DIR) if fn.endswith(".jpg"))
    missing = [s for s in image_ids if s not in attrs_index or s not in stim_ids]
    if missing:
        raise RuntimeError(f"STOP: images missing from attrs.mat/stimuli: {missing[:10]}")

    rows = []
    n_objects_total = 0
    n_objects_skipped = 0
    n_nan_inf = 0

    print(f"\n--- Computing low-level targets ({len(image_ids)} images) ---")
    for idx, stem in enumerate(image_ids):
        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        pil_img = Image.open(img_path).convert("RGB")
        if pil_img.size != (IMG_W, IMG_H):
            raise RuntimeError(f"STOP: {stem} unexpected PIL size {pil_img.size}")

        img_224_pil = image_pipeline(pil_img)
        img_float_224 = np.array(img_224_pil).astype(np.float64) / 255.0
        if img_float_224.shape != (224, 224, 3):
            raise RuntimeError(f"STOP: {stem} unexpected 224x224 image shape {img_float_224.shape}")

        masks = load_object_masks_for_image(f, attrs_index[stem], (IMG_H, IMG_W))
        for object_id, mask in masks:
            n_objects_total += 1
            mask_pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            mask_224 = np.array(mask_pipeline(mask_pil)) > 127

            targets = compute_object_lowlevel_targets(img_float_224, mask_224)
            if targets is None:
                n_objects_skipped += 1
                rows.append({"image_id": stem, "object_id": object_id, "valid": False,
                             "skip_reason": "empty_mask_after_transform",
                             **{k: "" for k in TARGET_NAMES + GEOMETRY_CONTROL_NAMES}, "n_mask_pixels_224": 0})
                continue

            for k, v in targets.items():
                if k == "n_mask_pixels_224":
                    continue
                if not np.isfinite(v):
                    n_nan_inf += 1
                    raise RuntimeError(f"STOP: non-finite {k} for {stem}/{object_id}: {v}")

            rows.append({"image_id": stem, "object_id": object_id, "valid": True, "skip_reason": "", **targets})

        if idx == 0 or (idx + 1) % 100 == 0 or idx == len(image_ids) - 1:
            print(f"  [{idx + 1:4d}/{len(image_ids)}] {stem}: OK  ({len(rows)} objects so far)")

    fieldnames = (["image_id", "object_id", "valid", "skip_reason"] + TARGET_NAMES
                  + GEOMETRY_CONTROL_NAMES + ["n_mask_pixels_224"])
    with open(OBJECT_TARGETS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n  Saved: {OBJECT_TARGETS_CSV}  ({len(rows)} rows, {n_objects_skipped} skipped)")

    valid_rows = [r for r in rows if r["valid"]]
    summary_rows = []
    for name in TARGET_NAMES + GEOMETRY_CONTROL_NAMES:
        vals = np.array([r[name] for r in valid_rows], dtype=float)
        summary_rows.append({
            "target": name, "n_valid": len(vals),
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()), "max": float(vals.max()), "median": float(np.median(vals)),
        })
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["target", "n_valid", "mean", "std", "min", "max", "median"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {SUMMARY_CSV}")
    for row in summary_rows:
        print(f"    {row['target']:20s} mean={row['mean']:.4f}  std={row['std']:.4f}  "
              f"[{row['min']:.4f}, {row['max']:.4f}]  n={row['n_valid']}")

    sanity = {
        "n_objects_total": n_objects_total, "n_objects_valid": len(valid_rows),
        "n_objects_skipped": n_objects_skipped, "n_nan_inf_detected": n_nan_inf,
        "image_pipeline_matches_clip_preprocess_resize_crop": True,
        "mask_pipeline_interpolation": "nearest",
        "sobel_laplacian_applied_to_full_image_before_masking": True,
    }
    with open(SANITY_JSON, "w", encoding="utf-8") as fh:
        json.dump(sanity, fh, indent=2)
    print(f"  Saved: {SANITY_JSON}")

    config = {
        "seed": SEED, "n_images": len(image_ids), "n_objects_total": n_objects_total,
        "n_objects_valid": len(valid_rows), "n_objects_skipped": n_objects_skipped,
        "target_definitions": {
            "mean_luminance": "0.2126*R + 0.7152*G + 0.0722*B (ITU-R BT.709), mean inside mask, R/G/B in [0,1]",
            "mean_red": "mean R inside mask, [0,1]",
            "mean_green": "mean G inside mask, [0,1]",
            "mean_blue": "mean B inside mask, [0,1]",
            "mean_saturation": "HSV saturation (matplotlib.colors.rgb_to_hsv), mean inside mask, [0,1]",
            "luminance_contrast": "std of luminance inside mask",
            "edge_strength": "Sobel gradient magnitude (scipy.ndimage.sobel, full image), mean inside mask",
            "fine_texture": "Laplacian (scipy.ndimage.laplace, full image) variance inside mask",
            "mask_area_fraction": "geometry control -- mask pixel count / 224*224",
            "centroid_x": "geometry control -- mean column index of mask pixels / 224",
            "centroid_y": "geometry control -- mean row index of mask pixels / 224",
        },
        "image_selection_method": "reused verbatim from outputs/osie_attribute_grounding_full700/full700_image_list.csv (all 700)",
        "wall_clock_total_s": time.time() - t_start,
        "no_new_dependency_added": "cv2/scikit-image are not installed; scipy.ndimage.sobel/laplace and "
                                    "matplotlib.colors.rgb_to_hsv (both already-installed) were used instead",
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Saved: {CONFIG_JSON}")
    print(f"\n  Wall clock: {time.time() - t_start:.1f}s")

    print("\n" + "=" * 70)
    print("  Phase 2: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
