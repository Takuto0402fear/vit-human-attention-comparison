"""
Phase 4 extraction: RAW (un-projected) L4/L8/L12 pooled object features for
ALL 700 OSIE images, for the linear-probe extension only.

Deliberately does NOT compute text embeddings, per-token projection
(ln_post/visual_projection), or spatial similarity maps -- Implementation
A (text alignment) is explicitly NOT extended to 700 images per this
task's instructions. Only the probe-relevant RAW hidden-state features are
produced, reusing:
  - the existing full700 image list
    (outputs/osie_attribute_grounding_full700/full700_image_list.csv, the
    same "existing full700 manifest and OSIE loader" already used for the
    earlier attribute-grounding-full700 pilot)
  - the same attrs.mat loader, mask -> patch-weight conversion, and
    positive/negative split as the 50-image text-alignment pilot
    (lib/osie_text_alignment.py, unmodified)
  - the same CLIP hidden-state extractor (lib/clip_vit_hidden.py, unmodified)

Writes only under outputs/osie_probe_full700/: full700_manifest.csv,
object_features.npz (raw_L4/L8/L12 only), objects_metadata.csv, config.json.
Does not touch outputs/osie_text_alignment_pilot/,
outputs/osie_text_alignment_crop_pilot/, or
outputs/osie_attribute_grounding_full700/.

Usage (PowerShell):
    python scripts\\extract_osie_probe_full700_features.py
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
import torch
from PIL import Image

from clip_extractor import CLIP_PATCH_SIZE, load_clip_vit_b16, load_osie_image_tensor
from lib.clip_vit import interpolate_patch_pos_embed, patch_grid_from_image_hw
from lib.clip_vit_hidden import visual_forward_with_hidden_states
from lib.osie_text_alignment import mask_to_patch_weights, split_positive_negative_attributes, weighted_pool_tokens

# ======================= CONFIG =======================
SEED = 42
LAYERS_DISPLAY = [4, 8, 12]
LAYERS_ZERO_BASED = [3, 7, 11]
assert [d - 1 for d in LAYERS_DISPLAY] == LAYERS_ZERO_BASED

GRID_HW_EXPECTED = (38, 50)
IMG_W, IMG_H = 800, 600
MIN_WEIGHT_SUM = 1e-6

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

FULL700_IMAGE_LIST = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_full700\full700_image_list.csv"
PRIOR_ATTRIBUTE_MAPPING = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot\attribute_mapping.json"

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_probe_full700"
MANIFEST_CSV = os.path.join(OUT_DIR, "full700_manifest.csv")
OBJECT_FEATURES_NPZ = os.path.join(OUT_DIR, "object_features.npz")
OBJECTS_METADATA_CSV = os.path.join(OUT_DIR, "objects_metadata.csv")
CONFIG_JSON = os.path.join(OUT_DIR, "config.json")
# ======================================================


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_objects_for_image(f, group, mat_attr_names, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    objects = []
    for j in range(objs_ds.shape[1]):
        obj_g = f[objs_ds[0, j]]
        mp = obj_g["map"][()].T.astype(bool)
        if mp.shape != (h, w):
            raise RuntimeError(f"STOP: object {j} mask shape {mp.shape} != {(h, w)}")
        feat = obj_g["features"][()].flatten()
        if feat.shape[0] != len(mat_attr_names):
            raise RuntimeError(f"STOP: feature length {feat.shape[0]} != {len(mat_attr_names)}")
        objects.append({"object_id": j, "mask": mp, "features": feat})
    return objects


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(SEED)

    print("=" * 70)
    print("  Phase 4 extraction: RAW L4/L8/L12 object features, ALL 700 images")
    print("=" * 70)

    for path in (FULL700_IMAGE_LIST, PRIOR_ATTRIBUTE_MAPPING):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found")

    with open(FULL700_IMAGE_LIST, encoding="utf-8") as f:
        image_ids = sorted(r["image_id"] for r in csv.DictReader(f))
    if len(image_ids) != 700:
        raise RuntimeError(f"STOP: expected 700 images in {FULL700_IMAGE_LIST}, got {len(image_ids)}")
    print(f"  Reusing {len(image_ids)} images from {FULL700_IMAGE_LIST}")

    with open(PRIOR_ATTRIBUTE_MAPPING, encoding="utf-8") as f:
        attribute_mapping = json.load(f)
    mat_attr_names = attribute_mapping["attrs_mat_order_as_stored"]
    mat_to_task = attribute_mapping["mat_name_to_task_name"]

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "reused_from", "source_manifest"])
        w.writeheader()
        for stem in image_ids:
            w.writerow({"image_id": stem, "reused_from": "osie_attribute_grounding_full700",
                        "source_manifest": FULL700_IMAGE_LIST})
    print(f"  Saved: {MANIFEST_CSV}")

    print("\n--- Loading CLIP ViT-B/16 ---")
    model, _ = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}  dtype: {model.dtype}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    f = h5py.File(ATTRS_PATH, "r")
    loaded_names, attrs_index = load_attrs_index(f)
    if loaded_names != mat_attr_names:
        raise RuntimeError("STOP: attrs.mat order changed since mapping was built")
    stim_ids = set(os.path.splitext(fn)[0] for fn in os.listdir(STIM_DIR) if fn.endswith(".jpg"))
    missing = [s for s in image_ids if s not in attrs_index or s not in stim_ids]
    if missing:
        raise RuntimeError(f"STOP: images missing from attrs.mat/stimuli: {missing[:10]}")

    object_keys = []
    raw_feats = {l: [] for l in LAYERS_DISPLAY}
    object_metadata_rows = []
    n_objects_total = 0
    n_objects_skipped = 0
    n_nan_inf = 0

    print(f"\n--- Extracting ({len(image_ids)} images) ---")
    for idx, stem in enumerate(image_ids):
        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        tensor, orig_hw, pad_hw = load_osie_image_tensor(img_path, CLIP_PATCH_SIZE)
        tensor = tensor.to(device)
        grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
        if grid_hw != GRID_HW_EXPECTED:
            raise RuntimeError(f"STOP: {stem} grid {grid_hw} != {GRID_HW_EXPECTED}")
        if orig_hw != (IMG_H, IMG_W):
            raise RuntimeError(f"STOP: {stem} orig_hw {orig_hw} != {(IMG_H, IMG_W)}")
        pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)

        with torch.inference_mode():
            hidden_states, _ = visual_forward_with_hidden_states(
                model.visual, tensor.type(model.dtype), pos_embed=pos_interp,
                layers_zero_based=LAYERS_ZERO_BASED)

        gh, gw = grid_hw
        raw_patch = {}
        for l_disp, l_zero in zip(LAYERS_DISPLAY, LAYERS_ZERO_BASED):
            hs = hidden_states[l_zero][0]
            patch_tokens = hs[1:].detach().cpu().numpy().reshape(gh, gw, 768).astype(np.float64)
            if np.isnan(patch_tokens).any() or np.isinf(patch_tokens).any():
                n_nan_inf += 1
                raise RuntimeError(f"STOP: NaN/Inf in {stem} layer L{l_disp}")
            raw_patch[l_disp] = patch_tokens

        objs = load_objects_for_image(f, attrs_index[stem], mat_attr_names, (IMG_H, IMG_W))
        for obj in objs:
            n_objects_total += 1
            weights = mask_to_patch_weights(obj["mask"], CLIP_PATCH_SIZE)
            if weights.sum() < MIN_WEIGHT_SUM:
                n_objects_skipped += 1
                continue

            pos_task, neg_task = split_positive_negative_attributes(obj["features"], mat_attr_names, mat_to_task)

            object_keys.append(f"{stem}_{obj['object_id']}")
            for l_disp in LAYERS_DISPLAY:
                pooled = weighted_pool_tokens(raw_patch[l_disp], weights, MIN_WEIGHT_SUM)
                raw_feats[l_disp].append(pooled)

            object_metadata_rows.append({
                "image_id": stem, "object_id": obj["object_id"],
                "mask_pixel_count": int(obj["mask"].sum()), "patch_weight_sum": float(weights.sum()),
                "n_positive_attrs": len(pos_task), "positive_attributes": ";".join(pos_task),
                "negative_attributes": ";".join(neg_task),
            })

        if idx == 0 or (idx + 1) % 100 == 0 or idx == len(image_ids) - 1:
            print(f"  [{idx + 1:4d}/{len(image_ids)}] {stem}: OK  ({len(object_keys)} objects so far)")

    peak_mem_mib = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else None)

    save_dict = {"object_keys": np.array(object_keys)}
    for l_disp in LAYERS_DISPLAY:
        save_dict[f"raw_L{l_disp}"] = np.stack(raw_feats[l_disp]).astype(np.float32)
    np.savez_compressed(OBJECT_FEATURES_NPZ, **save_dict)
    print(f"\n  Saved: {OBJECT_FEATURES_NPZ}  ({len(object_keys)} objects)")

    with open(OBJECTS_METADATA_CSV, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["image_id", "object_id", "mask_pixel_count", "patch_weight_sum",
                      "n_positive_attrs", "positive_attributes", "negative_attributes"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(object_metadata_rows)
    print(f"  Saved: {OBJECTS_METADATA_CSV}  ({len(object_metadata_rows)} rows)")

    config = {
        "model": "OpenAI CLIP ViT-B/16 (clip package)", "layers_display": LAYERS_DISPLAY,
        "layers_zero_based": LAYERS_ZERO_BASED, "seed": SEED,
        "n_images": len(image_ids), "n_objects_total": n_objects_total,
        "n_objects_with_features": len(object_keys), "n_objects_skipped_zero_weight": n_objects_skipped,
        "n_nan_inf_detected": n_nan_inf, "peak_vram_mib": peak_mem_mib, "device": str(device),
        "wall_clock_total_s": time.time() - t_start,
        "note": "RAW hidden states only -- no text embeddings, no projection, no spatial maps "
                "were computed at 700-image scale (out of scope per task instructions)",
        "image_selection_method": "reused verbatim from outputs/osie_attribute_grounding_full700/full700_image_list.csv (all 700)",
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Saved: {CONFIG_JSON}")
    print(f"\n  Objects: {len(object_keys)}/{n_objects_total} kept ({n_objects_skipped} skipped)")
    print(f"  Peak VRAM: {peak_mem_mib:.1f} MiB" if peak_mem_mib else "  Peak VRAM: n/a")
    print(f"  Wall clock: {time.time() - t_start:.1f}s")

    print("\n" + "=" * 70)
    print("  Phase 4 extraction: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
