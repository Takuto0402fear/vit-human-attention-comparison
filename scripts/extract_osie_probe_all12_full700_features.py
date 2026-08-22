"""
Phase 1 (all-12-layer extension) extraction: RAW (un-projected) pooled
object features for CLIP ViT-B/16's L1 through L12, ALL 700 OSIE images.

Each image is forwarded through CLIP exactly ONCE, extracting all 12
transformer-block outputs simultaneously
(lib.clip_vit_hidden.visual_forward_with_hidden_states with
layers_zero_based=list(range(12))) -- not 12 separate forward passes.
Only the mask-pooled per-object feature (768-dim) is kept per layer; the
underlying (38,50,768) patch grid is discarded after pooling for every
image, so nothing scales with image count beyond the small per-object
vectors (no raw patch tokens are ever written to disk).

Display layer L{i} (1-indexed) <-> zero-based index i-1 into
visual.transformer.resblocks, asserted below.

Reuses, unmodified, the exact same pipeline as
scripts/extract_osie_probe_full700_features.py (which this script does
NOT replace -- that script's outputs/osie_probe_full700/ are left
untouched):
  - clip_extractor.load_osie_image_tensor / CLIP_PATCH_SIZE
  - lib.clip_vit.interpolate_patch_pos_embed / patch_grid_from_image_hw
  - lib.clip_vit_hidden.visual_forward_with_hidden_states
  - lib.osie_text_alignment.mask_to_patch_weights / weighted_pool_tokens /
    split_positive_negative_attributes
  - the existing full700 image list
    (outputs/osie_attribute_grounding_full700/full700_image_list.csv)

A reproducibility check against outputs/osie_probe_full700/object_features.npz
(the existing L4/L8/L12-only extraction) is run automatically at the end:
if this script's L4/L8/L12 pooled features don't match the existing ones
within tolerance for every object, extraction STOPS and does not proceed
to write final outputs (per task instructions: "許容誤差を超えて違う場合
は全層解析を止め、原因を調べる").

Writes only under outputs/osie_probe_all12_full700/. Does not touch
outputs/osie_probe_full700/ or any other existing output.

Usage (PowerShell):
    python scripts\\extract_osie_probe_all12_full700_features.py
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
LAYERS_DISPLAY = list(range(1, 13))               # L1..L12
LAYERS_ZERO_BASED = list(range(12))               # 0..11
assert [d - 1 for d in LAYERS_DISPLAY] == LAYERS_ZERO_BASED, "STOP: layer mapping inconsistent"
REPRO_CHECK_LAYERS = [4, 8, 12]
REPRO_TOLERANCE = 1e-4  # generous float32 tolerance across independent runs

GRID_HW_EXPECTED = (38, 50)
IMG_W, IMG_H = 800, 600
MIN_WEIGHT_SUM = 1e-6

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

FULL700_IMAGE_LIST = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_full700\full700_image_list.csv"
PRIOR_ATTRIBUTE_MAPPING = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot\attribute_mapping.json"
EXISTING_L4L8L12_NPZ = r"C:\Users\user\gaze\outputs\osie_probe_full700\object_features.npz"

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_probe_all12_full700"
MANIFEST_CSV = os.path.join(OUT_DIR, "full700_manifest.csv")
OBJECT_FEATURES_NPZ = os.path.join(OUT_DIR, "object_features_all12.npz")
OBJECTS_METADATA_CSV = os.path.join(OUT_DIR, "object_features_metadata.csv")
CONFIG_JSON = os.path.join(OUT_DIR, "config.json")
REPRO_CHECK_JSON = os.path.join(OUT_DIR, "extraction_reproducibility_check.json")
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


def run_reproducibility_check(object_keys, raw_feats):
    if not os.path.isfile(EXISTING_L4L8L12_NPZ):
        return {"skipped": True, "reason": f"{EXISTING_L4L8L12_NPZ} not found"}
    existing = np.load(EXISTING_L4L8L12_NPZ, allow_pickle=False)
    existing_keys = list(existing["object_keys"])
    key_to_idx_existing = {k: i for i, k in enumerate(existing_keys)}
    key_to_idx_new = {k: i for i, k in enumerate(object_keys)}
    common = sorted(set(existing_keys) & set(object_keys))

    max_abs_diff = {}
    for l in REPRO_CHECK_LAYERS:
        existing_arr = existing[f"raw_L{l}"].astype(np.float64)
        new_arr = np.stack(raw_feats[l]).astype(np.float64)
        diffs = []
        for k in common:
            e = existing_arr[key_to_idx_existing[k]]
            n = new_arr[key_to_idx_new[k]]
            diffs.append(np.abs(e - n).max())
        max_abs_diff[f"L{l}"] = float(max(diffs)) if diffs else float("nan")

    passed = all(v <= REPRO_TOLERANCE for v in max_abs_diff.values())
    return {
        "skipped": False, "passed": passed, "tolerance": REPRO_TOLERANCE,
        "n_common_objects": len(common), "max_abs_diff_by_layer": max_abs_diff,
        "existing_source": EXISTING_L4L8L12_NPZ,
    }


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(SEED)

    print("=" * 70)
    print("  Phase 1 (all-12-layer) extraction: RAW L1..L12 object features")
    print("=" * 70)

    for path in (FULL700_IMAGE_LIST, PRIOR_ATTRIBUTE_MAPPING):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found")

    with open(FULL700_IMAGE_LIST, encoding="utf-8") as f:
        image_ids = sorted(r["image_id"] for r in csv.DictReader(f))
    if len(image_ids) != 700:
        raise RuntimeError(f"STOP: expected 700 images, got {len(image_ids)}")
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

    print(f"\n--- Extracting all 12 layers, one forward pass per image ({len(image_ids)} images) ---")
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
            hs = hidden_states[l_zero][0]  # (seq_len, 768) -- CLS at index 0
            patch_tokens = hs[1:].detach().cpu().numpy().reshape(gh, gw, 768).astype(np.float64)
            if patch_tokens.shape[0] * patch_tokens.shape[1] != gh * gw:
                raise RuntimeError(f"STOP: {stem} L{l_disp} unexpected patch count")
            if np.isnan(patch_tokens).any() or np.isinf(patch_tokens).any():
                n_nan_inf += 1
                raise RuntimeError(f"STOP: NaN/Inf in {stem} layer L{l_disp}")
            raw_patch[l_disp] = patch_tokens
        del hidden_states

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

    # ------------------------------------------------------------------
    # Reproducibility gate BEFORE writing final outputs.
    # ------------------------------------------------------------------
    print("\n--- Reproducibility check vs. outputs/osie_probe_full700/ (L4/L8/L12) ---")
    repro = run_reproducibility_check(object_keys, raw_feats)
    with open(REPRO_CHECK_JSON, "w", encoding="utf-8") as fh:
        json.dump(repro, fh, indent=2)
    print(json.dumps(repro, indent=2))
    if not repro.get("skipped") and not repro.get("passed"):
        raise RuntimeError(
            f"STOP: L4/L8/L12 reproducibility check FAILED against {EXISTING_L4L8L12_NPZ} -- "
            f"see {REPRO_CHECK_JSON}. Refusing to write final all-12-layer outputs until "
            "the discrepancy is investigated.")
    print(f"  Saved: {REPRO_CHECK_JSON}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dict = {"object_keys": np.array(object_keys)}
    for l_disp in LAYERS_DISPLAY:
        save_dict[f"raw_L{l_disp}"] = np.stack(raw_feats[l_disp]).astype(np.float32)
    np.savez_compressed(OBJECT_FEATURES_NPZ, **save_dict)
    print(f"\n  Saved: {OBJECT_FEATURES_NPZ}  ({len(object_keys)} objects, 12 layers)")
    npz_size_mb = os.path.getsize(OBJECT_FEATURES_NPZ) / 1024 ** 2
    print(f"  File size: {npz_size_mb:.1f} MiB")

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
        "wall_clock_total_s": time.time() - t_start, "object_features_npz_size_mib": npz_size_mb,
        "note": "RAW hidden states only (no ln_post/proj), pooled per object per layer; "
                "full patch grids are never written to disk -- discarded after pooling, per image.",
        "reproducibility_check": repro,
        "image_selection_method": "reused verbatim from outputs/osie_attribute_grounding_full700/full700_image_list.csv (all 700)",
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Saved: {CONFIG_JSON}")
    print(f"\n  Objects: {len(object_keys)}/{n_objects_total} kept ({n_objects_skipped} skipped)")
    print(f"  Peak VRAM: {peak_mem_mib:.1f} MiB" if peak_mem_mib else "  Peak VRAM: n/a")
    print(f"  Wall clock: {time.time() - t_start:.1f}s")

    print("\n" + "=" * 70)
    print("  Phase 1 (all-12-layer) extraction: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
