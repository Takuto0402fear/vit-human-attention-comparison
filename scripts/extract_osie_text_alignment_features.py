"""
Main GPU extraction pass for the OSIE text-alignment pilot: one CLIP
forward pass per image (reused for every object and every attribute in
that image), producing:

  - Per-OBJECT pooled features at L4/L8/L12, in two variants:
      raw  : weighted mean of the RAW (un-normalized, no ln_post/proj)
             patch hidden states -- for the linear probe (Implementation B).
      proj : weighted mean of the per-token PROJECTED+L2-normalized patch
             features (ln_post -> visual_projection -> normalize, applied
             per token before pooling), re-normalized after pooling -- for
             the text-alignment margin (Implementation A).
    Both use the SAME per-object mask -> patch-coverage weights (see
    lib/osie_text_alignment.py::mask_to_patch_weights).
  - Per-ATTRIBUTE (union-mask) spatial similarity scores at L4/L8/L12
    against that attribute's equal-ensemble text vector, plus a seeded
    circular-shift spatial baseline.
  - Text embeddings for all 192 prompts + 12 raw-label baselines + the 12
    equal-ensemble vectors.

Requires (gates, refuses to run otherwise):
  - outputs/osie_text_alignment_pilot/sanity_checks.json with passed=true
  - outputs/osie_text_alignment_pilot/attribute_mapping.json and
    prompt_bank.json (scripts/build_osie_text_alignment_prompts.py)

Reuses unmodified: clip_extractor.py, lib/clip_vit.py, lib/clip_vit_hidden.py.
Reuses the pilot image list from the EARLIER attribute-grounding pilot
(outputs/osie_attribute_grounding_pilot/pilot_image_list.csv) verbatim --
the exact same 50 images, seed=42.

Writes only under outputs/osie_text_alignment_pilot/. Does not touch
outputs/osie_attribute_grounding_pilot/, outputs/expA_clip/,
outputs/attn_cache/, or any OSIE source file.

Usage (PowerShell):
    python scripts\\extract_osie_text_alignment_features.py
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
import torch.nn.functional as F
from PIL import Image

import clip
from clip_extractor import CLIP_PATCH_SIZE, load_clip_vit_b16, load_osie_image_tensor
from lib.clip_vit import interpolate_patch_pos_embed, patch_grid_from_image_hw
from lib.clip_vit_hidden import project_and_normalize_tokens, visual_forward_with_hidden_states
from lib.osie_text_alignment import (
    build_equal_ensemble, circular_shift_2d, l2_normalize, mask_to_patch_weights,
    random_nonzero_shift, split_positive_negative_attributes, weighted_pool_tokens,
)

# ======================= CONFIG =======================
SEED = 42
LAYERS_DISPLAY = [4, 8, 12]
LAYERS_ZERO_BASED = [3, 7, 11]
assert [d - 1 for d in LAYERS_DISPLAY] == LAYERS_ZERO_BASED, "STOP: layer mapping inconsistent"

GRID_HW_EXPECTED = (38, 50)
IMG_W, IMG_H = 800, 600
MIN_WEIGHT_SUM = 1e-6

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

PRIOR_PILOT_IMAGE_LIST = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_pilot\pilot_image_list.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot"
SANITY_JSON = os.path.join(OUT_DIR, "sanity_checks.json")
ATTRIBUTE_MAPPING_JSON = os.path.join(OUT_DIR, "attribute_mapping.json")
PROMPT_BANK_JSON = os.path.join(OUT_DIR, "prompt_bank.json")

PILOT_MANIFEST_CSV = os.path.join(OUT_DIR, "pilot_manifest.csv")
OBJECTS_METADATA_CSV = os.path.join(OUT_DIR, "objects_metadata.csv")
SPATIAL_SCORES_CSV = os.path.join(OUT_DIR, "spatial_scores.csv")
OBJECT_FEATURES_NPZ = os.path.join(OUT_DIR, "object_features.npz")
TEXT_EMBEDDINGS_NPZ = os.path.join(OUT_DIR, "text_embeddings.npz")
EXTRACTION_LOG_JSON = os.path.join(OUT_DIR, "extraction_run_log.json")
CONFIG_JSON = os.path.join(OUT_DIR, "config.json")
# ======================================================


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_gate(path, what):
    if not os.path.isfile(path):
        raise RuntimeError(f"STOP: {what} not found at {path} -- run the prerequisite script first")
    return path


def load_prior_pilot_image_ids():
    require_gate(PRIOR_PILOT_IMAGE_LIST, "prior attribute-grounding pilot manifest")
    with open(PRIOR_PILOT_IMAGE_LIST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["image_id"] for r in rows]
    if len(ids) != 50:
        raise RuntimeError(f"STOP: expected 50 images, got {len(ids)}")
    return sorted(ids)


def h5_char_str(f, ref):
    arr = f[ref][()]
    return "".join(chr(int(c)) for c in arr.flatten())


def load_attrs_index(f):
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw_name = "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[os.path.splitext(raw_name)[0]] = g
    return attr_names, index


def load_objects_for_image(f, group, mat_attr_names, expected_hw):
    """Returns list of dicts: {object_id, mask (H,W) bool, features (len(mat_attr_names),) float}."""
    objs_ds = group["objs"]
    n_objs = objs_ds.shape[1]
    if n_objs == 0:
        raise RuntimeError("STOP: image with zero objects")
    h, w = expected_hw
    objects = []
    for j in range(n_objs):
        obj_g = f[objs_ds[0, j]]
        mp = obj_g["map"][()].T.astype(bool)
        if mp.shape != (h, w):
            raise RuntimeError(f"STOP: object {j} mask shape {mp.shape} != {(h, w)}")
        feat = obj_g["features"][()].flatten()
        if feat.shape[0] != len(mat_attr_names):
            raise RuntimeError(f"STOP: feature length {feat.shape[0]} != {len(mat_attr_names)}")
        objects.append({"object_id": j, "mask": mp, "features": feat})
    return objects


def encode_text_prompts(model, prompt_bank, attribute_order, device):
    """
    Returns dict:
      prompt_vectors      : {attr: (16, 512) L2-normalized}
      raw_label_vectors   : {attr: (512,) L2-normalized}
      equal_ensemble_vectors : {attr: (512,) L2-normalized mean-then-renormalized}
    """
    prompt_vectors, raw_label_vectors, equal_ensemble_vectors = {}, {}, {}
    for attr in attribute_order:
        data = prompt_bank["attributes"][attr]
        sentences = [p["sentence"] for p in sorted(data["prompts"], key=lambda p: p["prompt_index"])]
        tokens = clip.tokenize(sentences).to(device)
        with torch.inference_mode():
            emb = model.encode_text(tokens).float()
        emb = F.normalize(emb, dim=-1).detach().cpu().numpy()
        if emb.shape != (16, 512):
            raise RuntimeError(f"STOP: prompt embedding shape {emb.shape} != (16,512) for {attr}")
        prompt_vectors[attr] = emb
        equal_ensemble_vectors[attr] = build_equal_ensemble(emb)

        raw_tokens = clip.tokenize([data["raw_label_baseline"]]).to(device)
        with torch.inference_mode():
            raw_emb = model.encode_text(raw_tokens).float()
        raw_emb = F.normalize(raw_emb, dim=-1).detach().cpu().numpy()[0]
        raw_label_vectors[attr] = raw_emb
    return prompt_vectors, raw_label_vectors, equal_ensemble_vectors


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(SEED)

    print("=" * 70)
    print("  OSIE Text-Alignment Pilot: feature extraction (L4/L8/L12)")
    print("=" * 70)

    require_gate(SANITY_JSON, "sanity_checks.json")
    with open(SANITY_JSON, encoding="utf-8") as f:
        sanity = json.load(f)
    if not sanity.get("passed", False):
        raise RuntimeError(f"STOP: sanity check did not PASS ({SANITY_JSON}) -- refusing to extract")
    print(f"  Sanity check: PASS (min cosine check1={sanity['check1_native_224']['min_cosine']:.6f}, "
          f"check2={sanity['check2_osie_nonsquare']['min_cosine']:.6f})")

    require_gate(ATTRIBUTE_MAPPING_JSON, "attribute_mapping.json")
    require_gate(PROMPT_BANK_JSON, "prompt_bank.json")
    with open(ATTRIBUTE_MAPPING_JSON, encoding="utf-8") as f:
        attribute_mapping = json.load(f)
    with open(PROMPT_BANK_JSON, encoding="utf-8") as f:
        prompt_bank = json.load(f)

    mat_attr_names = attribute_mapping["attrs_mat_order_as_stored"]
    mat_to_task = attribute_mapping["mat_name_to_task_name"]
    task_attribute_order = attribute_mapping["task_attribute_order"]
    print(f"  attrs.mat order: {mat_attr_names}")
    print(f"  task attribute order: {task_attribute_order}")

    # ------------------------------------------------------------------
    # Reuse the EXACT prior 50-image, seed=42 manifest.
    # ------------------------------------------------------------------
    image_ids = load_prior_pilot_image_ids()
    print(f"\n--- Reusing prior pilot manifest: {len(image_ids)} images (seed=42) ---")
    with open(PILOT_MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "reused_from_prior_pilot", "source_manifest"])
        w.writeheader()
        for stem in image_ids:
            w.writerow({"image_id": stem, "reused_from_prior_pilot": True,
                        "source_manifest": PRIOR_PILOT_IMAGE_LIST})
    print(f"  Saved: {PILOT_MANIFEST_CSV}")

    # ------------------------------------------------------------------
    # Load attrs.mat objects for these 50 images
    # ------------------------------------------------------------------
    print("\n--- Loading attrs.mat objects ---")
    f = h5py.File(ATTRS_PATH, "r")
    loaded_names, attrs_index = load_attrs_index(f)
    if loaded_names != mat_attr_names:
        raise RuntimeError(f"STOP: attrs.mat order changed since mapping was built: {loaded_names}")
    stim_ids = set(os.path.splitext(fn)[0] for fn in os.listdir(STIM_DIR) if fn.endswith(".jpg"))
    missing = [s for s in image_ids if s not in attrs_index or s not in stim_ids]
    if missing:
        raise RuntimeError(f"STOP: pilot images missing from attrs.mat/stimuli: {missing}")

    objects_by_image = {}
    n_total_objects = 0
    for stem in image_ids:
        objs = load_objects_for_image(f, attrs_index[stem], mat_attr_names, expected_hw=(IMG_H, IMG_W))
        objects_by_image[stem] = objs
        n_total_objects += len(objs)
    print(f"  {n_total_objects} objects across {len(image_ids)} images")

    # ------------------------------------------------------------------
    # Load CLIP + encode all text prompts
    # ------------------------------------------------------------------
    print("\n--- Loading CLIP ViT-B/16 + encoding text prompts ---")
    model, _ = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}  dtype: {model.dtype}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prompt_vectors, raw_label_vectors, equal_ensemble_vectors = encode_text_prompts(
        model, prompt_bank, task_attribute_order, device)
    np.savez_compressed(
        TEXT_EMBEDDINGS_NPZ,
        attribute_order=np.array(task_attribute_order),
        prompt_vectors=np.stack([prompt_vectors[a] for a in task_attribute_order]),       # (12,16,512)
        raw_label_vectors=np.stack([raw_label_vectors[a] for a in task_attribute_order]), # (12,512)
        equal_ensemble_vectors=np.stack([equal_ensemble_vectors[a] for a in task_attribute_order]),  # (12,512)
    )
    print(f"  Saved: {TEXT_EMBEDDINGS_NPZ}")

    # ------------------------------------------------------------------
    # Main per-image extraction loop
    # ------------------------------------------------------------------
    print(f"\n--- Extracting L4/L8/L12 hidden states + object/attribute features ({len(image_ids)} images) ---")

    object_keys, raw_feats, proj_feats = [], {l: [] for l in LAYERS_DISPLAY}, {l: [] for l in LAYERS_DISPLAY}
    object_metadata_rows = []
    spatial_rows = []
    n_objects_skipped_zero_weight = 0
    n_attr_presence_skipped_zero_weight = 0
    n_nan_inf_detected = 0
    shift_rng = np.random.RandomState(SEED)
    timings = []

    for idx, stem in enumerate(image_ids):
        t0 = time.perf_counter()
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
        proj_patch = {}
        for l_disp, l_zero in zip(LAYERS_DISPLAY, LAYERS_ZERO_BASED):
            hs = hidden_states[l_zero][0]  # (seq_len, 768)
            patch_tokens_raw = hs[1:].detach().cpu().numpy().reshape(gh, gw, 768).astype(np.float64)
            with torch.inference_mode():
                proj = project_and_normalize_tokens(model.visual, hs[1:])  # (gh*gw, 512)
            patch_tokens_proj = proj.detach().cpu().numpy().reshape(gh, gw, 512).astype(np.float64)
            if np.isnan(patch_tokens_raw).any() or np.isinf(patch_tokens_raw).any() \
                    or np.isnan(patch_tokens_proj).any() or np.isinf(patch_tokens_proj).any():
                n_nan_inf_detected += 1
                raise RuntimeError(f"STOP: NaN/Inf in {stem} layer L{l_disp} patch tokens")
            raw_patch[l_disp] = patch_tokens_raw
            proj_patch[l_disp] = patch_tokens_proj

        # ---- per-object pooled features ----
        objs = objects_by_image[stem]
        for obj in objs:
            weights = mask_to_patch_weights(obj["mask"], CLIP_PATCH_SIZE)
            if weights.sum() < MIN_WEIGHT_SUM:
                n_objects_skipped_zero_weight += 1
                continue

            pos_task, neg_task = split_positive_negative_attributes(
                obj["features"], mat_attr_names, mat_to_task)

            object_keys.append(f"{stem}_{obj['object_id']}")
            for l_disp in LAYERS_DISPLAY:
                raw_pooled = weighted_pool_tokens(raw_patch[l_disp], weights, MIN_WEIGHT_SUM)
                proj_pooled = weighted_pool_tokens(proj_patch[l_disp], weights, MIN_WEIGHT_SUM)
                proj_pooled = l2_normalize(proj_pooled, axis=-1)
                raw_feats[l_disp].append(raw_pooled)
                proj_feats[l_disp].append(proj_pooled)

            object_metadata_rows.append({
                "image_id": stem, "object_id": obj["object_id"],
                "mask_pixel_count": int(obj["mask"].sum()),
                "patch_weight_sum": float(weights.sum()),
                "n_positive_attrs": len(pos_task),
                "positive_attributes": ";".join(pos_task),
                "negative_attributes": ";".join(neg_task),
            })

        # ---- per-attribute union-mask spatial scores ----
        for mat_idx, mat_name in enumerate(mat_attr_names):
            task_name = mat_to_task[mat_name]
            pos_masks = [o["mask"] for o in objs if o["features"][mat_idx] > 0]
            if not pos_masks:
                continue  # attribute not present in this image -- skip, not zero
            union_mask = np.zeros_like(objs[0]["mask"], dtype=bool)
            for m in pos_masks:
                union_mask |= m
            union_weights = mask_to_patch_weights(union_mask, CLIP_PATCH_SIZE)
            if union_weights.sum() < MIN_WEIGHT_SUM:
                n_attr_presence_skipped_zero_weight += 1
                continue

            dy, dx = random_nonzero_shift(gh, gw, shift_rng)
            shifted_weights = circular_shift_2d(union_weights, dy, dx)

            for l_disp in LAYERS_DISPLAY:
                text_vec = equal_ensemble_vectors[task_name]
                sim_map = proj_patch[l_disp].reshape(-1, 512) @ text_vec  # (gh*gw,)
                sim_map = sim_map.reshape(gh, gw)

                def region_stats(weight_grid):
                    w_flat = weight_grid.reshape(-1)
                    s_flat = sim_map.reshape(-1)
                    inside_w = w_flat
                    outside_w = 1.0 - w_flat
                    inside_mean = float((s_flat * inside_w).sum() / max(inside_w.sum(), 1e-12))
                    outside_mean = (float((s_flat * outside_w).sum() / outside_w.sum())
                                    if outside_w.sum() > 1e-9 else float("nan"))
                    binary = (w_flat > 0).astype(int)
                    if binary.min() == binary.max():
                        auc = float("nan")
                    else:
                        from sklearn.metrics import roc_auc_score
                        auc = float(roc_auc_score(binary, s_flat))
                    return inside_mean, outside_mean, auc

                inside_mean, outside_mean, auc = region_stats(union_weights)
                s_inside_mean, s_outside_mean, s_auc = region_stats(shifted_weights)

                if any(np.isnan(v) for v in [inside_mean]) or np.isinf(inside_mean):
                    n_nan_inf_detected += 1

                spatial_rows.append({
                    "image_id": stem, "attribute": task_name, "layer": l_disp,
                    "inside_mean": inside_mean, "outside_mean": outside_mean,
                    "diff": inside_mean - outside_mean if np.isfinite(outside_mean) else float("nan"),
                    "auc": auc,
                    "shift_dy": dy, "shift_dx": dx,
                    "shift_inside_mean": s_inside_mean, "shift_outside_mean": s_outside_mean,
                    "shift_diff": s_inside_mean - s_outside_mean if np.isfinite(s_outside_mean) else float("nan"),
                    "shift_auc": s_auc,
                })

        t1 = time.perf_counter()
        timings.append({"image": stem, "t_total_s": t1 - t0})
        if idx == 0 or (idx + 1) % 10 == 0 or idx == len(image_ids) - 1:
            print(f"  [{idx + 1:3d}/{len(image_ids)}] {stem}: {t1 - t0:.3f}s")

    peak_mem_mib = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else None)

    # ------------------------------------------------------------------
    # Save object features
    # ------------------------------------------------------------------
    print("\n--- Saving object features + metadata ---")
    save_dict = {"object_keys": np.array(object_keys)}
    for l_disp in LAYERS_DISPLAY:
        save_dict[f"raw_L{l_disp}"] = np.stack(raw_feats[l_disp]).astype(np.float32)
        save_dict[f"proj_L{l_disp}"] = np.stack(proj_feats[l_disp]).astype(np.float32)
    np.savez_compressed(OBJECT_FEATURES_NPZ, **save_dict)
    print(f"  Saved: {OBJECT_FEATURES_NPZ}  ({len(object_keys)} objects)")

    with open(OBJECTS_METADATA_CSV, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["image_id", "object_id", "mask_pixel_count", "patch_weight_sum",
                      "n_positive_attrs", "positive_attributes", "negative_attributes"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(object_metadata_rows)
    print(f"  Saved: {OBJECTS_METADATA_CSV}  ({len(object_metadata_rows)} rows)")

    with open(SPATIAL_SCORES_CSV, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["image_id", "attribute", "layer", "inside_mean", "outside_mean", "diff", "auc",
                      "shift_dy", "shift_dx", "shift_inside_mean", "shift_outside_mean", "shift_diff", "shift_auc"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in spatial_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {SPATIAL_SCORES_CSV}  ({len(spatial_rows)} rows)")

    # ------------------------------------------------------------------
    # Config + run log
    # ------------------------------------------------------------------
    config = {
        "model": "OpenAI CLIP ViT-B/16 (clip package)",
        "clip_module_path": clip.__file__,
        "visual_width": 768, "shared_embed_dim": 512, "depth": 12, "heads": 12, "patch_size": 16,
        "layers_display": LAYERS_DISPLAY, "layers_zero_based": LAYERS_ZERO_BASED,
        "seed": SEED,
        "n_images": len(image_ids), "image_ids": image_ids,
        "image_selection_method": "reused verbatim from outputs/osie_attribute_grounding_pilot/pilot_image_list.csv "
                                    "(seed=42, 50 images, sorted attrs.mat/stimuli common set)",
        "attribute_mapping_path": ATTRIBUTE_MAPPING_JSON, "prompt_bank_path": PROMPT_BANK_JSON,
        "grid_hw": list(GRID_HW_EXPECTED), "img_w": IMG_W, "img_h": IMG_H,
        "min_patch_weight_sum": MIN_WEIGHT_SUM,
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Saved: {CONFIG_JSON}")

    run_log = {
        "n_images": len(image_ids), "n_total_objects": n_total_objects,
        "n_objects_with_features": len(object_keys),
        "n_objects_skipped_zero_weight": n_objects_skipped_zero_weight,
        "n_attr_presence_skipped_zero_weight": n_attr_presence_skipped_zero_weight,
        "n_spatial_rows": len(spatial_rows),
        "n_nan_inf_detected": n_nan_inf_detected,
        "peak_vram_mib": peak_mem_mib, "device": str(device),
        "wall_clock_total_s": time.time() - t_start,
        "per_image_timings": timings,
    }
    with open(EXTRACTION_LOG_JSON, "w", encoding="utf-8") as fh:
        json.dump(run_log, fh, indent=2)
    print(f"  Saved: {EXTRACTION_LOG_JSON}")
    print(f"\n  Objects: {len(object_keys)}/{n_total_objects} kept "
          f"({n_objects_skipped_zero_weight} skipped, zero patch weight)")
    print(f"  Peak VRAM: {peak_mem_mib:.1f} MiB" if peak_mem_mib else "  Peak VRAM: n/a")
    print(f"  Wall clock: {time.time() - t_start:.1f}s")

    print("\n" + "=" * 70)
    print("  Feature extraction: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
