"""
Phase 1 (crop pilot) extraction: does OSIE-attribute <-> CLIP-text
alignment recover once we bypass patch tokens and intermediate-layer
projection entirely, feeding OBJECT CROPS through the OFFICIAL
`model.encode_image` (the exact production CLIP usage) instead?

For each of the 394 objects (same 50 pilot images, same attrs.mat labels,
same prompt_bank.json/text_embeddings.npz as
outputs/osie_text_alignment_pilot/ -- reused unmodified, no new prompts,
no new text encoding), generates 3 crop conditions from its mask's
bounding box, plus one whole-image "global_image" condition (a systematic,
reproducible re-check of the one-off whole-image CLS finding already
reported in outputs/osie_text_alignment_pilot/report.md section 9):

  global_image     : the ORIGINAL image (no crop), through the official
                      preprocess (Resize -> CenterCrop -> Normalize).
  tight_crop        : the object's own bounding box, padded to a square
                      with CLIP's own mean color (never plain black), then
                      through the same official preprocess.
  context20_crop    : bounding box expanded 20% each direction (clipped to
                      image bounds), same square-padding + preprocess.
  masked_context20  : same expanded region as context20_crop, but every
                      pixel OUTSIDE the object's own mask is replaced with
                      CLIP's mean color before padding + preprocess.

No intermediate-layer hook, no patch token, no manual visual_projection --
only `clip_extractor.load_clip_vit_b16()`'s model and its own
`model.encode_image`, exactly as CLIP is meant to be used.

Requires (gates): outputs/osie_text_alignment_pilot/{text_embeddings.npz,
attribute_mapping.json, objects_metadata.csv, pilot_manifest.csv} already
generated (read-only). Does not modify that directory or any of its files.

Writes only under outputs/osie_text_alignment_crop_pilot/.

Usage (PowerShell):
    python scripts\\extract_osie_text_alignment_crop_pilot.py
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
from sklearn.metrics import label_ranking_average_precision_score

from clip_extractor import load_clip_vit_b16
from lib.osie_text_alignment import (
    CLIP_MEAN_RGB_255, alignment_margin, bbox_from_mask, expand_bbox,
    pad_to_square, split_positive_negative_attributes,
)

# ======================= CONFIG =======================
SEED = 42
CONTEXT_FRAC = 0.20
MIN_BBOX_PX = 4  # bbox shorter side below this -> skip as "bbox_too_small"

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")
IMG_W, IMG_H = 800, 600

PRIOR_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot"
PRIOR_MANIFEST = os.path.join(PRIOR_DIR, "pilot_manifest.csv")
PRIOR_TEXT_EMBEDDINGS = os.path.join(PRIOR_DIR, "text_embeddings.npz")
PRIOR_ATTRIBUTE_MAPPING = os.path.join(PRIOR_DIR, "attribute_mapping.json")

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_crop_pilot"
CONFIG_JSON = os.path.join(OUT_DIR, "config.json")
CROP_MANIFEST_CSV = os.path.join(OUT_DIR, "crop_manifest.csv")
CROP_OBJECT_SCORES_CSV = os.path.join(OUT_DIR, "crop_object_scores.csv")
SANITY_JSON = os.path.join(OUT_DIR, "crop_sanity_checks.json")
FIG_DIR = os.path.join(OUT_DIR, "figures")

CROP_CONDITIONS = ["global_image", "tight_crop", "context20_crop", "masked_context20"]
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


def make_masked_region(img, mask, y0, y1, x0, x1):
    """Region [y0:y1+1, x0:x1+1] of img, with pixels outside `mask`
    (evaluated over the SAME region) replaced by CLIP's mean color."""
    region = img[y0:y1 + 1, x0:x1 + 1].copy()
    region_mask = mask[y0:y1 + 1, x0:x1 + 1]
    region[~region_mask] = CLIP_MEAN_RGB_255
    return region


def encode_official(model, preprocess, device, np_image_uint8):
    pil = Image.fromarray(np_image_uint8, mode="RGB")
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = model.encode_image(tensor).float()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.detach().cpu().numpy()[0]


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    set_all_seeds(SEED)

    print("=" * 70)
    print("  OSIE Text-Alignment CROP Pilot (Phase 1): official encode_image")
    print("=" * 70)

    for path in (PRIOR_MANIFEST, PRIOR_TEXT_EMBEDDINGS, PRIOR_ATTRIBUTE_MAPPING):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found -- run the base text-alignment pilot first")

    with open(PRIOR_MANIFEST, encoding="utf-8") as f:
        image_ids = sorted(r["image_id"] for r in csv.DictReader(f))
    if len(image_ids) != 50:
        raise RuntimeError(f"STOP: expected 50 images, got {len(image_ids)}")
    print(f"  Reusing {len(image_ids)} images from {PRIOR_MANIFEST}")

    with open(PRIOR_ATTRIBUTE_MAPPING, encoding="utf-8") as f:
        attribute_mapping = json.load(f)
    mat_attr_names = attribute_mapping["attrs_mat_order_as_stored"]
    mat_to_task = attribute_mapping["mat_name_to_task_name"]

    text_npz = np.load(PRIOR_TEXT_EMBEDDINGS, allow_pickle=False)
    attribute_order = [a for a in text_npz["attribute_order"]]
    equal_ensemble = {a: text_npz["equal_ensemble_vectors"][i].astype(np.float64)
                       for i, a in enumerate(attribute_order)}
    raw_label = {a: text_npz["raw_label_vectors"][i].astype(np.float64)
                 for i, a in enumerate(attribute_order)}
    prompt_vecs = {a: text_npz["prompt_vectors"][i].astype(np.float64)
                   for i, a in enumerate(attribute_order)}
    print(f"  Reused text embeddings for {len(attribute_order)} attributes "
          f"(NO new prompts, NO new text encoding): {PRIOR_TEXT_EMBEDDINGS}")
    equal_ensemble_matrix = np.stack([equal_ensemble[a] for a in attribute_order])  # (12,512)

    print("\n--- Loading CLIP ViT-B/16 (official preprocess + encode_image only) ---")
    model, preprocess = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}  dtype: {model.dtype}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    f = h5py.File(ATTRS_PATH, "r")
    loaded_names, attrs_index = load_attrs_index(f)
    if loaded_names != mat_attr_names:
        raise RuntimeError("STOP: attrs.mat order changed since mapping was built")

    manifest_rows = []
    score_rows = []
    n_objects_total = 0
    n_objects_skipped = 0
    n_nan_inf = 0

    for img_idx, stem in enumerate(image_ids):
        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        img = np.array(Image.open(img_path).convert("RGB"))
        if img.shape[:2] != (IMG_H, IMG_W):
            raise RuntimeError(f"STOP: {stem} unexpected shape {img.shape}")

        # global_image: computed ONCE per image, shared across its objects.
        global_emb = encode_official(model, preprocess, device, img)
        if np.isnan(global_emb).any() or np.isinf(global_emb).any():
            n_nan_inf += 1
            raise RuntimeError(f"STOP: NaN/Inf in global_image embedding for {stem}")

        objs = load_objects_for_image(f, attrs_index[stem], mat_attr_names, (IMG_H, IMG_W))
        for obj in objs:
            n_objects_total += 1
            mask = obj["mask"]
            pos_task, neg_task = split_positive_negative_attributes(obj["features"], mat_attr_names, mat_to_task)
            mask_area = int(mask.sum())
            bbox = bbox_from_mask(mask)

            manifest_row = {"image_id": stem, "object_id": obj["object_id"], "mask_area": mask_area,
                             "n_positive_attrs": len(pos_task)}

            embeddings = {"global_image": global_emb}
            valid_flags = {"global_image": True}
            skip_reasons = {"global_image": ""}
            bbox_str = ""
            context_bbox_str = ""

            if bbox is None:
                n_objects_skipped += 1
                for cond in ("tight_crop", "context20_crop", "masked_context20"):
                    valid_flags[cond] = False
                    skip_reasons[cond] = "empty_mask"
            else:
                y0, y1, x0, x1 = bbox
                bbox_str = f"{y0}:{y1}:{x0}:{x1}"
                h_box, w_box = y1 - y0 + 1, x1 - x0 + 1
                if min(h_box, w_box) < MIN_BBOX_PX:
                    n_objects_skipped += 1
                    for cond in ("tight_crop", "context20_crop", "masked_context20"):
                        valid_flags[cond] = False
                        skip_reasons[cond] = "bbox_too_small"
                else:
                    # tight_crop
                    tight = img[y0:y1 + 1, x0:x1 + 1]
                    embeddings["tight_crop"] = encode_official(model, preprocess, device,
                                                                pad_to_square(tight))
                    valid_flags["tight_crop"] = True
                    skip_reasons["tight_crop"] = ""

                    # context20_crop
                    cy0, cy1, cx0, cx1 = expand_bbox(bbox, CONTEXT_FRAC, IMG_H, IMG_W)
                    context_bbox_str = f"{cy0}:{cy1}:{cx0}:{cx1}"
                    context_region = img[cy0:cy1 + 1, cx0:cx1 + 1]
                    embeddings["context20_crop"] = encode_official(model, preprocess, device,
                                                                    pad_to_square(context_region))
                    valid_flags["context20_crop"] = True
                    skip_reasons["context20_crop"] = ""

                    # masked_context20
                    masked_region = make_masked_region(img, mask, cy0, cy1, cx0, cx1)
                    embeddings["masked_context20"] = encode_official(model, preprocess, device,
                                                                      pad_to_square(masked_region))
                    valid_flags["masked_context20"] = True
                    skip_reasons["masked_context20"] = ""

            manifest_row.update({"bbox": bbox_str, "context_bbox": context_bbox_str})
            for cond in CROP_CONDITIONS:
                manifest_row[f"valid_{cond}"] = valid_flags.get(cond, False)
                manifest_row[f"skip_reason_{cond}"] = skip_reasons.get(cond, "no_mask")
            manifest_rows.append(manifest_row)

            if not neg_task:
                continue  # object positive on every attribute -- no valid negative set

            for cond in CROP_CONDITIONS:
                if not valid_flags.get(cond, False):
                    score_rows.append({
                        "image_id": stem, "object_id": obj["object_id"], "crop_condition": cond,
                        "positive_attribute": "", "aggregation": "", "prompt_index": "",
                        "positive_similarity": "", "negative_similarity_mean": "", "alignment_margin": "",
                        "positive_rank": "", "label_ranking_average_precision": "",
                        "bbox": bbox_str, "context_bbox": context_bbox_str, "mask_area": mask_area,
                        "valid": False, "skip_reason": skip_reasons.get(cond, "no_mask"),
                    })
                    continue

                emb = embeddings[cond]
                if np.isnan(emb).any() or np.isinf(emb).any():
                    n_nan_inf += 1
                    raise RuntimeError(f"STOP: NaN/Inf in {stem}/{obj['object_id']}/{cond}")

                full_sim_equal = equal_ensemble_matrix @ emb  # (12,) in attribute_order
                y_true = np.array([1 if a in pos_task else 0 for a in attribute_order]).reshape(1, -1)
                lrap = float(label_ranking_average_precision_score(y_true, full_sim_equal.reshape(1, -1)))
                order_desc = np.argsort(-full_sim_equal)
                rank_of = {attribute_order[idx]: int(r + 1) for r, idx in enumerate(order_desc)}

                for pos_attr in pos_task:
                    # equal ensemble
                    pos_sim = float(emb @ equal_ensemble[pos_attr])
                    neg_sims = [float(emb @ equal_ensemble[n]) for n in neg_task]
                    margin = alignment_margin(pos_sim, neg_sims)
                    score_rows.append({
                        "image_id": stem, "object_id": obj["object_id"], "crop_condition": cond,
                        "positive_attribute": pos_attr, "aggregation": "equal_ensemble", "prompt_index": "",
                        "positive_similarity": pos_sim, "negative_similarity_mean": float(np.mean(neg_sims)),
                        "alignment_margin": margin, "positive_rank": rank_of[pos_attr],
                        "label_ranking_average_precision": lrap,
                        "bbox": bbox_str, "context_bbox": context_bbox_str, "mask_area": mask_area,
                        "valid": True, "skip_reason": "",
                    })
                    # raw label baseline
                    pos_sim_r = float(emb @ raw_label[pos_attr])
                    neg_sims_r = [float(emb @ raw_label[n]) for n in neg_task]
                    margin_r = alignment_margin(pos_sim_r, neg_sims_r)
                    score_rows.append({
                        "image_id": stem, "object_id": obj["object_id"], "crop_condition": cond,
                        "positive_attribute": pos_attr, "aggregation": "raw_label", "prompt_index": "",
                        "positive_similarity": pos_sim_r, "negative_similarity_mean": float(np.mean(neg_sims_r)),
                        "alignment_margin": margin_r, "positive_rank": "", "label_ranking_average_precision": "",
                        "bbox": bbox_str, "context_bbox": context_bbox_str, "mask_area": mask_area,
                        "valid": True, "skip_reason": "",
                    })
                    # each individual prompt
                    for p_idx in range(16):
                        pos_sim_p = float(emb @ prompt_vecs[pos_attr][p_idx])
                        neg_sims_p = [float(emb @ prompt_vecs[n][p_idx]) for n in neg_task]
                        margin_p = alignment_margin(pos_sim_p, neg_sims_p)
                        score_rows.append({
                            "image_id": stem, "object_id": obj["object_id"], "crop_condition": cond,
                            "positive_attribute": pos_attr, "aggregation": f"prompt_{p_idx}", "prompt_index": p_idx,
                            "positive_similarity": pos_sim_p, "negative_similarity_mean": float(np.mean(neg_sims_p)),
                            "alignment_margin": margin_p, "positive_rank": "", "label_ranking_average_precision": "",
                            "bbox": bbox_str, "context_bbox": context_bbox_str, "mask_area": mask_area,
                            "valid": True, "skip_reason": "",
                        })

        if img_idx == 0 or (img_idx + 1) % 10 == 0 or img_idx == len(image_ids) - 1:
            print(f"  [{img_idx + 1:2d}/{len(image_ids)}] {stem}: OK")

    peak_mem_mib = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else None)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    manifest_fields = (["image_id", "object_id", "mask_area", "n_positive_attrs", "bbox", "context_bbox"]
                        + [f"valid_{c}" for c in CROP_CONDITIONS] + [f"skip_reason_{c}" for c in CROP_CONDITIONS])
    with open(CROP_MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\n  Saved: {CROP_MANIFEST_CSV}  ({len(manifest_rows)} objects)")

    score_fields = ["image_id", "object_id", "crop_condition", "positive_attribute", "aggregation",
                    "prompt_index", "positive_similarity", "negative_similarity_mean", "alignment_margin",
                    "positive_rank", "label_ranking_average_precision", "bbox", "context_bbox",
                    "mask_area", "valid", "skip_reason"]
    with open(CROP_OBJECT_SCORES_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=score_fields)
        w.writeheader()
        for r in score_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  Saved: {CROP_OBJECT_SCORES_CSV}  ({len(score_rows)} rows)")

    n_valid_per_cond = {c: sum(1 for r in manifest_rows if r.get(f"valid_{c}")) for c in CROP_CONDITIONS}
    skip_reason_counts = {}
    for c in CROP_CONDITIONS:
        for r in manifest_rows:
            if not r.get(f"valid_{c}"):
                reason = r.get(f"skip_reason_{c}", "")
                skip_reason_counts[(c, reason)] = skip_reason_counts.get((c, reason), 0) + 1

    sanity = {
        "n_objects_total": n_objects_total,
        "n_valid_per_condition": n_valid_per_cond,
        "skip_reason_counts": {f"{c}/{r}": n for (c, r), n in skip_reason_counts.items()},
        "n_nan_inf_detected": n_nan_inf,
        "embedding_norms_are_unit": True,  # enforced by construction (L2-normalize after encode_image)
        "text_embeddings_reused_unmodified_from": PRIOR_TEXT_EMBEDDINGS,
        "prompt_bank_reused_unmodified": True,
        "clip_mean_fill_color_rgb_255": list(CLIP_MEAN_RGB_255),
        "min_bbox_px": MIN_BBOX_PX,
        "context_frac": CONTEXT_FRAC,
    }
    with open(SANITY_JSON, "w", encoding="utf-8") as fh:
        json.dump(sanity, fh, indent=2)
    print(f"  Saved: {SANITY_JSON}")
    print(f"  Valid objects per condition: {n_valid_per_cond}")

    config = {
        "model": "OpenAI CLIP ViT-B/16 (clip package), official model.encode_image only",
        "seed": SEED, "n_images": len(image_ids), "image_ids": image_ids,
        "crop_conditions": CROP_CONDITIONS, "context_frac": CONTEXT_FRAC, "min_bbox_px": MIN_BBOX_PX,
        "clip_mean_fill_color_rgb_255": list(CLIP_MEAN_RGB_255),
        "text_source": "outputs/osie_text_alignment_pilot/text_embeddings.npz (reused verbatim, no new prompts)",
        "peak_vram_mib": peak_mem_mib, "device": str(device),
        "wall_clock_total_s": time.time() - t_start,
        "n_objects_total": n_objects_total, "n_objects_skipped_any_crop": n_objects_skipped,
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Saved: {CONFIG_JSON}")
    print(f"\n  Peak VRAM: {peak_mem_mib:.1f} MiB" if peak_mem_mib else "  Peak VRAM: n/a")
    print(f"  Wall clock: {time.time() - t_start:.1f}s")

    print("\n" + "=" * 70)
    print("  Crop pilot extraction: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
