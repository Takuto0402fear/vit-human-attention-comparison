"""
CPU-only analysis for the OSIE text-alignment pilot: consumes the cached
object features (scripts/extract_osie_text_alignment_features.py) and text
embeddings, computes:

  - region_text_scores.csv: per (object, positive_attribute, layer,
    aggregation) Label Alignment Margin. aggregation in
    {equal_ensemble, raw_label, prompt_0..prompt_15}.
  - summary_by_attribute_layer.csv: aggregated over objects, for
    aggregation in {equal_ensemble, raw_label} (per-prompt aggregation is
    summarized separately in prompt_variant_stability.csv instead, to
    avoid duplicating the same per-prompt numbers in two files).
  - prompt_variant_stability.csv: for each (attribute, layer), the
    distribution (median/mean/std/min/max, and how many of the 16 are
    sign-positive) of the mean alignment margin computed with EACH
    individual prompt (not the ensemble) used alone as the text vector --
    the "per-prompt-then-median" robustness statistic from the task spec.
  - prompt_k_stability.csv: for K in {1,2,4,8,16}, up to 100 seed=42
    equal-ensemble subsets of the 16 templates (exhaustive if C(16,K)<=100),
    reused identically across L4/L8/L12: mean/std/95%CI of the resulting
    mean alignment margin, the sign-flip rate vs. the K=16 (full-ensemble)
    reference, and the rate at which the layer with the highest margin
    (the "top layer") differs from the K=16 reference's top layer.
    Negative-attribute comparisons always use the full K=16 equal ensemble
    (only the POSITIVE attribute's own text vector is varied by K/subset;
    see README.md).

Does not run CLIP (no GPU needed here). Reads only
outputs/osie_text_alignment_pilot/{object_features.npz,objects_metadata.csv,
text_embeddings.npz,prompt_bank.json,attribute_mapping.json}. Writes only
the 4 CSVs above under the same directory.

Usage (PowerShell):
    python scripts\\analyze_osie_text_alignment.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os

import numpy as np

from lib.osie_text_alignment import alignment_margin, build_equal_ensemble, enumerate_or_sample_subsets

# ======================= CONFIG =======================
SEED = 42
LAYERS_DISPLAY = [4, 8, 12]
K_VALUES = [1, 2, 4, 8, 16]
MAX_SUBSETS = 100

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot"
OBJECT_FEATURES_NPZ = os.path.join(OUT_DIR, "object_features.npz")
OBJECTS_METADATA_CSV = os.path.join(OUT_DIR, "objects_metadata.csv")
TEXT_EMBEDDINGS_NPZ = os.path.join(OUT_DIR, "text_embeddings.npz")

REGION_TEXT_SCORES_CSV = os.path.join(OUT_DIR, "region_text_scores.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary_by_attribute_layer.csv")
PROMPT_VARIANT_CSV = os.path.join(OUT_DIR, "prompt_variant_stability.csv")
PROMPT_K_CSV = os.path.join(OUT_DIR, "prompt_k_stability.csv")
# ======================================================


def load_everything():
    obj_npz = np.load(OBJECT_FEATURES_NPZ, allow_pickle=False)
    object_keys = list(obj_npz["object_keys"])
    proj_by_layer = {l: obj_npz[f"proj_L{l}"].astype(np.float64) for l in LAYERS_DISPLAY}

    with open(OBJECTS_METADATA_CSV, encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    if len(meta_rows) != len(object_keys):
        raise RuntimeError(
            f"STOP: objects_metadata.csv has {len(meta_rows)} rows, "
            f"object_features.npz has {len(object_keys)} objects")
    meta_by_key = {}
    for row, key in zip(meta_rows, object_keys):
        expected_key = f"{row['image_id']}_{row['object_id']}"
        if expected_key != key:
            raise RuntimeError(f"STOP: object key mismatch: metadata={expected_key} features={key}")
        meta_by_key[key] = row

    text_npz = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=False)
    attribute_order = [a for a in text_npz["attribute_order"]]
    prompt_vectors = {a: text_npz["prompt_vectors"][i].astype(np.float64) for i, a in enumerate(attribute_order)}
    raw_label_vectors = {a: text_npz["raw_label_vectors"][i].astype(np.float64) for i, a in enumerate(attribute_order)}
    equal_ensemble_vectors = {a: text_npz["equal_ensemble_vectors"][i].astype(np.float64) for i, a in enumerate(attribute_order)}

    return object_keys, proj_by_layer, meta_by_key, attribute_order, prompt_vectors, raw_label_vectors, equal_ensemble_vectors


def positive_negative_lists(meta_by_key, key):
    row = meta_by_key[key]
    pos = row["positive_attributes"].split(";") if row["positive_attributes"] else []
    neg = row["negative_attributes"].split(";") if row["negative_attributes"] else []
    return pos, neg


def compute_region_text_scores(object_keys, proj_by_layer, meta_by_key, attribute_order,
                                prompt_vectors, raw_label_vectors, equal_ensemble_vectors):
    rows = []
    for obj_idx, key in enumerate(object_keys):
        image_id, object_id = key.rsplit("_", 1)
        pos_attrs, neg_attrs = positive_negative_lists(meta_by_key, key)
        if not neg_attrs:
            continue  # object positive on every attribute -- no valid negative set
        for l_disp in LAYERS_DISPLAY:
            vec = proj_by_layer[l_disp][obj_idx]
            for pos_attr in pos_attrs:
                # equal ensemble
                pos_sim = float(vec @ equal_ensemble_vectors[pos_attr])
                neg_sims = [float(vec @ equal_ensemble_vectors[n]) for n in neg_attrs]
                margin = alignment_margin(pos_sim, neg_sims)
                rows.append({"image_id": image_id, "object_id": object_id, "layer": l_disp,
                             "positive_attribute": pos_attr, "aggregation": "equal_ensemble",
                             "prompt_index": "", "positive_similarity": pos_sim,
                             "negative_similarity_mean": float(np.mean(neg_sims)), "alignment_margin": margin})
                # raw label baseline
                pos_sim_r = float(vec @ raw_label_vectors[pos_attr])
                neg_sims_r = [float(vec @ raw_label_vectors[n]) for n in neg_attrs]
                margin_r = alignment_margin(pos_sim_r, neg_sims_r)
                rows.append({"image_id": image_id, "object_id": object_id, "layer": l_disp,
                             "positive_attribute": pos_attr, "aggregation": "raw_label",
                             "prompt_index": "", "positive_similarity": pos_sim_r,
                             "negative_similarity_mean": float(np.mean(neg_sims_r)), "alignment_margin": margin_r})
                # each individual prompt
                for p_idx in range(16):
                    pos_sim_p = float(vec @ prompt_vectors[pos_attr][p_idx])
                    neg_sims_p = [float(vec @ prompt_vectors[n][p_idx]) for n in neg_attrs]
                    margin_p = alignment_margin(pos_sim_p, neg_sims_p)
                    rows.append({"image_id": image_id, "object_id": object_id, "layer": l_disp,
                                 "positive_attribute": pos_attr, "aggregation": f"prompt_{p_idx}",
                                 "prompt_index": p_idx, "positive_similarity": pos_sim_p,
                                 "negative_similarity_mean": float(np.mean(neg_sims_p)), "alignment_margin": margin_p})
    return rows


def write_region_text_scores(rows):
    fieldnames = ["image_id", "object_id", "layer", "positive_attribute", "aggregation",
                  "prompt_index", "positive_similarity", "negative_similarity_mean", "alignment_margin"]
    with open(REGION_TEXT_SCORES_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  Saved: {REGION_TEXT_SCORES_CSV}  ({len(rows)} rows)")


def write_summary(rows):
    by_key = {}
    for r in rows:
        if r["aggregation"] not in ("equal_ensemble", "raw_label"):
            continue
        k = (r["positive_attribute"], r["layer"], r["aggregation"])
        by_key.setdefault(k, []).append(r["alignment_margin"])

    out_rows = []
    for (attr, layer, agg), margins in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        arr = np.array(margins, dtype=float)
        out_rows.append({
            "attribute": attr, "layer": layer, "aggregation": agg,
            "n_objects": len(arr), "mean_margin": float(arr.mean()),
            "median_margin": float(np.median(arr)), "std_margin": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n_positive_margin": int((arr > 0).sum()), "frac_positive_margin": float((arr > 0).mean()),
        })
    fieldnames = ["attribute", "layer", "aggregation", "n_objects", "mean_margin", "median_margin",
                  "std_margin", "n_positive_margin", "frac_positive_margin"]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {SUMMARY_CSV}  ({len(out_rows)} rows)")
    return out_rows


def write_prompt_variant_stability(region_rows):
    by_key = {}
    for r in region_rows:
        if not str(r["aggregation"]).startswith("prompt_"):
            continue
        k = (r["positive_attribute"], r["layer"], r["prompt_index"])
        by_key.setdefault(k, []).append(r["alignment_margin"])

    per_template_mean = {}  # (attr, layer, prompt_index) -> mean margin across objects
    for k, vals in by_key.items():
        per_template_mean[k] = float(np.mean(vals))

    attrs = sorted({k[0] for k in per_template_mean})
    layers = sorted({k[1] for k in per_template_mean})
    out_rows = []
    for attr in attrs:
        for layer in layers:
            vals = np.array([per_template_mean[(attr, layer, p)] for p in range(16)
                              if (attr, layer, p) in per_template_mean])
            if len(vals) == 0:
                continue
            out_rows.append({
                "attribute": attr, "layer": layer, "n_templates": len(vals),
                "median_margin_across_templates": float(np.median(vals)),
                "mean_margin_across_templates": float(vals.mean()),
                "std_margin_across_templates": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min_margin_across_templates": float(vals.min()),
                "max_margin_across_templates": float(vals.max()),
                "n_templates_with_positive_margin": int((vals > 0).sum()),
            })
    fieldnames = ["attribute", "layer", "n_templates", "median_margin_across_templates",
                  "mean_margin_across_templates", "std_margin_across_templates",
                  "min_margin_across_templates", "max_margin_across_templates",
                  "n_templates_with_positive_margin"]
    with open(PROMPT_VARIANT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {PROMPT_VARIANT_CSV}  ({len(out_rows)} rows)")


def write_prompt_k_stability(object_keys, proj_by_layer, meta_by_key, attribute_order,
                              prompt_vectors, equal_ensemble_vectors):
    out_rows = []
    for attr in attribute_order:
        # positive objects for this attribute, and each one's fixed negative set
        pos_entries = []  # (obj_idx, neg_attrs)
        for obj_idx, key in enumerate(object_keys):
            pos_attrs, neg_attrs = positive_negative_lists(meta_by_key, key)
            if attr in pos_attrs and neg_attrs:
                pos_entries.append((obj_idx, neg_attrs))
        if not pos_entries:
            continue

        def mean_margin_for_vector(text_vec, layer):
            vals = []
            vec_layer = proj_by_layer[layer]
            for obj_idx, neg_attrs in pos_entries:
                v = vec_layer[obj_idx]
                pos_sim = float(v @ text_vec)
                neg_sims = [float(v @ equal_ensemble_vectors[n]) for n in neg_attrs]
                vals.append(alignment_margin(pos_sim, neg_sims))
            return float(np.mean(vals))

        reference_margins = {l: mean_margin_for_vector(equal_ensemble_vectors[attr], l) for l in LAYERS_DISPLAY}
        reference_top_layer = max(reference_margins, key=reference_margins.get)
        reference_signs = {l: (1 if reference_margins[l] > 0 else (-1 if reference_margins[l] < 0 else 0))
                            for l in LAYERS_DISPLAY}

        for K in K_VALUES:
            subsets = enumerate_or_sample_subsets(16, K, seed=SEED, max_subsets=MAX_SUBSETS)
            per_layer_vals = {l: [] for l in LAYERS_DISPLAY}
            top_layer_matches = []
            for subset in subsets:
                ens_vec = build_equal_ensemble(prompt_vectors[attr][list(subset)])
                margins = {l: mean_margin_for_vector(ens_vec, l) for l in LAYERS_DISPLAY}
                for l in LAYERS_DISPLAY:
                    per_layer_vals[l].append(margins[l])
                top_layer_matches.append(max(margins, key=margins.get) == reference_top_layer)

            top_layer_match_rate = float(np.mean(top_layer_matches))
            for l in LAYERS_DISPLAY:
                vals = np.array(per_layer_vals[l])
                signs = np.sign(vals)
                signs[signs == 0] = reference_signs[l] if reference_signs[l] != 0 else 1
                sign_flip_rate = float(np.mean(signs != (reference_signs[l] if reference_signs[l] != 0 else 1)))
                ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
                out_rows.append({
                    "attribute": attr, "layer": l, "K": K, "n_subsets_evaluated": len(subsets),
                    "mean_margin": float(vals.mean()), "std_margin": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
                    "reference_margin_K16": reference_margins[l],
                    "sign_flip_rate_vs_K16": sign_flip_rate,
                    "top_layer_match_rate_vs_K16": top_layer_match_rate,
                    "reference_top_layer_K16": reference_top_layer,
                })

    fieldnames = ["attribute", "layer", "K", "n_subsets_evaluated", "mean_margin", "std_margin",
                  "ci95_lo", "ci95_hi", "reference_margin_K16", "sign_flip_rate_vs_K16",
                  "top_layer_match_rate_vs_K16", "reference_top_layer_K16"]
    with open(PROMPT_K_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {PROMPT_K_CSV}  ({len(out_rows)} rows)")


def main():
    print("=" * 65)
    print("  OSIE Text-Alignment Pilot: analysis (region scores + stability)")
    print("=" * 65)

    for path in (OBJECT_FEATURES_NPZ, OBJECTS_METADATA_CSV, TEXT_EMBEDDINGS_NPZ):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found -- run the extraction script first")

    (object_keys, proj_by_layer, meta_by_key, attribute_order,
     prompt_vectors, raw_label_vectors, equal_ensemble_vectors) = load_everything()
    print(f"  Loaded {len(object_keys)} objects, {len(attribute_order)} attributes")

    print("\n--- region_text_scores.csv ---")
    region_rows = compute_region_text_scores(
        object_keys, proj_by_layer, meta_by_key, attribute_order,
        prompt_vectors, raw_label_vectors, equal_ensemble_vectors)
    write_region_text_scores(region_rows)

    print("\n--- summary_by_attribute_layer.csv ---")
    summary_rows = write_summary(region_rows)

    print("\n--- prompt_variant_stability.csv ---")
    write_prompt_variant_stability(region_rows)

    print("\n--- prompt_k_stability.csv ---")
    write_prompt_k_stability(object_keys, proj_by_layer, meta_by_key, attribute_order,
                              prompt_vectors, equal_ensemble_vectors)

    print("\n=== equal_ensemble margin: mean (n_objects), by attribute x layer ===")
    header = f"{'attribute':15s}" + "".join(f"  L{l:<9d}" for l in LAYERS_DISPLAY)
    print(header)
    by_attr = {}
    for r in summary_rows:
        if r["aggregation"] == "equal_ensemble":
            by_attr.setdefault(r["attribute"], {})[r["layer"]] = r
    for attr in sorted(by_attr):
        line = f"{attr:15s}"
        for l in LAYERS_DISPLAY:
            row = by_attr[attr].get(l)
            if row:
                line += f"  {row['mean_margin']:+6.3f}({row['n_objects']:2d})"
            else:
                line += "  " + " " * 9
        print(line)

    print("\n" + "=" * 65)
    print("  Analysis: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
