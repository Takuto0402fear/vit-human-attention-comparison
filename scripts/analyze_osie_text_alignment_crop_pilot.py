"""
Phase 1 (crop pilot) analysis: attribute-level and group-level summaries,
plus paired crop-condition comparisons, all with IMAGE-level (not
object-level) bootstrap 95% CI to respect within-image correlation between
objects of the same image.

Reads only outputs/osie_text_alignment_crop_pilot/crop_object_scores.csv
(equal_ensemble aggregation for the primary summaries; prompt_0..15 for the
per-prompt-median robustness check, matching the base pilot's convention).
Writes only crop_attribute_summary.csv, crop_group_summary.csv, and
crop_condition_comparison.csv under the same directory.

Usage (PowerShell):
    python scripts\\analyze_osie_text_alignment_crop_pilot.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import os

import numpy as np

# ======================= CONFIG =======================
SEED = 42
N_BOOT = 10000
ALPHA = 0.05

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_crop_pilot"
SCORES_CSV = os.path.join(OUT_DIR, "crop_object_scores.csv")
ATTRIBUTE_SUMMARY_CSV = os.path.join(OUT_DIR, "crop_attribute_summary.csv")
GROUP_SUMMARY_CSV = os.path.join(OUT_DIR, "crop_group_summary.csv")
CONDITION_COMPARISON_CSV = os.path.join(OUT_DIR, "crop_condition_comparison.csv")

CROP_CONDITIONS = ["global_image", "tight_crop", "context20_crop", "masked_context20"]

ATTRIBUTE_GROUPS = {
    "direct_visual": ["Face", "Text"],
    "emotion": ["Emotion"],
    "relational": ["Gazed", "Touched"],
    "sensory_functional": ["Motion", "Sound", "Smell", "Taste", "Touch", "Operability", "Watchability"],
}
# ======================================================


def load_rows():
    with open(SCORES_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_level_bootstrap_ci(values, image_ids, seed=SEED, n_boot=N_BOOT, alpha=ALPHA):
    """
    values, image_ids : parallel 1D arrays (one alignment_margin value and
    its source image_id per row). Resamples IMAGE ids (not objects/rows),
    then averages every row whose image_id was drawn, so within-image
    correlation across objects is respected.
    """
    values = np.asarray(values, dtype=float)
    unique_images = sorted(set(image_ids))
    n_img = len(unique_images)
    if n_img < 2:
        return float("nan"), float("nan")
    rows_by_image = {img: np.where(np.array(image_ids) == img)[0] for img in unique_images}
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled_images = rng.choice(unique_images, size=n_img, replace=True)
        idx = np.concatenate([rows_by_image[img] for img in sampled_images])
        boot_means[b] = values[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def write_attribute_summary(rows):
    eq_rows = [r for r in rows if r["aggregation"] == "equal_ensemble" and r["valid"] == "True"]
    prompt_rows = [r for r in rows if r["aggregation"].startswith("prompt_") and r["valid"] == "True"]

    out_rows = []
    for cond in CROP_CONDITIONS:
        attrs = sorted({r["positive_attribute"] for r in eq_rows if r["crop_condition"] == cond})
        for attr in attrs:
            subset = [r for r in eq_rows if r["crop_condition"] == cond and r["positive_attribute"] == attr]
            margins = np.array([float(r["alignment_margin"]) for r in subset])
            image_ids = [r["image_id"] for r in subset]
            n_images = len(set(image_ids))
            ci_lo, ci_hi = image_level_bootstrap_ci(margins, image_ids)

            # per-prompt-then-median robustness stat (matches base pilot's prompt_variant_stability.csv)
            p_subset = [r for r in prompt_rows if r["crop_condition"] == cond and r["positive_attribute"] == attr]
            per_prompt_means = []
            for p in range(16):
                vals = [float(r["alignment_margin"]) for r in p_subset if int(r["prompt_index"]) == p]
                if vals:
                    per_prompt_means.append(float(np.mean(vals)))
            median_prompt = float(np.median(per_prompt_means)) if per_prompt_means else float("nan")

            out_rows.append({
                "crop_condition": cond, "attribute": attr, "n_objects": len(margins), "n_images": n_images,
                "mean_margin": float(margins.mean()), "median_margin": float(np.median(margins)),
                "std_margin": float(margins.std(ddof=1)) if len(margins) > 1 else 0.0,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "median_margin_across_prompts": median_prompt,
                "frac_positive_margin": float((margins > 0).mean()),
            })

    fieldnames = ["crop_condition", "attribute", "n_objects", "n_images", "mean_margin", "median_margin",
                  "std_margin", "ci95_lo", "ci95_hi", "median_margin_across_prompts", "frac_positive_margin"]
    with open(ATTRIBUTE_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {ATTRIBUTE_SUMMARY_CSV}  ({len(out_rows)} rows)")
    return out_rows


def write_group_summary(rows):
    eq_rows = [r for r in rows if r["aggregation"] == "equal_ensemble" and r["valid"] == "True"]
    out_rows = []
    for cond in CROP_CONDITIONS:
        for group_name, attrs in ATTRIBUTE_GROUPS.items():
            subset = [r for r in eq_rows if r["crop_condition"] == cond and r["positive_attribute"] in attrs]
            if not subset:
                continue
            margins = np.array([float(r["alignment_margin"]) for r in subset])
            image_ids = [r["image_id"] for r in subset]
            ci_lo, ci_hi = image_level_bootstrap_ci(margins, image_ids)
            out_rows.append({
                "crop_condition": cond, "attribute_group": group_name, "attributes": ";".join(attrs),
                "n_objects": len(margins), "n_images": len(set(image_ids)),
                "mean_margin": float(margins.mean()), "median_margin": float(np.median(margins)),
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "frac_positive_margin": float((margins > 0).mean()),
            })
    fieldnames = ["crop_condition", "attribute_group", "attributes", "n_objects", "n_images",
                  "mean_margin", "median_margin", "ci95_lo", "ci95_hi", "frac_positive_margin"]
    with open(GROUP_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {GROUP_SUMMARY_CSV}  ({len(out_rows)} rows)")
    return out_rows


def write_condition_comparison(rows):
    eq_rows = [r for r in rows if r["aggregation"] == "equal_ensemble" and r["valid"] == "True"]
    by_key = {}
    for r in eq_rows:
        key = (r["image_id"], r["object_id"], r["positive_attribute"])
        by_key.setdefault(key, {})[r["crop_condition"]] = float(r["alignment_margin"])

    comparisons = [("tight_crop", "context20_crop"), ("context20_crop", "masked_context20"),
                   ("tight_crop", "masked_context20"), ("global_image", "tight_crop")]

    out_rows = []
    for cond_b, cond_a in comparisons:
        diffs, image_ids = [], []
        for (img_id, obj_id, attr), by_cond in by_key.items():
            if cond_a in by_cond and cond_b in by_cond:
                diffs.append(by_cond[cond_b] - by_cond[cond_a])
                image_ids.append(img_id)
        if not diffs:
            continue
        diffs = np.array(diffs)
        ci_lo, ci_hi = image_level_bootstrap_ci(diffs, image_ids)
        out_rows.append({
            "comparison": f"{cond_b}-{cond_a}", "condition_b": cond_b, "condition_a": cond_a,
            "n_paired_objects": len(diffs), "n_images": len(set(image_ids)),
            "mean_diff": float(diffs.mean()), "median_diff": float(np.median(diffs)),
            "ci95_lo_diff": ci_lo, "ci95_hi_diff": ci_hi,
            "frac_positive_diff": float((diffs > 0).mean()),
        })
    fieldnames = ["comparison", "condition_b", "condition_a", "n_paired_objects", "n_images",
                  "mean_diff", "median_diff", "ci95_lo_diff", "ci95_hi_diff", "frac_positive_diff"]
    with open(CONDITION_COMPARISON_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {CONDITION_COMPARISON_CSV}  ({len(out_rows)} rows)")
    return out_rows


def main():
    print("=" * 65)
    print("  OSIE Text-Alignment CROP Pilot: analysis")
    print("=" * 65)
    if not os.path.isfile(SCORES_CSV):
        raise RuntimeError(f"STOP: {SCORES_CSV} not found -- run the extraction script first")
    rows = load_rows()
    print(f"  Loaded {len(rows)} rows")

    print("\n--- crop_attribute_summary.csv (image-level bootstrap 95% CI) ---")
    attr_rows = write_attribute_summary(rows)
    print("\n--- crop_group_summary.csv ---")
    write_group_summary(rows)
    print("\n--- crop_condition_comparison.csv ---")
    write_condition_comparison(rows)

    print("\n=== equal_ensemble margin, mean [95% CI], by attribute x condition ===")
    by_attr = {}
    for r in attr_rows:
        by_attr.setdefault(r["attribute"], {})[r["crop_condition"]] = r
    header = f"{'attribute':14s}" + "".join(f"{c:>24s}" for c in CROP_CONDITIONS)
    print(header)
    for attr in sorted(by_attr):
        line = f"{attr:14s}"
        for c in CROP_CONDITIONS:
            r = by_attr[attr].get(c)
            if r:
                line += f"  {r['mean_margin']:+6.3f}[{r['ci95_lo']:+5.3f},{r['ci95_hi']:+5.3f}]"
            else:
                line += " " * 24
        print(line)

    print("\n" + "=" * 65)
    print("  Analysis: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
