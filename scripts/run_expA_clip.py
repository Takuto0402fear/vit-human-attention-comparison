"""
Experiment A (CLIP pilot20): CLIP ViT-B/16 layerwise [CLS]->patch
attention vs. human fixations (OSIE, healthy) -- 20-image pilot.

Mirrors scripts/run_expA.py's evaluation convention (bilinear upsample to
800x600, NSS/AUC-Judd/sAUC via metrics.py imported unmodified, identical
fixation-coordinate extraction, identical center-Gaussian/uniform-noise
positive controls) but targets the CLIP ViT-B/16 extractor
(clip_extractor.py / lib/clip_vit.py) instead of DINO ViT-S/16.

Scope of this script (this stage): 20 images only, chosen
deterministically from the sorted unique image IDs already present in
outputs/expA/healthy/metrics_per_image.csv (the existing DINO production
run) -- not from directory-listing order, and not from a prior 20-image
pilot artifact (none was found under outputs/ or in git history for
either DINO or CLIP). The sAUC negative pool is still drawn from the
FULL 700-image fixmap set, matching run_expA.py's convention exactly:
one shared `all_pool` for every image, with NO self-image exclusion.

Does not modify scripts/run_expA.py, vit_extractor.py, config.py, or any
other DINO file. Does not write to outputs/expA_clip/healthy/,
outputs/attn_cache/dino_vits16_patchgrid.npz,
outputs/attn_cache/clip_vitb16_full700_patchgrid.npz, or results/.

Usage (PowerShell):
    python scripts\\run_expA_clip.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import gc
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from metrics import nss, auc_judd, sauc
from clip_extractor import CLIP_PATCH_SIZE, load_clip_vit_b16, load_osie_image_tensor
from lib.clip_vit import (
    interpolate_patch_pos_embed, patch_grid_from_image_hw,
    visual_forward_with_cls_patch_attention,
)

# ======================= CONFIG =======================
MODEL        = "clip_vitb16"
LAYERS       = list(range(1, 13))
ATTN         = "cls2patch"
HEAD_AGG     = "mean"
GROUP        = "healthy"
IMG_W        = 800
IMG_H        = 600
UPSAMPLE     = "bilinear"
METRICS_LIST = ["NSS", "AUC_Judd", "sAUC"]
SEED         = 42
N_PILOT      = 20
EXPECTED_GRID = (38, 50)
EXPECTED_TOKENS = 1901

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR    = os.path.join(DATA_BASE, "data", "stimuli")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
DINO_METRICS_CSV = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"

OUT_DIR    = r"C:\Users\user\gaze\outputs\expA_clip\pilot20"
# Reserved for a future stage; NOT written by this script.
ATTN_CACHE = r"C:\Users\user\gaze\outputs\attn_cache\clip_vitb16_pilot20_patchgrid.npz"
# ======================================================


def validate_image_id_sets():
    """
    Cross-check OSIE stimuli / fixmaps / existing-DINO-metrics image-ID
    sets. Stops if the intersection isn't exactly 700 -- CLIP production
    scope must match the DINO reference exactly.
    """
    stim_ids = set(os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(STIM_DIR, "*.jpg")))
    fixmap_ids = set(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(FIXMAP_DIR, "*.npz")))
    dino_ids = set()
    with open(DINO_METRICS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dino_ids.add(os.path.splitext(row["image"])[0])

    common = stim_ids & fixmap_ids & dino_ids
    if len(common) != 700:
        raise RuntimeError(
            f"STOP: expected exactly 700 common images, got {len(common)}. "
            f"stim={len(stim_ids)} fixmap={len(fixmap_ids)} dino={len(dino_ids)} "
            f"stim-common={sorted(stim_ids - common)} "
            f"fixmap-common={sorted(fixmap_ids - common)} "
            f"dino-common={sorted(dino_ids - common)}")
    return common, dino_ids


def select_pilot_image_ids():
    """
    Deterministic 20-image pilot set: sorted unique image IDs from the
    existing DINO metrics_per_image.csv, independent of directory-listing
    order.
    """
    ids = set()
    with open(DINO_METRICS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.add(os.path.splitext(row["image"])[0])
    return sorted(ids)[:N_PILOT], len(ids)


# ------------------------------------------------------------------
# Fixation data -- same extraction logic as scripts/run_expA.py's
# load_fixation_data(). Duplicated here (not imported), matching this
# repo's existing convention of each self-contained exp script keeping
# its own copy (see run_expB.py / run_expB_control.py / run_expB_lastk.py);
# run_expA.py itself is not modified to export this.
# ------------------------------------------------------------------

def load_fixation_data():
    """
    Returns
    -------
    fix_points : {stem: (N,2) int (x,y)}
    all_pool   : (M,2) int -- pooled across ALL 700 images. Matches
        run_expA.py's convention of using the SAME shared pool for every
        image's sAUC (no self-image exclusion).
    """
    fix_points, pool = {}, []
    for f in sorted(glob.glob(os.path.join(FIXMAP_DIR, "*.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)
        ys, xs = np.where(d["points"])
        coords = np.column_stack([xs, ys])
        fix_points[stem] = coords
        pool.append(coords)
    return fix_points, np.concatenate(pool)


# ------------------------------------------------------------------
# Baselines -- identical formulas to scripts/run_expA.py
# ------------------------------------------------------------------

def center_gaussian(h, w, sigma=None):
    if sigma is None:
        sigma = min(h, w) / 6
    ys = np.arange(h, dtype=np.float64).reshape(-1, 1)
    xs = np.arange(w, dtype=np.float64).reshape(1, -1)
    g = np.exp(-((xs - w / 2) ** 2 + (ys - h / 2) ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def uniform_noise(h, w):
    rng = np.random.RandomState(SEED)
    u = rng.rand(h, w).astype(np.float32)
    return u / u.sum()


# ------------------------------------------------------------------
# CLIP attention extraction for one image -> (12, IMG_H, IMG_W)
# ------------------------------------------------------------------

def extract_and_resize(model, image_path):
    """
    Mirrors run_expA.py's extract_attention_chunk: bilinear-upsample each
    layer's [CLS]->patch map directly to (IMG_H, IMG_W), then renormalize
    the UPSAMPLED map to sum=1. This is a post-hoc saliency-map
    convention applied AFTER the CLS column has been dropped -- distinct
    from (and unrelated to) the "no renormalization right after dropping
    CLS" rule enforced inside cls_to_patch_grid.

    Returns (resized (12,IMG_H,IMG_W) float32, grid_hw, t_io, t_infer,
    row_sums (12,)).
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

    stacked = torch.stack(cls_patch_maps, dim=1)  # (1, 12, gh, gw)
    resized = F.interpolate(
        stacked.reshape(12, 1, *grid_hw), size=(IMG_H, IMG_W),
        mode="bilinear", align_corners=False)
    resized = resized[:, 0].detach().cpu().numpy().astype(np.float32)  # (12, IMG_H, IMG_W)

    for l in range(resized.shape[0]):
        s = resized[l].sum()
        if s > 0:
            resized[l] /= s

    row_sums = torch.stack(full_row_sums).squeeze(-1).detach().cpu().numpy()  # (12,)
    return resized, grid_hw, t_io, t_infer, row_sums


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  Exp A (CLIP pilot20): CLIP ViT-B/16 vs Human Fixations")
    print("=" * 65)

    print("\n--- Validating image-ID sets (stimuli / fixmaps / DINO metrics) ---")
    common_700, dino_ids = validate_image_id_sets()
    print(f"  Intersection: {len(common_700)} images (expected 700)  OK")

    pilot_ids, dino_total = select_pilot_image_ids()
    print("\n--- Pilot image selection ---")
    print(f"  DINO metrics_per_image.csv unique images: {dino_total}")
    print(f"  Pilot (sorted, first {N_PILOT}): {pilot_ids}")
    for img_id in pilot_ids:
        if img_id not in common_700:
            raise RuntimeError(f"STOP: pilot image {img_id} missing from the 700-image intersection")

    print("\n--- Loading CLIP ViT-B/16 ---")
    model, _ = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}  dtype: {model.dtype}")

    print("\n--- Loading fixation data (700-image pool for sAUC) ---")
    fix_points, all_pool = load_fixation_data()
    print(f"  {len(fix_points)} images with fixations, {len(all_pool)} pooled points")
    if len(fix_points) != 700:
        raise RuntimeError(f"STOP: expected 700 fixmap images, got {len(fix_points)}")

    # ------------------------------------------------------------------
    # Positive controls (sanity check on the 20 pilot images)
    # ------------------------------------------------------------------
    print("\n--- Positive control: center-Gaussian vs uniform-noise ---")
    cg_map = center_gaussian(IMG_H, IMG_W)
    uf_map = uniform_noise(IMG_H, IMG_W)
    bl = {"center_gaussian": [], "uniform": []}
    for img_id in pilot_ids:
        pts = fix_points[img_id]
        for bname, bmap in [("center_gaussian", cg_map), ("uniform", uf_map)]:
            bl[bname].append({
                "NSS": nss(bmap, pts),
                "AUC_Judd": auc_judd(bmap, pts),
                "sAUC": sauc(bmap, pts, all_pool),
            })
    bl_summary = {}
    for bname, recs in bl.items():
        bl_summary[bname] = {m: float(np.nanmean([r[m] for r in recs])) for m in METRICS_LIST}
        print(f"  {bname:16s}  NSS={bl_summary[bname]['NSS']:.4f}  "
              f"AUC={bl_summary[bname]['AUC_Judd']:.4f}  "
              f"sAUC={bl_summary[bname]['sAUC']:.4f}")
    uniform_near_half = (0.4 <= bl_summary["uniform"]["AUC_Judd"] <= 0.6 and
                          0.4 <= bl_summary["uniform"]["sAUC"] <= 0.6)
    center_beats_uniform = (bl_summary["center_gaussian"]["NSS"] > bl_summary["uniform"]["NSS"] and
                             bl_summary["center_gaussian"]["AUC_Judd"] > bl_summary["uniform"]["AUC_Judd"])
    print(f"  Uniform AUC/sAUC ~0.5 (0.4-0.6 band): {'PASS' if uniform_near_half else 'WARN'}")
    print(f"  Center Gaussian > Uniform (NSS & AUC): {'PASS' if center_beats_uniform else 'WARN'}")

    # ------------------------------------------------------------------
    # CLIP extraction + metrics, 20 images.
    # Image[0] = warmup: metrics ARE saved, but its timing is excluded
    # from the steady-state aggregate / 700-image extrapolation.
    # ------------------------------------------------------------------
    print(f"\n--- Extracting CLIP attention + computing metrics ({N_PILOT} images) ---")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows = []
    timings = []
    seen_pairs = set()

    for idx, img_id in enumerate(pilot_ids):
        img_path = os.path.join(STIM_DIR, f"{img_id}.jpg")
        pts = fix_points[img_id]

        t_total0 = time.perf_counter()
        resized, grid_hw, t_io, t_infer, row_sums = extract_and_resize(model, img_path)

        if grid_hw != EXPECTED_GRID:
            raise RuntimeError(f"STOP: {img_id} grid {grid_hw} != {EXPECTED_GRID}")
        if resized.shape != (12, IMG_H, IMG_W):
            raise RuntimeError(f"STOP: {img_id} attention shape {resized.shape} != (12,{IMG_H},{IMG_W})")
        if np.isnan(resized).any() or np.isinf(resized).any():
            raise RuntimeError(f"STOP: NaN/Inf in {img_id} attention maps")
        if np.isnan(row_sums).any() or np.isinf(row_sums).any() or np.any(np.abs(row_sums - 1.0) > 1e-2):
            raise RuntimeError(f"STOP: {img_id} CLS-row-sum out of range: {row_sums}")

        t_metrics0 = time.perf_counter()
        for l in range(12):
            layer = l + 1
            key = (img_id, layer)
            if key in seen_pairs:
                raise RuntimeError(f"STOP: duplicate (image,layer) pair {key}")
            seen_pairs.add(key)
            sal = resized[l]
            row = {
                "image": f"{img_id}.jpg",
                "layer": layer,
                "NSS": nss(sal, pts),
                "AUC_Judd": auc_judd(sal, pts),
                "sAUC": sauc(sal, pts, all_pool),
            }
            for m in METRICS_LIST:
                if not np.isfinite(row[m]):
                    raise RuntimeError(f"STOP: non-finite {m} for {img_id} layer {layer}: {row[m]}")
            if not (0.0 <= row["AUC_Judd"] <= 1.0):
                raise RuntimeError(f"STOP: AUC_Judd out of [0,1] for {img_id} layer {layer}: {row['AUC_Judd']}")
            if not (0.0 <= row["sAUC"] <= 1.0):
                raise RuntimeError(f"STOP: sAUC out of [0,1] for {img_id} layer {layer}: {row['sAUC']}")
            rows.append(row)
        t_metrics = time.perf_counter() - t_metrics0
        t_total = time.perf_counter() - t_total0

        timings.append({
            "image": img_id, "is_warmup": idx == 0,
            "t_io_s": t_io, "t_infer_s": t_infer,
            "t_metrics_s": t_metrics, "t_total_s": t_total,
        })
        tag = "WARMUP" if idx == 0 else f"{idx + 1:2d}/{N_PILOT}"
        print(f"  [{tag}] {img_id}: io={t_io * 1000:6.1f}ms infer={t_infer * 1000:6.1f}ms "
              f"metrics={t_metrics * 1000:6.1f}ms total={t_total * 1000:6.1f}ms")

        del resized
        gc.collect()

    peak_mem_mib = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else None)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n--- Verification ---")
    if len(rows) != N_PILOT * 12:
        raise RuntimeError(f"STOP: expected {N_PILOT * 12} rows, got {len(rows)}")
    if len(seen_pairs) != N_PILOT * 12:
        raise RuntimeError("STOP: duplicate (image,layer) pairs found")
    n_missing = sum(1 for r in rows if any(v is None for v in r.values()))
    if n_missing:
        raise RuntimeError(f"STOP: {n_missing} rows with missing values")
    print(f"  rows: {len(rows)}  (expected {N_PILOT * 12})  OK")
    print(f"  unique (image,layer) pairs: {len(seen_pairs)}  OK")
    print(f"  missing values: {n_missing}  OK")

    # ------------------------------------------------------------------
    # Save metrics_per_image.csv
    # ------------------------------------------------------------------
    csv_img = os.path.join(OUT_DIR, "metrics_per_image.csv")
    with open(csv_img, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "layer"] + METRICS_LIST)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_img}")

    # ------------------------------------------------------------------
    # Layer summary
    # ------------------------------------------------------------------
    layer_agg = {l: {m: [] for m in METRICS_LIST} for l in range(1, 13)}
    for r in rows:
        for m in METRICS_LIST:
            layer_agg[r["layer"]][m].append(r[m])

    layer_rows = []
    for l in range(1, 13):
        row = {"layer": l}
        for m in METRICS_LIST:
            vals = layer_agg[l][m]
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals))
        layer_rows.append(row)

    csv_layer = os.path.join(OUT_DIR, "metrics_by_layer.csv")
    fieldnames = ["layer"] + [f"{m}_{s}" for m in METRICS_LIST for s in ("mean", "std")]
    with open(csv_layer, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in layer_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {csv_layer}")

    print("\n=== metrics_by_layer (CLIP ViT-B/16, pilot20) ===")
    print(f"{'layer':>5}  {'NSS':>8}  {'AUC_Judd':>10}  {'sAUC':>8}")
    for row in layer_rows:
        print(f"{row['layer']:5d}  {row['NSS_mean']:8.4f}  {row['AUC_Judd_mean']:10.4f}  {row['sAUC_mean']:8.4f}")

    # ------------------------------------------------------------------
    # Timing summary + 700-image extrapolation (image[0] = warmup, excluded)
    # ------------------------------------------------------------------
    steady = timings[1:]
    n_steady = len(steady)
    sum_io = sum(t["t_io_s"] for t in steady)
    sum_infer = sum(t["t_infer_s"] for t in steady)
    sum_metrics = sum(t["t_metrics_s"] for t in steady)
    sum_total = sum(t["t_total_s"] for t in steady)
    avg_total_per_image = sum_total / n_steady

    timing_summary = {
        "n_images_pilot": N_PILOT,
        "n_images_steady_state": n_steady,
        "warmup_image": timings[0]["image"],
        "warmup_total_s": timings[0]["t_total_s"],
        "steady_state": {
            "sum_io_s": sum_io, "sum_infer_s": sum_infer,
            "sum_metrics_s": sum_metrics, "sum_total_s": sum_total,
            "avg_io_s": sum_io / n_steady, "avg_infer_s": sum_infer / n_steady,
            "avg_metrics_s": sum_metrics / n_steady, "avg_total_s": avg_total_per_image,
        },
        "peak_vram_mib": peak_mem_mib,
        "peak_vram_scope": "entire pilot run including the warmup image",
        "extrapolated_700_images_s": avg_total_per_image * 700,
        "extrapolated_700_images_minutes": avg_total_per_image * 700 / 60,
        "per_image_timings": timings,
    }
    timing_path = os.path.join(OUT_DIR, "timing.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)
    print(f"\n  Saved: {timing_path}")
    ss = timing_summary["steady_state"]
    print(f"  Steady-state avg/image (n={n_steady}): io={ss['avg_io_s'] * 1000:.1f}ms "
          f"infer={ss['avg_infer_s'] * 1000:.1f}ms metrics={ss['avg_metrics_s'] * 1000:.1f}ms "
          f"total={avg_total_per_image * 1000:.1f}ms")
    print(f"  Peak VRAM: {peak_mem_mib:.1f} MiB" if peak_mem_mib is not None else "  Peak VRAM: n/a")
    print(f"  Extrapolated 700-image time: {timing_summary['extrapolated_700_images_s']:.1f}s "
          f"({timing_summary['extrapolated_700_images_minutes']:.1f} min)")

    # ------------------------------------------------------------------
    # run_config.json
    # ------------------------------------------------------------------
    run_config = {
        "model": MODEL,
        "layers": LAYERS,
        "attn": ATTN,
        "head_agg": HEAD_AGG,
        "group": GROUP,
        "img_w": IMG_W,
        "img_h": IMG_H,
        "upsample": UPSAMPLE,
        "metrics": METRICS_LIST,
        "seed": SEED,
        "batch_size": 1,
        "n_pilot_images": N_PILOT,
        "pilot_image_ids": pilot_ids,
        "pilot_selection_method":
            "sorted(unique(outputs/expA/healthy/metrics_per_image.csv image ids))[:20]; "
            "no prior 20-image pilot artifact found for DINO or CLIP",
        "sauc_negative_pool": {
            "source": "outputs/fixmaps/healthy (all 700 images)",
            "n_points": int(len(all_pool)),
            "self_image_excluded": False,
            "note": "matches scripts/run_expA.py: one shared all_pool for every image",
        },
        "positive_controls": bl_summary,
        "n_rows": len(rows),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(OUT_DIR, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"  Saved: {config_path}")

    print("\n" + "=" * 65)
    print("  Exp A (CLIP pilot20): DONE")
    print("  outputs/expA_clip/healthy/ and "
          "outputs/attn_cache/clip_vitb16_full700_patchgrid.npz were NOT touched.")
    print("=" * 65)


if __name__ == "__main__":
    main()
