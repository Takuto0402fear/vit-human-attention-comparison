"""
Experiment A (DINO ViT-B/16 pilot20): official DINO ViT-B/16 layerwise
[CLS]->patch attention vs. human fixations (OSIE, healthy) -- 20-image
pilot, at the SAME Base model scale as the existing CLIP ViT-B/16
experiment (isolating training-method vs. model-scale effects).

Same evaluation convention as scripts/run_expA.py (DINO-S) and
scripts/run_expA_clip.py / run_expA_clip_full700.py (CLIP-B): bilinear
upsample to 800x600, metrics.py's nss/auc_judd/sauc imported unmodified,
identical fixation-coordinate extraction, identical center-Gaussian/
uniform-noise positive controls, identical sAUC negative-pool convention
(one shared 700-image pool, no self-image exclusion, metrics.py's
internal RandomState(0) untouched). None of run_expA.py,
run_expA_clip.py, run_expA_clip_full700.py, vit_extractor.py,
dino_vitb16_extractor.py, clip_extractor.py, or lib/vision_transformer.py
is modified by this script.

Attention extraction uses ONLY the already-validated memory-conscious
extractor (dino_vitb16_extractor.py::extract_layerwise_cls_attention_
memory_conscious), unmodified -- no speed/refactor changes this stage.

Image selection: the SAME 20 images as the existing CLIP pilot
(outputs/expA_clip/pilot20/run_config.json's pilot_image_ids, read
read-only), not re-derived from directory listing order, so any future
drift in that file would be caught rather than silently diverging.

Explicit pilot/full mode switch (RUN_MODE below) -- only "pilot20" is
exercised this stage; "full700" selects the production output path and
image count but is not run here.

Output confined to outputs/expA_dino_vitb16/pilot20/. Does not write to
outputs/expA/, outputs/expA_clip/, outputs/expA_dino_vitb16/healthy/,
outputs/attn_cache/, or results/.

Usage (PowerShell):
    python scripts\\run_expA_dino_vitb16.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import nss, auc_judd, sauc
from vit_extractor import _pad_to_patch  # read-only reuse; DINO-S file untouched
from lib.clip_vit import patch_grid_from_image_hw  # read-only reuse; generic grid arithmetic
from dino_vitb16_extractor import (
    DINO_PATCH_SIZE, extract_layerwise_cls_attention_memory_conscious, load_dino_vitb16,
)

# ======================= CONFIG =======================
RUN_MODE = "pilot20"  # "pilot20" | "full700" -- explicit switch; only "pilot20" is run this stage

MODEL        = "dino_vitb16"
LAYERS       = list(range(1, 13))
ATTN         = "cls2patch"
HEAD_AGG     = "mean"
GROUP        = "healthy"
IMG_W        = 800
IMG_H        = 600
UPSAMPLE     = "bilinear"
METRICS_LIST = ["NSS", "AUC_Judd", "sAUC"]
SEED         = 42
EXPECTED_GRID = (38, 50)
EXPECTED_TOKENS = 1901

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR    = os.path.join(DATA_BASE, "data", "stimuli")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
DINO_S_METRICS_CSV = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"
CLIP_B_PILOT20_CONFIG = r"C:\Users\user\gaze\outputs\expA_clip\pilot20\run_config.json"
CLIP_B_PILOT20_CSV = r"C:\Users\user\gaze\outputs\expA_clip\pilot20\metrics_per_image.csv"

if RUN_MODE == "pilot20":
    OUT_DIR = r"C:\Users\user\gaze\outputs\expA_dino_vitb16\pilot20"
    N_IMAGES = 20
elif RUN_MODE == "full700":
    OUT_DIR = r"C:\Users\user\gaze\outputs\expA_dino_vitb16\healthy"
    N_IMAGES = 700
else:
    raise ValueError(f"unknown RUN_MODE {RUN_MODE!r}")
# ======================================================


def validate_image_id_sets():
    stim_ids = set(os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(STIM_DIR, "*.jpg")))
    fixmap_ids = set(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(FIXMAP_DIR, "*.npz")))
    dino_s_ids = set()
    with open(DINO_S_METRICS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dino_s_ids.add(os.path.splitext(row["image"])[0])
    common = stim_ids & fixmap_ids & dino_s_ids
    if len(common) != 700:
        raise RuntimeError(
            f"STOP: expected exactly 700 common images, got {len(common)}. "
            f"stim={len(stim_ids)} fixmap={len(fixmap_ids)} dino_s={len(dino_s_ids)}")
    return common


def load_pilot_image_ids_from_clip():
    """Same 20 images as the existing CLIP pilot -- read-only, not
    re-derived from directory order."""
    with open(CLIP_B_PILOT20_CONFIG, encoding="utf-8") as f:
        clip_config = json.load(f)
    ids = clip_config["pilot_image_ids"]
    if len(ids) != 20:
        raise RuntimeError(f"STOP: CLIP pilot20 image list has {len(ids)} images, expected 20")
    return ids


def load_fixation_data():
    """Identical extraction logic to scripts/run_expA.py / run_expA_clip*.py
    (duplicated, not imported -- matches this repo's per-script convention)."""
    fix_points, pool = {}, []
    for f in sorted(glob.glob(os.path.join(FIXMAP_DIR, "*.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)
        ys, xs = np.where(d["points"])
        coords = np.column_stack([xs, ys])
        fix_points[stem] = coords
        pool.append(coords)
    return fix_points, np.concatenate(pool)


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


def load_osie_image_tensor(image_path, patch_size=DINO_PATCH_SIZE):
    """DINO-side ImageNet-normalized counterpart of
    clip_extractor.py::load_osie_image_tensor -- same _pad_to_patch
    geometry, DINO-S's own ImageNet mean/std."""
    from PIL import Image
    from torchvision import transforms as pth_transforms

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)
    img_np = _pad_to_patch(img_np, patch_size)
    pad_h, pad_w = img_np.shape[:2]

    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(Image.fromarray(img_np)).unsqueeze(0)
    return tensor, (orig_h, orig_w), (pad_h, pad_w)


def extract_and_resize(model, image_path, device):
    """
    Mirrors run_expA_clip_full700.py's extract_and_resize: bilinear-
    upsample each layer's [CLS]->patch map directly to (IMG_H, IMG_W),
    then renormalize the UPSAMPLED map to sum=1 (the same post-hoc
    saliency-map convention already used by the official evaluation
    pipelines, applied AFTER the CLS column has already been dropped
    inside the memory-conscious extractor -- not a re-normalization of
    that step).

    Returns (resized (12,IMG_H,IMG_W) float32, grid_hw, t_io, t_infer,
    row_sums (12,)).
    """
    t0 = time.perf_counter()
    tensor, orig_hw, pad_hw = load_osie_image_tensor(image_path, DINO_PATCH_SIZE)
    tensor = tensor.to(device)
    grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], DINO_PATCH_SIZE)
    if device == "cuda":
        torch.cuda.synchronize()
    t_io = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.inference_mode():
        layer_maps, full_row_sums = extract_layerwise_cls_attention_memory_conscious(
            model, tensor, grid_hw)
    if device == "cuda":
        torch.cuda.synchronize()
    t_infer = time.perf_counter() - t1

    stacked = torch.stack(layer_maps, dim=1)  # (1, 12, gh, gw)
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


def read_layer_means_for_images(csv_path, image_ids, layers=range(1, 13)):
    """Read-only: per-layer mean of each metric, restricted to image_ids."""
    wanted = set(image_ids)
    data = {m: {l: [] for l in layers} for m in METRICS_LIST}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_id = os.path.splitext(row["image"])[0]
            if img_id not in wanted:
                continue
            layer = int(row["layer"])
            if layer not in data["NSS"]:
                continue
            for m in METRICS_LIST:
                data[m][layer].append(float(row[m]))
    means = {m: {l: float(np.mean(v)) if v else float("nan") for l, v in data[m].items()} for m in METRICS_LIST}
    counts = {m: {l: len(v) for l, v in data[m].items()} for m in METRICS_LIST}
    return means, counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print(f"  Exp A (DINO ViT-B/16 {RUN_MODE}): DINO-B vs Human Fixations")
    print("=" * 65)

    print("\n--- Validating image-ID sets (stimuli / fixmaps / DINO-S metrics) ---")
    common_700 = validate_image_id_sets()
    print(f"  Intersection: {len(common_700)} images (expected 700)  OK")

    if RUN_MODE == "pilot20":
        pilot_ids = load_pilot_image_ids_from_clip()
    else:
        raise NotImplementedError("full700 mode is not exercised at this stage")

    print(f"\n--- Image selection ({RUN_MODE}) ---")
    print(f"  Images (same as CLIP pilot20): {pilot_ids}")
    for img_id in pilot_ids:
        if img_id not in common_700:
            raise RuntimeError(f"STOP: pilot image {img_id} missing from the 700-image intersection")
    if len(pilot_ids) != N_IMAGES:
        raise RuntimeError(f"STOP: {len(pilot_ids)} images selected, expected {N_IMAGES}")

    print("\n--- Loading official DINO ViT-B/16 (backbone-only, strict=True) ---")
    model, device = load_dino_vitb16()
    print(f"  Device: {device}  dtype: {next(model.parameters()).dtype}")

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
              f"AUC={bl_summary[bname]['AUC_Judd']:.4f}  sAUC={bl_summary[bname]['sAUC']:.4f}")
    uniform_near_half = (0.4 <= bl_summary["uniform"]["AUC_Judd"] <= 0.6 and
                          0.4 <= bl_summary["uniform"]["sAUC"] <= 0.6)
    center_beats_uniform = (bl_summary["center_gaussian"]["NSS"] > bl_summary["uniform"]["NSS"] and
                             bl_summary["center_gaussian"]["AUC_Judd"] > bl_summary["uniform"]["AUC_Judd"])
    print(f"  Uniform AUC/sAUC ~0.5 (0.4-0.6 band): {'PASS' if uniform_near_half else 'FAIL'}")
    print(f"  Center Gaussian > Uniform (NSS & AUC): {'PASS' if center_beats_uniform else 'FAIL'}")
    if not uniform_near_half or not center_beats_uniform:
        raise RuntimeError("STOP: positive control failed")

    # ------------------------------------------------------------------
    # DINO-B extraction + metrics, 20 images. Image[0] = warmup: metrics
    # ARE saved, timing excluded from the steady-state aggregate.
    # ------------------------------------------------------------------
    print(f"\n--- Extracting DINO-B attention + computing metrics ({N_IMAGES} images) ---")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    timings = []
    seen_pairs = set()

    for idx, img_id in enumerate(pilot_ids):
        img_path = os.path.join(STIM_DIR, f"{img_id}.jpg")
        pts = fix_points[img_id]

        t_total0 = time.perf_counter()
        resized, grid_hw, t_io, t_infer, row_sums = extract_and_resize(model, img_path, device)

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
        tag = "WARMUP" if idx == 0 else f"{idx + 1:2d}/{N_IMAGES}"
        print(f"  [{tag}] {img_id}: io={t_io * 1000:6.1f}ms infer={t_infer * 1000:6.1f}ms "
              f"metrics={t_metrics * 1000:6.1f}ms total={t_total * 1000:6.1f}ms")

        del resized
        gc.collect()

    peak_alloc_mib = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else None
    peak_reserved_mib = (torch.cuda.max_memory_reserved() / 1024 ** 2) if device == "cuda" else None

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n--- Verification ---")
    if len(rows) != N_IMAGES * 12:
        raise RuntimeError(f"STOP: expected {N_IMAGES * 12} rows, got {len(rows)}")
    if len(seen_pairs) != N_IMAGES * 12:
        raise RuntimeError("STOP: duplicate (image,layer) pairs found")
    n_missing = sum(1 for r in rows if any(v is None for v in r.values()))
    if n_missing:
        raise RuntimeError(f"STOP: {n_missing} rows with missing values")
    row_images = set(r["image"] for r in rows)
    if len(row_images) != N_IMAGES:
        raise RuntimeError(f"STOP: {len(row_images)} unique images in output, expected {N_IMAGES}")
    per_image_layers = {}
    for r in rows:
        per_image_layers.setdefault(r["image"], set()).add(r["layer"])
    bad = {img: l for img, l in per_image_layers.items() if l != set(range(1, 13))}
    if bad:
        raise RuntimeError(f"STOP: images without exactly layers 1-12: {bad}")
    print(f"  rows: {len(rows)}  (expected {N_IMAGES * 12})  OK")
    print(f"  unique images: {len(row_images)}  (expected {N_IMAGES})  OK")
    print(f"  every image has layers 1-12 exactly once: OK")
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
            if len(vals) != N_IMAGES:
                raise RuntimeError(f"STOP: layer {l} metric {m} has N={len(vals)}, expected {N_IMAGES}")
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
    print(f"  Saved: {csv_layer}  ({len(layer_rows)} layers, N={N_IMAGES} each)")

    print(f"\n=== metrics_by_layer (DINO ViT-B/16, {RUN_MODE}) ===")
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
        "n_images": N_IMAGES,
        "n_images_steady_state": n_steady,
        "warmup_image": timings[0]["image"],
        "warmup_total_s": timings[0]["t_total_s"],
        "steady_state": {
            "sum_io_s": sum_io, "sum_infer_s": sum_infer,
            "sum_metrics_s": sum_metrics, "sum_total_s": sum_total,
            "avg_io_s": sum_io / n_steady, "avg_infer_s": sum_infer / n_steady,
            "avg_metrics_s": sum_metrics / n_steady, "avg_total_s": avg_total_per_image,
        },
        "peak_vram_allocated_mib": peak_alloc_mib,
        "peak_vram_reserved_mib": peak_reserved_mib,
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
    print(f"  Peak VRAM allocated: {peak_alloc_mib:.1f} MiB" if peak_alloc_mib is not None else "  Peak VRAM: n/a")
    print(f"  Peak VRAM reserved:  {peak_reserved_mib:.1f} MiB" if peak_reserved_mib is not None else "")
    print(f"  Extrapolated 700-image time: {timing_summary['extrapolated_700_images_s']:.1f}s "
          f"({timing_summary['extrapolated_700_images_minutes']:.1f} min)")

    # ------------------------------------------------------------------
    # run_config.json
    # ------------------------------------------------------------------
    run_config = {
        "model": MODEL, "run_mode": RUN_MODE, "layers": LAYERS, "attn": ATTN, "head_agg": HEAD_AGG,
        "group": GROUP, "img_w": IMG_W, "img_h": IMG_H, "upsample": UPSAMPLE,
        "metrics": METRICS_LIST, "seed": SEED, "batch_size": 1,
        "n_images": N_IMAGES,
        "pilot_image_ids": pilot_ids,
        "pilot_selection_method": (
            "same 20 images as outputs/expA_clip/pilot20/run_config.json's "
            "pilot_image_ids (read read-only), not directory-listing order"),
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

    # ------------------------------------------------------------------
    # 3-model pilot comparison (read-only) + confirmation-only plot
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  3-model pilot comparison (N=20, implementation check only -- "
          "NOT for research conclusions)")
    print("=" * 65)

    dino_s_means, dino_s_counts = read_layer_means_for_images(DINO_S_METRICS_CSV, pilot_ids)
    clip_b_means, clip_b_counts = read_layer_means_for_images(CLIP_B_PILOT20_CSV, pilot_ids)
    dino_b_means = {m: {row["layer"]: row[f"{m}_mean"] for row in layer_rows} for m in METRICS_LIST}

    for m in METRICS_LIST:
        for l in range(1, 13):
            if dino_s_counts[m][l] != N_IMAGES:
                raise RuntimeError(f"STOP: DINO-S {m} L{l} has N={dino_s_counts[m][l]}, expected {N_IMAGES}")
            if clip_b_counts[m][l] != N_IMAGES:
                raise RuntimeError(f"STOP: CLIP-B {m} L{l} has N={clip_b_counts[m][l]}, expected {N_IMAGES}")

    peak_layers = {}
    for model_name, means in [("DINO-S", dino_s_means), ("DINO-B", dino_b_means), ("CLIP-B", clip_b_means)]:
        peak_layers[model_name] = {}
        for m in METRICS_LIST:
            best_l = max(means[m], key=lambda l: means[m][l])
            peak_layers[model_name][m] = (best_l, means[m][best_l])
            print(f"  {model_name:8s} {m:10s} peak: L{best_l} ({means[m][best_l]:.4f})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    style = {"DINO-S": ("steelblue", "o", "-"), "DINO-B": ("seagreen", "^", "-."), "CLIP-B": ("darkorange", "s", "--")}
    layers_arr = np.arange(1, 13)
    for ax, m in zip(axes, METRICS_LIST):
        for model_name, means in [("DINO-S", dino_s_means), ("DINO-B", dino_b_means), ("CLIP-B", clip_b_means)]:
            color, marker, ls = style[model_name]
            vals = [means[m][l] for l in range(1, 13)]
            ax.plot(layers_arr, vals, marker=marker, linestyle=ls, color=color, lw=2, ms=5, label=model_name)
        ax.set_xlabel("Layer")
        ax.set_ylabel(m if m != "AUC_Judd" else "AUC-Judd")
        ax.set_title(m if m != "AUC_Judd" else "AUC-Judd")
        ax.set_xticks(layers_arr)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("DINO-S vs DINO-B vs CLIP-B -- Pilot N=20 / not for inference", fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "pilot20_3models_layerwise.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {plot_path}")

    print("\n" + "=" * 65)
    print(f"  Exp A (DINO ViT-B/16 {RUN_MODE}): DONE")
    print("  outputs/expA/, outputs/expA_clip/, outputs/expA_dino_vitb16/healthy/, "
          "outputs/attn_cache/, results/ were NOT touched.")
    print("=" * 65)


if __name__ == "__main__":
    main()
