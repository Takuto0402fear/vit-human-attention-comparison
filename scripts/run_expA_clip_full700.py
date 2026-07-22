"""
Experiment A (CLIP full700): CLIP ViT-B/16 layerwise [CLS]->patch
attention vs. human fixations (OSIE, healthy) -- full 700-image
production run.

Same evaluation convention as scripts/run_expA_clip.py (the 20-image
pilot, validated and committed at 78a36b2) and scripts/run_expA.py (the
DINO production run): bilinear upsample to 800x600, metrics.py's
nss/auc_judd/sauc imported unmodified, identical fixation-coordinate
extraction, identical center-Gaussian/uniform-noise positive controls,
identical sAUC negative-pool convention (one shared 700-image pool, no
self-image exclusion, metrics.py's internal RandomState(0) untouched).

Differences from scripts/run_expA_clip.py (intentional, this stage):
  - All 700 images (sorted, deterministic), not a 20-image subset.
  - Output confined to outputs/expA_clip/healthy/ (pilot20/ is untouched
    -- this is a separate script, not an edit to run_expA_clip.py).
  - Periodic partial-CSV checkpointing every 50 images (flush + fsync),
    but partial files are never treated as the official result and this
    script does NOT implement resume: if interrupted, partial files are
    left in place (never auto-deleted) and the run must be restarted
    from scratch after they are manually moved aside or removed.
  - Refuses to run if any FINAL output file already exists (no
    overwrite), and refuses to run if a partial file from a previous
    attempt is still present (no silent resume/append).
  - No attention cache is written this stage.

Does not modify scripts/run_expA.py, scripts/run_expA_clip.py,
vit_extractor.py, config.py, or any other DINO file. Does not touch
outputs/expA_clip/pilot20/, outputs/expA/, outputs/attn_cache/, or
results/.

Usage (PowerShell):
    python scripts\\run_expA_clip_full700.py
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
N_EXPECTED   = 700
EXPECTED_GRID = (38, 50)
CHECKPOINT_EVERY = 50

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR    = os.path.join(DATA_BASE, "data", "stimuli")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
DINO_METRICS_CSV = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"

OUT_DIR = r"C:\Users\user\gaze\outputs\expA_clip\healthy"

PARTIAL_CSV    = os.path.join(OUT_DIR, "metrics_per_image.partial.csv")
PARTIAL_CONFIG = os.path.join(OUT_DIR, "run_config.partial.json")
PARTIAL_TIMING = os.path.join(OUT_DIR, "timing.partial.json")

FINAL_CSV       = os.path.join(OUT_DIR, "metrics_per_image.csv")
FINAL_LAYER_CSV = os.path.join(OUT_DIR, "metrics_by_layer.csv")
FINAL_CONFIG    = os.path.join(OUT_DIR, "run_config.json")
FINAL_TIMING    = os.path.join(OUT_DIR, "timing.json")
FINAL_PLOT      = os.path.join(OUT_DIR, "layerwise_agreement.png")

CSV_FIELDS = ["image", "layer"] + METRICS_LIST
# ======================================================


def guard_no_overwrite_and_no_stale_partials():
    finals = [FINAL_CSV, FINAL_LAYER_CSV, FINAL_CONFIG, FINAL_TIMING, FINAL_PLOT]
    existing_finals = [p for p in finals if os.path.isfile(p)]
    if existing_finals:
        raise RuntimeError(
            "STOP: final output(s) already exist, refusing to overwrite:\n  "
            + "\n  ".join(existing_finals))

    partials = [PARTIAL_CSV, PARTIAL_CONFIG, PARTIAL_TIMING]
    existing_partials = [p for p in partials if os.path.isfile(p)]
    if existing_partials:
        raise RuntimeError(
            "STOP: partial file(s) from a previous attempt are still present "
            "(no auto-resume, no auto-delete). Move these aside or remove them "
            "after manual review, then restart from scratch:\n  "
            + "\n  ".join(existing_partials))


def validate_image_id_sets():
    stim_ids = set(os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(STIM_DIR, "*.jpg")))
    fixmap_ids = set(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(FIXMAP_DIR, "*.npz")))
    dino_ids = set()
    with open(DINO_METRICS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dino_ids.add(os.path.splitext(row["image"])[0])

    common = stim_ids & fixmap_ids & dino_ids
    if len(common) != N_EXPECTED:
        raise RuntimeError(
            f"STOP: expected exactly {N_EXPECTED} common images, got {len(common)}. "
            f"stim={len(stim_ids)} fixmap={len(fixmap_ids)} dino={len(dino_ids)}")
    return common


def load_fixation_data():
    """Identical extraction logic to scripts/run_expA.py / run_expA_clip.py
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


def extract_and_resize(model, image_path):
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

    stacked = torch.stack(cls_patch_maps, dim=1)
    resized = F.interpolate(
        stacked.reshape(12, 1, *grid_hw), size=(IMG_H, IMG_W),
        mode="bilinear", align_corners=False)
    resized = resized[:, 0].detach().cpu().numpy().astype(np.float32)

    for l in range(resized.shape[0]):
        s = resized[l].sum()
        if s > 0:
            resized[l] /= s

    row_sums = torch.stack(full_row_sums).squeeze(-1).detach().cpu().numpy()
    return resized, grid_hw, t_io, t_infer, row_sums


def _append_partial_csv(path, rows):
    write_header = (not os.path.isfile(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def compute_positive_controls(image_ids, fix_points, all_pool):
    cg_map = center_gaussian(IMG_H, IMG_W)
    uf_map = uniform_noise(IMG_H, IMG_W)
    bl = {"center_gaussian": [], "uniform": []}
    for img_id in image_ids:
        pts = fix_points[img_id]
        for bname, bmap in [("center_gaussian", cg_map), ("uniform", uf_map)]:
            bl[bname].append({
                "NSS": nss(bmap, pts),
                "AUC_Judd": auc_judd(bmap, pts),
                "sAUC": sauc(bmap, pts, all_pool),
            })
    summary = {}
    for bname, recs in bl.items():
        summary[bname] = {m: float(np.nanmean([r[m] for r in recs])) for m in METRICS_LIST}
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wall_t0 = time.perf_counter()

    print("=" * 65)
    print("  Exp A (CLIP full700): CLIP ViT-B/16 vs Human Fixations")
    print("=" * 65)

    print("\n--- Overwrite / stale-partial guard ---")
    guard_no_overwrite_and_no_stale_partials()
    print("  No existing final outputs, no stale partial files.  OK")

    print("\n--- Validating image-ID sets (stimuli / fixmaps / DINO metrics) ---")
    common_700 = validate_image_id_sets()
    image_ids = sorted(common_700)
    print(f"  Intersection: {len(common_700)} images (expected {N_EXPECTED})  OK")
    if len(image_ids) != N_EXPECTED:
        raise RuntimeError(f"STOP: {len(image_ids)} images selected, expected {N_EXPECTED}")

    print("\n--- Loading CLIP ViT-B/16 ---")
    model, _ = load_clip_vit_b16()
    device = next(model.parameters()).device
    print(f"  Device: {device}  dtype: {model.dtype}")

    print("\n--- Loading fixation data (700-image pool for sAUC) ---")
    fix_points, all_pool = load_fixation_data()
    print(f"  {len(fix_points)} images with fixations, {len(all_pool)} pooled points")
    if len(fix_points) != N_EXPECTED:
        raise RuntimeError(f"STOP: expected {N_EXPECTED} fixmap images, got {len(fix_points)}")

    # Record a partial run_config immediately, marking the run as in progress.
    with open(PARTIAL_CONFIG, "w", encoding="utf-8") as f:
        json.dump({
            "status": "in_progress", "model": MODEL, "n_images_target": N_EXPECTED,
            "image_ids": image_ids, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    print("\n--- Positive control: center-Gaussian vs uniform-noise (all 700 images) ---")
    bl_summary = compute_positive_controls(image_ids, fix_points, all_pool)
    for bname, vals in bl_summary.items():
        print(f"  {bname:16s}  NSS={vals['NSS']:.4f}  AUC={vals['AUC_Judd']:.4f}  sAUC={vals['sAUC']:.4f}")
    uniform_near_half = (0.4 <= bl_summary["uniform"]["AUC_Judd"] <= 0.6 and
                          0.4 <= bl_summary["uniform"]["sAUC"] <= 0.6)
    center_beats_uniform = (bl_summary["center_gaussian"]["NSS"] > bl_summary["uniform"]["NSS"] and
                             bl_summary["center_gaussian"]["AUC_Judd"] > bl_summary["uniform"]["AUC_Judd"])
    print(f"  Uniform AUC/sAUC ~0.5 (0.4-0.6 band): {'PASS' if uniform_near_half else 'WARN'}")
    print(f"  Center Gaussian > Uniform (NSS & AUC): {'PASS' if center_beats_uniform else 'WARN'}")

    # ------------------------------------------------------------------
    # Main extraction loop. image_ids[0] = warmup (metrics saved, timing
    # excluded from the steady-state aggregate).
    # ------------------------------------------------------------------
    print(f"\n--- Extracting CLIP attention + computing metrics ({N_EXPECTED} images) ---")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows = []
    pending_rows = []
    timings = []
    seen_pairs = set()
    loop_t0 = time.perf_counter()

    for idx, img_id in enumerate(image_ids):
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
        img_rows = []
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
            img_rows.append(row)
        t_metrics = time.perf_counter() - t_metrics0
        t_total = time.perf_counter() - t_total0

        rows.extend(img_rows)
        pending_rows.extend(img_rows)
        timings.append({
            "image": img_id, "is_warmup": idx == 0,
            "t_io_s": t_io, "t_infer_s": t_infer,
            "t_metrics_s": t_metrics, "t_total_s": t_total,
        })

        del resized
        gc.collect()

        n_done = idx + 1
        if n_done % CHECKPOINT_EVERY == 0 or n_done == N_EXPECTED:
            _append_partial_csv(PARTIAL_CSV, pending_rows)
            pending_rows = []

            elapsed = time.perf_counter() - loop_t0
            steady_so_far = timings[1:] if len(timings) > 1 else []
            avg_so_far = (sum(t["t_total_s"] for t in steady_so_far) / len(steady_so_far)
                          if steady_so_far else timings[0]["t_total_s"])
            eta_s = avg_so_far * (N_EXPECTED - n_done)
            print(f"  [{n_done:4d}/{N_EXPECTED}] elapsed={elapsed:7.1f}s  "
                  f"avg/img={avg_so_far * 1000:6.1f}ms  ETA={eta_s:6.1f}s  "
                  f"(checkpoint written: {PARTIAL_CSV})")

            peak_so_far = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                           if device.type == "cuda" else None)
            with open(PARTIAL_TIMING, "w", encoding="utf-8") as f:
                json.dump({
                    "status": "in_progress", "n_done": n_done, "n_target": N_EXPECTED,
                    "elapsed_s": elapsed, "avg_total_s_so_far": avg_so_far,
                    "peak_vram_mib_so_far": peak_so_far,
                }, f, indent=2)

    peak_mem_mib = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else None)
    wall_elapsed_s = time.perf_counter() - wall_t0

    # ------------------------------------------------------------------
    # Verification (hard stop conditions -- final files NOT written on failure)
    # ------------------------------------------------------------------
    print("\n--- Verification ---")
    if len(rows) != N_EXPECTED * 12:
        raise RuntimeError(f"STOP: expected {N_EXPECTED * 12} rows, got {len(rows)}")
    if len(seen_pairs) != N_EXPECTED * 12:
        raise RuntimeError("STOP: duplicate (image,layer) pairs found")
    row_image_ids = set(r["image"] for r in rows)
    if len(row_image_ids) != N_EXPECTED:
        raise RuntimeError(f"STOP: {len(row_image_ids)} unique images in output, expected {N_EXPECTED}")
    per_image_layers = {}
    for r in rows:
        per_image_layers.setdefault(r["image"], set()).add(r["layer"])
    bad = {img: layers for img, layers in per_image_layers.items() if layers != set(range(1, 13))}
    if bad:
        raise RuntimeError(f"STOP: {len(bad)} images do not have exactly layers 1-12: {list(bad)[:5]}")
    n_missing = sum(1 for r in rows if any(v is None for v in r.values()))
    if n_missing:
        raise RuntimeError(f"STOP: {n_missing} rows with missing values")
    print(f"  rows: {len(rows)}  (expected {N_EXPECTED * 12})  OK")
    print(f"  unique images: {len(row_image_ids)}  (expected {N_EXPECTED})  OK")
    print(f"  every image has layers 1-12 exactly once: OK")
    print(f"  missing values: {n_missing}  OK")

    # ------------------------------------------------------------------
    # Save final metrics_per_image.csv
    # ------------------------------------------------------------------
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {FINAL_CSV}")

    # ------------------------------------------------------------------
    # Layer summary
    # ------------------------------------------------------------------
    layer_agg = {l: {m: [] for m in METRICS_LIST} for l in range(1, 13)}
    for r in rows:
        for m in METRICS_LIST:
            layer_agg[r["layer"]][m].append(r[m])

    layer_rows = []
    peak_layer = {}
    for m in METRICS_LIST:
        best_l, best_v = None, None
        for l in range(1, 13):
            v = float(np.mean(layer_agg[l][m]))
            if best_v is None or v > best_v:
                best_l, best_v = l, v
        peak_layer[m] = (best_l, best_v)

    for l in range(1, 13):
        row = {"layer": l}
        for m in METRICS_LIST:
            vals = layer_agg[l][m]
            if len(vals) != N_EXPECTED:
                raise RuntimeError(f"STOP: layer {l} metric {m} has N={len(vals)}, expected {N_EXPECTED}")
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals))
        layer_rows.append(row)

    fieldnames = ["layer"] + [f"{m}_{s}" for m in METRICS_LIST for s in ("mean", "std")]
    with open(FINAL_LAYER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in layer_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  Saved: {FINAL_LAYER_CSV}  ({len(layer_rows)} layers, N={N_EXPECTED} each)")

    print("\n=== metrics_by_layer (CLIP ViT-B/16, full700) ===")
    print(f"{'layer':>5}  {'NSS':>8}  {'AUC_Judd':>10}  {'sAUC':>8}")
    for row in layer_rows:
        print(f"{row['layer']:5d}  {row['NSS_mean']:8.4f}  {row['AUC_Judd_mean']:10.4f}  {row['sAUC_mean']:8.4f}")
    print("\nPeak layer per metric:")
    for m in METRICS_LIST:
        print(f"  {m}: layer {peak_layer[m][0]}  ({peak_layer[m][1]:.4f})")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_defs = [("NSS", "NSS_mean", "NSS_std"),
                 ("AUC_Judd", "AUC_Judd_mean", "AUC_Judd_std"),
                 ("sAUC", "sAUC_mean", "sAUC_std")]
    layers_arr = np.arange(1, 13)
    for ax, (mname, mc, sc) in zip(axes, plot_defs):
        means = np.array([layer_rows[l][mc] for l in range(12)])
        stds = np.array([layer_rows[l][sc] for l in range(12)])
        ax.plot(layers_arr, means, "o-", color="darkorange", lw=2, ms=5, label="CLIP ViT-B/16")
        ax.fill_between(layers_arr, means - stds, means + stds, alpha=0.15, color="darkorange")
        ax.axhline(bl_summary["center_gaussian"][mname], color="steelblue", ls="--", lw=1.5, label="Center Gaussian")
        ax.axhline(bl_summary["uniform"][mname], color="gray", ls=":", lw=1.5, label="Uniform")
        ax.set_xlabel("Layer"); ax.set_ylabel(mname); ax.set_title(mname)
        ax.set_xticks(layers_arr); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Exp-A (CLIP): CLIP ViT-B/16 vs Human Fixations ({GROUP}, full700)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FINAL_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {FINAL_PLOT}")

    # ------------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------------
    steady = timings[1:]
    n_steady = len(steady)
    sum_io = sum(t["t_io_s"] for t in steady)
    sum_infer = sum(t["t_infer_s"] for t in steady)
    sum_metrics = sum(t["t_metrics_s"] for t in steady)
    sum_total = sum(t["t_total_s"] for t in steady)
    avg_total_per_image = sum_total / n_steady

    timing_summary = {
        "n_images": N_EXPECTED,
        "n_images_steady_state": n_steady,
        "warmup_image": timings[0]["image"],
        "warmup_total_s": timings[0]["t_total_s"],
        "wall_clock_total_s": wall_elapsed_s,
        "steady_state": {
            "sum_io_s": sum_io, "sum_infer_s": sum_infer,
            "sum_metrics_s": sum_metrics, "sum_total_s": sum_total,
            "avg_io_s": sum_io / n_steady, "avg_infer_s": sum_infer / n_steady,
            "avg_metrics_s": sum_metrics / n_steady, "avg_total_s": avg_total_per_image,
        },
        "peak_vram_mib": peak_mem_mib,
    }
    with open(FINAL_TIMING, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)
    print(f"\n  Saved: {FINAL_TIMING}")
    ss = timing_summary["steady_state"]
    print(f"  Wall-clock total: {wall_elapsed_s:.1f}s ({wall_elapsed_s / 60:.1f} min)")
    print(f"  Steady-state avg/image (n={n_steady}): io={ss['avg_io_s'] * 1000:.1f}ms "
          f"infer={ss['avg_infer_s'] * 1000:.1f}ms metrics={ss['avg_metrics_s'] * 1000:.1f}ms "
          f"total={avg_total_per_image * 1000:.1f}ms")
    print(f"  Peak VRAM: {peak_mem_mib:.1f} MiB" if peak_mem_mib is not None else "  Peak VRAM: n/a")
    if wall_elapsed_s > 30 * 60:
        print("  NOTE: wall-clock exceeded 30 minutes -- see per-phase breakdown above for the bottleneck.")

    # ------------------------------------------------------------------
    # Final run_config.json
    # ------------------------------------------------------------------
    run_config = {
        "model": MODEL, "layers": LAYERS, "attn": ATTN, "head_agg": HEAD_AGG,
        "group": GROUP, "img_w": IMG_W, "img_h": IMG_H, "upsample": UPSAMPLE,
        "metrics": METRICS_LIST, "seed": SEED, "batch_size": 1,
        "n_images": N_EXPECTED, "image_ids": image_ids,
        "sauc_negative_pool": {
            "source": "outputs/fixmaps/healthy (all 700 images)",
            "n_points": int(len(all_pool)), "self_image_excluded": False,
            "note": "matches scripts/run_expA.py: one shared all_pool for every image",
        },
        "positive_controls": bl_summary,
        "peak_layer": {m: {"layer": peak_layer[m][0], "value": peak_layer[m][1]} for m in METRICS_LIST},
        "n_rows": len(rows),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(FINAL_CONFIG, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"  Saved: {FINAL_CONFIG}")

    print("\n" + "=" * 65)
    print("  Exp A (CLIP full700): DONE")
    print("  outputs/expA_clip/pilot20/, outputs/expA/, outputs/attn_cache/, "
          "results/ were NOT touched.")
    print("=" * 65)


if __name__ == "__main__":
    main()
