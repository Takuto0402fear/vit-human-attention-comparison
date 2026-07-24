"""
Experiment A (SL ViT-S/16 pilot20): official Yamamoto-paper SL (DeiT-
trained, standard/no-distillation-token) ViT-S/16 layerwise [CLS]->patch
attention vs. human fixations (OSIE, healthy) -- 20-image pilot, trial 01,
depth=12, ahead of the 6-trial x 700-image production run.

Same evaluation convention as scripts/run_expA.py (DINO-S),
scripts/run_expA_clip*.py (CLIP-B), and scripts/run_expA_dino_vitb16*.py
(DINO-B): bilinear upsample to 800x600, metrics.py's nss/auc_judd/sauc
imported unmodified, identical fixation-coordinate extraction, identical
center-Gaussian/uniform-noise positive controls, identical sAUC
negative-pool convention (one shared 700-image pool, no self-image
exclusion, metrics.py's internal RandomState(0) untouched). None of
run_expA.py, run_expA_clip*.py, run_expA_dino_vitb16*.py, vit_extractor.py,
dino_vitb16_extractor.py, clip_extractor.py, sl_extractor.py, or
lib/vision_transformer.py is modified by this script.

Attention extraction uses ONLY sl_extractor.py's
extract_layerwise_cls_attention_memory_conscious (re-exported from
dino_vitb16_extractor.py, architecture-generic, unmodified).

Image selection: the SAME 20 images as the existing CLIP/DINO-B pilots
(outputs/expA_clip/pilot20/run_config.json's pilot_image_ids, read
read-only) -- 1001..1020.

Resumability: uses the SAME done-set/append-rows pattern as
scripts/run_expA.py (not the guard-only pattern of the *_full700.py
scripts) -- re-running this script with metrics_per_image.csv already
complete skips extraction entirely and re-derives the layer summary from
the existing CSV, so it can be safely re-run without duplicating rows.
This is deliberately exercised at the end of main() (an explicit second
call) so the pilot itself demonstrates the property it will rely on
during the (potentially long, thermally-throttled) 6x700 production run.

Output confined to outputs/expA_sl/pilot20/. Does not write to
outputs/expA/, outputs/expA_clip/, outputs/expA_dino_vitb16/,
outputs/expA_sl/validation/, outputs/expA_sl/trial_*/,
outputs/expA_sl/combined/, outputs/attn_cache/, or results/.

Usage (PowerShell):
    python scripts\\run_expA_sl_pilot20.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import gc
import glob
import hashlib
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
from vit_extractor import _pad_to_patch  # read-only reuse
from lib.clip_vit import patch_grid_from_image_hw  # read-only reuse; generic grid arithmetic
from sl_extractor import (
    SL_PATCH_SIZE, checkpoint_path, extract_layerwise_cls_attention_memory_conscious,
    load_sl_vit,
)

# ======================= CONFIG =======================
MODEL        = "sl_vits16"
TRIAL_NUM    = 1
DEPTH        = 12
LAYERS       = list(range(1, 13))
ATTN         = "cls2patch"
HEAD_AGG     = "mean"
GROUP        = "healthy"
IMG_W        = 800
IMG_H        = 600
UPSAMPLE     = "bilinear"
METRICS_LIST = ["NSS", "AUC_Judd", "sAUC"]
SEED         = 42
N_IMAGES     = 20
EXPECTED_GRID = (38, 50)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR    = os.path.join(DATA_BASE, "data", "stimuli")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
DINO_S_METRICS_CSV = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"
CLIP_B_PILOT20_CONFIG = r"C:\Users\user\gaze\outputs\expA_clip\pilot20\run_config.json"

OUT_DIR = r"C:\Users\user\gaze\outputs\expA_sl\pilot20"
CSV_FIELDS = ["image", "layer", "relative_depth"] + METRICS_LIST
# ======================================================


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """Same 20 images as the existing CLIP/DINO-B pilots -- read-only."""
    with open(CLIP_B_PILOT20_CONFIG, encoding="utf-8") as f:
        clip_config = json.load(f)
    ids = clip_config["pilot_image_ids"]
    if len(ids) != 20:
        raise RuntimeError(f"STOP: CLIP pilot20 image list has {len(ids)} images, expected 20")
    return ids


def load_fixation_data():
    """Identical extraction logic to scripts/run_expA.py / run_expA_clip*.py
    / run_expA_dino_vitb16*.py (duplicated, not imported -- matches this
    repo's per-script convention)."""
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


def load_osie_image_tensor(image_path, patch_size=SL_PATCH_SIZE):
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
    t0 = time.perf_counter()
    tensor, orig_hw, pad_hw = load_osie_image_tensor(image_path, SL_PATCH_SIZE)
    tensor = tensor.to(device)
    grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], SL_PATCH_SIZE)
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

    stacked = torch.stack(layer_maps, dim=1)
    resized = F.interpolate(
        stacked.reshape(DEPTH, 1, *grid_hw), size=(IMG_H, IMG_W),
        mode="bilinear", align_corners=False)
    resized = resized[:, 0].detach().cpu().numpy().astype(np.float32)

    for l in range(resized.shape[0]):
        s = resized[l].sum()
        if s > 0:
            resized[l] /= s

    row_sums = torch.stack(full_row_sums).squeeze(-1).detach().cpu().numpy()
    return resized, grid_hw, t_io, t_infer, row_sums


def _csv_path():
    return os.path.join(OUT_DIR, "metrics_per_image.csv")


def _load_done_set(csv_path):
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["image"], int(row["layer"])))
    return done


def _append_rows(csv_path, rows):
    write_header = (not os.path.isfile(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def _load_all_records(csv_path):
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "image": row["image"], "layer": int(row["layer"]),
                "relative_depth": float(row["relative_depth"]),
                "NSS": float(row["NSS"]), "AUC_Judd": float(row["AUC_Judd"]),
                "sAUC": float(row["sAUC"]),
            })
    return records


def run_extraction_pass(model, device, pilot_ids, fix_points, all_pool):
    """One extraction pass: skips (image,layer) pairs already present in
    the CSV. Returns (n_newly_written, n_total_in_csv_after)."""
    csv_img = _csv_path()
    done = _load_done_set(csv_img)
    n_before = len(done)

    rows_written = 0
    for img_id in pilot_ids:
        img_name = f"{img_id}.jpg"
        if all((img_name, l + 1) in done for l in range(DEPTH)):
            continue  # this image fully done -- skip re-extraction entirely

        img_path = os.path.join(STIM_DIR, img_name)
        pts = fix_points[img_id]
        resized, grid_hw, t_io, t_infer, row_sums = extract_and_resize(model, img_path, device)

        if grid_hw != EXPECTED_GRID:
            raise RuntimeError(f"STOP: {img_id} grid {grid_hw} != {EXPECTED_GRID}")
        if np.isnan(resized).any() or np.isinf(resized).any():
            raise RuntimeError(f"STOP: NaN/Inf in {img_id} attention maps")
        if np.isnan(row_sums).any() or np.isinf(row_sums).any() or np.any(np.abs(row_sums - 1.0) > 1e-2):
            raise RuntimeError(f"STOP: {img_id} CLS-row-sum out of range: {row_sums}")

        img_rows = []
        for l in range(DEPTH):
            layer = l + 1
            key = (img_name, layer)
            if key in done:
                continue
            sal = resized[l]
            row = {
                "image": img_name, "layer": layer,
                "relative_depth": (layer - 1) / (DEPTH - 1),
                "NSS": nss(sal, pts), "AUC_Judd": auc_judd(sal, pts),
                "sAUC": sauc(sal, pts, all_pool),
            }
            for m in METRICS_LIST:
                if not np.isfinite(row[m]):
                    raise RuntimeError(f"STOP: non-finite {m} for {img_id} layer {layer}: {row[m]}")
            if not (0.0 <= row["AUC_Judd"] <= 1.0):
                raise RuntimeError(f"STOP: AUC_Judd out of [0,1] for {img_id} layer {layer}")
            if not (0.0 <= row["sAUC"] <= 1.0):
                raise RuntimeError(f"STOP: sAUC out of [0,1] for {img_id} layer {layer}")
            img_rows.append(row)
            done.add(key)

        if img_rows:
            _append_rows(csv_img, img_rows)
            rows_written += len(img_rows)

        del resized
        gc.collect()

    return rows_written, len(done)


def save_attention_fixation_overlays(model, device, pilot_ids, fix_points, sample_ids):
    """Saves original image + attention-map(final layer) overlay + human
    fixation-point scatter, for a handful of pilot images, so coordinate
    alignment can be visually confirmed."""
    from PIL import Image

    os.makedirs(OUT_DIR, exist_ok=True)
    saved = []
    for img_id in sample_ids:
        img_path = os.path.join(STIM_DIR, f"{img_id}.jpg")
        resized, grid_hw, _, _, _ = extract_and_resize(model, img_path, device)
        heat = resized[-1]  # final layer

        img = Image.open(img_path).convert("RGB")
        pts = fix_points[img_id]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        axes[0].imshow(img)
        axes[0].scatter(pts[:, 0], pts[:, 1], s=28, facecolors="none", edgecolors="lime", linewidths=1.5)
        axes[0].set_title(f"{img_id}.jpg  human fixations (n={len(pts)})")
        axes[0].axis("off")

        h = heat.copy()
        if h.max() > 0:
            h = h / h.max()
        axes[1].imshow(img)
        axes[1].imshow(h, cmap="jet", alpha=0.45)
        axes[1].scatter(pts[:, 0], pts[:, 1], s=28, facecolors="none", edgecolors="lime", linewidths=1.5)
        axes[1].set_title(f"SL ViT-S/16 L{DEPTH} attention + fixations overlay")
        axes[1].axis("off")
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"overlay_{img_id}.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        saved.append(out_path)
    return saved


def read_layer_means_for_images(csv_path, image_ids, layers=range(1, 13)):
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
    return means


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  Exp A (SL ViT-S/16 pilot20): SL vs Human Fixations")
    print("=" * 65)

    ckpt_path = checkpoint_path(TRIAL_NUM, DEPTH)
    ckpt_hash = _md5(ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"  Checkpoint MD5: {ckpt_hash}")

    print("\n--- Validating image-ID sets (stimuli / fixmaps / DINO-S metrics) ---")
    common_700 = validate_image_id_sets()
    print(f"  Intersection: {len(common_700)} images (expected 700)  OK")

    pilot_ids = load_pilot_image_ids_from_clip()
    print(f"\n--- Image selection (pilot20, same as CLIP/DINO-B pilots) ---")
    print(f"  Images: {pilot_ids}")
    for img_id in pilot_ids:
        if img_id not in common_700:
            raise RuntimeError(f"STOP: pilot image {img_id} missing from the 700-image intersection")
    if len(pilot_ids) != N_IMAGES:
        raise RuntimeError(f"STOP: {len(pilot_ids)} images selected, expected {N_IMAGES}")

    print(f"\n--- Loading SL ViT-S/16 (trial={TRIAL_NUM}, depth={DEPTH}, strict=True) ---")
    model, device, loaded_ckpt_path = load_sl_vit(TRIAL_NUM, DEPTH)
    assert loaded_ckpt_path == ckpt_path
    print(f"  Device: {device}  dtype: {next(model.parameters()).dtype}")

    print("\n--- Loading fixation data (700-image pool for sAUC) ---")
    fix_points, all_pool = load_fixation_data()
    print(f"  {len(fix_points)} images with fixations, {len(all_pool)} pooled points")
    if len(fix_points) != 700:
        raise RuntimeError(f"STOP: expected 700 fixmap images, got {len(fix_points)}")

    # ------------------------------------------------------------------
    # Positive controls
    # ------------------------------------------------------------------
    print("\n--- Positive control: center-Gaussian vs uniform-noise ---")
    cg_map = center_gaussian(IMG_H, IMG_W)
    uf_map = uniform_noise(IMG_H, IMG_W)
    bl = {"center_gaussian": [], "uniform": []}
    for img_id in pilot_ids:
        pts = fix_points[img_id]
        for bname, bmap in [("center_gaussian", cg_map), ("uniform", uf_map)]:
            bl[bname].append({
                "NSS": nss(bmap, pts), "AUC_Judd": auc_judd(bmap, pts), "sAUC": sauc(bmap, pts, all_pool),
            })
    bl_summary = {bn: {m: float(np.nanmean([r[m] for r in recs])) for m in METRICS_LIST}
                  for bn, recs in bl.items()}
    for bname, vals in bl_summary.items():
        print(f"  {bname:16s}  NSS={vals['NSS']:.4f}  AUC={vals['AUC_Judd']:.4f}  sAUC={vals['sAUC']:.4f}")
    uniform_near_half = (0.4 <= bl_summary["uniform"]["AUC_Judd"] <= 0.6 and
                          0.4 <= bl_summary["uniform"]["sAUC"] <= 0.6)
    center_beats_uniform = (bl_summary["center_gaussian"]["NSS"] > bl_summary["uniform"]["NSS"] and
                             bl_summary["center_gaussian"]["AUC_Judd"] > bl_summary["uniform"]["AUC_Judd"])
    print(f"  Uniform AUC/sAUC ~0.5 (0.4-0.6 band): {'PASS' if uniform_near_half else 'FAIL'}")
    print(f"  Center Gaussian > Uniform (NSS & AUC): {'PASS' if center_beats_uniform else 'FAIL'}")
    if not uniform_near_half or not center_beats_uniform:
        raise RuntimeError("STOP: positive control failed")

    # ------------------------------------------------------------------
    # Pass 1: extraction + metrics (resumable)
    # ------------------------------------------------------------------
    print(f"\n--- Pass 1: extracting SL attention + computing metrics ({N_IMAGES} images) ---")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_pass1_0 = time.perf_counter()
    n_written_1, n_total_1 = run_extraction_pass(model, device, pilot_ids, fix_points, all_pool)
    t_pass1 = time.perf_counter() - t_pass1_0
    print(f"  Pass 1: wrote {n_written_1} new rows, {n_total_1} total rows in CSV, {t_pass1:.1f}s")

    # ------------------------------------------------------------------
    # Resume-safety check: re-run the SAME extraction pass. Must write 0
    # new rows and leave the total unchanged (no duplicate append).
    # ------------------------------------------------------------------
    print(f"\n--- Pass 2 (resume-safety check): re-running the identical extraction ---")
    t_pass2_0 = time.perf_counter()
    n_written_2, n_total_2 = run_extraction_pass(model, device, pilot_ids, fix_points, all_pool)
    t_pass2 = time.perf_counter() - t_pass2_0
    print(f"  Pass 2: wrote {n_written_2} new rows, {n_total_2} total rows in CSV, {t_pass2:.1f}s")
    if n_written_2 != 0:
        raise RuntimeError(f"STOP: resume-safety check FAILED -- pass 2 wrote {n_written_2} new rows (expected 0)")
    if n_total_2 != n_total_1:
        raise RuntimeError(f"STOP: resume-safety check FAILED -- row count changed {n_total_1} -> {n_total_2}")
    print(f"  Resume-safety check: PASS (0 new rows written, {n_total_2} rows unchanged, "
          f"pass2 time {t_pass2:.2f}s << pass1 {t_pass1:.2f}s since all images were skipped)")

    # ------------------------------------------------------------------
    # Verification against the mechanical pass conditions
    # ------------------------------------------------------------------
    print("\n--- Verification ---")
    records = _load_all_records(_csv_path())
    if len(records) != N_IMAGES * DEPTH:
        raise RuntimeError(f"STOP: expected {N_IMAGES * DEPTH} rows, got {len(records)}")
    seen_pairs = set((r["image"], r["layer"]) for r in records)
    if len(seen_pairs) != len(records):
        raise RuntimeError("STOP: duplicate (image,layer) pairs found in CSV")
    row_images = set(r["image"] for r in records)
    if len(row_images) != N_IMAGES:
        raise RuntimeError(f"STOP: {len(row_images)} unique images in output, expected {N_IMAGES}")
    per_image_layers = {}
    for r in records:
        per_image_layers.setdefault(r["image"], set()).add(r["layer"])
    bad = {img: l for img, l in per_image_layers.items() if l != set(range(1, DEPTH + 1))}
    if bad:
        raise RuntimeError(f"STOP: images without exactly layers 1-{DEPTH}: {bad}")
    for r in records:
        for m in METRICS_LIST:
            if not np.isfinite(r[m]):
                raise RuntimeError(f"STOP: non-finite {m} in {r['image']} layer {r['layer']}")
        if not (0.0 <= r["AUC_Judd"] <= 1.0):
            raise RuntimeError(f"STOP: AUC_Judd out of [0,1]: {r}")
        if not (0.0 <= r["sAUC"] <= 1.0):
            raise RuntimeError(f"STOP: sAUC out of [0,1]: {r}")
    print(f"  rows: {len(records)}  (expected {N_IMAGES * DEPTH})  OK")
    print(f"  unique (image,layer) pairs: {len(seen_pairs)}  (no duplicates)  OK")
    print(f"  unique images: {len(row_images)}  (expected {N_IMAGES})  OK")
    print(f"  every image has layers 1-{DEPTH} exactly once: OK")
    print(f"  NSS/AUC_Judd/sAUC: no NaN/Inf, AUC_Judd & sAUC in [0,1]: OK")

    # ------------------------------------------------------------------
    # Attention + fixation overlay images (coordinate-alignment check)
    # ------------------------------------------------------------------
    print("\n--- Saving attention+fixation overlay images (coordinate check) ---")
    sample_ids = pilot_ids[:3]
    overlay_paths = save_attention_fixation_overlays(model, device, pilot_ids, fix_points, sample_ids)
    for p in overlay_paths:
        print(f"  Saved: {p}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # Layer summary
    # ------------------------------------------------------------------
    layer_agg = {l: {m: [] for m in METRICS_LIST} for l in range(1, DEPTH + 1)}
    for r in records:
        for m in METRICS_LIST:
            layer_agg[r["layer"]][m].append(r[m])

    layer_rows = []
    for l in range(1, DEPTH + 1):
        row = {"layer": l, "relative_depth": (l - 1) / (DEPTH - 1)}
        for m in METRICS_LIST:
            vals = layer_agg[l][m]
            if len(vals) != N_IMAGES:
                raise RuntimeError(f"STOP: layer {l} metric {m} has N={len(vals)}, expected {N_IMAGES}")
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals))
        layer_rows.append(row)

    csv_layer = os.path.join(OUT_DIR, "metrics_by_layer.csv")
    fieldnames = ["layer", "relative_depth"] + [f"{m}_{s}" for m in METRICS_LIST for s in ("mean", "std")]
    with open(csv_layer, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in layer_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\n  Saved: {csv_layer}  ({len(layer_rows)} layers, N={N_IMAGES} each)")

    print(f"\n=== metrics_by_layer (SL ViT-S/16, pilot20) ===")
    print(f"{'layer':>5} {'rel_d':>6}  {'NSS':>8}  {'AUC_Judd':>10}  {'sAUC':>8}")
    for row in layer_rows:
        print(f"{row['layer']:5d} {row['relative_depth']:6.3f}  {row['NSS_mean']:8.4f}  "
              f"{row['AUC_Judd_mean']:10.4f}  {row['sAUC_mean']:8.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_defs = [("NSS", "NSS_mean", "NSS_std"), ("AUC_Judd", "AUC_Judd_mean", "AUC_Judd_std"),
                 ("sAUC", "sAUC_mean", "sAUC_std")]
    layers_arr = np.arange(1, DEPTH + 1)
    for ax, (mname, mc, sc) in zip(axes, plot_defs):
        means = np.array([layer_rows[l][mc] for l in range(DEPTH)])
        stds = np.array([layer_rows[l][sc] for l in range(DEPTH)])
        ax.plot(layers_arr, means, "o-", color="firebrick", lw=2, ms=5, label="SL ViT-S/16 (trial01)")
        ax.fill_between(layers_arr, means - stds, means + stds, alpha=0.15, color="firebrick")
        ax.axhline(bl_summary["center_gaussian"][mname], color="steelblue", ls="--", lw=1.5, label="Center Gaussian")
        ax.axhline(bl_summary["uniform"][mname], color="gray", ls=":", lw=1.5, label="Uniform")
        ax.set_xlabel("Layer")
        ax.set_ylabel(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_title(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_xticks(layers_arr)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-A (SL ViT-S/16, trial01): SL vs Human Fixations -- Pilot N=20", fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "pilot20_layerwise.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {plot_path}")

    # ------------------------------------------------------------------
    # run_config.json
    # ------------------------------------------------------------------
    run_config = {
        "model": MODEL, "trial_num": TRIAL_NUM, "depth": DEPTH,
        "checkpoint_path": ckpt_path, "checkpoint_md5": ckpt_hash,
        "layers": LAYERS, "attn": ATTN, "head_agg": HEAD_AGG,
        "group": GROUP, "img_w": IMG_W, "img_h": IMG_H, "upsample": UPSAMPLE,
        "metrics": METRICS_LIST, "seed": SEED, "batch_size": 1,
        "n_images": N_IMAGES, "pilot_image_ids": pilot_ids,
        "pilot_selection_method": (
            "same 20 images as outputs/expA_clip/pilot20 / outputs/expA_dino_vitb16/pilot20 "
            "run_config.json's pilot_image_ids (read read-only)"),
        "sauc_negative_pool": {
            "source": "outputs/fixmaps/healthy (all 700 images)",
            "n_points": int(len(all_pool)), "self_image_excluded": False,
        },
        "positive_controls": bl_summary,
        "mechanical_checks": {
            "n_rows": len(records), "n_rows_expected": N_IMAGES * DEPTH,
            "duplicate_pairs": 0, "missing_images_or_layers": 0,
            "nan_inf_in_metrics": 0, "auc_judd_in_range": True, "sauc_in_range": True,
            "resume_safety_pass2_new_rows": n_written_2,
        },
        "n_rows": len(records),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(OUT_DIR, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"  Saved: {config_path}")

    print("\n" + "=" * 65)
    print("  ALL PILOT20 MECHANICAL CHECKS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
