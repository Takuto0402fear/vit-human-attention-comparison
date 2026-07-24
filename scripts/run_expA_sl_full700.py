"""
Experiment A (SL ViT-S/16 full700): official Yamamoto-paper SL (DeiT-
trained, standard/no-distillation-token) ViT-S/16 layerwise [CLS]->patch
attention vs. human fixations (OSIE, healthy) -- full 700-image
production run, looped over all 6 official trials
(supervised/{01..06}/12layers/checkpoint.pth).

Same evaluation convention as scripts/run_expA.py (DINO-S),
scripts/run_expA_clip_full700.py (CLIP-B), and
scripts/run_expA_dino_vitb16_full700.py (DINO-B): bilinear upsample to
800x600, metrics.py's nss/auc_judd/sauc imported unmodified, identical
fixation-coordinate extraction, identical center-Gaussian/uniform-noise
positive controls, identical sAUC negative-pool convention (one shared
700-image pool, no self-image exclusion, metrics.py's internal
RandomState(0) untouched). Same 700 images as the existing DINO-S/
DINO-B/CLIP-B production runs (derived the same way: stimuli inter
fixmaps inter DINO-S's own metrics_per_image.csv image list).

Resumability: unlike run_expA_clip_full700.py / run_expA_dino_vitb16_
full700.py (guard-only, no auto-resume), this script uses the same
done-set/append-rows pattern already exercised and verified in
scripts/run_expA_sl_pilot20.py's resume-safety check -- each trial's
metrics_per_image.csv is the resume ledger: already-done (image,layer)
pairs are skipped, new rows are appended and fsync'd immediately after
each image. Safe to interrupt (thermal throttling, etc.) and re-run
unchanged; never duplicates rows. A trial already fully complete (700x12
rows present) is skipped entirely and only re-aggregated.

4-layer/8-layer checkpoints are NOT used here (out of scope per this
run's spec) -- only supervised/{trial:02d}/12layers/checkpoint.pth.

Output: one subdirectory per trial, outputs/expA_sl/trial_{01..06}/,
each with metrics_per_image.csv, metrics_by_layer.csv, run_config.json
(checkpoint path + MD5, timestamp, n images, mechanical-check summary),
and layerwise_agreement.png. Does not touch outputs/expA/,
outputs/expA_clip/, outputs/expA_dino_vitb16/, outputs/expA_sl/pilot20/,
outputs/expA_sl/validation/, outputs/attn_cache/, or results/.

Usage (PowerShell):
    python scripts\\run_expA_sl_full700.py            # all 6 trials
    python scripts\\run_expA_sl_full700.py 1          # trial 1 only
    python scripts\\run_expA_sl_full700.py 1 2 3      # trials 1-3 only
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
from lib.clip_vit import patch_grid_from_image_hw  # read-only reuse
from sl_extractor import (
    SL_PATCH_SIZE, checkpoint_path, extract_layerwise_cls_attention_memory_conscious,
    load_sl_vit,
)

# ======================= CONFIG =======================
MODEL        = "sl_vits16"
ALL_TRIALS   = [1, 2, 3, 4, 5, 6]
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
N_EXPECTED   = 700
EXPECTED_GRID = (38, 50)
PROGRESS_EVERY = 50

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR    = os.path.join(DATA_BASE, "data", "stimuli")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
DINO_S_METRICS_CSV = r"C:\Users\user\gaze\outputs\expA\healthy\metrics_per_image.csv"

OUT_DIR_ROOT = r"C:\Users\user\gaze\outputs\expA_sl"
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
    if len(common) != N_EXPECTED:
        raise RuntimeError(
            f"STOP: expected exactly {N_EXPECTED} common images, got {len(common)}. "
            f"stim={len(stim_ids)} fixmap={len(fixmap_ids)} dino_s={len(dino_s_ids)}")
    return common


def load_fixation_data():
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


def compute_positive_controls(image_ids, fix_points, all_pool):
    cg_map = center_gaussian(IMG_H, IMG_W)
    uf_map = uniform_noise(IMG_H, IMG_W)
    bl = {"center_gaussian": [], "uniform": []}
    for img_id in image_ids:
        pts = fix_points[img_id]
        for bname, bmap in [("center_gaussian", cg_map), ("uniform", uf_map)]:
            bl[bname].append({
                "NSS": nss(bmap, pts), "AUC_Judd": auc_judd(bmap, pts), "sAUC": sauc(bmap, pts, all_pool),
            })
    return {bn: {m: float(np.nanmean([r[m] for r in recs])) for m in METRICS_LIST}
            for bn, recs in bl.items()}


def run_trial(trial_num, image_ids, fix_points, all_pool, bl_summary):
    out_dir = os.path.join(OUT_DIR_ROOT, f"trial_{trial_num:02d}")
    os.makedirs(out_dir, exist_ok=True)
    csv_img = os.path.join(out_dir, "metrics_per_image.csv")

    ckpt_path = checkpoint_path(trial_num, DEPTH)
    ckpt_hash = _md5(ckpt_path)

    print("\n" + "=" * 65)
    print(f"  Trial {trial_num:02d}: {ckpt_path}")
    print(f"  Checkpoint MD5: {ckpt_hash}")
    print("=" * 65)

    done = _load_done_set(csv_img)
    n_done_start = len(done)
    if n_done_start == N_EXPECTED * DEPTH:
        print(f"  Trial {trial_num:02d} already fully complete ({n_done_start} rows) -- skipping extraction.")
    else:
        print(f"  Resuming: {n_done_start}/{N_EXPECTED * DEPTH} rows already present.")

        model, device, loaded_ckpt_path = load_sl_vit(trial_num, DEPTH)
        assert loaded_ckpt_path == ckpt_path
        print(f"  Device: {device}  dtype: {next(model.parameters()).dtype}")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t_loop0 = time.perf_counter()
        n_images_done = sum(1 for img_id in image_ids
                             if all((f"{img_id}.jpg", l + 1) in done for l in range(DEPTH)))

        for idx, img_id in enumerate(image_ids):
            img_name = f"{img_id}.jpg"
            if all((img_name, l + 1) in done for l in range(DEPTH)):
                continue  # fully done -- skip re-extraction

            img_path = os.path.join(STIM_DIR, img_name)
            pts = fix_points[img_id]
            resized, grid_hw, t_io, t_infer, row_sums = extract_and_resize(model, img_path, device)

            if grid_hw != EXPECTED_GRID:
                raise RuntimeError(f"STOP [trial {trial_num:02d}]: {img_id} grid {grid_hw} != {EXPECTED_GRID}")
            if np.isnan(resized).any() or np.isinf(resized).any():
                raise RuntimeError(f"STOP [trial {trial_num:02d}]: NaN/Inf in {img_id} attention maps")
            if np.isnan(row_sums).any() or np.isinf(row_sums).any() or np.any(np.abs(row_sums - 1.0) > 1e-2):
                raise RuntimeError(f"STOP [trial {trial_num:02d}]: {img_id} CLS-row-sum out of range: {row_sums}")

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
                        raise RuntimeError(
                            f"STOP [trial {trial_num:02d}]: non-finite {m} for {img_id} layer {layer}: {row[m]}")
                if not (0.0 <= row["AUC_Judd"] <= 1.0):
                    raise RuntimeError(
                        f"STOP [trial {trial_num:02d}]: AUC_Judd out of [0,1] for {img_id} layer {layer}: "
                        f"{row['AUC_Judd']}")
                if not (0.0 <= row["sAUC"] <= 1.0):
                    raise RuntimeError(
                        f"STOP [trial {trial_num:02d}]: sAUC out of [0,1] for {img_id} layer {layer}: {row['sAUC']}")
                img_rows.append(row)
                done.add(key)

            if img_rows:
                _append_rows(csv_img, img_rows)

            n_images_done += 1
            del resized
            gc.collect()

            if n_images_done % PROGRESS_EVERY == 0 or n_images_done == N_EXPECTED:
                elapsed = time.perf_counter() - t_loop0
                done_this_run = n_images_done - n_done_start // DEPTH
                avg = elapsed / max(done_this_run, 1)
                remaining = N_EXPECTED - n_images_done
                eta = avg * remaining
                print(f"  [trial {trial_num:02d}] [{n_images_done:4d}/{N_EXPECTED}] "
                      f"elapsed={elapsed:7.1f}s  avg/img={avg * 1000:6.1f}ms  ETA={eta:6.1f}s")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Verification (hard stop on any anomaly)
    # ------------------------------------------------------------------
    records = _load_all_records(csv_img)
    if len(records) != N_EXPECTED * DEPTH:
        raise RuntimeError(f"STOP [trial {trial_num:02d}]: expected {N_EXPECTED * DEPTH} rows, got {len(records)}")
    seen_pairs = set((r["image"], r["layer"]) for r in records)
    if len(seen_pairs) != len(records):
        raise RuntimeError(f"STOP [trial {trial_num:02d}]: duplicate (image,layer) pairs found")
    row_images = set(r["image"] for r in records)
    if len(row_images) != N_EXPECTED:
        raise RuntimeError(f"STOP [trial {trial_num:02d}]: {len(row_images)} unique images, expected {N_EXPECTED}")
    per_image_layers = {}
    for r in records:
        per_image_layers.setdefault(r["image"], set()).add(r["layer"])
    bad = {img: l for img, l in per_image_layers.items() if l != set(range(1, DEPTH + 1))}
    if bad:
        raise RuntimeError(f"STOP [trial {trial_num:02d}]: images without exactly layers 1-{DEPTH}: "
                            f"{list(bad.items())[:5]}")
    n_nan_inf = 0
    n_range_bad = 0
    bad_examples = []
    for r in records:
        for m in METRICS_LIST:
            if not np.isfinite(r[m]):
                n_nan_inf += 1
                bad_examples.append((r["image"], r["layer"], m, r[m]))
        if not (0.0 <= r["AUC_Judd"] <= 1.0) or not (0.0 <= r["sAUC"] <= 1.0):
            n_range_bad += 1
            bad_examples.append((r["image"], r["layer"], "range", (r["AUC_Judd"], r["sAUC"])))
    if n_nan_inf or n_range_bad:
        raise RuntimeError(f"STOP [trial {trial_num:02d}]: {n_nan_inf} NaN/Inf, {n_range_bad} out-of-range "
                            f"values. Examples: {bad_examples[:10]}")

    print(f"  [trial {trial_num:02d}] rows: {len(records)}  (expected {N_EXPECTED * DEPTH})  OK")
    print(f"  [trial {trial_num:02d}] unique images: {len(row_images)}  (expected {N_EXPECTED})  OK")
    print(f"  [trial {trial_num:02d}] every image has layers 1-{DEPTH} exactly once: OK")
    print(f"  [trial {trial_num:02d}] NaN/Inf: 0, out-of-range: 0  OK")

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
            if len(vals) != N_EXPECTED:
                raise RuntimeError(f"STOP [trial {trial_num:02d}]: layer {l} metric {m} has N={len(vals)}")
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals))
        layer_rows.append(row)

    csv_layer = os.path.join(out_dir, "metrics_by_layer.csv")
    fieldnames = ["layer", "relative_depth"] + [f"{m}_{s}" for m in METRICS_LIST for s in ("mean", "std")]
    with open(csv_layer, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in layer_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  [trial {trial_num:02d}] Saved: {csv_layer}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_defs = [("NSS", "NSS_mean", "NSS_std"), ("AUC_Judd", "AUC_Judd_mean", "AUC_Judd_std"),
                 ("sAUC", "sAUC_mean", "sAUC_std")]
    layers_arr = np.arange(1, DEPTH + 1)
    for ax, (mname, mc, sc) in zip(axes, plot_defs):
        means = np.array([layer_rows[l][mc] for l in range(DEPTH)])
        stds = np.array([layer_rows[l][sc] for l in range(DEPTH)])
        ax.plot(layers_arr, means, "o-", color="firebrick", lw=2, ms=5, label=f"SL ViT-S/16 (trial{trial_num:02d})")
        ax.fill_between(layers_arr, means - stds, means + stds, alpha=0.15, color="firebrick")
        ax.axhline(bl_summary["center_gaussian"][mname], color="steelblue", ls="--", lw=1.5, label="Center Gaussian")
        ax.axhline(bl_summary["uniform"][mname], color="gray", ls=":", lw=1.5, label="Uniform")
        ax.set_xlabel("Layer")
        ax.set_ylabel(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_title(mname if mname != "AUC_Judd" else "AUC-Judd")
        ax.set_xticks(layers_arr)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Exp-A (SL ViT-S/16, trial{trial_num:02d}): SL vs Human Fixations (full700)", fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "layerwise_agreement.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [trial {trial_num:02d}] Saved: {plot_path}")

    # run_config.json
    run_config = {
        "model": MODEL, "trial_num": trial_num, "depth": DEPTH,
        "checkpoint_path": ckpt_path, "checkpoint_md5": ckpt_hash,
        "layers": LAYERS, "attn": ATTN, "head_agg": HEAD_AGG,
        "group": GROUP, "img_w": IMG_W, "img_h": IMG_H, "upsample": UPSAMPLE,
        "metrics": METRICS_LIST, "seed": SEED, "batch_size": 1,
        "n_images": N_EXPECTED, "image_ids": image_ids,
        "sauc_negative_pool": {
            "source": "outputs/fixmaps/healthy (all 700 images)",
            "n_points": int(len(all_pool)), "self_image_excluded": False,
        },
        "positive_controls": bl_summary,
        "mechanical_checks": {
            "n_rows": len(records), "n_rows_expected": N_EXPECTED * DEPTH,
            "duplicate_pairs": len(records) - len(seen_pairs),
            "missing_images_or_layers": len(bad), "nan_inf_in_metrics": n_nan_inf,
            "out_of_range": n_range_bad,
        },
        "n_rows": len(records),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(out_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"  [trial {trial_num:02d}] Saved: {config_path}")
    print(f"  [trial {trial_num:02d}] DONE")


def main():
    trials = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else ALL_TRIALS
    for t in trials:
        if t not in ALL_TRIALS:
            raise ValueError(f"unknown trial {t}, expected one of {ALL_TRIALS}")

    print("=" * 65)
    print(f"  Exp A (SL ViT-S/16 full700): trials {trials}")
    print("=" * 65)

    print("\n--- Validating image-ID sets (stimuli / fixmaps / DINO-S metrics) ---")
    common_700 = validate_image_id_sets()
    image_ids = sorted(common_700)
    print(f"  Intersection: {len(common_700)} images (expected {N_EXPECTED})  OK")
    if len(image_ids) != N_EXPECTED:
        raise RuntimeError(f"STOP: {len(image_ids)} images selected, expected {N_EXPECTED}")

    print("\n--- Loading fixation data (700-image pool for sAUC) ---")
    fix_points, all_pool = load_fixation_data()
    print(f"  {len(fix_points)} images with fixations, {len(all_pool)} pooled points")
    if len(fix_points) != N_EXPECTED:
        raise RuntimeError(f"STOP: expected {N_EXPECTED} fixmap images, got {len(fix_points)}")

    print("\n--- Positive control: center-Gaussian vs uniform-noise (all 700 images) ---")
    bl_summary = compute_positive_controls(image_ids, fix_points, all_pool)
    for bname, vals in bl_summary.items():
        print(f"  {bname:16s}  NSS={vals['NSS']:.4f}  AUC={vals['AUC_Judd']:.4f}  sAUC={vals['sAUC']:.4f}")
    uniform_near_half = (0.4 <= bl_summary["uniform"]["AUC_Judd"] <= 0.6 and
                          0.4 <= bl_summary["uniform"]["sAUC"] <= 0.6)
    center_beats_uniform = (bl_summary["center_gaussian"]["NSS"] > bl_summary["uniform"]["NSS"] and
                             bl_summary["center_gaussian"]["AUC_Judd"] > bl_summary["uniform"]["AUC_Judd"])
    if not uniform_near_half or not center_beats_uniform:
        raise RuntimeError("STOP: positive control failed")
    print("  Positive controls: PASS")

    wall_t0 = time.perf_counter()
    for t in trials:
        run_trial(t, image_ids, fix_points, all_pool, bl_summary)
    wall_elapsed = time.perf_counter() - wall_t0

    print("\n" + "=" * 65)
    print(f"  Exp A (SL ViT-S/16 full700): trials {trials} DONE")
    print(f"  Wall-clock (this invocation): {wall_elapsed:.1f}s ({wall_elapsed / 60:.1f} min)")
    print("  outputs/expA_sl/pilot20/, outputs/expA_sl/validation/, outputs/expA/, "
          "outputs/expA_clip/, outputs/expA_dino_vitb16/, outputs/attn_cache/, results/ were NOT touched.")
    print("=" * 65)


if __name__ == "__main__":
    main()
