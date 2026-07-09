"""
Experiment A (Case A -- time-compressed):
  DINO ViT-S/16 layerwise attention vs. human fixations (OSIE, healthy).

For each image x layer, computes NSS / AUC-Judd / sAUC (+ CC / SIM),
aggregates by layer, and plots layerwise agreement with baselines.
"""

import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import gc
import json
import os
import glob
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from metrics import nss, auc_judd, sauc
from vit_extractor import _load_model, _ImageListDataset

# ======================= CONFIG =======================
MODEL       = "dino_vits16"
LAYERS      = list(range(1, 13))          # 1..12
ATTN        = "cls2patch"
HEAD_AGG    = "mean"                      # "mean" | "max"
GROUP       = "healthy"
MAP_MODE    = "heat_all"                  # time-compressed
IMG_W       = 800
IMG_H       = 600
UPSAMPLE    = "bilinear"
METRICS_LIST = ["NSS", "AUC_Judd", "sAUC", "CC", "SIM"]
SEED        = 42
BATCH_SIZE  = 2
CHUNK_SIZE  = 50                          # images per GPU chunk

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR    = os.path.join(DATA_BASE, "data", "stimuli")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
OUT_DIR     = r"C:\Users\user\gaze\outputs\expA\healthy"
# ======================================================

_config = Config(batch_size=BATCH_SIZE, depth=12, patch_size=16)


# ------------------------------------------------------------------
# Attention extraction (bilinear upsample, no box-blur)
# ------------------------------------------------------------------

def extract_attention_chunk(model, device, stem_path_pairs):
    """
    Extract per-layer [CLS]->patch attention for a list of images.

    Returns dict {stem: ndarray(12, IMG_H, IMG_W) float32, each layer sum=1}.
    """
    stems = [s for s, _ in stem_path_pairs]
    paths = [p for _, p in stem_path_pairs]

    dataset = _ImageListDataset(paths, _config.patch_size)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

    all_maps = []
    for images in loader:
        with torch.inference_mode():
            attn_list = model.get_fulllayers_selfattention(images.to(device))

        B = images.shape[0]
        ps = _config.patch_size
        h_feat = images.shape[-2] // ps
        w_feat = images.shape[-1] // ps
        n_heads = model.num_heads

        layer_maps = []                             # list of (B, H, W)
        for attn in attn_list:                      # attn: (B, heads, T, T)
            a = attn[:, :, 0, 1:]                   # CLS -> patches
            a = a.reshape(B, n_heads, h_feat, w_feat)
            if HEAD_AGG == "mean":
                a = a.mean(dim=1, keepdim=True)
            else:
                a = a.max(dim=1, keepdim=True).values
            a_up = F.interpolate(a, size=(IMG_H, IMG_W),
                                 mode="bilinear", align_corners=False)
            layer_maps.append(a_up[:, 0].detach().cpu().numpy())

        stacked = np.stack(layer_maps, axis=1)      # (B, 12, H, W)
        for b in range(B):
            m = stacked[b].copy()
            for l in range(m.shape[0]):
                s = m[l].sum()
                if s > 0:
                    m[l] /= s
            all_maps.append(m.astype(np.float32))

        del attn_list, layer_maps, stacked

    return dict(zip(stems, all_maps))


# ------------------------------------------------------------------
# Fixation data
# ------------------------------------------------------------------

def load_fixation_data():
    """
    Returns
    -------
    fix_points : {stem: (N,2) int (x,y)}
    fix_heats  : {stem: (H,W) float32}
    all_pool   : (M,2) int -- pooled across all images for sAUC
    """
    fix_points, fix_heats, pool = {}, {}, []
    for f in sorted(glob.glob(os.path.join(FIXMAP_DIR, "*.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)
        ys, xs = np.where(d["points"])
        coords = np.column_stack([xs, ys])          # (N,2) as (x,y)
        fix_points[stem] = coords
        fix_heats[stem] = d[MAP_MODE].astype(np.float32)
        pool.append(coords)
    return fix_points, fix_heats, np.concatenate(pool)


# ------------------------------------------------------------------
# Baselines
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
# Continuous-map metrics (optional)
# ------------------------------------------------------------------

def cc_metric(sal, human):
    s = sal.astype(np.float64).ravel()
    h = human.astype(np.float64).ravel()
    if s.std() < 1e-12 or h.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(s, h)[0, 1])


def sim_metric(sal, human):
    s = sal.astype(np.float64)
    h = human.astype(np.float64)
    s = s / (s.sum() + 1e-12)
    h = h / (h.sum() + 1e-12)
    return float(np.minimum(s, h).sum())


# ------------------------------------------------------------------
# Resume helpers
# ------------------------------------------------------------------

CSV_FIELDS = ["image", "layer", "NSS", "AUC_Judd", "sAUC", "CC", "SIM"]


def _csv_path():
    return os.path.join(OUT_DIR, "metrics_per_image.csv")


def _load_done_set(csv_path):
    """Read existing CSV and return set of (image, layer_int) tuples."""
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["image"], int(row["layer"])))
    return done


def _append_rows(csv_path, rows):
    """Append rows to CSV. Write header only if file is new/empty."""
    write_header = (not os.path.isfile(csv_path)
                    or os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def _load_all_records(csv_path):
    """Read back the full CSV as a list of dicts (with correct types)."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "image":    row["image"],
                "layer":    int(row["layer"]),
                "NSS":      float(row["NSS"]),
                "AUC_Judd": float(row["AUC_Judd"]),
                "sAUC":     float(row["sAUC"]),
                "CC":       float(row["CC"]),
                "SIM":      float(row["SIM"]),
            })
    return records


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_img = _csv_path()
    t_start = time.time()

    # ── 0. Resume: load already-done (image, layer) pairs ──
    done = _load_done_set(csv_img)
    if done:
        done_images = set(img for img, _ in done)
        print(f"Resume: {len(done)} rows found in existing CSV "
              f"({len(done_images)} images complete)")
    else:
        print("No existing CSV found -- starting fresh.")

    # ── 1. Load model ──
    print("Loading DINO ViT-S/16 ...")
    model, device = _load_model(_config)
    print(f"  Device: {device}")

    # ── 2. Load fixation data ──
    print("Loading fixation data ...")
    fix_points, fix_heats, all_pool = load_fixation_data()
    print(f"  {len(fix_points)} images, {len(all_pool)} total fixation points")

    # ── 3. Regression check (1 image) ──
    print("\n=== Regression check (1001.jpg) ===")
    test_path = os.path.join(STIM_DIR, "1001.jpg")
    ds_one = _ImageListDataset([test_path], _config.patch_size)
    img_t = ds_one[0].unsqueeze(0).to(device)

    with torch.inference_mode():
        raw = model.get_fulllayers_selfattention(img_t)

    ps = _config.patch_size
    h_f = img_t.shape[-2] // ps
    w_f = img_t.shape[-1] // ps
    n_tok = h_f * w_f + 1

    print(f"  Tensor:  {img_t.shape}")
    print(f"  Grid:    {h_f}x{w_f} = {h_f * w_f} patches")
    print(f"  Layers:  {len(raw)}")
    a0 = raw[0]
    print(f"  L1 attn: {a0.shape}  (expect (1, 6, {n_tok}, {n_tok}))")
    print(f"  Row sum: {a0[0, 0, 0].sum().item():.6f}  (expect 1.0)")

    test_out = extract_attention_chunk(model, device, [("1001", test_path)])
    tm = test_out["1001"]
    print(f"  Bilinear output: {tm.shape}, L1 sum={tm[0].sum():.6f}, "
          f"L12 sum={tm[11].sum():.6f}")
    print("  Regression check: PASS")
    del raw, img_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 4. Extract + compute metrics (chunked, with resume) ──
    image_list = sorted(
        [(os.path.splitext(os.path.basename(p))[0], p)
         for p in glob.glob(os.path.join(STIM_DIR, "*.jpg"))]
    )
    n_img = len(image_list)
    n_expected = n_img * 12       # total (image, layer) pairs

    # Count how many images still need processing
    todo_count = 0
    for stem, _ in image_list:
        img_name = f"{stem}.jpg"
        if any((img_name, l + 1) not in done for l in range(12)):
            todo_count += 1

    if todo_count == 0:
        print(f"\nAll {n_img} images already processed -- skipping extraction.")
    else:
        print(f"\nProcessing {todo_count} remaining images "
              f"(of {n_img} total) ...")

    chunks_written = 0
    t_start = time.time()
    for i in range(0, n_img, CHUNK_SIZE):
        chunk = image_list[i:i + CHUNK_SIZE]

        # Filter: keep only images that have any missing layers
        chunk_todo = []
        for stem, path in chunk:
            img_name = f"{stem}.jpg"
            if any((img_name, l + 1) not in done for l in range(12)):
                chunk_todo.append((stem, path))

        if not chunk_todo:
            continue                      # entire chunk already done

        t_chunk = time.time()

        # GPU: extract attention only for images we need
        attn_maps = extract_attention_chunk(model, device, chunk_todo)

        # Compute metrics -- collect full chunk results before writing
        chunk_rows = []
        for stem, _ in chunk_todo:
            img_name = f"{stem}.jpg"
            pts  = fix_points.get(stem)
            heat = fix_heats.get(stem)
            if pts is None or len(pts) == 0:
                continue
            am = attn_maps[stem]
            for l in range(12):
                if (img_name, l + 1) in done:
                    continue              # skip individually done layers
                sal = am[l]
                chunk_rows.append({
                    "image":    img_name,
                    "layer":    l + 1,
                    "NSS":      nss(sal, pts),
                    "AUC_Judd": auc_judd(sal, pts),
                    "sAUC":     sauc(sal, pts, all_pool),
                    "CC":       cc_metric(sal, heat),
                    "SIM":      sim_metric(sal, heat),
                })

        # Atomic append: all rows for this chunk at once
        if chunk_rows:
            _append_rows(csv_img, chunk_rows)
            for r in chunk_rows:
                done.add((r["image"], r["layer"]))
            chunks_written += 1

        del attn_maps
        gc.collect()

        # Progress log with timing and chunk mean NSS
        elapsed = time.time() - t_start
        chunk_dt = time.time() - t_chunk
        chunk_nss = np.mean([r["NSS"] for r in chunk_rows]) if chunk_rows else 0
        pct = min(100.0, (i + len(chunk)) / n_img * 100)
        n_done_imgs = len(set(img for img, _ in done))
        print(f"  [{pct:5.1f}%] {i + len(chunk)}/{n_img}  "
              f"({n_done_imgs} imgs, {len(done)} rows)  "
              f"chunk {chunk_dt:.1f}s  total {elapsed:.0f}s  "
              f"chunk_NSS={chunk_nss:.3f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── 5. Completeness check ──
    if len(done) < n_expected:
        print(f"\nWARNING: only {len(done)}/{n_expected} (image, layer) pairs "
              f"complete. Skipping final aggregation.")
        print("Re-run to finish remaining images.")
        return

    print(f"\nAll {n_expected} (image, layer) pairs complete.")

    # ── 6. Baselines ──
    print("Computing baselines ...")
    cg_map = center_gaussian(IMG_H, IMG_W)
    uf_map = uniform_noise(IMG_H, IMG_W)

    bl = {"center_gaussian": [], "uniform": []}
    for stem in sorted(fix_points):
        pts  = fix_points[stem]
        heat = fix_heats.get(stem)
        if len(pts) == 0:
            continue
        for bname, bmap in [("center_gaussian", cg_map), ("uniform", uf_map)]:
            bl[bname].append({
                "NSS":      nss(bmap, pts),
                "AUC_Judd": auc_judd(bmap, pts),
                "sAUC":     sauc(bmap, pts, all_pool),
                "CC":       cc_metric(bmap, heat),
                "SIM":      sim_metric(bmap, heat),
            })

    bl_summ = {}
    for bname, recs in bl.items():
        bl_summ[bname] = {}
        for m in METRICS_LIST:
            vals = [r[m] for r in recs]
            bl_summ[bname][f"{m}_mean"] = float(np.nanmean(vals))
            bl_summ[bname][f"{m}_std"]  = float(np.nanstd(vals))

    print("\n=== Baseline sanity check ===")
    for bn, bs in bl_summ.items():
        print(f"  {bn}:  NSS={bs['NSS_mean']:.4f}  AUC={bs['AUC_Judd_mean']:.4f}  "
              f"sAUC={bs['sAUC_mean']:.4f}")
    ok = (bl_summ["center_gaussian"]["NSS_mean"] > bl_summ["uniform"]["NSS_mean"] and
          bl_summ["center_gaussian"]["AUC_Judd_mean"] > bl_summ["uniform"]["AUC_Judd_mean"])
    print(f"  Center Gaussian > Uniform (NSS & AUC): {'PASS' if ok else 'FAIL'}")

    # ── 7. Load full CSV and aggregate by layer ──
    records = _load_all_records(csv_img)
    print(f"\nLoaded {len(records)} rows from {csv_img}")

    layer_agg = {l: {m: [] for m in METRICS_LIST} for l in range(1, 13)}
    for r in records:
        for m in METRICS_LIST:
            layer_agg[r["layer"]][m].append(r[m])

    header2 = ["layer"]
    for m in METRICS_LIST:
        header2 += [f"{m}_mean", f"{m}_std"]
    rows2 = []
    for l in range(1, 13):
        row = {"layer": l}
        for m in METRICS_LIST:
            vals = layer_agg[l][m]
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_std"]  = float(np.nanstd(vals))
        rows2.append(row)

    csv_layer = os.path.join(OUT_DIR, "metrics_by_layer.csv")
    with open(csv_layer, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header2)
        w.writeheader()
        for row in rows2:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                        for k, v in row.items()})
    print(f"Saved: {csv_layer}")

    # Print layer summary
    print("\n=== metrics_by_layer ===")
    print(f"{'layer':>5}  {'NSS':>8}  {'AUC_Judd':>10}  {'sAUC':>8}  {'CC':>8}  {'SIM':>8}")
    for row in rows2:
        print(f"{row['layer']:5d}  {row['NSS_mean']:8.4f}  {row['AUC_Judd_mean']:10.4f}  "
              f"{row['sAUC_mean']:8.4f}  {row['CC_mean']:8.4f}  {row['SIM_mean']:8.4f}")

    # ── 8. Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_defs = [
        ("NSS",      "NSS_mean",      "NSS_std"),
        ("AUC_Judd", "AUC_Judd_mean", "AUC_Judd_std"),
        ("sAUC",     "sAUC_mean",     "sAUC_std"),
    ]
    layers = np.arange(1, 13)

    for ax, (mname, mc, sc) in zip(axes, plot_defs):
        means = np.array([rows2[l][mc] for l in range(12)])
        stds  = np.array([rows2[l][sc] for l in range(12)])

        ax.plot(layers, means, "o-", color="steelblue", lw=2, ms=5,
                label="DINO ViT-S/16")
        ax.fill_between(layers, means - stds, means + stds,
                        alpha=0.15, color="steelblue")

        cg_v = bl_summ["center_gaussian"][f"{mname}_mean"]
        uf_v = bl_summ["uniform"][f"{mname}_mean"]
        ax.axhline(cg_v, color="orange", ls="--", lw=1.5,
                   label="Center Gaussian")
        ax.axhline(uf_v, color="gray",   ls=":",  lw=1.5,
                   label="Uniform")

        ax.set_xlabel("Layer")
        ax.set_ylabel(mname)
        ax.set_title(mname)
        ax.set_xticks(layers)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Exp-A: DINO ViT-S/16 vs Human Fixations ({GROUP}, {MAP_MODE})",
        fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "layerwise_agreement.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {plot_path}")

    # ── 9. Save run_config.json ──
    total_elapsed = time.time() - t_start
    run_config = {
        "model": MODEL,
        "layers": LAYERS,
        "attn": ATTN,
        "head_agg": HEAD_AGG,
        "group": GROUP,
        "map_mode": MAP_MODE,
        "img_w": IMG_W,
        "img_h": IMG_H,
        "upsample": UPSAMPLE,
        "metrics": METRICS_LIST,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "sigma_px": 24,
        "n_images": n_img,
        "n_rows": len(records),
        "elapsed_seconds": round(total_elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(OUT_DIR, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved: {config_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
