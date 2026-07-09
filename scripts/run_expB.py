"""
Experiment B: Compare 'firstk' (initial saccade, K=2) vs 'all' (time-compressed)
fixation maps against DINO ViT-S/16 layerwise attention.

Only the human-side ground truth differs between conditions.
AI-side attention, metrics functions, sigma, sAUC seed are identical to Exp A.
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
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from metrics import nss, auc_judd, sauc
from vit_extractor import _load_model, _ImageListDataset

# ======================= CONFIG =======================
MODEL       = "dino_vits16"
LAYERS      = list(range(1, 13))
ATTN        = "cls2patch"
HEAD_AGG    = "mean"
GROUP       = "healthy"
IMG_W       = 800
IMG_H       = 600
SIGMA_PX    = 24
FIRST_K     = 2
SEED        = 42
BATCH_SIZE  = 2
CHUNK_SIZE  = 50
N_BOOT      = 10000

CONDITIONS  = ["all", "firstk"]
METRICS     = ["NSS", "AUC_Judd", "sAUC"]
CSV_FIELDS  = ["image", "layer", "condition", "NSS", "AUC_Judd", "sAUC"]

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR    = os.path.join(DATA_BASE, "data", "stimuli")
FIX_MAT     = os.path.join(DATA_BASE, "data", "eye", "fixations.mat")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
CACHE_DIR   = r"C:\Users\user\gaze\outputs\attn_cache"
CACHE_FILE  = os.path.join(CACHE_DIR, "dino_vits16_patchgrid.npz")
OUT_DIR     = r"C:\Users\user\gaze\outputs\expB\healthy"
# ======================================================

_config = Config(batch_size=BATCH_SIZE, depth=12, patch_size=16)
H_FEAT = (IMG_H + 15) // 16   # 38  (after padding 600->608)
W_FEAT = IMG_W // 16           # 50


# ==============================================================
# Step 1: Add points_firstk to .npz files
# ==============================================================

def ensure_points_firstk():
    """Add points_firstk to each .npz if not already present."""
    # Check if already done
    sample = np.load(os.path.join(FIXMAP_DIR, "1001.npz"))
    if "points_firstk" in sample.files:
        print("points_firstk already present in .npz files -- skipping.")
        return

    print("Adding points_firstk to .npz files ...")
    mat = scipy.io.loadmat(FIX_MAT)
    fixations = mat["fixations"]

    count = 0
    for idx in range(fixations.shape[0]):
        struct = fixations[idx, 0]
        img_name = str(struct["img"][0, 0][0])
        stem = os.path.splitext(img_name)[0]
        npz_path = os.path.join(FIXMAP_DIR, f"{stem}.npz")

        # Extract first-K fixation coordinates
        subjects = struct["subjects"][0, 0]
        xs, ys = [], []
        for j in range(subjects.shape[0]):
            sj = subjects[j, 0]
            fx = sj["fix_x"][0, 0].flatten()[:FIRST_K]
            fy = sj["fix_y"][0, 0].flatten()[:FIRST_K]
            xs.append(fx)
            ys.append(fy)
        xs_all = np.concatenate(xs)
        ys_all = np.concatenate(ys)

        # Convert to 0-indexed pixel coordinates
        px = np.clip(np.round(xs_all).astype(int) - 1, 0, IMG_W - 1)
        py = np.clip(np.round(ys_all).astype(int) - 1, 0, IMG_H - 1)

        # Binary map
        bmap = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        bmap[py, px] = 1

        # Re-save npz with new key
        d = dict(np.load(npz_path))
        d["points_firstk"] = bmap
        np.savez_compressed(npz_path, **d)
        count += 1

    print(f"  Added points_firstk to {count} files.")


# ==============================================================
# Step 2: Attention cache (patch-grid resolution)
# ==============================================================

def get_attention_cache():
    """
    Load or create attention cache at patch-grid resolution.
    Cache: {stem: (12, H_FEAT, W_FEAT) float32} before upsample/normalize.
    Returns dict {stem: (12, IMG_H, IMG_W) float32 normalized}.
    """
    if os.path.isfile(CACHE_FILE):
        print(f"Loading attention cache from {CACHE_FILE} ...")
        cache = np.load(CACHE_FILE)
        stems = list(cache["stems"])
        attn_pg = cache["attn"]         # (N, 12, H_FEAT, W_FEAT)
        print(f"  {len(stems)} images loaded. Upsampling to {IMG_H}x{IMG_W} ...")
        result = {}
        for i, stem in enumerate(stems):
            # upsample + normalize
            t = torch.from_numpy(attn_pg[i]).unsqueeze(1).float()  # (12, 1, h, w)
            t_up = F.interpolate(t, size=(IMG_H, IMG_W),
                                 mode="bilinear", align_corners=False)
            m = t_up[:, 0].numpy()     # (12, H, W)
            for l in range(12):
                s = m[l].sum()
                if s > 0:
                    m[l] /= s
            result[stem] = m.astype(np.float32)
        print(f"  Done.")
        return result

    # Extract from scratch
    print("No attention cache found. Extracting from model ...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    model, device = _load_model(_config)

    image_list = sorted(
        [(os.path.splitext(os.path.basename(p))[0], p)
         for p in glob.glob(os.path.join(STIM_DIR, "*.jpg"))]
    )

    all_stems = []
    all_patchgrid = []   # list of (12, H_FEAT, W_FEAT)
    result = {}

    for i in range(0, len(image_list), CHUNK_SIZE):
        chunk = image_list[i:i + CHUNK_SIZE]
        stems = [s for s, _ in chunk]
        paths = [p for _, p in chunk]
        ds = _ImageListDataset(paths, _config.patch_size)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        chunk_maps_pg = []   # patch-grid
        chunk_maps_up = []   # upsampled + normalized
        for images in loader:
            with torch.inference_mode():
                attn_list = model.get_fulllayers_selfattention(images.to(device))
            B = images.shape[0]
            h_f = images.shape[-2] // _config.patch_size
            w_f = images.shape[-1] // _config.patch_size
            n_h = model.num_heads

            pg_layers = []
            up_layers = []
            for attn in attn_list:
                a = attn[:, :, 0, 1:].reshape(B, n_h, h_f, w_f)
                a_mean = a.mean(dim=1, keepdim=True)      # (B,1,h,w)
                pg_layers.append(a_mean[:, 0].detach().cpu().numpy())
                a_up = F.interpolate(a_mean, size=(IMG_H, IMG_W),
                                     mode="bilinear", align_corners=False)
                up_layers.append(a_up[:, 0].detach().cpu().numpy())

            pg_stacked = np.stack(pg_layers, axis=1)   # (B,12,h,w)
            up_stacked = np.stack(up_layers, axis=1)
            for b in range(B):
                chunk_maps_pg.append(pg_stacked[b].astype(np.float32))
                m = up_stacked[b].copy()
                for l in range(12):
                    s = m[l].sum()
                    if s > 0:
                        m[l] /= s
                chunk_maps_up.append(m.astype(np.float32))
            del attn_list

        for stem, pg, up in zip(stems, chunk_maps_pg, chunk_maps_up):
            all_stems.append(stem)
            all_patchgrid.append(pg)
            result[stem] = up

        gc.collect()
        pct = min(100, (i + len(chunk)) / len(image_list) * 100)
        print(f"  [{pct:5.1f}%] {i + len(chunk)}/{len(image_list)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save cache
    np.savez_compressed(
        CACHE_FILE,
        stems=np.array(all_stems),
        attn=np.stack(all_patchgrid),    # (700, 12, H_FEAT, W_FEAT)
    )
    print(f"  Cache saved: {CACHE_FILE} "
          f"({os.path.getsize(CACHE_FILE) / 1e6:.1f} MB)")
    return result


# ==============================================================
# Step 3: Load fixation data (both conditions)
# ==============================================================

def load_fixation_data():
    """
    Returns
    -------
    fix : dict with keys 'all' and 'firstk', each {stem: (N,2) (x,y)}
    pools : dict with keys 'all' and 'firstk', each (M,2) pooled
    """
    fix = {"all": {}, "firstk": {}}
    pool_all, pool_fk = [], []

    for f in sorted(glob.glob(os.path.join(FIXMAP_DIR, "*.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)

        # all condition
        ys, xs = np.where(d["points"])
        pts_all = np.column_stack([xs, ys])
        fix["all"][stem] = pts_all
        pool_all.append(pts_all)

        # firstk condition
        ys_fk, xs_fk = np.where(d["points_firstk"])
        pts_fk = np.column_stack([xs_fk, ys_fk])
        fix["firstk"][stem] = pts_fk
        pool_fk.append(pts_fk)

    pools = {
        "all": np.concatenate(pool_all),
        "firstk": np.concatenate(pool_fk),
    }
    return fix, pools


# ==============================================================
# Resume helpers (same pattern as Exp A)
# ==============================================================

def _csv_path():
    return os.path.join(OUT_DIR, "metrics_per_image.csv")


def _load_done_set(csv_path):
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["image"], int(row["layer"]), row["condition"]))
    return done


def _append_rows(csv_path, rows):
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
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "image":     row["image"],
                "layer":     int(row["layer"]),
                "condition": row["condition"],
                "NSS":       float(row["NSS"]),
                "AUC_Judd":  float(row["AUC_Judd"]),
                "sAUC":      float(row["sAUC"]),
            })
    return records


# ==============================================================
# Statistics
# ==============================================================

def wilcoxon_test(x, y, alternative="two-sided"):
    diff = np.array(x) - np.array(y)
    diff_nz = diff[diff != 0]
    n = len(diff_nz)
    if n < 10:
        return np.nan, np.nan, np.nan
    res = sp_stats.wilcoxon(diff_nz, alternative=alternative)
    stat, p = res.statistic, res.pvalue
    total_rank = n * (n + 1) / 2
    r = abs(1 - (2 * stat) / total_rank)
    return float(stat), float(p), float(r)


def holm_correction(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, pvals[idx] * (n - rank))
    running_max = 0
    for idx in order:
        running_max = max(running_max, corrected[idx])
        corrected[idx] = running_max
    return corrected


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    means = np.array([rng.choice(arr, n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


def run_statistics(data):
    """
    data: {condition: {metric: {layer: [values]}}}
    Returns test_rows, peak_summary, z_corr
    """
    test_rows = []

    # 1) Per-layer condition comparison (Wilcoxon) for each metric
    print("\n=== Condition comparison (all vs firstk) per layer ===")
    for m in METRICS:
        layer_tests = []
        for l in range(1, 13):
            vals_all = data["all"][m][l]
            vals_fk  = data["firstk"][m][l]
            stat, p, r = wilcoxon_test(vals_all, vals_fk)
            row = {
                "test": "condition_compare",
                "metric": m,
                "comparison": f"L{l} all vs firstk",
                "layer": l,
                "statistic": stat,
                "p": p,
                "p_corrected": np.nan,
                "effect_r": r,
                "mean_all": np.mean(vals_all),
                "mean_firstk": np.mean(vals_fk),
            }
            layer_tests.append(row)
            test_rows.append(row)
        # Holm per metric
        ps = np.array([r["p"] for r in layer_tests])
        ps_corr = holm_correction(ps)
        for r, pc in zip(layer_tests, ps_corr):
            r["p_corrected"] = pc

    # Print NSS summary
    for r in test_rows:
        if r["metric"] == "NSS":
            sig = "***" if r["p_corrected"] < 0.001 else "**" if r["p_corrected"] < 0.01 else "*" if r["p_corrected"] < 0.05 else "ns"
            print(f"  L{r['layer']:2d} NSS: all={r['mean_all']:.3f} fk={r['mean_firstk']:.3f}  "
                  f"p_holm={r['p_corrected']:.2e} r={r['effect_r']:.3f} {sig}")

    # 2) Peak layer bootstrap
    print("\n=== Peak layer bootstrap (NSS) ===")
    rng = np.random.RandomState(SEED)
    n_img = len(data["all"]["NSS"][1])
    peaks = {c: [] for c in CONDITIONS}
    for _ in range(N_BOOT):
        idx = rng.choice(n_img, n_img, replace=True)
        for c in CONDITIONS:
            layer_means = [np.mean(np.array(data[c]["NSS"][l])[idx]) for l in range(1, 13)]
            peaks[c].append(np.argmax(layer_means) + 1)

    peak_summary = {}
    for c in CONDITIONS:
        pk = np.array(peaks[c])
        peak_summary[c] = {
            "peak_mean": float(np.mean(pk)),
            "peak_median": float(np.median(pk)),
            "peak_ci_lo": float(np.percentile(pk, 2.5)),
            "peak_ci_hi": float(np.percentile(pk, 97.5)),
            "peak_mode": int(sp_stats.mode(pk, keepdims=False).mode),
        }
        print(f"  {c:7s}: mode={peak_summary[c]['peak_mode']}, "
              f"mean={peak_summary[c]['peak_mean']:.1f}, "
              f"95%CI=[{peak_summary[c]['peak_ci_lo']:.0f}, {peak_summary[c]['peak_ci_hi']:.0f}]")

    delta_peak = np.array(peaks["all"]) - np.array(peaks["firstk"])
    peak_summary["delta"] = {
        "mean": float(np.mean(delta_peak)),
        "ci_lo": float(np.percentile(delta_peak, 2.5)),
        "ci_hi": float(np.percentile(delta_peak, 97.5)),
    }
    print(f"  delta(all-firstk): mean={peak_summary['delta']['mean']:.2f}, "
          f"95%CI=[{peak_summary['delta']['ci_lo']:.1f}, {peak_summary['delta']['ci_hi']:.1f}]")

    # 3) Z-normalized profile correlation
    print("\n=== Z-normalized profile correlation ===")
    profiles = {}
    for c in CONDITIONS:
        prof = np.array([np.mean(data[c]["NSS"][l]) for l in range(1, 13)])
        profiles[c] = (prof - prof.mean()) / prof.std()
    z_corr = float(np.corrcoef(profiles["all"], profiles["firstk"])[0, 1])
    print(f"  Pearson r(z_all, z_firstk) = {z_corr:.4f}")

    return test_rows, peak_summary, z_corr


# ==============================================================
# Plots (Exp A plot_3metrics style, two conditions overlaid)
# ==============================================================

def plot_comparison(layer_summary, out_path):
    """3-panel plot: two conditions overlaid, +/- SD bands."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_display = {"NSS": "NSS", "AUC_Judd": "AUC-Judd", "sAUC": "sAUC"}
    colors = {"all": "steelblue", "firstk": "coral"}
    labels = {"all": "All fixations", "firstk": f"First-K (K={FIRST_K})"}
    layers = np.arange(1, 13)

    for ax, m in zip(axes, METRICS):
        for c in CONDITIONS:
            means = layer_summary[c][m]["mean"]
            stds  = layer_summary[c][m]["std"]
            ax.plot(layers, means, "o-", color=colors[c], lw=2, ms=5,
                    label=labels[c])
            ax.fill_between(layers, means - stds, means + stds,
                            alpha=0.15, color=colors[c])
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_display[m])
        ax.set_title(metric_display[m])
        ax.set_xticks(layers)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Exp-B: All vs First-K fixations ({GROUP}, 700 images)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================
# Main
# ==============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    # ── Step 1: Ensure points_firstk ──
    ensure_points_firstk()

    # ── Step 2: Get attention maps ──
    attn_maps = get_attention_cache()
    print(f"Attention maps ready: {len(attn_maps)} images")

    # ── Step 3: Load fixation data ──
    print("Loading fixation data (both conditions) ...")
    fix, pools = load_fixation_data()
    n_fix_all = len(pools["all"])
    n_fix_fk  = len(pools["firstk"])
    print(f"  all: {n_fix_all} fix points,  firstk: {n_fix_fk} fix points")

    # ── Step 4: Compute metrics (with resume) ──
    csv_img = _csv_path()
    done = _load_done_set(csv_img)
    if done:
        print(f"Resume: {len(done)} rows in existing CSV")

    stems_sorted = sorted(attn_maps.keys())
    n_img = len(stems_sorted)
    n_expected = n_img * 12 * 2  # 2 conditions

    todo_count = sum(
        1 for stem in stems_sorted
        if any((f"{stem}.jpg", l + 1, c) not in done
               for l in range(12) for c in CONDITIONS)
    )

    if todo_count == 0:
        print(f"All {n_expected} rows already computed -- skipping.")
    else:
        print(f"Computing metrics for {todo_count} images ...")

    for i in range(0, n_img, CHUNK_SIZE):
        chunk_stems = stems_sorted[i:i + CHUNK_SIZE]
        chunk_rows = []

        for stem in chunk_stems:
            img_name = f"{stem}.jpg"
            am = attn_maps[stem]
            for c in CONDITIONS:
                pts = fix[c].get(stem)
                if pts is None or len(pts) == 0:
                    continue
                pool = pools[c]
                for l in range(12):
                    if (img_name, l + 1, c) in done:
                        continue
                    sal = am[l]
                    chunk_rows.append({
                        "image":     img_name,
                        "layer":     l + 1,
                        "condition": c,
                        "NSS":       nss(sal, pts),
                        "AUC_Judd":  auc_judd(sal, pts),
                        "sAUC":      sauc(sal, pts, pool),
                    })

        if chunk_rows:
            _append_rows(csv_img, chunk_rows)
            for r in chunk_rows:
                done.add((r["image"], r["layer"], r["condition"]))

        if todo_count > 0:
            pct = min(100, (i + len(chunk_stems)) / n_img * 100)
            print(f"  [{pct:5.1f}%] {i + len(chunk_stems)}/{n_img}  "
                  f"({len(done)} rows)")

    # Free attention maps
    del attn_maps
    gc.collect()

    # ── Completeness check ──
    if len(done) < n_expected:
        print(f"WARNING: {len(done)}/{n_expected} complete. Re-run to finish.")
        return

    print(f"\nAll {n_expected} rows complete.")

    # ── Step 5: Aggregate & statistics ──
    records = _load_all_records(csv_img)
    print(f"Loaded {len(records)} rows")

    # Build data structure: {condition: {metric: {layer: [values]}}}
    data = {c: {m: {l: [] for l in range(1, 13)} for m in METRICS}
            for c in CONDITIONS}
    for r in records:
        for m in METRICS:
            data[r["condition"]][m][r["layer"]].append(r[m])

    # Layer summary
    layer_summary = {}
    summary_rows = []
    for c in CONDITIONS:
        layer_summary[c] = {}
        for m in METRICS:
            means = np.array([np.mean(data[c][m][l]) for l in range(1, 13)])
            stds  = np.array([np.std(data[c][m][l], ddof=1) for l in range(1, 13)])
            layer_summary[c][m] = {"mean": means, "std": stds}
        for l in range(1, 13):
            row = {"layer": l, "condition": c}
            for m in METRICS:
                vals = data[c][m][l]
                ci_lo, ci_hi = bootstrap_ci(vals)
                row[f"{m}_mean"] = np.mean(vals)
                row[f"{m}_std"]  = np.std(vals, ddof=1)
                row[f"{m}_ci_lo"] = ci_lo
                row[f"{m}_ci_hi"] = ci_hi
            summary_rows.append(row)

    # Save metrics_by_layer.csv
    layer_fields = ["layer", "condition"]
    for m in METRICS:
        layer_fields += [f"{m}_mean", f"{m}_std", f"{m}_ci_lo", f"{m}_ci_hi"]
    csv_layer = os.path.join(OUT_DIR, "metrics_by_layer.csv")
    with open(csv_layer, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=layer_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                        for k, v in row.items()})
    print(f"Saved: {csv_layer}")

    # Print summary
    print("\n=== Layer summary (NSS mean +/- SD) ===")
    print(f"{'Layer':>5}  {'all':>16}  {'firstk':>16}")
    for l in range(1, 13):
        a = [r for r in summary_rows if r["layer"] == l and r["condition"] == "all"][0]
        f = [r for r in summary_rows if r["layer"] == l and r["condition"] == "firstk"][0]
        print(f"{l:5d}  {a['NSS_mean']:7.3f}+/-{a['NSS_std']:.3f}  "
              f"{f['NSS_mean']:7.3f}+/-{f['NSS_std']:.3f}")

    # Statistics
    test_rows, peak_summary, z_corr = run_statistics(data)

    # Save stats_condition_tests.csv
    test_fields = ["test", "metric", "comparison", "layer", "statistic",
                   "p", "p_corrected", "effect_r", "mean_all", "mean_firstk"]
    csv_tests = os.path.join(OUT_DIR, "stats_condition_tests.csv")
    with open(csv_tests, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=test_fields)
        w.writeheader()
        for row in test_rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"Saved: {csv_tests}")

    # Save peak_layer_summary.csv
    csv_peak = os.path.join(OUT_DIR, "peak_layer_summary.csv")
    with open(csv_peak, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        for c in CONDITIONS:
            ps = peak_summary[c]
            for k, v in ps.items():
                w.writerow([f"{c}_{k}", f"{v:.4f}" if isinstance(v, float) else v])
        w.writerow(["delta_mean", f"{peak_summary['delta']['mean']:.4f}"])
        w.writerow(["delta_ci_lo", f"{peak_summary['delta']['ci_lo']:.4f}"])
        w.writerow(["delta_ci_hi", f"{peak_summary['delta']['ci_hi']:.4f}"])
        w.writerow(["z_profile_corr", f"{z_corr:.6f}"])
    print(f"Saved: {csv_peak}")

    # ── Step 6: Plot ──
    plot_path = os.path.join(OUT_DIR, "layerwise_compare_3metrics.png")
    plot_comparison(layer_summary, plot_path)

    # ── Step 7: run_config.json ──
    elapsed = time.time() - t_start
    cfg = {
        "model": MODEL, "layers": LAYERS, "attn": ATTN,
        "head_agg": HEAD_AGG, "group": GROUP,
        "img_w": IMG_W, "img_h": IMG_H, "sigma_px": SIGMA_PX,
        "first_k": FIRST_K, "seed": SEED,
        "conditions": CONDITIONS, "metrics": METRICS,
        "n_images": n_img, "n_rows": len(records),
        "n_boot": N_BOOT,
        "sAUC_neg": "per-condition pool (all->points, firstk->points_firstk)",
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg_path = os.path.join(OUT_DIR, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved: {cfg_path}")

    # ── Judgment summary ──
    print("\n" + "=" * 60)
    print("JUDGMENT SUMMARY")
    print("=" * 60)
    pk_all = peak_summary["all"]["peak_mode"]
    pk_fk  = peak_summary["firstk"]["peak_mode"]
    print(f"  Peak layer (NSS):  all={pk_all},  firstk={pk_fk}")
    print(f"  Delta peak (all-firstk): {peak_summary['delta']['mean']:.2f} "
          f"[{peak_summary['delta']['ci_lo']:.1f}, {peak_summary['delta']['ci_hi']:.1f}]")
    print(f"  Z-profile correlation: r = {z_corr:.4f}")

    # Significant differences?
    nss_tests = [r for r in test_rows if r["metric"] == "NSS"]
    n_sig = sum(1 for r in nss_tests if r["p_corrected"] < 0.05)
    print(f"  Layers with significant NSS difference (Holm p<0.05): {n_sig}/12")

    if z_corr > 0.95 and abs(pk_all - pk_fk) <= 1:
        print("  -> Profile shape highly similar. Peak position stable.")
        print("  -> Time compression does NOT distort layer-level agreement structure.")
    elif z_corr > 0.85:
        print("  -> Profile shape similar but with detectable differences.")
        print("  -> Partial information loss from time compression.")
    else:
        print("  -> Profile shape substantially different.")
        print("  -> Time compression meaningfully changes layer-agreement pattern.")

    print(f"\nTotal elapsed: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
