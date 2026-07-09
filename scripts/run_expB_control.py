"""
Experiment B Control: Disentangle 'earliness' from 'fewer points'.

Compare firstk (first K=2 fixations) vs randk (random K=2 fixations)
against frozen DINO ViT-S/16 layerwise attention.

If firstk > randk systematically  --> temporal position (earliness) matters.
If firstk ~ randk                 --> the advantage was just fewer, concentrated points.

Reuses attention cache from Exp B. Outputs to outputs/expB_control/healthy/.
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
import scipy.ndimage
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import nss, auc_judd, sauc

# ======================= CONFIG =======================
LAYERS      = list(range(1, 13))
GROUP       = "healthy"
IMG_W       = 800
IMG_H       = 600
SIGMA_PX    = 24
FIRST_K     = 2
SEED        = 42
CHUNK_SIZE  = 50
N_BOOT      = 10000

CONDITIONS  = ["all", "firstk", "randk"]
METRICS     = ["NSS", "AUC_Judd", "sAUC"]
CSV_FIELDS  = ["image", "layer", "condition", "NSS", "AUC_Judd", "sAUC"]

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
FIX_MAT     = os.path.join(DATA_BASE, "data", "eye", "fixations.mat")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
CACHE_FILE  = r"C:\Users\user\gaze\outputs\attn_cache\dino_vits16_patchgrid.npz"
EXPB_CSV    = r"C:\Users\user\gaze\outputs\expB\healthy\metrics_per_image.csv"
OUT_DIR     = r"C:\Users\user\gaze\outputs\expB_control\healthy"

H_FEAT = (IMG_H + 15) // 16   # 38
W_FEAT = IMG_W // 16           # 50
# ======================================================


# ==============================================================
# Step 1: Generate randk fixation maps (points_randk + heat_randk)
# ==============================================================

def ensure_randk():
    """Add points_randk and heat_randk to each .npz if not already present."""
    sample = np.load(os.path.join(FIXMAP_DIR, "1001.npz"))
    if "points_randk" in sample.files:
        print("points_randk already present in .npz files -- skipping.")
        return

    print("Generating randk fixation maps (K=%d, seed=%d) ..." % (FIRST_K, SEED))
    mat = scipy.io.loadmat(FIX_MAT)
    fixations = mat["fixations"]

    count = 0
    for idx in range(fixations.shape[0]):
        struct = fixations[idx, 0]
        img_name = str(struct["img"][0, 0][0])
        stem = os.path.splitext(img_name)[0]
        npz_path = os.path.join(FIXMAP_DIR, f"{stem}.npz")

        subjects = struct["subjects"][0, 0]
        xs, ys = [], []
        for j in range(subjects.shape[0]):
            sj = subjects[j, 0]
            fx = sj["fix_x"][0, 0].flatten()
            fy = sj["fix_y"][0, 0].flatten()
            n_fix = len(fx)
            # Per-subject, per-image deterministic seed
            rng = np.random.RandomState(SEED + idx * 1000 + j)
            k = min(FIRST_K, n_fix)
            chosen = rng.choice(n_fix, k, replace=False)
            xs.append(fx[chosen])
            ys.append(fy[chosen])

        xs_all = np.concatenate(xs)
        ys_all = np.concatenate(ys)

        # Convert to 0-indexed pixel coordinates (same logic as generate_fixmaps.py)
        px = np.clip(np.round(xs_all).astype(int) - 1, 0, IMG_W - 1)
        py = np.clip(np.round(ys_all).astype(int) - 1, 0, IMG_H - 1)

        # Binary map
        bmap = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        bmap[py, px] = 1

        # Heatmap (same sigma as heat_firstk)
        heat = scipy.ndimage.gaussian_filter(bmap.astype(np.float64), sigma=SIGMA_PX)
        s = heat.sum()
        if s > 0:
            heat /= s
        heat = heat.astype(np.float32)

        # Re-save npz with new keys
        d = dict(np.load(npz_path))
        d["points_randk"] = bmap
        d["heat_randk"] = heat
        np.savez_compressed(npz_path, **d)
        count += 1

    print(f"  Added points_randk + heat_randk to {count} files.")

    # Verify
    d = np.load(os.path.join(FIXMAP_DIR, "1001.npz"))
    fk_count = int(d["points_firstk"].sum())
    rk_count = int(d["points_randk"].sum())
    print(f"  Verification (1001): points_firstk={fk_count}, points_randk={rk_count}")


# ==============================================================
# Step 2: Load attention cache (reuse from Exp B)
# ==============================================================

def load_attention_cache():
    """Load attention cache and upsample to image resolution."""
    print(f"Loading attention cache from {CACHE_FILE} ...")
    cache = np.load(CACHE_FILE)
    stems = list(cache["stems"])
    attn_pg = cache["attn"]   # (N, 12, H_FEAT, W_FEAT)
    print(f"  {len(stems)} images loaded. Upsampling to {IMG_H}x{IMG_W} ...")
    result = {}
    for i, stem in enumerate(stems):
        t = torch.from_numpy(attn_pg[i]).unsqueeze(1).float()  # (12, 1, h, w)
        t_up = F.interpolate(t, size=(IMG_H, IMG_W),
                             mode="bilinear", align_corners=False)
        m = t_up[:, 0].numpy()   # (12, H, W)
        for l in range(12):
            s = m[l].sum()
            if s > 0:
                m[l] /= s
        result[stem] = m.astype(np.float32)
    print(f"  Done.")
    return result


# ==============================================================
# Step 3: Load fixation data (all three conditions)
# ==============================================================

def load_fixation_data():
    """
    Returns
    -------
    fix : dict {condition: {stem: (N,2) (x,y)}}
    pools : dict {condition: pooled (M,2)}
    """
    fix = {c: {} for c in CONDITIONS}
    pool = {c: [] for c in CONDITIONS}

    for f in sorted(glob.glob(os.path.join(FIXMAP_DIR, "*.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)

        for cond, key in [("all", "points"), ("firstk", "points_firstk"),
                          ("randk", "points_randk")]:
            ys, xs = np.where(d[key])
            pts = np.column_stack([xs, ys])
            fix[cond][stem] = pts
            pool[cond].append(pts)

    pools = {c: np.concatenate(pool[c]) for c in CONDITIONS}
    return fix, pools


# ==============================================================
# Resume helpers
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
# Load Exp B results for all/firstk (avoid recomputation)
# ==============================================================

def load_expB_records():
    """Load existing Exp B per-image metrics for 'all' and 'firstk' conditions."""
    print(f"Loading Exp B results from {EXPB_CSV} ...")
    records = []
    with open(EXPB_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "image":     row["image"],
                "layer":     int(row["layer"]),
                "condition": row["condition"],
                "NSS":       float(row["NSS"]),
                "AUC_Judd":  float(row["AUC_Judd"]),
                "sAUC":      float(row["sAUC"]),
            })
    n_all = sum(1 for r in records if r["condition"] == "all")
    n_fk = sum(1 for r in records if r["condition"] == "firstk")
    print(f"  Loaded {len(records)} rows (all={n_all}, firstk={n_fk})")
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
    Focus: firstk vs randk comparison.
    """
    test_rows = []

    # 1) Per-layer firstk vs randk Wilcoxon for each metric
    print("\n=== firstk vs randk per layer ===")
    for m in METRICS:
        layer_tests = []
        for l in range(1, 13):
            vals_fk = data["firstk"][m][l]
            vals_rk = data["randk"][m][l]
            stat, p, r = wilcoxon_test(vals_fk, vals_rk)
            delta = np.mean(vals_fk) - np.mean(vals_rk)
            delta_ci_lo, delta_ci_hi = bootstrap_ci(
                np.array(vals_fk) - np.array(vals_rk))
            row = {
                "test": "firstk_vs_randk",
                "metric": m,
                "comparison": f"L{l}",
                "layer": l,
                "statistic": stat,
                "p": p,
                "p_corrected": np.nan,
                "effect_r": r,
                "mean_firstk": np.mean(vals_fk),
                "mean_randk": np.mean(vals_rk),
                "delta": delta,
                "delta_ci_lo": delta_ci_lo,
                "delta_ci_hi": delta_ci_hi,
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
            sig = ("***" if r["p_corrected"] < 0.001 else
                   "**"  if r["p_corrected"] < 0.01  else
                   "*"   if r["p_corrected"] < 0.05  else "ns")
            print(f"  L{r['layer']:2d} NSS: fk={r['mean_firstk']:.3f} "
                  f"rk={r['mean_randk']:.3f}  "
                  f"delta={r['delta']:+.3f} [{r['delta_ci_lo']:+.3f}, "
                  f"{r['delta_ci_hi']:+.3f}]  "
                  f"p_holm={r['p_corrected']:.2e} r={r['effect_r']:.3f} {sig}")

    # 2) Peak layer bootstrap
    print("\n=== Peak layer bootstrap (NSS) ===")
    rng = np.random.RandomState(SEED)
    n_img = len(data["firstk"]["NSS"][1])
    peaks = {c: [] for c in CONDITIONS}
    for _ in range(N_BOOT):
        idx = rng.choice(n_img, n_img, replace=True)
        for c in CONDITIONS:
            layer_means = [np.mean(np.array(data[c]["NSS"][l])[idx])
                           for l in range(1, 13)]
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
              f"95%CI=[{peak_summary[c]['peak_ci_lo']:.0f}, "
              f"{peak_summary[c]['peak_ci_hi']:.0f}]")

    delta_fk_rk = np.array(peaks["firstk"]) - np.array(peaks["randk"])
    peak_summary["delta_fk_rk"] = {
        "mean": float(np.mean(delta_fk_rk)),
        "ci_lo": float(np.percentile(delta_fk_rk, 2.5)),
        "ci_hi": float(np.percentile(delta_fk_rk, 97.5)),
    }
    print(f"  delta(firstk-randk): mean={peak_summary['delta_fk_rk']['mean']:.2f}, "
          f"95%CI=[{peak_summary['delta_fk_rk']['ci_lo']:.1f}, "
          f"{peak_summary['delta_fk_rk']['ci_hi']:.1f}]")

    # 3) Z-normalized profile correlations
    print("\n=== Z-normalized profile correlations ===")
    profiles = {}
    for c in CONDITIONS:
        prof = np.array([np.mean(data[c]["NSS"][l]) for l in range(1, 13)])
        profiles[c] = (prof - prof.mean()) / prof.std()

    z_corrs = {}
    for a, b in [("firstk", "randk"), ("all", "firstk"), ("all", "randk")]:
        r_val = float(np.corrcoef(profiles[a], profiles[b])[0, 1])
        z_corrs[f"{a}_vs_{b}"] = r_val
        print(f"  r(z_{a}, z_{b}) = {r_val:.4f}")

    return test_rows, peak_summary, z_corrs


# ==============================================================
# Plot: 3 conditions overlaid (Exp A style)
# ==============================================================

def plot_3conditions(layer_summary, out_path):
    """3-panel plot: three conditions overlaid, +/- SD bands."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_display = {"NSS": "NSS", "AUC_Judd": "AUC-Judd", "sAUC": "sAUC"}
    colors = {"all": "steelblue", "firstk": "coral", "randk": "mediumseagreen"}
    labels = {"all": "All fixations", "firstk": f"First-K (K={FIRST_K})",
              "randk": f"Random-K (K={FIRST_K})"}
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

    fig.suptitle(f"Exp-B Control: All vs First-K vs Random-K ({GROUP}, 700 images)",
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

    # -- Step 1: Generate randk --
    ensure_randk()

    # -- Step 2: Load attention cache --
    attn_maps = load_attention_cache()
    print(f"Attention maps ready: {len(attn_maps)} images")

    # -- Step 3: Load fixation data --
    print("Loading fixation data (3 conditions) ...")
    fix, pools = load_fixation_data()
    for c in CONDITIONS:
        print(f"  {c}: {len(pools[c])} fix points")

    # -- Step 4: Compute randk metrics (with resume) --
    csv_img = _csv_path()
    done = _load_done_set(csv_img)
    if done:
        print(f"Resume: {len(done)} rows in existing CSV")

    stems_sorted = sorted(attn_maps.keys())
    n_img = len(stems_sorted)
    n_expected = n_img * 12   # randk only: 700 x 12

    todo_count = sum(
        1 for stem in stems_sorted
        if any((f"{stem}.jpg", l + 1, "randk") not in done for l in range(12))
    )

    if todo_count == 0:
        print(f"All {n_expected} randk rows already computed -- skipping.")
    else:
        print(f"Computing randk metrics for {todo_count} images ...")

    for i in range(0, n_img, CHUNK_SIZE):
        chunk_stems = stems_sorted[i:i + CHUNK_SIZE]
        chunk_rows = []

        for stem in chunk_stems:
            img_name = f"{stem}.jpg"
            am = attn_maps[stem]
            pts = fix["randk"].get(stem)
            if pts is None or len(pts) == 0:
                continue
            pool = pools["randk"]
            for l in range(12):
                if (img_name, l + 1, "randk") in done:
                    continue
                sal = am[l]
                chunk_rows.append({
                    "image":     img_name,
                    "layer":     l + 1,
                    "condition": "randk",
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

    del attn_maps
    gc.collect()

    if len(done) < n_expected:
        print(f"WARNING: {len(done)}/{n_expected} complete. Re-run to finish.")
        return

    print(f"\nAll {n_expected} randk rows complete.")

    # -- Step 5: Merge with Exp B data --
    randk_records = _load_all_records(csv_img)
    expb_records = load_expB_records()
    all_records = expb_records + randk_records
    print(f"Total records: {len(all_records)} "
          f"(expB={len(expb_records)}, randk={len(randk_records)})")

    # Build data structure: {condition: {metric: {layer: [values]}}}
    # Align by image order to ensure paired comparisons
    data = {c: {m: {l: [] for l in range(1, 13)} for m in METRICS}
            for c in CONDITIONS}
    for r in all_records:
        c = r["condition"]
        if c not in CONDITIONS:
            continue
        for m in METRICS:
            data[c][m][r["layer"]].append(r[m])

    # Verify alignment
    for l in range(1, 13):
        n_a = len(data["all"]["NSS"][l])
        n_f = len(data["firstk"]["NSS"][l])
        n_r = len(data["randk"]["NSS"][l])
        if not (n_a == n_f == n_r == n_img):
            print(f"WARNING: Layer {l} count mismatch: "
                  f"all={n_a}, firstk={n_f}, randk={n_r}")

    # -- Layer summary --
    layer_summary = {}
    summary_rows = []
    for c in CONDITIONS:
        layer_summary[c] = {}
        for m in METRICS:
            means = np.array([np.mean(data[c][m][l]) for l in range(1, 13)])
            stds  = np.array([np.std(data[c][m][l], ddof=1)
                              for l in range(1, 13)])
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

    # Print summary table
    print("\n=== Layer summary (NSS mean +/- SD) ===")
    print(f"{'Layer':>5}  {'all':>16}  {'firstk':>16}  {'randk':>16}")
    for l in range(1, 13):
        vals = {}
        for c in CONDITIONS:
            sr = [r for r in summary_rows
                  if r["layer"] == l and r["condition"] == c][0]
            vals[c] = sr
        print(f"{l:5d}  "
              f"{vals['all']['NSS_mean']:7.3f}+/-{vals['all']['NSS_std']:.3f}  "
              f"{vals['firstk']['NSS_mean']:7.3f}+/-{vals['firstk']['NSS_std']:.3f}  "
              f"{vals['randk']['NSS_mean']:7.3f}+/-{vals['randk']['NSS_std']:.3f}")

    # -- Step 6: Statistics --
    test_rows, peak_summary, z_corrs = run_statistics(data)

    # Save stats_firstk_vs_randk.csv
    test_fields = ["test", "metric", "comparison", "layer", "statistic",
                   "p", "p_corrected", "effect_r",
                   "mean_firstk", "mean_randk", "delta",
                   "delta_ci_lo", "delta_ci_hi"]
    csv_tests = os.path.join(OUT_DIR, "stats_firstk_vs_randk.csv")
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
                w.writerow([f"{c}_{k}",
                            f"{v:.4f}" if isinstance(v, float) else v])
        d = peak_summary["delta_fk_rk"]
        w.writerow(["delta_fk_rk_mean", f"{d['mean']:.4f}"])
        w.writerow(["delta_fk_rk_ci_lo", f"{d['ci_lo']:.4f}"])
        w.writerow(["delta_fk_rk_ci_hi", f"{d['ci_hi']:.4f}"])
        for k, v in z_corrs.items():
            w.writerow([f"z_corr_{k}", f"{v:.6f}"])
    print(f"Saved: {csv_peak}")

    # -- Step 7: Plot --
    plot_path = os.path.join(OUT_DIR, "layerwise_compare_3conditions.png")
    plot_3conditions(layer_summary, plot_path)

    # -- Step 8: run_config.json --
    elapsed = time.time() - t_start
    cfg = {
        "experiment": "expB_control",
        "purpose": "Disentangle earliness from fewer-points effect",
        "conditions": CONDITIONS,
        "K": FIRST_K,
        "randk_seed": SEED,
        "sigma_px": SIGMA_PX,
        "metrics": METRICS,
        "n_images": n_img,
        "n_rows_randk": len(randk_records),
        "n_rows_total": len(all_records),
        "n_boot": N_BOOT,
        "sAUC_neg": "per-condition pool",
        "attn_cache": CACHE_FILE,
        "expB_csv": EXPB_CSV,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg_path = os.path.join(OUT_DIR, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved: {cfg_path}")

    # -- Judgment --
    print("\n" + "=" * 60)
    print("JUDGMENT SUMMARY")
    print("=" * 60)

    pk_fk = peak_summary["firstk"]["peak_mode"]
    pk_rk = peak_summary["randk"]["peak_mode"]
    pk_all = peak_summary["all"]["peak_mode"]
    print(f"  Peak layer (NSS): all={pk_all}, firstk={pk_fk}, randk={pk_rk}")
    print(f"  Delta peak (firstk-randk): "
          f"{peak_summary['delta_fk_rk']['mean']:.2f} "
          f"[{peak_summary['delta_fk_rk']['ci_lo']:.1f}, "
          f"{peak_summary['delta_fk_rk']['ci_hi']:.1f}]")

    for k, v in z_corrs.items():
        print(f"  Z-profile correlation ({k}): r = {v:.4f}")

    nss_tests = [r for r in test_rows if r["metric"] == "NSS"]
    n_sig = sum(1 for r in nss_tests if r["p_corrected"] < 0.05)
    n_fk_higher = sum(1 for r in nss_tests
                      if r["p_corrected"] < 0.05 and r["delta"] > 0)
    print(f"  Layers with significant firstk>randk (NSS, Holm p<0.05): "
          f"{n_fk_higher}/12")
    print(f"  Layers with any significant difference: {n_sig}/12")

    # Deep layers (L7-L12) check
    deep_tests = [r for r in nss_tests if r["layer"] >= 7]
    n_deep_sig = sum(1 for r in deep_tests
                     if r["p_corrected"] < 0.05 and r["delta"] > 0)
    mean_deep_delta = np.mean([r["delta"] for r in deep_tests])

    print(f"  Deep layers (L7-L12) firstk>randk significant: {n_deep_sig}/6")
    print(f"  Deep layers mean delta(firstk-randk): {mean_deep_delta:+.3f}")

    if n_fk_higher >= 8 and mean_deep_delta > 0.05:
        print("\n  -> firstk is systematically higher than randk, "
              "especially in deep layers.")
        print("  -> The 'earliness' of initial saccades has a genuine effect.")
        print("  -> Initial saccades align more strongly with ViT deep-layer "
              "attention than random fixation subsets of equal size.")
    elif n_sig <= 3 and abs(mean_deep_delta) < 0.05:
        print("\n  -> firstk ~ randk (no systematic difference).")
        print("  -> The firstk advantage over 'all' was primarily an artifact "
              "of fewer, more concentrated points.")
        print("  -> Earliness per se has limited effect on layer alignment.")
    else:
        print("\n  -> Mixed results: some layers show firstk > randk, "
              "others do not.")
        print("  -> The earliness effect is partial -- present but not dominant.")

    print(f"\nTotal elapsed: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
