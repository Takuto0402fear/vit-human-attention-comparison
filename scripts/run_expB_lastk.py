"""
Experiment B - lastk: Compare firstk (first K=2 fixations) vs lastk (last K=2
fixations) against frozen DINO ViT-S/16 layerwise attention.

Purpose: Exp B control showed the firstk advantage over randk/all is mostly a
center-bias artifact once sAUC is used, except for a small residual effect at
L8-L10. This experiment tests whether that residual is specific to *early*
saccades by comparing against a point-count-matched *late* condition (each
subject's last 2 fixations instead of their first 2).

Boundary case (n_fix < 4): a subject with fewer than 4 total fixations would
have overlapping first-2/last-2 points. Such subjects are excluded from BOTH
conditions on the affected images (matched exclusion), so per-image N stays
identical between firstk and lastk everywhere. Only images with >=1 excluded
subject need firstk recomputed from the reduced subject set; all other images
reuse the existing Exp B firstk values unchanged.

Reuses the Exp B attention cache (no ViT re-inference).
Outputs to outputs/expB_lastk/healthy/.
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
GROUP       = "healthy"
IMG_W       = 800
IMG_H       = 600
SIGMA_PX    = 24
K           = 2
MIN_FIX     = 4      # subjects with n_fix < MIN_FIX excluded from BOTH conditions
SEED        = 42
N_BOOT      = 10000

CONDITIONS  = ["firstk", "lastk"]
METRICS     = ["NSS", "AUC_Judd", "sAUC"]
CSV_FIELDS  = ["image", "layer", "condition", "NSS", "AUC_Judd", "sAUC"]

DATA_BASE   = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
FIX_MAT     = os.path.join(DATA_BASE, "data", "eye", "fixations.mat")
FIXMAP_DIR  = r"C:\Users\user\gaze\outputs\fixmaps\healthy"
CACHE_FILE  = r"C:\Users\user\gaze\outputs\attn_cache\dino_vits16_patchgrid.npz"
EXPB_CSV    = r"C:\Users\user\gaze\outputs\expB\healthy\metrics_per_image.csv"
OUT_DIR     = r"C:\Users\user\gaze\outputs\expB_lastk\healthy"
# ======================================================


# ==============================================================
# Step 1: Generate points_lastk/heat_lastk (all images) and the
#         matched-exclusion firstk points (affected images only).
# ==============================================================

def _to_pixels(xs, ys):
    px = np.clip(np.round(xs).astype(int) - 1, 0, IMG_W - 1)
    py = np.clip(np.round(ys).astype(int) - 1, 0, IMG_H - 1)
    return px, py


def prepare_data():
    """
    Ensure points_lastk/heat_lastk exist in every .npz (generated once, then
    reused on subsequent runs). Always recomputes (cheap) the list of
    affected images and the matched-exclusion firstk points for them, since
    those aren't persisted to the .npz (they only apply to this experiment).

    Returns
    -------
    affected_stems : sorted list of image stems with >=1 excluded subject
    excluded_log   : list of {image, subject_index, n_fix} for excluded subjects
    points_firstk_matched : {stem: (H,W) uint8 binary map} for affected_stems only
    """
    mat = scipy.io.loadmat(FIX_MAT)
    fixations = mat["fixations"]
    n_images = fixations.shape[0]

    sample = np.load(os.path.join(FIXMAP_DIR, "1001.npz"))
    need_generate = "points_lastk" not in sample.files
    if need_generate:
        print(f"Generating points_lastk/heat_lastk (K={K}, min_fix={MIN_FIX}) ...")
    else:
        print("points_lastk already present in .npz files -- skipping generation.")

    affected_stems = []
    excluded_log = []
    points_firstk_matched = {}

    for idx in range(n_images):
        struct = fixations[idx, 0]
        img_name = str(struct["img"][0, 0][0])
        stem = os.path.splitext(img_name)[0]
        subjects = struct["subjects"][0, 0]
        n_subj = subjects.shape[0]

        eligible = []
        any_excluded = False
        for j in range(n_subj):
            sj = subjects[j, 0]
            fx = sj["fix_x"][0, 0].flatten()
            fy = sj["fix_y"][0, 0].flatten()
            n_fix = len(fx)
            if n_fix < MIN_FIX:
                any_excluded = True
                excluded_log.append({"image": img_name, "subject_index": j,
                                      "n_fix": n_fix})
            else:
                eligible.append((fx, fy))

        if any_excluded:
            affected_stems.append(stem)

        if need_generate:
            xs_lk = (np.concatenate([fx[-K:] for fx, fy in eligible])
                      if eligible else np.empty(0))
            ys_lk = (np.concatenate([fy[-K:] for fx, fy in eligible])
                      if eligible else np.empty(0))
            px_lk, py_lk = _to_pixels(xs_lk, ys_lk)
            bmap_lk = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            bmap_lk[py_lk, px_lk] = 1
            heat_lk = scipy.ndimage.gaussian_filter(bmap_lk.astype(np.float64),
                                                     sigma=SIGMA_PX)
            s = heat_lk.sum()
            if s > 0:
                heat_lk /= s
            heat_lk = heat_lk.astype(np.float32)

            npz_path = os.path.join(FIXMAP_DIR, f"{stem}.npz")
            d = dict(np.load(npz_path))
            d["points_lastk"] = bmap_lk
            d["heat_lastk"] = heat_lk
            np.savez_compressed(npz_path, **d)

        if any_excluded:
            xs_fk = (np.concatenate([fx[:K] for fx, fy in eligible])
                       if eligible else np.empty(0))
            ys_fk = (np.concatenate([fy[:K] for fx, fy in eligible])
                       if eligible else np.empty(0))
            px_fk, py_fk = _to_pixels(xs_fk, ys_fk)
            bmap_fk = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            bmap_fk[py_fk, px_fk] = 1
            points_firstk_matched[stem] = bmap_fk

    if need_generate:
        print(f"  points_lastk/heat_lastk added to {n_images} files.")

    print(f"  Excluded subjects (n_fix<{MIN_FIX}): {len(excluded_log)} "
          f"across {len(affected_stems)} images.")

    return sorted(affected_stems), excluded_log, points_firstk_matched


# ==============================================================
# Step 2: Load attention cache (reuse from Exp B, no re-inference)
# ==============================================================

def load_attention_cache():
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
# Step 3: Load fixation data (firstk uses the matched-exclusion
#         override for affected images; lastk always from .npz)
# ==============================================================

def load_fixation_data(points_firstk_matched):
    fix = {"firstk": {}, "lastk": {}}
    pool_fk, pool_lk = [], []

    for f in sorted(glob.glob(os.path.join(FIXMAP_DIR, "*.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)

        fk_map = points_firstk_matched.get(stem, d["points_firstk"])
        ys, xs = np.where(fk_map)
        pts_fk = np.column_stack([xs, ys])
        fix["firstk"][stem] = pts_fk
        pool_fk.append(pts_fk)

        ys, xs = np.where(d["points_lastk"])
        pts_lk = np.column_stack([xs, ys])
        fix["lastk"][stem] = pts_lk
        pool_lk.append(pts_lk)

    pools = {"firstk": np.concatenate(pool_fk), "lastk": np.concatenate(pool_lk)}
    return fix, pools


def load_expb_firstk_lookup():
    """{(image, layer): {NSS, AUC_Judd, sAUC}} for condition == 'firstk'."""
    lookup = {}
    with open(EXPB_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["condition"] != "firstk":
                continue
            lookup[(row["image"], int(row["layer"]))] = {
                "NSS": float(row["NSS"]),
                "AUC_Judd": float(row["AUC_Judd"]),
                "sAUC": float(row["sAUC"]),
            }
    return lookup


def load_expb_all_lookup():
    """layer -> list of NSS/AUC_Judd/sAUC values for condition == 'all' (plot context only)."""
    data = {m: {l: [] for l in range(1, 13)} for m in METRICS}
    with open(EXPB_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["condition"] != "all":
                continue
            l = int(row["layer"])
            for m in METRICS:
                data[m][l].append(float(row[m]))
    return data


# ==============================================================
# Step 4: Compute metrics
#   - lastk: recomputed for all 700 images
#   - firstk: recomputed for affected images only; reused from Exp B otherwise
# ==============================================================

def compute_metrics(attn_maps, fix, pools, affected_stems, expb_firstk_lookup):
    rows = []
    affected_set = set(affected_stems)
    stems_sorted = sorted(attn_maps.keys())
    n_img = len(stems_sorted)
    t0 = time.time()
    n_recomputed_firstk = 0
    n_reused_firstk = 0

    for i, stem in enumerate(stems_sorted):
        img_name = f"{stem}.jpg"
        am = attn_maps[stem]

        pts_lk = fix["lastk"][stem]
        for l in range(12):
            sal = am[l]
            rows.append({
                "image": img_name, "layer": l + 1, "condition": "lastk",
                "NSS": nss(sal, pts_lk),
                "AUC_Judd": auc_judd(sal, pts_lk),
                "sAUC": sauc(sal, pts_lk, pools["lastk"]),
            })

        if stem in affected_set:
            n_recomputed_firstk += 1
            pts_fk = fix["firstk"][stem]
            for l in range(12):
                sal = am[l]
                rows.append({
                    "image": img_name, "layer": l + 1, "condition": "firstk",
                    "NSS": nss(sal, pts_fk),
                    "AUC_Judd": auc_judd(sal, pts_fk),
                    "sAUC": sauc(sal, pts_fk, pools["firstk"]),
                })
        else:
            n_reused_firstk += 1
            for l in range(12):
                key = (img_name, l + 1)
                vals = expb_firstk_lookup[key]
                rows.append({
                    "image": img_name, "layer": l + 1, "condition": "firstk",
                    "NSS": vals["NSS"], "AUC_Judd": vals["AUC_Judd"],
                    "sAUC": vals["sAUC"],
                })

        if (i + 1) % 100 == 0 or i == n_img - 1:
            print(f"  [{i + 1}/{n_img}]  elapsed={time.time() - t0:.0f}s")

    print(f"  firstk: recomputed for {n_recomputed_firstk} images, "
          f"reused Exp B values for {n_reused_firstk} images.")
    return rows


# ==============================================================
# Statistics (mirrors run_expB_control.py's pattern; adapted to
# firstk vs lastk instead of firstk vs randk)
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
    """data: {condition: {metric: {layer: [values]}}}. Focus: firstk vs lastk."""
    test_rows = []

    print("\n=== firstk vs lastk per layer ===")
    for m in METRICS:
        layer_tests = []
        for l in range(1, 13):
            vals_fk = data["firstk"][m][l]
            vals_lk = data["lastk"][m][l]
            stat, p, r = wilcoxon_test(vals_fk, vals_lk)
            delta = np.mean(vals_fk) - np.mean(vals_lk)
            delta_ci_lo, delta_ci_hi = bootstrap_ci(
                np.array(vals_fk) - np.array(vals_lk))
            row = {
                "test": "firstk_vs_lastk",
                "metric": m,
                "comparison": f"L{l}",
                "layer": l,
                "statistic": stat,
                "p": p,
                "p_corrected": np.nan,
                "effect_r": r,
                "mean_firstk": np.mean(vals_fk),
                "mean_lastk": np.mean(vals_lk),
                "delta": delta,
                "delta_ci_lo": delta_ci_lo,
                "delta_ci_hi": delta_ci_hi,
            }
            layer_tests.append(row)
            test_rows.append(row)
        ps = np.array([r["p"] for r in layer_tests])
        ps_corr = holm_correction(ps)
        for r, pc in zip(layer_tests, ps_corr):
            r["p_corrected"] = pc

    for r in test_rows:
        if r["metric"] == "sAUC":
            sig = ("***" if r["p_corrected"] < 0.001 else
                   "**" if r["p_corrected"] < 0.01 else
                   "*" if r["p_corrected"] < 0.05 else "ns")
            print(f"  L{r['layer']:2d} sAUC: fk={r['mean_firstk']:.4f} "
                  f"lk={r['mean_lastk']:.4f}  "
                  f"delta={r['delta']:+.4f} [{r['delta_ci_lo']:+.4f}, "
                  f"{r['delta_ci_hi']:+.4f}]  "
                  f"p_holm={r['p_corrected']:.2e} r={r['effect_r']:.3f} {sig}")

    # Peak layer bootstrap (NSS)
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

    delta_fk_lk = np.array(peaks["firstk"]) - np.array(peaks["lastk"])
    peak_summary["delta_fk_lk"] = {
        "mean": float(np.mean(delta_fk_lk)),
        "ci_lo": float(np.percentile(delta_fk_lk, 2.5)),
        "ci_hi": float(np.percentile(delta_fk_lk, 97.5)),
    }
    print(f"  delta(firstk-lastk): mean={peak_summary['delta_fk_lk']['mean']:.2f}, "
          f"95%CI=[{peak_summary['delta_fk_lk']['ci_lo']:.1f}, "
          f"{peak_summary['delta_fk_lk']['ci_hi']:.1f}]")

    # Z-normalized profile correlation
    print("\n=== Z-normalized NSS profile correlation ===")
    profiles = {}
    for c in CONDITIONS:
        prof = np.array([np.mean(data[c]["NSS"][l]) for l in range(1, 13)])
        profiles[c] = (prof - prof.mean()) / prof.std()
    z_corr = float(np.corrcoef(profiles["firstk"], profiles["lastk"])[0, 1])
    print(f"  r(z_firstk, z_lastk) = {z_corr:.4f}")

    return test_rows, peak_summary, {"firstk_vs_lastk": z_corr}


# ==============================================================
# Plot
# ==============================================================

def plot_conditions(layer_summary, all_ref, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_display = {"NSS": "NSS", "AUC_Judd": "AUC-Judd", "sAUC": "sAUC"}
    colors = {"firstk": "coral", "lastk": "mediumpurple"}
    labels = {"firstk": f"First-K (K={K})", "lastk": f"Last-K (K={K})"}
    layers = np.arange(1, 13)

    for ax, m in zip(axes, METRICS):
        # thin reference line: all-fixations condition, no error band (context only)
        ref_means = np.array([np.mean(all_ref[m][l]) for l in range(1, 13)])
        ax.plot(layers, ref_means, "--", color="gray", lw=1, alpha=0.6,
                label="All fixations (ref.)")

        for c in CONDITIONS:
            means = layer_summary[c][m]["mean"]
            stds = layer_summary[c][m]["std"]
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

    fig.suptitle(f"Exp-B lastk: First-K vs Last-K ({GROUP}, 700 images)",
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

    # -- Step 1: lastk generation + matched firstk override --
    affected_stems, excluded_log, points_firstk_matched = prepare_data()

    # excluded-subjects audit log
    csv_excl = os.path.join(OUT_DIR, "excluded_subjects.csv")
    with open(csv_excl, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "subject_index", "n_fix"])
        w.writeheader()
        w.writerows(excluded_log)
    print(f"Saved: {csv_excl}")

    # -- Step 2: attention cache --
    attn_maps = load_attention_cache()
    print(f"Attention maps ready: {len(attn_maps)} images")

    # -- Step 3: fixation data + pools --
    print("Loading fixation data (firstk matched-override, lastk) ...")
    fix, pools = load_fixation_data(points_firstk_matched)
    for c in CONDITIONS:
        print(f"  {c}: {len(pools[c])} fix points pooled")

    expb_firstk_lookup = load_expb_firstk_lookup()

    # -- Step 4: metrics --
    print("Computing metrics (lastk: all images; firstk: affected images only) ...")
    rows = compute_metrics(attn_maps, fix, pools, affected_stems, expb_firstk_lookup)
    del attn_maps
    gc.collect()

    csv_img = os.path.join(OUT_DIR, "metrics_per_image.csv")
    with open(csv_img, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_img} ({len(rows)} rows)")

    # -- Build data structure: {condition: {metric: {layer: [values]}}} --
    data = {c: {m: {l: [] for l in range(1, 13)} for m in METRICS}
            for c in CONDITIONS}
    for r in rows:
        c = r["condition"]
        for m in METRICS:
            data[c][m][r["layer"]].append(r[m])

    n_img = len(set(r["image"] for r in rows))
    for l in range(1, 13):
        n_f = len(data["firstk"]["NSS"][l])
        n_k = len(data["lastk"]["NSS"][l])
        if not (n_f == n_k == n_img):
            print(f"WARNING: Layer {l} count mismatch: firstk={n_f}, lastk={n_k}")

    # -- Layer summary --
    layer_summary = {}
    summary_rows = []
    for c in CONDITIONS:
        layer_summary[c] = {}
        for m in METRICS:
            means = np.array([np.mean(data[c][m][l]) for l in range(1, 13)])
            stds = np.array([np.std(data[c][m][l], ddof=1) for l in range(1, 13)])
            layer_summary[c][m] = {"mean": means, "std": stds}
        for l in range(1, 13):
            row = {"layer": l, "condition": c}
            for m in METRICS:
                vals = data[c][m][l]
                ci_lo, ci_hi = bootstrap_ci(vals)
                row[f"{m}_mean"] = np.mean(vals)
                row[f"{m}_std"] = np.std(vals, ddof=1)
                row[f"{m}_ci_lo"] = ci_lo
                row[f"{m}_ci_hi"] = ci_hi
            summary_rows.append(row)

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

    print("\n=== Layer summary (sAUC mean +/- SD) ===")
    print(f"{'Layer':>5}  {'firstk':>16}  {'lastk':>16}")
    for l in range(1, 13):
        vals = {}
        for c in CONDITIONS:
            sr = [r for r in summary_rows if r["layer"] == l and r["condition"] == c][0]
            vals[c] = sr
        print(f"{l:5d}  "
              f"{vals['firstk']['sAUC_mean']:7.4f}+/-{vals['firstk']['sAUC_std']:.4f}  "
              f"{vals['lastk']['sAUC_mean']:7.4f}+/-{vals['lastk']['sAUC_std']:.4f}")

    # -- Statistics --
    test_rows, peak_summary, z_corrs = run_statistics(data)

    test_fields = ["test", "metric", "comparison", "layer", "statistic",
                   "p", "p_corrected", "effect_r",
                   "mean_firstk", "mean_lastk", "delta",
                   "delta_ci_lo", "delta_ci_hi"]
    csv_tests = os.path.join(OUT_DIR, "stats_firstk_vs_lastk.csv")
    with open(csv_tests, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=test_fields)
        w.writeheader()
        for row in test_rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"Saved: {csv_tests}")

    csv_peak = os.path.join(OUT_DIR, "peak_layer_summary.csv")
    with open(csv_peak, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        for c in CONDITIONS:
            for k, v in peak_summary[c].items():
                w.writerow([f"{c}_{k}", f"{v:.4f}" if isinstance(v, float) else v])
        d = peak_summary["delta_fk_lk"]
        w.writerow(["delta_fk_lk_mean", f"{d['mean']:.4f}"])
        w.writerow(["delta_fk_lk_ci_lo", f"{d['ci_lo']:.4f}"])
        w.writerow(["delta_fk_lk_ci_hi", f"{d['ci_hi']:.4f}"])
        for k, v in z_corrs.items():
            w.writerow([f"z_corr_{k}", f"{v:.6f}"])
    print(f"Saved: {csv_peak}")

    # -- Plot --
    all_ref = load_expb_all_lookup()
    plot_path = os.path.join(OUT_DIR, "layerwise_compare_firstk_lastk.png")
    plot_conditions(layer_summary, all_ref, plot_path)

    # -- run_config.json --
    elapsed = time.time() - t_start
    cfg = {
        "experiment": "expB_lastk",
        "purpose": "Test whether the sAUC L8-L10 firstk residual is specific "
                   "to early saccades (firstk) vs a point-count-matched late "
                   "condition (lastk)",
        "conditions": CONDITIONS,
        "K": K,
        "min_fix_for_inclusion": MIN_FIX,
        "n_images": n_img,
        "n_excluded_subjects": len(excluded_log),
        "n_affected_images": len(affected_stems),
        "n_firstk_recomputed_images": len(affected_stems),
        "n_firstk_reused_images": n_img - len(affected_stems),
        "sigma_px": SIGMA_PX,
        "metrics": METRICS,
        "n_boot": N_BOOT,
        "sAUC_neg": "per-condition pool, no self-image exclusion, matched "
                    "subject-exclusion on affected images for both conditions",
        "attn_cache": CACHE_FILE,
        "expb_csv": EXPB_CSV,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg_path = os.path.join(OUT_DIR, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved: {cfg_path}")

    print(f"\nTotal elapsed: {elapsed:.0f}s")
    print(f"Excluded subjects: {len(excluded_log)} across {len(affected_stems)} images "
          f"(firstk recomputed for these {len(affected_stems)}, reused for "
          f"{n_img - len(affected_stems)}).")


if __name__ == "__main__":
    main()
