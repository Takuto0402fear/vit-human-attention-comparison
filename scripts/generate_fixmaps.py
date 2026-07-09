"""
Generate Fixation Maps (human gaze ground truth) from OSIE fixations.mat.

Outputs per image (.npz):
  - points     : binary fixation map (H x W, uint8 0/1)
  - heat_all   : Gaussian-blurred heatmap using ALL fixations (float32, sum=1)
  - heat_firstk: Gaussian-blurred heatmap using first-K fixations (float32, sum=1)
  - img_name   : stimulus filename
  - n_subjects : number of observers
  - n_fix      : total fixation count (all mode)

QA outputs:
  - Overlay PNGs for sample images
  - Sigma sweep comparison PNGs
"""

import os
import sys
import glob
import numpy as np
import scipy.io
import scipy.ndimage
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────── CONFIG ───────────────────────────
IMG_W = 800
IMG_H = 600
GROUP = "healthy"

SIGMA_PX = 24                  # default Gaussian sigma (px), ~1 dva
SIGMA_SWEEP = [16, 24, 32]    # QA: sigmas to compare
FIRST_K = 2                   # first-K fixations per subject

COORD_ORIGIN_1INDEXED = True   # subtract 1 to convert to 0-indexed

DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR  = os.path.join(DATA_BASE, "data", "stimuli")
FIX_PATH  = os.path.join(DATA_BASE, "data", "eye", "fixations.mat")

OUT_DIR   = rf"C:\Users\user\gaze\outputs\fixmaps\{GROUP}"
QA_DIR    = os.path.join(OUT_DIR, "_qa")

QA_SAMPLES = 5  # number of sample images for QA overlay
# ──────────────────────────────────────────────────────────────


def load_fixations(fix_path):
    """Load fixations.mat and return the (700,1) struct array."""
    m = scipy.io.loadmat(fix_path)
    return m["fixations"]


def extract_fixation_points(struct, first_k=None):
    """
    Extract (x, y) fixation coordinates from one image's struct.

    Parameters
    ----------
    struct : fixations[i, 0]
    first_k : int or None
        If set, use only the first K fixations per subject.

    Returns
    -------
    xs, ys : 1-D arrays of float coordinates (raw, before coord conversion)
    n_subjects : int
    """
    subjects = struct["subjects"][0, 0]  # (n_subj, 1) object array
    n_subjects = subjects.shape[0]
    xs, ys = [], []
    for j in range(n_subjects):
        sj = subjects[j, 0]
        fx = sj["fix_x"][0, 0].flatten()
        fy = sj["fix_y"][0, 0].flatten()
        if first_k is not None:
            fx = fx[:first_k]
            fy = fy[:first_k]
        xs.append(fx)
        ys.append(fy)
    return np.concatenate(xs), np.concatenate(ys), n_subjects


def coords_to_pixels(xs, ys, w, h, one_indexed=True):
    """Convert float coords to integer pixel indices, clipped to [0, dim-1]."""
    px = np.round(xs).astype(int)
    py = np.round(ys).astype(int)
    if one_indexed:
        px -= 1
        py -= 1
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    return px, py


def make_binary_map(px, py, h, w):
    """Create a binary fixation map (H x W, uint8)."""
    bmap = np.zeros((h, w), dtype=np.uint8)
    bmap[py, px] = 1
    return bmap


def make_heatmap(binary_map, sigma):
    """Gaussian-blur a binary map and normalise to sum=1 (probability map)."""
    heat = scipy.ndimage.gaussian_filter(binary_map.astype(np.float64), sigma=sigma)
    s = heat.sum()
    if s > 0:
        heat /= s
    return heat.astype(np.float32)


def save_overlay_png(stim_path, heatmap, out_path, title=""):
    """Save stimulus + heatmap overlay as PNG."""
    img = Image.open(stim_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    ax.imshow(img)
    # normalise heatmap to [0,1] for display
    h_disp = heatmap.copy()
    hmax = h_disp.max()
    if hmax > 0:
        h_disp /= hmax
    ax.imshow(h_disp, cmap="jet", alpha=0.45)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def save_sigma_comparison(stim_path, binary_map, sigmas, out_path, img_name):
    """Side-by-side sigma comparison PNG."""
    img = Image.open(stim_path).convert("RGB")
    n = len(sigmas)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=100)
    if n == 1:
        axes = [axes]
    for ax, sigma in zip(axes, sigmas):
        heat = make_heatmap(binary_map, sigma)
        h_disp = heat / heat.max() if heat.max() > 0 else heat
        ax.imshow(img)
        ax.imshow(h_disp, cmap="jet", alpha=0.45)
        ax.set_title(f"sigma={sigma}", fontsize=11)
        ax.set_axis_off()
    fig.suptitle(img_name, fontsize=12)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(QA_DIR, exist_ok=True)

    print(f"Loading fixations from {FIX_PATH} ...")
    fixations = load_fixations(FIX_PATH)
    n_images = fixations.shape[0]
    print(f"  {n_images} images found.")

    # Pick QA sample indices (first one is index 0 = 1001.jpg, rest evenly spaced)
    qa_indices = [0] + list(np.linspace(1, n_images - 1, QA_SAMPLES - 1, dtype=int))
    qa_indices = sorted(set(qa_indices))[:QA_SAMPLES]
    print(f"  QA sample indices: {qa_indices}")

    generated = 0
    report_interval = max(1, n_images // 10)

    for i in range(n_images):
        struct = fixations[i, 0]
        img_name = str(struct["img"][0, 0][0])

        # --- all fixations ---
        xs_all, ys_all, n_subj = extract_fixation_points(struct, first_k=None)
        px_all, py_all = coords_to_pixels(xs_all, ys_all, IMG_W, IMG_H,
                                           one_indexed=COORD_ORIGIN_1INDEXED)
        binary_map = make_binary_map(px_all, py_all, IMG_H, IMG_W)
        heat_all = make_heatmap(binary_map, SIGMA_PX)

        # --- first-K fixations ---
        xs_fk, ys_fk, _ = extract_fixation_points(struct, first_k=FIRST_K)
        px_fk, py_fk = coords_to_pixels(xs_fk, ys_fk, IMG_W, IMG_H,
                                          one_indexed=COORD_ORIGIN_1INDEXED)
        binary_fk = make_binary_map(px_fk, py_fk, IMG_H, IMG_W)
        heat_firstk = make_heatmap(binary_fk, SIGMA_PX)

        # --- save npz ---
        stem = os.path.splitext(img_name)[0]
        npz_path = os.path.join(OUT_DIR, f"{stem}.npz")
        np.savez_compressed(
            npz_path,
            points=binary_map,
            heat_all=heat_all,
            heat_firstk=heat_firstk,
            img_name=img_name,
            n_subjects=n_subj,
            n_fix=len(xs_all),
        )
        generated += 1

        # --- QA overlays ---
        if i in qa_indices:
            stim_path = os.path.join(STIM_DIR, img_name)
            # overlay with default sigma
            overlay_path = os.path.join(QA_DIR, f"{stem}_overlay.png")
            save_overlay_png(stim_path, heat_all, overlay_path,
                             title=f"{img_name}  sigma={SIGMA_PX}  fix={len(xs_all)}")
            # sigma sweep
            sweep_path = os.path.join(QA_DIR, f"{stem}_sigma_sweep.png")
            save_sigma_comparison(stim_path, binary_map, SIGMA_SWEEP, sweep_path, img_name)

        # progress
        if (i + 1) % report_interval == 0 or i == n_images - 1:
            pct = (i + 1) / n_images * 100
            print(f"  [{pct:5.1f}%] {i+1}/{n_images}  ({img_name})")

    print(f"\nDone. Generated {generated} .npz files in {OUT_DIR}")

    # ─── Verification ───
    print("\n=== QA Verification ===")
    # 1) count
    npz_files = glob.glob(os.path.join(OUT_DIR, "*.npz"))
    print(f"1) .npz count: {len(npz_files)}")

    # 2) sum check on a few files
    print("2) heat_all sum check (should be ~1.0):")
    for idx in qa_indices[:5]:
        struct = fixations[idx, 0]
        stem = os.path.splitext(str(struct["img"][0, 0][0]))[0]
        d = np.load(os.path.join(OUT_DIR, f"{stem}.npz"))
        s = float(d["heat_all"].sum())
        print(f"   {d['img_name']}:  sum={s:.6f}")

    # 3) QA paths
    qa_files = sorted(glob.glob(os.path.join(QA_DIR, "*.png")))
    print(f"3) QA PNGs saved in: {QA_DIR}")
    for f in qa_files:
        print(f"   {os.path.basename(f)}")

    print("\n4) Sigma comparison -- see the _sigma_sweep.png files above.")


if __name__ == "__main__":
    main()
