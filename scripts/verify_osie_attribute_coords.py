"""
QA gate: OSIE attrs.mat <-> stimuli coordinate-alignment verification.

Must be run (and visually inspected) BEFORE
scripts/run_osie_attribute_grounding_pilot.py. Confirms, on 3 sample
images, that attrs.mat object masks align with the original OSIE stimuli
correctly, since no existing code in this repo reads attrs.mat:

  - image-name correspondence: attrs.mat's per-entry 'img' field vs
    data/stimuli/*.jpg filenames, checked for ALL 700 entries (not just
    the 3 samples).
  - object-mask array orientation: attrs.mat is MATLAB v7.3 (HDF5), so
    scipy.io.loadmat cannot read it (raises NotImplementedError) -- h5py
    is used instead. h5py does NOT apply the axis-order swap that
    scipy.io.loadmat applies for older .mat files, so a MATLAB array
    stored as (rows=600, cols=800) is read back by h5py with shape
    (800, 600). Investigation (manual overlay of the union mask and the
    'face'/'text' attribute masks on the original image for 1001/1023/
    1139) confirmed that a plain `.T` -- no flip -- recovers exact
    pixel-for-pixel alignment with the image's native (H, W) = (600, 800)
    numpy layout.
  - no interaction with clip_extractor.py's ViT-input padding/CenterCrop
    logic: attrs.mat masks are at native stimulus resolution and are
    never passed through clip_extractor.py.

Read-only w.r.t. attrs.mat, fixations.mat, and stimuli/*.jpg. Does not
import or modify clip_extractor.py, vit_extractor.py, lib/, or any
existing output directory. Writes only under
outputs/osie_attribute_grounding_pilot/qa/.

Usage (PowerShell):
    python scripts\\verify_osie_attribute_coords.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import glob
import json
import os

import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
DATA_BASE  = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR   = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_pilot\qa"

SAMPLE_IMAGE_IDS = ["1001", "1023", "1139"]
HIGHLIGHT_ATTRS = ["face", "text"]
EXPECTED_ATTR_NAMES = [
    "text", "face", "emotion", "sound", "smell", "taste", "touch",
    "motion", "operability", "watchability", "touched", "gazed",
]
EXPECTED_N_IMAGES = 700
EXPECTED_IMG_WH = (800, 600)
# ======================================================


def h5_char_str(f, ref):
    arr = f[ref][()]
    return "".join(chr(int(c)) for c in arr.flatten())


def load_attrs_index(f):
    """Returns (attr_names list[12], {img_stem: h5py.Group}) -- read-only."""
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]

    attrs_ds = f["attrs"]
    n_img = attrs_ds.shape[1]
    index = {}
    for i in range(n_img):
        g = f[attrs_ds[0, i]]
        img_arr = g["img"][()]
        raw_name = "".join(chr(int(c)) for c in img_arr.flatten())
        stem = os.path.splitext(raw_name)[0]
        if stem in index:
            raise RuntimeError(f"STOP: duplicate attrs.mat image id {stem}")
        index[stem] = g
    return attr_names, index


def object_masks_for_image(f, group, attr_names):
    """
    Returns (fg_mask (H,W) bool, per_attr_masks {name: (H,W) bool}, n_objects).

    group['objs'][0, j]['map'] is read as-is from h5py (shape (800, 600))
    and TRANSPOSED (.T) to match the image's native (H, W) = (600, 800)
    layout -- see module docstring. No flip is applied.
    """
    objs_ds = group["objs"]
    n_objs = objs_ds.shape[1]
    if n_objs == 0:
        raise RuntimeError("STOP: image with zero objects")

    sample_map = f[objs_ds[0, 0]]["map"][()]
    h, w = sample_map.T.shape
    fg = np.zeros((h, w), dtype=bool)
    per_attr = {name: np.zeros((h, w), dtype=bool) for name in attr_names}
    for j in range(n_objs):
        obj_g = f[objs_ds[0, j]]
        raw_map = obj_g["map"][()]
        mp = raw_map.T.astype(bool)
        if mp.shape != (h, w):
            raise RuntimeError(f"STOP: object {j} mask shape {mp.shape} != {(h, w)}")
        fg |= mp
        feat = obj_g["features"][()].flatten()
        if feat.shape[0] != len(attr_names):
            raise RuntimeError(
                f"STOP: feature vector length {feat.shape[0]} != "
                f"{len(attr_names)} attrNames")
        for k, name in enumerate(attr_names):
            if feat[k] > 0:
                per_attr[name] |= mp
    return fg, per_attr, n_objs


def check_full_image_name_correspondence(index):
    """Cross-check ALL attrs.mat image ids against data/stimuli/*.jpg (not just the 3 samples)."""
    stim_ids = set(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(STIM_DIR, "*.jpg")))
    attrs_ids = set(index.keys())

    only_in_attrs = sorted(attrs_ids - stim_ids)
    only_in_stim = sorted(stim_ids - attrs_ids)
    return {
        "n_stimuli": len(stim_ids),
        "n_attrs": len(attrs_ids),
        "n_common": len(attrs_ids & stim_ids),
        "only_in_attrs": only_in_attrs,
        "only_in_stim_not_in_attrs": only_in_stim,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  QA: OSIE attrs.mat <-> stimuli coordinate verification")
    print("=" * 65)

    if not os.path.isfile(ATTRS_PATH):
        raise RuntimeError(f"STOP: attrs.mat not found at {ATTRS_PATH}")

    f = h5py.File(ATTRS_PATH, "r")
    attr_names, index = load_attrs_index(f)
    if attr_names != EXPECTED_ATTR_NAMES:
        raise RuntimeError(f"STOP: attrNames mismatch, got {attr_names}")
    print(f"  attrNames ({len(attr_names)}): {attr_names}")
    print(f"  attrs.mat image count: {len(index)}")
    if len(index) != EXPECTED_N_IMAGES:
        raise RuntimeError(
            f"STOP: expected {EXPECTED_N_IMAGES} attrs.mat images, got {len(index)}")

    print("\n--- Full-dataset image-name correspondence (attrs.mat vs stimuli/) ---")
    name_check = check_full_image_name_correspondence(index)
    print(f"  stimuli files: {name_check['n_stimuli']}")
    print(f"  attrs.mat entries: {name_check['n_attrs']}")
    print(f"  common: {name_check['n_common']}")
    if name_check["only_in_attrs"] or name_check["only_in_stim_not_in_attrs"]:
        raise RuntimeError(
            f"STOP: image-id mismatch between attrs.mat and stimuli/. "
            f"only_in_attrs={name_check['only_in_attrs'][:10]} "
            f"only_in_stim={name_check['only_in_stim_not_in_attrs'][:10]}")
    print("  PASS: attrs.mat and data/stimuli/ cover exactly the same 700 image ids.")

    warnings = []
    results = []
    print("\n--- Per-sample-image overlay generation ---")
    for stem in SAMPLE_IMAGE_IDS:
        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        if not os.path.isfile(img_path):
            raise RuntimeError(f"STOP: stimulus image missing for {stem}")
        if stem not in index:
            raise RuntimeError(f"STOP: {stem} not found in attrs.mat image index")

        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        if (W, H) != EXPECTED_IMG_WH:
            warnings.append(f"{stem}: unexpected stimulus size {(W, H)}, expected {EXPECTED_IMG_WH}")

        group = index[stem]
        fg_mask, per_attr, n_objs = object_masks_for_image(f, group, attr_names)
        if fg_mask.shape != (H, W):
            raise RuntimeError(
                f"STOP: {stem} mask shape {fg_mask.shape} != image shape {(H, W)} "
                f"after transpose -- coordinate alignment FAILED")

        fg_area_frac = float(fg_mask.mean())
        print(f"  [{stem}] objects={n_objs}  mask_shape={fg_mask.shape}  "
              f"image_shape={(H, W)}  fg_area_fraction={fg_area_frac:.4f}")

        fig, axes = plt.subplots(1, 2 + len(HIGHLIGHT_ATTRS),
                                  figsize=(6 * (2 + len(HIGHLIGHT_ATTRS)), 5))
        axes[0].imshow(img)
        axes[0].set_title(f"{stem}: original ({W}x{H})")
        axes[0].axis("off")

        axes[1].imshow(img)
        axes[1].imshow(np.where(fg_mask, 1.0, np.nan), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title(f"foreground union ({n_objs} objects, area={fg_area_frac:.3f})")
        axes[1].axis("off")

        for a_idx, attr_name in enumerate(HIGHLIGHT_ATTRS):
            ax = axes[2 + a_idx]
            ax.imshow(img)
            mask = per_attr[attr_name]
            if mask.any():
                ax.imshow(np.where(mask, 1.0, np.nan), cmap="cool", alpha=0.6, vmin=0, vmax=1)
            ax.set_title(f"{attr_name} (pixels={int(mask.sum())})")
            ax.axis("off")

        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f"coord_check_{stem}.png")
        plt.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"    saved: {out_png}")

        results.append({
            "image": stem,
            "n_objects": n_objs,
            "image_shape_hw": [H, W],
            "mask_shape_hw": list(fg_mask.shape),
            "fg_area_fraction": fg_area_frac,
            "attr_pixel_counts": {k: int(v.sum()) for k, v in per_attr.items()},
        })

    print("\n--- Verdict ---")
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("  No shape/size warnings.")
    print("  Mask <-> image shape matched for all sample images after .T transpose: PASS")
    print("  Manual visual check required before proceeding: open the PNGs above and")
    print("  confirm masks fall exactly on object silhouettes with no offset/flip/transpose error.")

    summary_path = os.path.join(OUT_DIR, "coord_check_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "attrs_path": ATTRS_PATH,
            "attr_names": attr_names,
            "full_dataset_name_check": name_check,
            "sample_images": SAMPLE_IMAGE_IDS,
            "results": results,
            "warnings": warnings,
            "mask_orientation_rule": (
                "h5py-read obj.map has shape (800,600) (MATLAB v7.3/HDF5 "
                "column-major vs row-major storage order, NOT scipy.io.loadmat- "
                "compatible). Transpose (.T) recovers the image's native "
                "(H,W)=(600,800) layout. No flip needed. Verified against PIL "
                "stimulus image shape and visual overlay for 1001/1023/1139."
            ),
        }, fh, indent=2)
    print(f"  Saved: {summary_path}")
    print("\n" + "=" * 65)
    print("  QA coordinate check: DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
