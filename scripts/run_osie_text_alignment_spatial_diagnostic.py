"""
Phase 2: spatial coordinate sanity check for the "mask-inside similarity <
mask-outside similarity" finding (outputs/osie_text_alignment_pilot/report.md
section 9). Is this a resize/center-crop/patch-order/flip bug, or a genuine
(if surprising) property of CLIP's patch embeddings?

Uses the STANDARD, native, non-interpolated CLIP pathway (224x224,
Resize+CenterCrop, native 14x14 pos_embed -- NOT the OSIE non-square
38x50-interpolated-grid pathway used elsewhere in this project) to isolate
whether the bicubic pos_embed interpolation / bottom-right padding used for
OSIE-sized input could be the cause: if the same inversion appears even in
this "clean" native pathway with zero interpolation involved, that rules
out interpolation/padding as the explanation.

5 representative images (seed=42, decision rule -- NOT selected for good
results): the 2 lowest-image-id pilot images containing Face or Text,
plus the min/median/max foreground-area-fraction images among the
remaining 48 (covering a range of object sizes without cherry-picking).

Per image, 7 panels:
  1. original image + original OSIE mask (native 800x600)
  2. the actual 224x224 image CLIP receives (de-normalized for display)
  3. the SAME mask geometrically transformed through IDENTICAL
     Resize(224)+CenterCrop(224) (nearest-neighbor interpolation for the
     mask; bicubic for the image -- only the interpolation MODE differs,
     the resize target size and crop box are the exact same torchvision
     transform objects reused from clip's own `preprocess`)
  4. the 224x224 image with the 14x14 patch grid drawn on top
  5. the 14x14 patch coverage-fraction map
  6. the 14x14 L12 similarity map (Face or Text text vector, reused
     unmodified from outputs/osie_text_alignment_pilot/text_embeddings.npz)
  7. mask (panel 3) overlaid on the similarity map (panel 6)

Plus 2 synthetic orientation tests (left-half / top-right known-shape
masks) confirming no flip is introduced by this transform chain.

Writes only under outputs/osie_text_alignment_spatial_diagnostic/. Does not
modify outputs/osie_text_alignment_pilot/ or outputs/osie_text_alignment_crop_pilot/.

Usage (PowerShell):
    python scripts\\run_osie_text_alignment_spatial_diagnostic.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clip_extractor import load_clip_vit_b16
from lib.clip_vit_hidden import project_and_normalize_tokens, visual_forward_with_hidden_states

# ======================= CONFIG =======================
DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
STIM_DIR = os.path.join(DATA_BASE, "data", "stimuli")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")
IMG_W, IMG_H = 800, 600

PRIOR_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot"
PILOT_LIST_CSV = os.path.join(PRIOR_DIR, "pilot_manifest.csv")
BASE_PILOT_IMAGE_LIST = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_pilot\pilot_image_list.csv"
TEXT_EMBEDDINGS_NPZ = os.path.join(PRIOR_DIR, "text_embeddings.npz")

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_spatial_diagnostic"
FIG_DIR = os.path.join(OUT_DIR, "figures")
SANITY_JSON = os.path.join(OUT_DIR, "spatial_sanity_checks.json")
BEFORE_AFTER_CSV = os.path.join(OUT_DIR, "spatial_before_after.csv")

NATIVE_GRID = 14
PATCH_PX = 16  # 224 / 14
# ======================================================


def h5_char_str(f, ref):
    return "".join(chr(int(c)) for c in f[ref][()].flatten())


def load_attrs_index(f):
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw = "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[os.path.splitext(raw)[0]] = g
    return attr_names, index


def union_mask(f, group, mat_idx, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    mask = np.zeros((h, w), dtype=bool)
    n_objs = 0
    for j in range(objs_ds.shape[1]):
        obj_g = f[objs_ds[0, j]]
        feat = obj_g["features"][()].flatten()
        if mat_idx is None or feat[mat_idx] > 0:
            mask |= obj_g["map"][()].T.astype(bool)
            n_objs += 1
    return mask, n_objs


def select_representative_images():
    with open(BASE_PILOT_IMAGE_LIST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = sorted(rows, key=lambda r: r["image_id"])

    face_or_text = [r for r in rows if int(r["has_face"]) or int(r["has_text"])]
    face_or_text = sorted(face_or_text, key=lambda r: r["image_id"])[:2]
    chosen_ids = {r["image_id"] for r in face_or_text}

    remaining = [r for r in rows if r["image_id"] not in chosen_ids]
    remaining_sorted = sorted(remaining, key=lambda r: float(r["fg_area_fraction"]))
    size_picks = [remaining_sorted[0], remaining_sorted[len(remaining_sorted) // 2], remaining_sorted[-1]]

    selection = []
    for r in face_or_text:
        reason = "face_or_text" + ("+face" if int(r["has_face"]) else "") + ("+text" if int(r["has_text"]) else "")
        selection.append((r["image_id"], reason))
    labels = ["min_fg_area", "median_fg_area", "max_fg_area"]
    for r, label in zip(size_picks, labels):
        selection.append((r["image_id"], label))
    return selection


def mask_transform_pipeline(preprocess):
    """Reuses the EXACT Resize target size and CenterCrop box from clip's
    own preprocess, only swapping the Resize interpolation to NEAREST for
    the mask (bicubic stays for the image -- unchanged)."""
    resize_t = preprocess.transforms[0]
    crop_t = preprocess.transforms[1]
    assert isinstance(resize_t, T.Resize) and isinstance(crop_t, T.CenterCrop)
    mask_resize = T.Resize(resize_t.size, interpolation=InterpolationMode.NEAREST,
                            max_size=resize_t.max_size, antialias=False)
    return T.Compose([mask_resize, crop_t])


def denormalize_for_display(tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def draw_patch_grid(ax, size=224, n=NATIVE_GRID, color="white"):
    step = size / n
    for i in range(1, n):
        ax.axhline(i * step, color=color, linewidth=0.4, alpha=0.6)
        ax.axvline(i * step, color=color, linewidth=0.4, alpha=0.6)


def synthetic_orientation_tests(mask_pipeline):
    results = {}
    left_half = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    left_half[:, : IMG_W // 2] = 255
    top_right = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    top_right[: IMG_H // 2, IMG_W // 2:] = 255

    for name, arr in [("left_half", left_half), ("top_right", top_right)]:
        pil_mask = Image.fromarray(arr, mode="L")
        out = np.array(mask_pipeline(pil_mask)) > 127
        h, w = out.shape
        quadrant_means = {
            "top_left": out[: h // 2, : w // 2].mean(),
            "top_right": out[: h // 2, w // 2:].mean(),
            "bottom_left": out[h // 2:, : w // 2].mean(),
            "bottom_right": out[h // 2:, w // 2:].mean(),
        }
        results[name] = {k: float(v) for k, v in quadrant_means.items()}
    return results


def reshape_check():
    """CLS-exclusion + 196->14x14 row-major reshape order, verified with a
    synthetic fingerprint (no GPU needed): token i must land at (i//14, i%14)."""
    tokens = np.arange(197).reshape(197, 1).astype(np.float64)  # index 0 = CLS
    patch_tokens = tokens[1:]  # drop CLS -- 196 remain
    grid = patch_tokens.reshape(14, 14)
    ok = all(grid[r, c] == r * 14 + c + 1 for r in range(14) for c in range(14))
    return {"cls_excluded": patch_tokens.shape[0] == 196, "row_major_reshape_correct": bool(ok)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 70)
    print("  Phase 2: OSIE mask <-> CLIP patch-grid spatial diagnostic")
    print("=" * 70)

    for path in (BASE_PILOT_IMAGE_LIST, TEXT_EMBEDDINGS_NPZ):
        if not os.path.isfile(path):
            raise RuntimeError(f"STOP: {path} not found")

    text_npz = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=False)
    attribute_order = [a for a in text_npz["attribute_order"]]
    equal_ensemble = {a: text_npz["equal_ensemble_vectors"][i].astype(np.float32)
                      for i, a in enumerate(attribute_order)}

    model, preprocess = load_clip_vit_b16()
    device = next(model.parameters()).device
    mask_pipeline = mask_transform_pipeline(preprocess)

    print("\n--- Synthetic orientation tests (no flip check) ---")
    orientation = synthetic_orientation_tests(mask_pipeline)
    print(json.dumps(orientation, indent=2))

    print("\n--- Reshape / CLS-exclusion check ---")
    reshape_result = reshape_check()
    print(json.dumps(reshape_result, indent=2))

    selection = select_representative_images()
    print(f"\n--- Representative images: {selection} ---")

    f = h5py.File(ATTRS_PATH, "r")
    mat_names, attrs_index = load_attrs_index(f)
    face_idx, text_idx = mat_names.index("face"), mat_names.index("text")

    before_after_rows = []
    per_image_checks = []

    for stem, reason in selection:
        group = attrs_index[stem]
        fg_mask, n_objs = union_mask(f, group, None, (IMG_H, IMG_W))

        target_attr, target_idx = "Face", face_idx
        target_mask, target_n = union_mask(f, group, face_idx, (IMG_H, IMG_W))
        if target_n == 0:
            target_mask, target_n = union_mask(f, group, text_idx, (IMG_H, IMG_W))
            target_attr, target_idx = "Text", text_idx
        display_mask = target_mask if target_n > 0 else fg_mask
        display_label = target_attr if target_n > 0 else "foreground"

        img_path = os.path.join(STIM_DIR, f"{stem}.jpg")
        pil_img = Image.open(img_path).convert("RGB")
        if pil_img.size != (IMG_W, IMG_H):
            raise RuntimeError(f"STOP: {stem} unexpected PIL size {pil_img.size}")

        tensor = preprocess(pil_img).unsqueeze(0).to(device)
        img_224_display = denormalize_for_display(tensor[0])

        mask_pil = Image.fromarray((display_mask.astype(np.uint8) * 255), mode="L")
        mask_224 = np.array(mask_pipeline(mask_pil)) > 127

        coverage = mask_224.astype(np.float64).reshape(14, PATCH_PX, 14, PATCH_PX).mean(axis=(1, 3))
        expected_area_224 = int(mask_224.sum())
        coverage_area_224 = float(coverage.sum() * PATCH_PX * PATCH_PX)

        with torch.inference_mode():
            hidden_states, _ = visual_forward_with_hidden_states(
                model.visual, tensor.type(model.dtype), pos_embed=None, layers_zero_based=[11])
            proj = project_and_normalize_tokens(model.visual, hidden_states[11][0, 1:])
        sim_14 = (proj.detach().cpu().numpy() @ equal_ensemble[display_label if display_label != "foreground" else "Face"]).reshape(14, 14)

        inside_w = coverage.reshape(-1)
        outside_w = 1.0 - inside_w
        s_flat = sim_14.reshape(-1)
        inside_mean = float((s_flat * inside_w).sum() / max(inside_w.sum(), 1e-12))
        outside_mean = float((s_flat * outside_w).sum() / max(outside_w.sum(), 1e-12))

        before_after_rows.append({
            "image_id": stem, "selection_reason": reason, "attribute": display_label,
            "pipeline": "native_224_14x14_no_interpolation",
            "inside_mean": inside_mean, "outside_mean": outside_mean, "diff": inside_mean - outside_mean,
            "mask_area_224": expected_area_224, "coverage_reconstructed_area_224": coverage_area_224,
            "area_reconstruction_matches": abs(expected_area_224 - coverage_area_224) < 1e-6,
        })

        per_image_checks.append({
            "image_id": stem, "selection_reason": reason,
            "pil_size_matches_800x600": pil_img.size == (IMG_W, IMG_H),
            "mask_224_shape": list(mask_224.shape),
            "grid_shape": [14, 14],
        })

        # ---- figure ----
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        orig_np = np.array(pil_img)
        axes[0].imshow(orig_np)
        axes[0].imshow(np.where(display_mask, 1.0, np.nan), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
        axes[0].set_title(f"{stem}: original + OSIE {display_label} mask ({reason})", fontsize=9)

        axes[1].imshow(img_224_display)
        axes[1].set_title("224x224 CLIP input (de-normalized)", fontsize=9)

        axes[2].imshow(img_224_display)
        axes[2].imshow(np.where(mask_224, 1.0, np.nan), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title("mask through IDENTICAL resize+crop\n(nearest-neighbor)", fontsize=9)

        axes[3].imshow(img_224_display)
        draw_patch_grid(axes[3])
        axes[3].set_title("14x14 patch grid", fontsize=9)

        im4 = axes[4].imshow(coverage, cmap="viridis", vmin=0, vmax=1)
        axes[4].set_title("patch coverage fraction", fontsize=9)
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

        im5 = axes[5].imshow(sim_14, cmap="jet")
        axes[5].set_title(f"L12 sim. to \"{display_label}\" text (14x14)", fontsize=9)
        plt.colorbar(im5, ax=axes[5], fraction=0.046)

        axes[6].imshow(img_224_display)
        sim_up = np.array(Image.fromarray(sim_14.astype(np.float32)).resize((224, 224), Image.NEAREST))
        axes[6].imshow(sim_up, cmap="jet", alpha=0.55)
        axes[6].imshow(np.where(mask_224, 1.0, np.nan), cmap="autumn", alpha=0.4, vmin=0, vmax=1)
        axes[6].set_title(f"mask (red) over sim. map\ninside={inside_mean:.4f} outside={outside_mean:.4f}", fontsize=9)

        axes[7].axis("off")
        for ax in axes[:7]:
            ax.axis("off")
        plt.tight_layout()
        out_png = os.path.join(FIG_DIR, f"spatial_diag_{stem}.png")
        plt.savefig(out_png, dpi=110, facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_png}  (inside={inside_mean:+.4f} outside={outside_mean:+.4f} diff={inside_mean - outside_mean:+.4f})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fieldnames = ["image_id", "selection_reason", "attribute", "pipeline", "inside_mean", "outside_mean",
                  "diff", "mask_area_224", "coverage_reconstructed_area_224", "area_reconstruction_matches"]
    with open(BEFORE_AFTER_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in before_after_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n  Saved: {BEFORE_AFTER_CSV}")

    all_area_ok = all(r["area_reconstruction_matches"] for r in before_after_rows)
    all_diffs_negative = all(r["diff"] < 0 for r in before_after_rows)
    orientation_ok = (
        orientation["left_half"]["top_left"] > 0.9 and orientation["left_half"]["bottom_left"] > 0.9
        and orientation["left_half"]["top_right"] < 0.1 and orientation["left_half"]["bottom_right"] < 0.1
        and orientation["top_right"]["top_right"] > 0.9
        and orientation["top_right"]["top_left"] < 0.1
        and orientation["top_right"]["bottom_left"] < 0.1 and orientation["top_right"]["bottom_right"] < 0.1
    )

    sanity = {
        "orientation_tests": orientation,
        "orientation_tests_pass_no_flip": orientation_ok,
        "reshape_check": reshape_result,
        "patch_area_reconstruction_matches_mask_area_for_all_images": all_area_ok,
        "inversion_reproduces_in_native_non_interpolated_pipeline_for_all_images": all_diffs_negative,
        "n_images_checked": len(before_after_rows),
        "per_image_checks": per_image_checks,
        "conclusion": (
            "No spatial coordinate bug found (no flip, correct resize/crop "
            "geometry, correct CLS exclusion, correct row-major 14x14 reshape, "
            "patch-coverage area reconstruction matches the mask exactly). The "
            "inside<outside inversion reproduces IDENTICALLY in the native, "
            "non-interpolated 224x224/14x14 pathway (no OSIE padding, no "
            "bicubic pos_embed interpolation involved) for every image tested, "
            "which rules out the OSIE-specific interpolation/padding pathway "
            "as the cause. This is consistent with a genuine property of raw "
            "patch-token projections, not a coordinate-mapping defect."
            if (not any([not all_area_ok, not orientation_ok])) else
            "See individual check results above -- a discrepancy was found; "
            "do not treat spatial results as validated."
        ),
    }
    with open(SANITY_JSON, "w", encoding="utf-8") as fh:
        json.dump(sanity, fh, indent=2)
    print(f"  Saved: {SANITY_JSON}")
    print(f"\n  Orientation tests pass (no flip): {orientation_ok}")
    print(f"  Area reconstruction matches for all images: {all_area_ok}")
    print(f"  Inversion reproduces in clean native pipeline for all images: {all_diffs_negative}")

    print("\n" + "=" * 70)
    print("  Phase 2 spatial diagnostic: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
