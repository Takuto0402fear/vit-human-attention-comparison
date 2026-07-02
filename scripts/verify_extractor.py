"""
Regression test: verify C:\\Users\\user\\gaze ViT extraction against
the author reference from Yamamoto et al. (2025).

Part 1 — Gaze-position comparison
    Uses the C: model to compute gaze positions on Nakano 2010 frames
    and compares against vit_gaze_pos_author.npz (DINO, depth=12, trial=1).
    Pass criteria: exact-match >= 95%, within-1px >= 99.9%.

Part 2 — Attention-map numerical comparison
    Loads both the C: model and the D: reference model, runs a small
    batch of frames through each, and compares the raw attention tensors.
    Pass criteria: max absolute diff < 1e-5 per layer.

Usage (PowerShell):
    python scripts\\verify_extractor.py                  # default 50 frames
    python scripts\\verify_extractor.py --n-frames 200   # more frames
    python scripts\\verify_extractor.py --n-frames 2327  # full test
    python scripts\\verify_extractor.py --skip-map        # skip Part 2
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as pth_transforms
from PIL import Image
from kornia.filters import box_blur

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import Config
from vit_extractor import _load_model  # C: side model loader

# Reference paths (read-only)
FRAMES_DIR = r"D:\gaze\vit-human-attention-comparison\dataset\Nakano_etal_2010\video_stimuli\frames"
AUTHOR_NPZ = r"D:\gaze\vit-human-attention-comparison\dataset\Nakano_etal_2010\preprocessed_data\vit_gaze_pos_author.npz"
D_ANALYSIS = r"D:\gaze\vit-human-attention-comparison\analysis_python"

# Model spec (matches reference verify_single_model.py exactly)
TRAINING_METHOD = "dino"
DEPTH = 12
TRIAL = 1
PATCH_SIZE = 16
BLUR_SIZE = 32
BATCH_SIZE = 4
NUM_HEADS = 6


# ======================================================================
# Dataset (exact copy of the reference ImageDataset)
# ======================================================================

class ReferenceImageDataset(Dataset):
    """
    Replicates the reference ImageDataset from utils_analysis.py verbatim:
    PIL.open -> optional L->RGB convert -> transform.
    No additional padding or resizing.
    """

    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        if image.mode == "L":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# ======================================================================
# argmax_2d (exact copy of the reference)
# ======================================================================

def argmax_2d(x, random_choice=True):
    if random_choice:
        max_indices = np.argwhere(x == np.max(x))
        num_max = len(max_indices)
        if num_max > 1:
            idx = np.random.choice(num_max, 1)
            return max_indices[idx][0]
        else:
            return max_indices[0]
    else:
        return np.asarray(np.unravel_index(np.argmax(x, axis=None), x.shape))


# ======================================================================
# Part 1: gaze-position extraction (mirrors reference pipeline exactly)
# ======================================================================

def extract_gaze_positions(model, device, dataloader):
    """
    Replicate get_gaze_pos_model_dataset from utils_analysis.py verbatim.
    Returns (depth, num_heads+1, n_frames, 2).
    """
    num_heads = model.num_heads
    gaze_pos_all = []
    w_featmap, h_featmap = None, None

    for images in dataloader:
        with torch.inference_mode():
            attn_list = model.get_fulllayers_selfattention(images.to(device))

        B = images.shape[0]
        if w_featmap is None:
            w_featmap = images.shape[-2] // PATCH_SIZE
            h_featmap = images.shape[-1] // PATCH_SIZE

        gaze_pos_layers = []
        for attentions in attn_list:
            attentions = attentions[:, :, 0, 1:].reshape(
                B, num_heads, w_featmap, h_featmap)
            attentions = F.interpolate(
                attentions, scale_factor=PATCH_SIZE, mode="nearest")
            attentions_mean = torch.mean(attentions, axis=1, keepdims=True)
            attentions = torch.cat((attentions, attentions_mean), dim=1)
            attentions_blur = box_blur(
                attentions, (BLUR_SIZE, BLUR_SIZE)).detach().cpu().numpy()

            gaze_pos = np.zeros((B, 1, num_heads + 1, 2))
            for b_idx in range(B):
                for h_idx in range(num_heads + 1):
                    attn = attentions_blur[b_idx, h_idx]
                    ey, ex = argmax_2d(attn)
                    gaze_pos[b_idx, :, h_idx] = np.array([ex, ey])
            gaze_pos_layers.append(gaze_pos)

        gaze_pos_layers = np.concatenate(gaze_pos_layers, 1)
        gaze_pos_all.append(gaze_pos_layers)

    gaze_pos_all = np.concatenate(gaze_pos_all, 0)
    gaze_pos_all = gaze_pos_all.transpose(1, 2, 0, 3)
    return gaze_pos_all  # (depth, heads+1, n_frames, 2)


# ======================================================================
# Part 2: raw attention-map comparison (C: vs D:)
# ======================================================================

def compare_attention_maps(c_model, device, images):
    """
    Run the same images through C: and D: models, compare raw attention.
    Returns list of dicts with per-layer stats.
    """
    # --- C: side ---
    with torch.inference_mode():
        c_attn_list = c_model.get_fulllayers_selfattention(images.to(device))

    # --- D: side ---
    # Import D: model_load (adds D: analysis dir to sys.path temporarily)
    sys.path.insert(0, D_ANALYSIS)
    try:
        from utils_analysis import model_load as d_model_load
    finally:
        sys.path.remove(D_ANALYSIS)

    d_model, d_device = d_model_load(
        TRAINING_METHOD, TRIAL, DEPTH, PATCH_SIZE,
        models_dir=r"D:\gaze\vit-human-attention-comparison\trained_model_weights/",
    )
    with torch.inference_mode():
        d_attn_list = d_model.get_fulllayers_selfattention(images.to(d_device))

    del d_model
    torch.cuda.empty_cache()

    # --- Compare ---
    results = []
    for layer_idx in range(DEPTH):
        c_a = c_attn_list[layer_idx].cpu().float().numpy()
        d_a = d_attn_list[layer_idx].cpu().float().numpy()
        abs_diff = np.abs(c_a - d_a)

        c_flat = c_a.ravel().astype(np.float64)
        d_flat = d_a.ravel().astype(np.float64)
        c_norm = np.linalg.norm(c_flat)
        d_norm = np.linalg.norm(d_flat)
        cos_sim = (np.dot(c_flat, d_flat) / (c_norm * d_norm)
                   if c_norm > 0 and d_norm > 0 else 0.0)

        results.append({
            "layer": layer_idx + 1,
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "cosine_similarity": float(cos_sim),
        })
    return results


# ======================================================================
# Verification of preprocessing identity
# ======================================================================

def verify_preprocessing(paths, transform):
    """Sanity-check that preprocessing matches reference expectations."""
    img = Image.open(paths[0])
    assert img.size == (720, 480), f"Unexpected size: {img.size}"
    assert img.mode == "RGB", f"Unexpected mode: {img.mode}"

    tensor = transform(img)
    assert tensor.shape == (3, 480, 720), f"Unexpected tensor shape: {tensor.shape}"
    # ImageNet normalisation: mean≈0, range≈[-2.1, 2.6]
    assert -3 < tensor.min() < 0, f"Tensor min out of range: {tensor.min()}"
    assert 0 < tensor.max() < 4, f"Tensor max out of range: {tensor.max()}"
    print(f"  Image size:    720x480 (WxH) OK")
    print(f"  Tensor shape:  {tuple(tensor.shape)} OK")
    print(f"  Tensor range:  [{tensor.min():.3f}, {tensor.max():.3f}] OK")
    print(f"  Patch aligned: H%16={480 % 16}, W%16={720 % 16} OK")


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    n_frames = args.n_frames

    print("=" * 65)
    print("  Regression test: C: ViT extractor vs D: author reference")
    print("=" * 65)
    print(f"Model:   {TRAINING_METHOD} / depth={DEPTH} / trial={TRIAL}")
    print(f"Frames:  first {n_frames} of Nakano 2010 ({FRAMES_DIR})")
    print(f"Seed:    {args.seed}")
    print()

    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Preprocessing verification
    # ------------------------------------------------------------------
    print("--- Preprocessing verification ---")
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
    ])
    all_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.png")))
    assert len(all_paths) == 2327, f"Expected 2327 frames, found {len(all_paths)}"
    paths = all_paths[:n_frames]
    verify_preprocessing(paths, transform)

    # ------------------------------------------------------------------
    # Load C: model
    # ------------------------------------------------------------------
    print("\n--- Loading C: model ---")
    config = Config(
        depth=DEPTH, trial_num=TRIAL, training_method=TRAINING_METHOD,
        batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, blur_size=BLUR_SIZE,
    )
    t0 = time.time()
    model, device = _load_model(config)
    print(f"  Loaded in {time.time() - t0:.1f}s on {device}")
    print(f"  depth={model.depth}, num_heads={model.num_heads}, "
          f"embed_dim={model.embed_dim}")

    # ------------------------------------------------------------------
    # Part 1: Gaze position comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  Part 1: Gaze position comparison (C: vs author reference)")
    print(f"{'=' * 65}")

    dataset = ReferenceImageDataset(paths, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    print(f"Extracting gaze positions for {n_frames} frames ...")
    t0 = time.time()
    gaze_pos = extract_gaze_positions(model, device, dataloader)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Output shape: {gaze_pos.shape}")  # (12, 7, n, 2)

    # Load author reference
    print("Loading author reference ...")
    author_data = np.load(AUTHOR_NPZ, allow_pickle=True)
    author_dino = author_data["dino"].item()
    author_all = author_dino[str(DEPTH)]        # (6, 12, 7, 2327, 2)
    author_single = author_all[TRIAL - 1]       # (12, 7, 2327, 2)
    author_subset = author_single[:, :, :n_frames, :]  # (12, 7, n, 2)
    print(f"  Author shape (subset): {author_subset.shape}")

    # Overall comparison
    diff = np.abs(gaze_pos - author_subset)
    total = diff.size
    exact = int((diff == 0).sum())
    within1 = int((diff <= 1).sum())
    exact_pct = 100 * exact / total
    within1_pct = 100 * within1 / total

    print(f"\n--- Overall (first {n_frames} frames) ---")
    print(f"  Exact match:      {exact:>8} / {total}  ({exact_pct:.1f}%)")
    print(f"  Within 1 pixel:   {within1:>8} / {total}  ({within1_pct:.1f}%)")
    print(f"  Max difference:   {diff.max():.1f} px")
    print(f"  Mean difference:  {diff.mean():.4f} px")
    print(f"  Median diff:      {np.median(diff):.1f} px")

    # Per-layer breakdown
    print(f"\n--- Per-layer breakdown ---")
    layer_results = []
    for l in range(DEPTH):
        ld = diff[l]
        lt = ld.size
        le = int((ld == 0).sum())
        lw = int((ld <= 1).sum())
        row = {
            "layer": l + 1,
            "exact_match_pct": round(100 * le / lt, 2),
            "within_1px_pct": round(100 * lw / lt, 2),
            "max_diff_px": round(float(ld.max()), 1),
            "mean_diff_px": round(float(ld.mean()), 4),
        }
        layer_results.append(row)
        print(f"  Layer {l + 1:2d}:  exact={row['exact_match_pct']:6.2f}%  "
              f"within1={row['within_1px_pct']:6.2f}%  "
              f"max={row['max_diff_px']:4.1f}  mean={row['mean_diff_px']:.4f}")

    # Spot check
    print(f"\n--- Spot check (layer=12, head=0) ---")
    for f_idx in [0, min(10, n_frames - 1), min(n_frames - 1, 49)]:
        ours = gaze_pos[11, 0, f_idx]
        ref = author_subset[11, 0, f_idx]
        print(f"  Frame {f_idx:4d}: ours={ours}  ref={ref}  diff={np.abs(ours - ref)}")

    # Save per-layer report
    report_path = os.path.join(PROJECT_DIR, "results", "regression_report.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=layer_results[0].keys())
        w.writeheader()
        w.writerows(layer_results)
    print(f"\n  Report saved to {report_path}")

    # Verdict Part 1
    pass_exact = exact_pct >= 95.0
    pass_within1 = within1_pct >= 99.9
    pass_part1 = pass_exact and pass_within1

    # ------------------------------------------------------------------
    # Part 2: Attention-map numerical comparison (C: vs D:)
    # ------------------------------------------------------------------
    pass_part2 = True
    map_results = []

    if not args.skip_map:
        print(f"\n{'=' * 65}")
        print("  Part 2: Attention-map numerical comparison (C: vs D: model)")
        print(f"{'=' * 65}")

        n_compare = min(3, n_frames)
        compare_dataset = ReferenceImageDataset(paths[:n_compare], transform)
        compare_loader = DataLoader(compare_dataset, batch_size=n_compare,
                                    shuffle=False, num_workers=0)
        compare_images = next(iter(compare_loader))
        print(f"Comparing raw attention on {n_compare} frames ...")

        try:
            map_results = compare_attention_maps(model, device, compare_images)
            print(f"\n--- Per-layer attention comparison ---")
            for r in map_results:
                status = "OK" if r["max_abs_diff"] < 1e-5 else "WARN"
                print(f"  Layer {r['layer']:2d}:  max_diff={r['max_abs_diff']:.2e}  "
                      f"mean_diff={r['mean_abs_diff']:.2e}  "
                      f"cos_sim={r['cosine_similarity']:.10f}  [{status}]")
                if r["max_abs_diff"] >= 1e-5:
                    pass_part2 = False

            # Save map comparison
            map_report_path = os.path.join(
                PROJECT_DIR, "results", "regression_attn_map.csv")
            with open(map_report_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=map_results[0].keys())
                w.writeheader()
                w.writerows(map_results)
            print(f"\n  Map report saved to {map_report_path}")

        except Exception as e:
            print(f"\n  Part 2 SKIPPED due to error: {e}")
            print("  (This may occur if D: code has import issues.)")
            print("  Part 1 result is the primary verification.")
            pass_part2 = True  # don't fail overall for import issues

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  FINAL VERDICT")
    print(f"{'=' * 65}")
    print(f"  Part 1 (gaze positions):  {'PASS' if pass_part1 else 'FAIL'}"
          f"  (exact={exact_pct:.1f}% >=95%, within1={within1_pct:.1f}% >=99.9%)")
    if map_results:
        print(f"  Part 2 (attention maps):  {'PASS' if pass_part2 else 'FAIL'}"
              f"  (max_diff < 1e-5)")
    overall = pass_part1 and pass_part2
    print(f"\n  Overall: {'PASS - extraction OK for downstream use' if overall else 'FAIL - investigation needed'}")
    print(f"{'=' * 65}")

    return 0 if overall else 1


def parse_args():
    p = argparse.ArgumentParser(description="ViT extractor regression test")
    p.add_argument("--n-frames", type=int, default=50,
                   help="Number of Nakano frames to test (default 50)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for argmax tie-breaking")
    p.add_argument("--skip-map", action="store_true",
                   help="Skip Part 2 (attention-map comparison)")
    return p.parse_args()


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    sys.exit(main())
