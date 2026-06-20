"""
Verify gaze position computation by running DINO/depth=12/trial=1
and comparing against author's pre-computed data.

Usage:
  python verify_single_model.py

Saves intermediate results per-model and final comparison report.
"""
import os
import sys
import time
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms

# Must run from analysis_python directory for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_analysis import model_load, get_gaze_pos_model_dataset, ImageDataset

# ─── Configuration ───────────────────────────────────────────────
DATASET_DIR = "../dataset/Nakano_etal_2010/video_stimuli/frames"
SAVE_DIR = "../dataset/Nakano_etal_2010/preprocessed_data"
AUTHOR_FILE = os.path.join(SAVE_DIR, "vit_gaze_pos_author.npz")

# Which models to compute (minimal verification set)
MODELS_TO_VERIFY = [
    {"training_method": "dino", "depth": 12, "trial": 1},
]

PATCH_SIZE = 16
BLUR_SIZE = PATCH_SIZE * 2
BATCH_SIZE = 4  # safe for GTX 1060 6GB
NUM_FRAMES = 2327
NUM_HEADS = 6

# ─── Setup ───────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Single-model verification: self-compute vs author data")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU available, running on CPU (will be slow)")

    # Transform
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Dataset
    image_path_list = sorted(glob.glob(f"{DATASET_DIR}/*.png"))
    print(f"Found {len(image_path_list)} frames")
    assert len(image_path_list) == NUM_FRAMES, f"Expected {NUM_FRAMES}, got {len(image_path_list)}"

    dataset = ImageDataset(image_path_list, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load author data for comparison
    print(f"Loading author data from {AUTHOR_FILE}")
    author_data = np.load(AUTHOR_FILE, allow_pickle=True)

    # ─── Compute each model ─────────────────────────────────────
    for model_spec in MODELS_TO_VERIFY:
        tm = model_spec["training_method"]
        depth = model_spec["depth"]
        trial = model_spec["trial"]
        trial_idx = trial - 1  # 0-based index for author data

        print(f"\n{'─' * 60}")
        print(f"Computing: {tm} / depth={depth} / trial={trial}")
        print(f"{'─' * 60}")

        # Load model
        t0 = time.time()
        print("Loading model weights...")
        model, device = model_load(tm, trial, depth, PATCH_SIZE)
        print(f"Model loaded in {time.time() - t0:.1f}s on {device}")

        # Compute gaze positions
        t0 = time.time()
        print(f"Computing gaze positions (batch_size={BATCH_SIZE})...")
        gaze_pos = get_gaze_pos_model_dataset(model, device, dataloader, PATCH_SIZE, BLUR_SIZE)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Result shape: {gaze_pos.shape}")
        # Expected: (depth, num_heads+1, num_frames, 2)

        # Save intermediate result
        save_path = os.path.join(SAVE_DIR, f"vit_gaze_pos_verify_{tm}_d{depth}_t{trial}.npz")
        np.savez_compressed(save_path, gaze_pos=gaze_pos)
        print(f"Saved to {save_path}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # ─── Compare with author data ───────────────────────────
        print(f"\nComparing with author data...")
        author_tm = author_data[tm].item()
        author_gaze = author_tm[str(depth)]  # (6, depth, 7, 2327, 2)
        author_single = author_gaze[trial_idx]  # (depth, 7, 2327, 2)
        print(f"Author data shape for trial {trial}: {author_single.shape}")
        print(f"Self-computed shape:                  {gaze_pos.shape}")

        # Compute differences
        diff = np.abs(gaze_pos - author_single)
        print(f"\n--- Numerical comparison ---")
        print(f"Max absolute difference:  {diff.max():.6f}")
        print(f"Mean absolute difference: {diff.mean():.6f}")
        print(f"Median absolute diff:     {np.median(diff):.6f}")
        print(f"Exact matches:            {(diff == 0).sum()} / {diff.size} ({100*(diff==0).sum()/diff.size:.1f}%)")
        print(f"Within 1 pixel:           {(diff <= 1).sum()} / {diff.size} ({100*(diff<=1).sum()/diff.size:.1f}%)")

        # Per-layer breakdown
        print(f"\n--- Per-layer max|mean diff ---")
        for layer_idx in range(depth):
            layer_diff = diff[layer_idx]
            print(f"  Layer {layer_idx:2d}: max={layer_diff.max():.1f}, mean={layer_diff.mean():.4f}")

        # Check head-mean channel (index 6)
        mean_diff = diff[:, 6, :, :]  # mean-head across all layers
        print(f"\n--- Mean-head (head=6) ---")
        print(f"  max={mean_diff.max():.1f}, mean={mean_diff.mean():.4f}")

        # Spot check some frames
        print(f"\n--- Sample frames (layer=11, head=0) ---")
        for frame in [0, 100, 500, 1000, 2000]:
            self_val = gaze_pos[11, 0, frame]
            auth_val = author_single[11, 0, frame]
            print(f"  Frame {frame:4d}: self={self_val}, author={auth_val}, diff={np.abs(self_val-auth_val)}")

    print(f"\n{'=' * 60}")
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    # Flush print output immediately so log file gets progressive updates
    import functools
    print = functools.partial(print, flush=True)
    main()
