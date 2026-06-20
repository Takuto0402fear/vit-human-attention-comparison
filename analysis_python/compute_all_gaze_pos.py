"""
Compute ViT gaze positions for all 36 models (2 methods x 3 depths x 6 trials).
Saves results per-model incrementally to avoid data loss on interruption.

Usage (no timeout, from analysis_python directory):
  python -u compute_all_gaze_pos.py

Resume after interruption:
  python -u compute_all_gaze_pos.py --resume

Estimated time: ~30 min/model x 36 models = ~18 hours on GTX 1060 (thermal throttled)
With cooling improvement (GPU <80C): ~12 hours
"""
import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_analysis import model_load, get_gaze_pos_model_dataset, ImageDataset

# ─── Configuration ───────────────────────────────────────────────
DATASET_DIR = "../dataset/Nakano_etal_2010/video_stimuli/frames"
SAVE_DIR = "../dataset/Nakano_etal_2010/preprocessed_data"
PARTIAL_DIR = os.path.join(SAVE_DIR, "partial")

TRAINING_METHODS = ["dino", "supervised"]
DEPTH_LIST = [4, 8, 12]
NUM_MODELS = 6
NUM_HEADS = 6
NUM_FRAMES = 2327
PATCH_SIZE = 16
BLUR_SIZE = PATCH_SIZE * 2
BATCH_SIZE = 4  # safe for GTX 1060 6GB


def partial_path(tm, depth, trial):
    return os.path.join(PARTIAL_DIR, f"gaze_pos_{tm}_d{depth}_t{trial}.npz")


def is_done(tm, depth, trial):
    return os.path.exists(partial_path(tm, depth, trial))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-computed models (resume after interruption)")
    args = parser.parse_args()

    os.makedirs(PARTIAL_DIR, exist_ok=True)

    print("=" * 60, flush=True)
    print("Full gaze position computation (all 36 models)", flush=True)
    print("=" * 60, flush=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}", flush=True)
    else:
        print("WARNING: No GPU, running on CPU", flush=True)

    # Transform & dataset
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_path_list = sorted(glob.glob(f"{DATASET_DIR}/*.png"))
    assert len(image_path_list) == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {len(image_path_list)}"
    dataset = ImageDataset(image_path_list, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Count total work
    total_models = len(TRAINING_METHODS) * len(DEPTH_LIST) * NUM_MODELS
    done_count = sum(1 for tm in TRAINING_METHODS for d in DEPTH_LIST
                     for t in range(1, NUM_MODELS + 1) if is_done(tm, d, t))
    if args.resume:
        print(f"Resume mode: {done_count}/{total_models} already computed", flush=True)

    model_idx = 0
    t_start_all = time.time()

    for tm in TRAINING_METHODS:
        for depth in DEPTH_LIST:
            for trial in range(1, NUM_MODELS + 1):
                model_idx += 1

                if args.resume and is_done(tm, depth, trial):
                    print(f"[{model_idx}/{total_models}] {tm}/d{depth}/t{trial} — SKIP (already done)", flush=True)
                    continue

                print(f"\n[{model_idx}/{total_models}] {tm}/d{depth}/t{trial}", flush=True)

                # Load model
                t0 = time.time()
                model, device = model_load(tm, trial, depth, PATCH_SIZE)
                print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

                # Compute
                t0 = time.time()
                gaze_pos = get_gaze_pos_model_dataset(model, device, dataloader, PATCH_SIZE, BLUR_SIZE)
                elapsed = time.time() - t0
                print(f"  Computed in {elapsed:.1f}s ({elapsed/60:.1f} min), shape={gaze_pos.shape}", flush=True)

                # Save partial result
                np.savez_compressed(partial_path(tm, depth, trial), gaze_pos=gaze_pos)
                print(f"  Saved to {partial_path(tm, depth, trial)}", flush=True)

                # Free GPU
                del model
                torch.cuda.empty_cache()

    # ─── Assemble final file ─────────────────────────────────────
    print(f"\nAssembling final file...", flush=True)
    res_dict = {}
    res_dict["info"] = np.array(["num_models", "depth", "num_head+mean", "num_frames", "xy"])

    for tm in TRAINING_METHODS:
        tm_dict = {}
        for depth in DEPTH_LIST:
            gaze_all = np.zeros((NUM_MODELS, depth, NUM_HEADS + 1, NUM_FRAMES, 2))
            for trial in range(1, NUM_MODELS + 1):
                data = np.load(partial_path(tm, depth, trial))
                gaze_all[trial - 1] = data["gaze_pos"]
            tm_dict[str(depth)] = gaze_all
        res_dict[tm] = tm_dict

    out_path = os.path.join(SAVE_DIR, "vit_gaze_pos.npz")
    np.savez_compressed(out_path, **res_dict)
    total_time = time.time() - t_start_all
    print(f"\nDone! Saved to {out_path}", flush=True)
    print(f"Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
