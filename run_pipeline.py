"""
Main entry point for the gaze x ViT attention comparison pipeline.

Usage (PowerShell)
------------------
# Dummy end-to-end test (no GPU / model weights needed):
python run_pipeline.py --dummy

# Real extraction (requires DINO weights + GPU):
python run_pipeline.py --image-dir path/to/images

# Limit image count for quick tests:
python run_pipeline.py --dummy --limit 2
"""
from __future__ import annotations

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np

from config import Config
from data_loader import load_gaze
from vit_extractor import extract_attention_maps
from aggregation import compute_all_metrics, summarise_group_layer, save_csv, print_table


def main():
    args = parse_args()
    config = Config(
        seed=args.seed,
        depth=args.depth,
        batch_size=args.batch,
        limit=args.limit,
        gaussian_sigma=args.sigma,
    )
    np.random.seed(config.seed)

    # ---- 1. Load gaze data ----
    print("[pipeline] Loading gaze data ...")
    records = load_gaze("dummy", config)
    print(f"  {len(records)} records  "
          f"({len(set(r.subject_id for r in records))} subjects, "
          f"{len(set(r.image_id for r in records))} images)")

    # ---- 2. Determine images to process ----
    image_ids = sorted(set(r.image_id for r in records))
    if config.limit > 0:
        image_ids = image_ids[:config.limit]
        records = [r for r in records if r.image_id in set(image_ids)]
        print(f"  --limit {config.limit}: using {len(image_ids)} images, "
              f"{len(records)} records")

    # Build image_id -> path mapping
    if args.dummy:
        # dummy mode: paths are irrelevant, but API requires the dict
        image_paths = {img_id: "" for img_id in image_ids}
    else:
        if not args.image_dir:
            print("ERROR: --image-dir is required in real mode", file=sys.stderr)
            sys.exit(1)
        image_paths = _discover_images(args.image_dir, image_ids)

    # ---- 3. Extract ViT attention maps ----
    print(f"[pipeline] Extracting attention maps (dummy={args.dummy}) ...")
    attn_maps = extract_attention_maps(image_paths, config, dummy=args.dummy)
    sample_key = next(iter(attn_maps))
    print(f"  shape per image: {attn_maps[sample_key].shape}  "
          f"(layers={config.depth})")

    # ---- 4. Compute metrics ----
    print("[pipeline] Computing metrics (NSS, AUC-Judd, sAUC) ...")
    results = compute_all_metrics(records, attn_maps, config)
    print(f"  {len(results)} metric rows computed")

    # ---- 5. Aggregate & output ----
    summary = summarise_group_layer(results)
    print_table(summary)

    out_dir = str(config.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    detail_path = os.path.join(out_dir, "detail_results.csv")
    save_csv(results, detail_path)

    summary_path = os.path.join(out_dir, "group_layer_summary.csv")
    save_csv(summary, summary_path)

    print("[pipeline] Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Gaze x ViT attention comparison pipeline")
    p.add_argument("--dummy", action="store_true",
                   help="Use dummy gaze + dummy ViT (no GPU needed)")
    p.add_argument("--image-dir", type=str, default=None,
                   help="Directory of stimulus images (real mode)")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap the number of images (0 = all)")
    p.add_argument("--batch", type=int, default=2,
                   help="Batch size for ViT extraction")
    p.add_argument("--depth", type=int, default=12,
                   help="Number of ViT layers")
    p.add_argument("--sigma", type=float, default=30.0,
                   help="Gaussian sigma for fixation heatmap")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


def _discover_images(image_dir: str, image_ids):
    """Map image_ids to actual file paths in image_dir."""
    found = {}
    for img_id in image_ids:
        # try common extensions
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            p = os.path.join(image_dir, img_id + ext)
            if os.path.isfile(p):
                found[img_id] = p
                break
    if len(found) < len(image_ids):
        missing = set(image_ids) - set(found.keys())
        print(f"WARNING: {len(missing)} images not found: {list(missing)[:5]}...")
    return found


if __name__ == "__main__":
    main()
