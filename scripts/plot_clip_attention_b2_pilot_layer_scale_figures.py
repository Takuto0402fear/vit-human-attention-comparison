"""
[B-2] Visualization fix: adds per_layer_scale / shared_scale figure pairs
for the pilot images already rendered by scripts/run_clip_attention_b2_pilot.py
(by default: 1156, 1159, 1213), without touching or overwriting the
existing *_overlay.png files (those used a per-layer-only color scale
with no shared-scale counterpart, which made cross-layer color intensity
comparisons invalid).

Reads (all read-only):
  - <pilot-config>  (default: outputs/clip_attention_b2_distribution_pilot/config.json),
    for figure_image_ids
  - <cache-path>    (default: the existing 700-image L4/L8/L12 attention cache),
    re-read directly -- not recomputed, consistent with the bit-exact
    match already established by scripts/verify_clip_attention_b2_cache.py
    and the pilot's own per-image cross-check
  - <attrs-path>    OSIE attrs.mat (foreground masks)

All paths are resolved relative to the repository root or overridable
via CLI flags -- no machine-specific absolute path is hardcoded. Writes
only new files under --output-dir (both a batch pre-check here AND the
per-file check inside lib.clip_attention_b2_plots.
render_per_layer_and_shared_scale_figures refuse to overwrite any
existing file; the pilot's existing figures/*_overlay.png are never
touched by this script):
  <output-dir>/{stem}_per_layer_scale.png
  <output-dir>/{stem}_shared_scale.png

Usage (PowerShell):
    python scripts\\plot_clip_attention_b2_pilot_layer_scale_figures.py
    python scripts\\plot_clip_attention_b2_pilot_layer_scale_figures.py --output-dir D:\\tmp\\b2_smoke\\pilot_figs
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import argparse
import json
import os

import h5py
import numpy as np
from PIL import Image

from lib.clip_attention_b2 import refuse_if_exists
from lib.clip_attention_b2_plots import render_per_layer_and_shared_scale_figures
from lib.osie_text_alignment import mask_to_patch_weights

CLIP_PATCH_SIZE = 16
LAYERS_DISPLAY = [4, 8, 12]
LAYERS_ZERO_BASED = [3, 7, 11]
IMG_W, IMG_H = 800, 600
N_IMAGES_EXPECTED = 700
GRID_HW = (38, 50)

DEFAULT_CACHE_PATH = (
    REPO_ROOT / "outputs" / "osie_attribute_grounding_full700" / "attn_cache"
    / "clip_vitb16_full700_L4L8L12_patchgrid.npz")
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "osie" / "predicting-human-gaze-beyond-pixels"
DEFAULT_STIMULI_DIR = DEFAULT_DATASET_DIR / "data" / "stimuli"
DEFAULT_ATTRS_PATH = DEFAULT_DATASET_DIR / "data" / "attrs.mat"
DEFAULT_PILOT_CONFIG_JSON = REPO_ROOT / "outputs" / "clip_attention_b2_distribution_pilot" / "config.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "clip_attention_b2_distribution_pilot" / "figures"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="[B-2] Add per_layer_scale / shared_scale figure pairs for the "
                    "pilot's representative images, without touching the existing "
                    "*_overlay.png figures.")
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                    help=f"Path to the existing L4/L8/L12 attention cache .npz, read-only "
                        f"(default: {DEFAULT_CACHE_PATH}).")
    p.add_argument("--stimuli-dir", type=Path, default=DEFAULT_STIMULI_DIR,
                    help=f"Directory containing OSIE stimulus images (<id>.jpg), read-only "
                        f"(default: {DEFAULT_STIMULI_DIR}).")
    p.add_argument("--attrs-path", type=Path, default=DEFAULT_ATTRS_PATH,
                    help=f"Path to OSIE's attrs.mat (object masks), read-only "
                        f"(default: {DEFAULT_ATTRS_PATH}).")
    p.add_argument("--pilot-config", type=Path, default=DEFAULT_PILOT_CONFIG_JSON,
                    help="Path to the pilot's config.json, read-only -- its "
                        "figure_image_ids are the images rendered here "
                        f"(default: {DEFAULT_PILOT_CONFIG_JSON}).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Directory to write the new {stem}_per_layer_scale.png / "
                        "{stem}_shared_scale.png figures into (created if missing; "
                        "refuses to overwrite any file that already exists there) "
                        f"(default: {DEFAULT_OUTPUT_DIR}).")
    return p.parse_args(argv)


def load_attrs_index(f):
    def h5_char_str(ref):
        return "".join(chr(int(c)) for c in f[ref][()].flatten())
    names_ds = f["attrNames"]
    attr_names = [h5_char_str(names_ds[i, 0]) for i in range(names_ds.shape[0])]
    attrs_ds = f["attrs"]
    index = {}
    for i in range(attrs_ds.shape[1]):
        g = f[attrs_ds[0, i]]
        raw_name = "".join(chr(int(c)) for c in g["img"][()].flatten())
        index[os.path.splitext(raw_name)[0]] = g
    return attr_names, index


def foreground_mask_for_image(f, group, expected_hw):
    objs_ds = group["objs"]
    h, w = expected_hw
    n_objs = objs_ds.shape[1]
    if n_objs == 0:
        raise RuntimeError("STOP: image with zero objects")
    fg = np.zeros((h, w), dtype=bool)
    for j in range(n_objs):
        obj_g = f[objs_ds[0, j]]
        mp = obj_g["map"][()].T.astype(bool)
        if mp.shape != (h, w):
            raise RuntimeError(f"STOP: object {j} mask shape {mp.shape} != {(h, w)}")
        fg |= mp
    return fg


def verify_cache_basic_health(cache):
    attn = cache["attn"]
    stems = cache["stems"].tolist()
    layers_display = cache["layers_display"].tolist()
    layers_zero_based = cache["layers_zero_based"].tolist()

    if attn.shape != (N_IMAGES_EXPECTED, 3, *GRID_HW):
        raise RuntimeError(f"STOP: cache attn.shape {attn.shape} != {(N_IMAGES_EXPECTED, 3, *GRID_HW)}")
    if layers_display != LAYERS_DISPLAY:
        raise RuntimeError(f"STOP: cache layers_display {layers_display} != {LAYERS_DISPLAY}")
    if layers_zero_based != LAYERS_ZERO_BASED:
        raise RuntimeError(f"STOP: cache layers_zero_based {layers_zero_based} != {LAYERS_ZERO_BASED}")
    if len(stems) != N_IMAGES_EXPECTED:
        raise RuntimeError(f"STOP: cache has {len(stems)} stems, expected {N_IMAGES_EXPECTED}")
    if len(set(stems)) != len(stems):
        raise RuntimeError("STOP: cache stems contain duplicates")
    if not np.isfinite(attn).all():
        raise RuntimeError("STOP: cache attention contains non-finite values")
    if (attn < 0).any():
        raise RuntimeError("STOP: cache attention contains negative values")
    return stems


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(args.cache_path)
    stimuli_dir = Path(args.stimuli_dir)
    attrs_path = Path(args.attrs_path)
    pilot_config_path = Path(args.pilot_config)
    output_dir = Path(args.output_dir)

    if not pilot_config_path.is_file():
        raise RuntimeError(f"STOP: {pilot_config_path} not found -- run the pilot first")
    with open(pilot_config_path, encoding="utf-8") as fh:
        pilot_config = json.load(fh)
    stems = pilot_config["figure_image_ids"]
    print(f"[B-2] Adding per_layer_scale/shared_scale figures for pilot images: {stems}")
    print(f"  cache_path   = {cache_path}")
    print(f"  stimuli_dir  = {stimuli_dir}")
    print(f"  attrs_path   = {attrs_path}")
    print(f"  pilot_config = {pilot_config_path}")
    print(f"  output_dir   = {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        str(output_dir / f"{stem}_{mode}.png")
        for stem in stems for mode in ("per_layer_scale", "shared_scale")
    ]
    refuse_if_exists(expected_outputs)

    cache = np.load(str(cache_path), allow_pickle=False)
    stems_all = verify_cache_basic_health(cache)
    print("  Cache self-check (shape/layers/stems/finite/non-negative): OK")
    attn_all = cache["attn"]

    f = h5py.File(str(attrs_path), "r")
    _, attrs_index = load_attrs_index(f)

    for stem in stems:
        img_idx = stems_all.index(stem)
        img_arr = np.array(Image.open(stimuli_dir / f"{stem}.jpg").convert("RGB"))
        H, W = img_arr.shape[:2]
        if (W, H) != (IMG_W, IMG_H):
            raise RuntimeError(f"STOP: {stem} unexpected size {(W, H)}")

        fg_mask = foreground_mask_for_image(f, attrs_index[stem], (H, W))
        coverage = mask_to_patch_weights(fg_mask, CLIP_PATCH_SIZE)

        layer_maps = {
            layer_display: attn_all[img_idx, l_idx].astype(np.float64)
            for l_idx, layer_display in enumerate(LAYERS_DISPLAY)
        }

        per_layer_path, shared_path = render_per_layer_and_shared_scale_figures(
            stem=stem, img_arr=img_arr, coverage=coverage, layer_maps=layer_maps,
            layers_display=LAYERS_DISPLAY, out_dir=str(output_dir), img_wh=(IMG_W, IMG_H))
        print(f"  {stem}: saved {per_layer_path}")
        print(f"  {stem}: saved {shared_path}")

    print("\n[B-2] Done. Existing *_overlay.png files were not touched.")


if __name__ == "__main__":
    main()
