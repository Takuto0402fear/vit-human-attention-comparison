"""
ViT attention-map extraction.

Two modes:
  - Real:  loads a trained DINO ViT-S/16 and extracts [CLS] attention heatmaps
           for every layer (averaged across heads).
  - Dummy: generates random centre-biased heatmaps (no GPU / weights needed).

Output
------
Dict[image_id, np.ndarray]  where each array has shape (num_layers, H, W).
Values are non-negative and sum to 1 per map (probability distribution).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as pth_transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import Config


# ======================================================================
# Public API
# ======================================================================

def extract_attention_maps(
    image_paths: Dict[str, str],
    config: Config,
    dummy: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extract per-layer attention heatmaps for a set of images.

    Parameters
    ----------
    image_paths : dict
        {image_id: file_path} mapping.  Ignored when dummy=True.
    config : Config
    dummy : bool
        If True, return synthetic heatmaps without running the model.

    Returns
    -------
    dict
        {image_id: np.ndarray of shape (num_layers, H, W)}
        Each (H, W) map sums to 1.
    """
    if dummy:
        return _generate_dummy_maps(list(image_paths.keys()), config)
    return _extract_real(image_paths, config)


# ======================================================================
# Dummy mode
# ======================================================================

def _generate_dummy_maps(
    image_ids: List[str], config: Config
) -> Dict[str, np.ndarray]:
    """Centre-biased Gaussian attention maps for pipeline testing."""
    rng = np.random.RandomState(config.seed + 1000)
    H, W = config.heatmap_height, config.heatmap_width
    n_layers = config.depth
    result: Dict[str, np.ndarray] = {}

    ys = np.arange(H).reshape(-1, 1)
    xs = np.arange(W).reshape(1, -1)

    for img_id in image_ids:
        maps = np.zeros((n_layers, H, W), dtype=np.float32)
        for layer in range(n_layers):
            # random centre with small jitter per layer
            cx = W / 2 + rng.normal(0, W / 10)
            cy = H / 2 + rng.normal(0, H / 10)
            sigma = rng.uniform(W / 8, W / 3)
            g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
            # add noise
            g += rng.uniform(0, 0.01, size=(H, W))
            g = g / g.sum()
            maps[layer] = g.astype(np.float32)
        result[img_id] = maps
    return result


# ======================================================================
# Real extraction (DINO ViT)
# ======================================================================

class _ImageListDataset(Dataset):
    """Loads images from a list of paths with padding to patch-size multiples."""

    def __init__(self, paths: List[str], patch_size: int = 16):
        self.paths = paths
        self.patch_size = patch_size
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_np = np.array(img)
        img_np = _pad_to_patch(img_np, self.patch_size)
        img_pil = Image.fromarray(img_np)
        return self.transform(img_pil)


def _pad_to_patch(frame: np.ndarray, patch_size: int) -> np.ndarray:
    """Pad image so H and W are multiples of patch_size."""
    h, w = frame.shape[:2]
    hp = (patch_size - h % patch_size) % patch_size
    wp = (patch_size - w % patch_size) % patch_size
    if hp or wp:
        frame = np.pad(frame, ((0, hp), (0, wp), (0, 0)))
    return frame


def _load_model(config: Config):
    """Load a frozen DINO ViT-S/16 model and return (model, device)."""
    import lib.vision_transformer as vits
    from lib.vision_transformer import DINOHead

    model_dir = os.path.join(
        str(config.models_dir),
        config.training_method,
        f"{config.trial_num:02d}",
        f"{config.depth}layers",
    )
    weight_path = os.path.join(model_dir, "checkpoint.pth")
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    model = vits.vit_small(patch_size=config.patch_size, depth=config.depth,
                           drop_path_rate=0.1)
    model.head = DINOHead(model.embed_dim, 65536, use_bn=False,
                          norm_last_layer=True)

    if "teacher" in checkpoint:
        state = {}
        for k, v in checkpoint["teacher"].items():
            k = k.replace("projection_head", "mlp")
            k = k.replace("prototypes", "last_layer")
            state[k] = v
        state = {k.replace("module.", "").replace("backbone.", ""): v
                 for k, v in state.items()}
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)
    return model, device


def _extract_real(
    image_paths: Dict[str, str], config: Config
) -> Dict[str, np.ndarray]:
    """Run the actual ViT and return attention heatmaps."""
    try:
        from kornia.filters import box_blur
    except ImportError:
        raise ImportError("kornia is required for real ViT extraction: "
                          "pip install kornia")

    model, device = _load_model(config)
    num_heads = model.num_heads
    patch_size = config.patch_size
    blur_size = config.blur_size

    ids = list(image_paths.keys())
    paths = [image_paths[i] for i in ids]

    dataset = _ImageListDataset(paths, patch_size)
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=0)

    all_maps: List[np.ndarray] = []     # one per image
    for images in loader:
        with torch.inference_mode():
            attn_list = model.get_fulllayers_selfattention(images.to(device))

        B = images.shape[0]
        h_feat = images.shape[-2] // patch_size
        w_feat = images.shape[-1] // patch_size

        # collect per-layer mean-head attention maps
        layer_maps = []  # list of (B, H_px, W_px) per layer
        for attn in attn_list:
            # attn: (B, heads, tokens, tokens)
            # [CLS] -> patches
            a = attn[:, :, 0, 1:].reshape(B, num_heads, h_feat, w_feat)
            a = F.interpolate(a, scale_factor=patch_size, mode="nearest")
            a_mean = a.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            a_blur = box_blur(a_mean, (blur_size, blur_size))
            layer_maps.append(a_blur[:, 0].detach().cpu().numpy())  # (B, H, W)

        # stack layers -> (B, layers, H, W) then split per image
        stacked = np.stack(layer_maps, axis=1)  # (B, L, H, W)
        for b in range(B):
            m = stacked[b]  # (L, H, W)
            # normalise each layer to sum to 1
            for l in range(m.shape[0]):
                s = m[l].sum()
                if s > 0:
                    m[l] /= s
            all_maps.append(m)

    result = {img_id: mp for img_id, mp in zip(ids, all_maps)}
    return result
