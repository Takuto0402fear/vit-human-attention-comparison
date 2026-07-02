"""
Configuration for the gaze-ViT comparison pipeline.
All tunable parameters are centralized here with sensible defaults
for a GTX 1060 6GB environment.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    # --- Paths ---
    models_dir: Path = Path(r"D:\gaze\vit-human-attention-comparison\trained_model_weights")
    output_dir: Path = Path(r".\results")

    # --- ViT model settings ---
    training_method: str = "dino"
    trial_num: int = 1
    depth: int = 12          # number of transformer layers
    arch: str = "vit_small"  # DINO ViT-S/16
    patch_size: int = 16
    blur_size: int = 32      # box-blur kernel for attention maps (matches original code)
    batch_size: int = 2      # conservative for GTX 1060 6 GB

    # --- Gaze heatmap generation ---
    gaussian_sigma: float = 30.0   # pixels, for fixation -> heatmap
    heatmap_height: int = 480
    heatmap_width: int = 720

    # --- Pipeline control ---
    seed: int = 42
    limit: int = 0           # 0 = use all images; >0 = cap image count
    common_images_only: bool = True  # restrict to images viewed by both groups

    # --- Group labels (extensible) ---
    groups: List[str] = field(default_factory=lambda: ["healthy", "schizophrenia"])

    # --- Dummy-data generation ---
    dummy_n_subjects_per_group: int = 5
    dummy_n_images: int = 3
    dummy_fixations_range: Tuple[int, int] = (5, 15)  # min/max fixations per trial

    @property
    def image_size(self) -> Tuple[int, int]:
        """(width, height) in pixels."""
        return (self.heatmap_width, self.heatmap_height)
