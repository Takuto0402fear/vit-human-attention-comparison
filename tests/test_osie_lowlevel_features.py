"""
Tests for Phase 2: lib/osie_lowlevel_features.py and
scripts/extract_osie_lowlevel_targets.py.

Run:  python -m pytest tests/test_osie_lowlevel_features.py -v
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.osie_lowlevel_features import (
    compute_gradient_magnitude, compute_laplacian, compute_luminance,
    compute_object_lowlevel_targets, compute_saturation,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "osie_lowlevel_targets_full700")


def full_mask(size=32):
    return np.ones((size, size), dtype=bool)


# ======================================================================
# Luminance ordering (black < gray < white)
# ======================================================================
class TestLuminanceOrdering:
    def test_black_lower_than_white(self):
        black = np.zeros((16, 16, 3))
        white = np.ones((16, 16, 3))
        lum_black = compute_luminance(black)
        lum_white = compute_luminance(white)
        assert lum_black.mean() < lum_white.mean()
        assert np.allclose(lum_black, 0.0)
        assert np.allclose(lum_white, 1.0)

    def test_green_weighted_more_than_blue(self):
        # pure green vs pure blue at the same intensity -- BT.709 weights
        # green (0.7152) far more than blue (0.0722).
        green = np.zeros((8, 8, 3)); green[..., 1] = 1.0
        blue = np.zeros((8, 8, 3)); blue[..., 2] = 1.0
        assert compute_luminance(green).mean() > compute_luminance(blue).mean()


# ======================================================================
# Saturation ordering (grayscale < saturated color)
# ======================================================================
class TestSaturationOrdering:
    def test_grayscale_has_zero_saturation(self):
        gray = np.full((16, 16, 3), 0.5)
        sat = compute_saturation(gray)
        assert np.allclose(sat, 0.0, atol=1e-9)

    def test_pure_color_has_higher_saturation_than_grayscale(self):
        gray = np.full((16, 16, 3), 0.5)
        red = np.zeros((16, 16, 3)); red[..., 0] = 1.0
        assert compute_saturation(red).mean() > compute_saturation(gray).mean()


# ======================================================================
# Edge strength: checkerboard > uniform
# ======================================================================
class TestEdgeStrengthOrdering:
    def test_checkerboard_has_higher_edge_strength_than_uniform(self):
        size = 32
        uniform = np.full((size, size), 0.5)
        checker = np.indices((size, size)).sum(axis=0) % 2
        checker = checker.astype(np.float64)
        edge_uniform = compute_gradient_magnitude(uniform)
        edge_checker = compute_gradient_magnitude(checker)
        assert edge_checker.mean() > edge_uniform.mean()
        assert np.allclose(edge_uniform, 0.0, atol=1e-9)


# ======================================================================
# Fine texture: high-frequency pattern > uniform
# ======================================================================
class TestFineTextureOrdering:
    def test_fine_pattern_has_higher_laplacian_variance_than_uniform(self):
        size = 32
        uniform = np.full((size, size), 0.5)
        rng = np.random.RandomState(0)
        fine_pattern = rng.rand(size, size)  # high-frequency noise
        lap_uniform = compute_laplacian(uniform)
        lap_fine = compute_laplacian(fine_pattern)
        assert lap_fine.var() > lap_uniform.var()
        assert np.allclose(lap_uniform, 0.0, atol=1e-9)


# ======================================================================
# Mask-exterior pixels never leak into a target
# ======================================================================
class TestMaskExteriorDoesNotLeak:
    def test_exterior_pixels_excluded_from_mean(self):
        size = 16
        img = np.zeros((size, size, 3))
        img[:, : size // 2] = 1.0  # left half white, right half black
        mask = np.zeros((size, size), dtype=bool)
        mask[:, : size // 2] = True  # mask covers ONLY the white half
        targets = compute_object_lowlevel_targets(img, mask)
        assert targets["mean_luminance"] == pytest.approx(1.0, abs=1e-9)

    def test_exterior_pixels_excluded_when_mask_covers_black_half(self):
        size = 16
        img = np.zeros((size, size, 3))
        img[:, : size // 2] = 1.0
        mask = np.zeros((size, size), dtype=bool)
        mask[:, size // 2:] = True  # mask covers ONLY the black half
        targets = compute_object_lowlevel_targets(img, mask)
        assert targets["mean_luminance"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_mask_returns_none(self):
        img = np.random.RandomState(0).rand(16, 16, 3)
        mask = np.zeros((16, 16), dtype=bool)
        assert compute_object_lowlevel_targets(img, mask) is None


# ======================================================================
# Sobel/Laplacian never treat the mask boundary as an artificial edge
# ======================================================================
class TestNoArtificialMaskBoundaryEdge:
    def test_edge_strength_computed_on_full_image_not_masked_crop(self):
        # A smoothly-varying (no real edges) image, with an irregular mask
        # covering only part of it. If edge_strength were (incorrectly)
        # computed by zeroing/cropping to the mask FIRST, the mask
        # boundary itself would register as a sharp edge, inflating the
        # result far above the gradient of the smooth underlying image.
        size = 40
        y, x = np.mgrid[0:size, 0:size]
        smooth = (x / size)  # a linear ramp -- constant, small gradient everywhere
        smooth3 = np.stack([smooth] * 3, axis=-1)

        mask = np.zeros((size, size), dtype=bool)
        mask[10:20, 10:20] = True  # an interior square, well away from image borders

        targets = compute_object_lowlevel_targets(smooth3, mask)
        # the true gradient magnitude of a linear ramp with slope 1/size is
        # a small constant (~1/size in one direction after Sobel scaling);
        # a boundary-artifact implementation would show a value orders of
        # magnitude larger due to the mask's hard edges.
        assert targets["edge_strength"] < 1.0

    def test_masked_crop_computation_would_have_differed(self):
        # Direct demonstration: compute gradient on the full image vs. on
        # a mask-zeroed version of the SAME image: they must differ inside
        # the mask interior whenever the mask itself introduces a boundary
        # (proving our implementation's full-image-first order matters).
        size = 40
        img = np.full((size, size), 0.5)
        mask = np.zeros((size, size), dtype=bool)
        mask[10:20, 10:20] = True

        full_grad = compute_gradient_magnitude(img)
        zeroed = img.copy()
        zeroed[~mask] = 0.0
        cropped_grad = compute_gradient_magnitude(zeroed)

        # full-image gradient inside the mask is exactly 0 (uniform image);
        # mask-then-gradient introduces spurious high gradients at the
        # mask's own boundary, visible even in the interior near the edge.
        assert np.allclose(full_grad[mask], 0.0, atol=1e-9)
        assert cropped_grad[mask].max() > 0.01


# ======================================================================
# Geometry controls
# ======================================================================
class TestGeometryControls:
    def test_area_fraction_matches_mask_size(self):
        size = 20
        img = np.random.RandomState(0).rand(size, size, 3)
        mask = np.zeros((size, size), dtype=bool)
        mask[:10, :10] = True  # 100 / 400 = 0.25
        targets = compute_object_lowlevel_targets(img, mask)
        assert targets["mask_area_fraction"] == pytest.approx(0.25)

    def test_centroid_at_mask_center(self):
        size = 20
        img = np.random.RandomState(0).rand(size, size, 3)
        mask = np.zeros((size, size), dtype=bool)
        mask[0:2, 0:2] = True  # small mask at the very top-left corner
        targets = compute_object_lowlevel_targets(img, mask)
        assert targets["centroid_x"] < 0.2
        assert targets["centroid_y"] < 0.2


# ======================================================================
# Real-data integration
# ======================================================================
class TestRealLowlevelIntegration:
    def test_no_nan_inf_in_sanity_checks(self):
        path = os.path.join(OUT_DIR, "sanity_checks.json")
        if not os.path.isfile(path):
            pytest.skip("sanity_checks.json not generated yet")
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["n_nan_inf_detected"] == 0

    def test_no_new_dependency_declared(self):
        path = os.path.join(OUT_DIR, "config.json")
        if not os.path.isfile(path):
            pytest.skip("config.json not generated yet")
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        assert "cv2" not in config["no_new_dependency_added"].lower() or \
            "not installed" in config["no_new_dependency_added"].lower()
