"""
Phase-1 tests for CLIP ViT-B/16 layerwise attention extraction:
standard 224x224 input, native 14x14 patch grid only. Non-square
input and OSIE-scale runs are covered separately once Phase 2 lands.

Requires network access to download the CLIP checkpoint on first run
(cached by the `clip` package under ~/.cache/clip afterwards).

Run:  python -m pytest tests/test_clip_extractor.py -v
"""
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clip_extractor import (
    CLIP_DEPTH, CLIP_HEADS, CLIP_NATIVE_GRID, CLIP_NUM_TOKENS,
    CLIP_PATCH_SIZE, assert_vit_b16_shape, load_clip_vit_b16, load_osie_image_tensor,
)
from lib.clip_vit import (
    cls_to_patch_grid, interpolate_patch_pos_embed, patch_grid_from_image_hw,
    visual_forward_with_attention, visual_forward_with_cls_patch_attention,
)

STIM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "osie",
    "predicting-human-gaze-beyond-pixels", "data", "stimuli", "1001.jpg")


@pytest.fixture(scope="module")
def loaded_model():
    return load_clip_vit_b16()


@pytest.fixture(scope="module")
def sample_image(loaded_model):
    model, preprocess = loaded_model
    device = next(model.parameters()).device
    img = Image.open(STIM_PATH).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)


@pytest.fixture(scope="module")
def attn_per_layer(loaded_model, sample_image):
    model, _ = loaded_model
    with torch.inference_mode():
        _, attn = visual_forward_with_attention(
            model.visual, sample_image.type(model.dtype))
    return attn


# ==== architecture ====
class TestArchitecture:
    def test_shape_spec(self, loaded_model):
        model, _ = loaded_model
        assert_vit_b16_shape(model)

    def test_constants_match_model(self, loaded_model):
        model, _ = loaded_model
        visual = model.visual
        assert len(visual.transformer.resblocks) == CLIP_DEPTH == 12
        assert visual.transformer.resblocks[0].attn.num_heads == CLIP_HEADS == 12
        assert visual.conv1.kernel_size[0] == CLIP_PATCH_SIZE == 16
        assert visual.positional_embedding.shape[0] == CLIP_NUM_TOKENS == 197


# ==== custom forward parity ====
class TestCustomForwardMatchesOfficial:
    def test_pooled_matches_encode_image(self, loaded_model, sample_image):
        model, _ = loaded_model
        with torch.inference_mode():
            official = model.encode_image(sample_image)
            pooled, _ = visual_forward_with_attention(
                model.visual, sample_image.type(model.dtype))
        torch.testing.assert_close(pooled, official, rtol=1e-4, atol=1e-4)


# ==== attention shape / normalization ====
class TestAttentionShapesAndNormalization:
    def test_returns_one_matrix_per_layer(self, attn_per_layer):
        assert len(attn_per_layer) == 12

    def test_full_row_sums_to_one(self, attn_per_layer):
        for attn in attn_per_layer:
            full_row_sum = attn[:, 0, :].sum(dim=-1)
            torch.testing.assert_close(
                full_row_sum, torch.ones_like(full_row_sum), rtol=1e-4, atol=1e-4)

    def test_cls_to_patch_in_zero_to_one_range(self, attn_per_layer):
        # CLS excluded: not guaranteed to sum to 1 (some mass stays on CLS
        # itself), so only (0, 1] is asserted -- and it is NOT renormalized
        # here, matching vit_extractor.py's DINO convention of using
        # attn[:, :, 0, 1:] as-is with no re-sum-to-1 step.
        for attn in attn_per_layer:
            cls_to_patch = attn[:, 0, 1:]
            cls_to_patch_sum = cls_to_patch.sum(dim=-1)
            assert bool((cls_to_patch > 0).all())
            assert bool((cls_to_patch_sum <= 1.0 + 1e-6).all())

    def test_cls_to_patch_reshapes_to_native_grid(self, attn_per_layer):
        for attn in attn_per_layer:
            grid = cls_to_patch_grid(attn, CLIP_NATIVE_GRID)
            assert tuple(grid.shape[-2:]) == CLIP_NATIVE_GRID


# ==== position-embedding interpolation ====
class TestPositionEmbeddingInterpolation:
    def test_identity_at_native_grid(self, loaded_model):
        model, _ = loaded_model
        pos = model.visual.positional_embedding.detach()
        pos_interp = interpolate_patch_pos_embed(pos, CLIP_NATIVE_GRID)
        torch.testing.assert_close(pos_interp, pos)

    def test_rejects_non_square_patch_token_count(self):
        bad_pos = torch.randn(200, 8)  # 199 patch tokens, not a perfect square
        with pytest.raises(ValueError):
            interpolate_patch_pos_embed(bad_pos, (14, 14))


# ==== Phase 2: OSIE single image, non-square input ====
@pytest.fixture(scope="module")
def osie_padded_tensor(loaded_model):
    model, _ = loaded_model
    device = next(model.parameters()).device
    tensor, orig_hw, pad_hw = load_osie_image_tensor(STIM_PATH, CLIP_PATCH_SIZE)
    return tensor.to(device), orig_hw, pad_hw


@pytest.fixture(scope="module")
def osie_layerwise_result(loaded_model, osie_padded_tensor):
    model, _ = loaded_model
    tensor, _, pad_hw = osie_padded_tensor
    grid_hw = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
    pos_interp = interpolate_patch_pos_embed(model.visual.positional_embedding, grid_hw)
    with torch.inference_mode():
        pooled, cls_patch_maps, full_row_sums = visual_forward_with_cls_patch_attention(
            model.visual, tensor.type(model.dtype), grid_hw, pos_embed=pos_interp)
    return pooled, cls_patch_maps, full_row_sums, grid_hw


class TestOsiePaddingAndGrid:
    def test_padding_matches_dino_geometry(self, osie_padded_tensor):
        # OSIE stimuli are 800x600 (W,H); 600 is not a multiple of 16, so
        # only the height gets padded (bottom), matching
        # vit_extractor.py::_pad_to_patch exactly.
        _, orig_hw, pad_hw = osie_padded_tensor
        assert orig_hw == (600, 800)
        assert pad_hw == (608, 800)

    def test_grid_and_token_count(self, osie_padded_tensor):
        _, _, pad_hw = osie_padded_tensor
        grid_h, grid_w = patch_grid_from_image_hw(pad_hw[0], pad_hw[1], CLIP_PATCH_SIZE)
        assert (grid_h, grid_w) == (38, 50)
        assert grid_h * grid_w + 1 == 1901

    def test_pos_embed_interpolated_shape_and_cls_untouched(self, loaded_model):
        model, _ = loaded_model
        pos = model.visual.positional_embedding.detach()
        pos_interp = interpolate_patch_pos_embed(pos, (38, 50))
        assert tuple(pos_interp.shape) == (1901, 768)
        torch.testing.assert_close(pos_interp[0], pos[0])


class TestOsieLayerwiseAttention:
    def test_shape_is_12_38_50(self, osie_layerwise_result):
        _, cls_patch_maps, _, grid_hw = osie_layerwise_result
        assert grid_hw == (38, 50)
        stacked = torch.stack(cls_patch_maps, dim=1)
        assert tuple(stacked.shape[1:]) == (12, 38, 50)

    def test_full_row_sums_to_one(self, osie_layerwise_result):
        _, _, full_row_sums, _ = osie_layerwise_result
        assert len(full_row_sums) == 12
        for row_sum in full_row_sums:
            torch.testing.assert_close(
                row_sum, torch.ones_like(row_sum), rtol=1e-3, atol=1e-3)

    def test_no_nan_or_inf(self, osie_layerwise_result):
        _, cls_patch_maps, full_row_sums, _ = osie_layerwise_result
        for m in cls_patch_maps:
            assert not torch.isnan(m).any()
            assert not torch.isinf(m).any()
        for r in full_row_sums:
            assert not torch.isnan(r).any()
            assert not torch.isinf(r).any()

    def test_bilinear_resize_to_800x600_preserves_axes(self, osie_layerwise_result):
        _, cls_patch_maps, _, grid_hw = osie_layerwise_result
        stacked = torch.stack(cls_patch_maps, dim=1)  # (1, 12, 38, 50)
        resized = torch.nn.functional.interpolate(
            stacked.reshape(12, 1, *grid_hw), size=(600, 800),
            mode="bilinear", align_corners=False)
        assert tuple(resized.shape) == (12, 1, 600, 800)
