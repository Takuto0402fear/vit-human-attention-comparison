"""
Phase-1 tests for DINO ViT-B/16 layerwise attention extraction: standard
224x224 input, native 14x14 patch grid only. Mirrors
tests/test_clip_extractor.py's structure for the CLIP side.

Requires network access to download the official DINO ViT-B/16 backbone
checkpoint on first run (cached by torch.hub under
~/.cache/torch/hub/checkpoints/ afterwards).

Run:  python -m pytest tests/test_dino_vitb16_extractor.py -v
"""
import os
import sys

import pytest
import torch
from PIL import Image
from torchvision import transforms as pth_transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dino_vitb16_extractor import (
    DINO_DEPTH, DINO_HEADS, DINO_NATIVE_GRID, DINO_PATCH_SIZE, DINO_VITB16_EXPECTED_MD5,
    assert_vitb16_shape, audit_official_dino_vitb16_config,
    check_native_pos_embed_untouched_at_grid, cls_to_patch_grid_mean_heads,
    extract_layerwise_cls_attention, extract_layerwise_cls_attention_memory_conscious,
    get_last_selfattention_reference, load_dino_vitb16,
)
from lib.clip_vit import patch_grid_from_image_hw
from vit_extractor import _pad_to_patch

STIM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "osie",
    "predicting-human-gaze-beyond-pixels", "data", "stimuli", "1001.jpg")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@pytest.fixture(scope="module")
def loaded_model():
    return load_dino_vitb16()


@pytest.fixture(scope="module")
def sample_image(loaded_model):
    _, device = loaded_model
    transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(STIM_PATH).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


@pytest.fixture(scope="module")
def extraction_result(loaded_model, sample_image):
    model, _ = loaded_model
    return extract_layerwise_cls_attention(model, sample_image, DINO_NATIVE_GRID)


class TestArchitectureAndMode:
    def test_shape_spec(self, loaded_model):
        model, _ = loaded_model
        assert_vitb16_shape(model)

    def test_eval_mode_and_no_grad(self, loaded_model):
        model, _ = loaded_model
        assert model.training is False
        assert all(not p.requires_grad for p in model.parameters())

    def test_clean_backbone_load(self, loaded_model):
        # load_dino_vitb16() itself raises if missing/unexpected keys are
        # non-empty; reaching this point means the checkpoint loaded clean.
        model, _ = loaded_model
        assert model is not None


class TestCheckpointIdentity:
    def test_cached_checkpoint_md5(self):
        hub_dir = torch.hub.get_dir()
        path = os.path.join(hub_dir, "checkpoints", "dino_vitbase16_pretrain.pth")
        assert os.path.isfile(path), f"expected cached checkpoint at {path}"
        import hashlib
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == DINO_VITB16_EXPECTED_MD5


class TestPositionEmbeddingNativeNoOp:
    def test_interpolate_pos_encoding_224_is_identity(self, loaded_model, sample_image):
        model, _ = loaded_model
        with torch.inference_mode():
            patch_tokens = model.patch_embed(sample_image)
            cls_tokens = model.cls_token.expand(patch_tokens.shape[0], -1, -1)
            x_with_cls = torch.cat((cls_tokens, patch_tokens), dim=1)
            pos_interp = model.interpolate_pos_encoding(x_with_cls, 224, 224)
        torch.testing.assert_close(pos_interp, model.pos_embed)


class TestLayerwiseAttention:
    def test_returns_12_layers(self, extraction_result):
        layer_maps, attn_raw = extraction_result
        assert len(layer_maps) == 12
        assert len(attn_raw) == 12

    def test_shapes(self, extraction_result):
        layer_maps, attn_raw = extraction_result
        for attn in attn_raw:
            assert attn.shape[1] == DINO_HEADS
            assert tuple(attn.shape[-2:]) == (197, 197)
        for grid_map in layer_maps:
            assert tuple(grid_map.shape[-2:]) == DINO_NATIVE_GRID

    def test_no_nan_or_inf(self, extraction_result):
        layer_maps, attn_raw = extraction_result
        for attn in attn_raw:
            assert not torch.isnan(attn).any()
            assert not torch.isinf(attn).any()
        for grid_map in layer_maps:
            assert not torch.isnan(grid_map).any()
            assert not torch.isinf(grid_map).any()

    def test_full_row_sums_to_one(self, extraction_result):
        _, attn_raw = extraction_result
        for attn in attn_raw:
            head_mean_full = attn.mean(dim=1)
            full_row_sum = head_mean_full[:, 0, :].sum(dim=-1)
            torch.testing.assert_close(
                full_row_sum, torch.ones_like(full_row_sum), rtol=1e-4, atol=1e-4)


class TestDeterminism:
    def test_two_runs_bit_identical(self, loaded_model, sample_image):
        model, _ = loaded_model
        with torch.inference_mode():
            attn_1 = model.get_fulllayers_selfattention(sample_image)
            attn_2 = model.get_fulllayers_selfattention(sample_image)
        for a1, a2 in zip(attn_1, attn_2):
            torch.testing.assert_close(a1, a2, rtol=0, atol=0)


class TestLastLayerMatchesReference:
    def test_matches_independent_reference(self, loaded_model, sample_image, extraction_result):
        model, _ = loaded_model
        _, attn_raw = extraction_result
        with torch.inference_mode():
            last_ref = get_last_selfattention_reference(model, sample_image)
        torch.testing.assert_close(last_ref, attn_raw[-1], rtol=1e-5, atol=1e-6)

    def test_matches_on_head_averaged_grid(self, loaded_model, sample_image, extraction_result):
        model, _ = loaded_model
        layer_maps, _ = extraction_result
        with torch.inference_mode():
            last_ref = get_last_selfattention_reference(model, sample_image)
        grid_last_ref = cls_to_patch_grid_mean_heads(last_ref, DINO_NATIVE_GRID)
        torch.testing.assert_close(grid_last_ref, layer_maps[-1], rtol=1e-5, atol=1e-6)


# ==== Phase 2: official-config audit ====
class TestOfficialConfigAudit:
    def test_matches_official_vit_base_config(self, loaded_model):
        model, _ = loaded_model
        audit = audit_official_dino_vitb16_config(model)
        assert all(r["matches"] for r in audit.values())

    def test_layernorm_eps_is_1e6_everywhere(self, loaded_model):
        import torch.nn as nn
        model, _ = loaded_model
        norm_modules = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
        assert len(norm_modules) == 2 * DINO_DEPTH + 1
        assert all(abs(m.eps - 1e-6) < 1e-12 for m in norm_modules)


# ==== Phase 2: memory-conscious extractor regression @ 224x224 ====
class TestMemoryConsciousMatchesFullExtractor:
    def test_matches_at_native_grid(self, loaded_model, sample_image, extraction_result):
        model, _ = loaded_model
        layer_maps, _ = extraction_result
        mc_layer_maps, mc_full_row_sums = extract_layerwise_cls_attention_memory_conscious(
            model, sample_image, DINO_NATIVE_GRID)
        assert len(mc_layer_maps) == 12 and len(mc_full_row_sums) == 12
        for a_old, a_new in zip(layer_maps, mc_layer_maps):
            torch.testing.assert_close(a_old, a_new, rtol=1e-5, atol=1e-6)

    def test_deterministic(self, loaded_model, sample_image):
        model, _ = loaded_model
        mc_1, _ = extract_layerwise_cls_attention_memory_conscious(model, sample_image, DINO_NATIVE_GRID)
        mc_2, _ = extract_layerwise_cls_attention_memory_conscious(model, sample_image, DINO_NATIVE_GRID)
        for a1, a2 in zip(mc_1, mc_2):
            torch.testing.assert_close(a1, a2, rtol=0, atol=0)


# ==== Phase 2: OSIE single image, non-square (608x800) ====
@pytest.fixture(scope="module")
def osie_padded_tensor(loaded_model):
    import numpy as np
    from PIL import Image as PILImage

    model, device = loaded_model
    img = PILImage.open(STIM_PATH).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)
    img_np = _pad_to_patch(img_np, DINO_PATCH_SIZE)
    pad_h, pad_w = img_np.shape[:2]
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(PILImage.fromarray(img_np)).unsqueeze(0).to(device)
    return tensor, (orig_h, orig_w), (pad_h, pad_w)


@pytest.fixture(scope="module")
def osie_extraction_result(loaded_model, osie_padded_tensor):
    model, _ = loaded_model
    tensor, _, (pad_h, pad_w) = osie_padded_tensor
    grid_hw = patch_grid_from_image_hw(pad_h, pad_w, DINO_PATCH_SIZE)
    layer_maps, full_row_sums = extract_layerwise_cls_attention_memory_conscious(model, tensor, grid_hw)
    return layer_maps, full_row_sums, grid_hw


class TestOsiePaddingAndGrid:
    def test_padding_matches_dino_geometry(self, osie_padded_tensor):
        _, orig_hw, pad_hw = osie_padded_tensor
        assert orig_hw == (600, 800)
        assert pad_hw == (608, 800)

    def test_grid_and_token_count(self, osie_padded_tensor):
        _, _, (pad_h, pad_w) = osie_padded_tensor
        grid_h, grid_w = patch_grid_from_image_hw(pad_h, pad_w, DINO_PATCH_SIZE)
        assert (grid_h, grid_w) == (38, 50)
        assert grid_h * grid_w + 1 == 1901

    def test_pos_embed_interpolated_shape_and_cls_untouched(self, loaded_model, osie_padded_tensor):
        model, _ = loaded_model
        tensor, _, (pad_h, pad_w) = osie_padded_tensor
        grid_hw = patch_grid_from_image_hw(pad_h, pad_w, DINO_PATCH_SIZE)
        pos_interp = check_native_pos_embed_untouched_at_grid(model, tensor, grid_hw)
        assert tuple(pos_interp.shape) == (1, 1901, 768)
        torch.testing.assert_close(pos_interp[:, 0], model.pos_embed[:, 0])


class TestOsieLayerwiseAttentionMemoryConscious:
    def test_shape_is_12_38_50(self, osie_extraction_result):
        layer_maps, _, grid_hw = osie_extraction_result
        assert grid_hw == (38, 50)
        stacked = torch.stack(layer_maps, dim=1)
        assert tuple(stacked.shape[1:]) == (12, 38, 50)

    def test_full_row_sums_to_one(self, osie_extraction_result):
        _, full_row_sums, _ = osie_extraction_result
        assert len(full_row_sums) == 12
        for row_sum in full_row_sums:
            torch.testing.assert_close(row_sum, torch.ones_like(row_sum), rtol=1e-3, atol=1e-3)

    def test_no_nan_or_inf(self, osie_extraction_result):
        layer_maps, full_row_sums, _ = osie_extraction_result
        for m in layer_maps:
            assert not torch.isnan(m).any()
            assert not torch.isinf(m).any()
        for r in full_row_sums:
            assert not torch.isnan(r).any()
            assert not torch.isinf(r).any()

    def test_deterministic(self, loaded_model, osie_padded_tensor, osie_extraction_result):
        model, _ = loaded_model
        tensor, _, (pad_h, pad_w) = osie_padded_tensor
        grid_hw = patch_grid_from_image_hw(pad_h, pad_w, DINO_PATCH_SIZE)
        layer_maps, _, _ = osie_extraction_result
        layer_maps_2, _ = extract_layerwise_cls_attention_memory_conscious(model, tensor, grid_hw)
        for a1, a2 in zip(layer_maps, layer_maps_2):
            torch.testing.assert_close(a1, a2, rtol=0, atol=0)

    def test_bilinear_resize_to_800x600_preserves_axes(self, osie_extraction_result):
        layer_maps, _, grid_hw = osie_extraction_result
        stacked = torch.stack(layer_maps, dim=1)
        resized = torch.nn.functional.interpolate(
            stacked.reshape(12, 1, *grid_hw), size=(600, 800),
            mode="bilinear", align_corners=False)
        assert tuple(resized.shape) == (12, 1, 600, 800)

    def test_last_layer_matches_independent_reference(
            self, loaded_model, osie_padded_tensor, osie_extraction_result):
        model, _ = loaded_model
        tensor, _, (pad_h, pad_w) = osie_padded_tensor
        grid_hw = patch_grid_from_image_hw(pad_h, pad_w, DINO_PATCH_SIZE)
        layer_maps, _, _ = osie_extraction_result
        with torch.inference_mode():
            last_ref = get_last_selfattention_reference(model, tensor)
        grid_last_ref = cls_to_patch_grid_mean_heads(last_ref, grid_hw)
        torch.testing.assert_close(grid_last_ref, layer_maps[-1], rtol=1e-5, atol=1e-6)
