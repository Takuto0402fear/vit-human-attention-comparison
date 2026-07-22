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
    DINO_DEPTH, DINO_HEADS, DINO_NATIVE_GRID, DINO_VITB16_EXPECTED_MD5,
    assert_vitb16_shape, cls_to_patch_grid_mean_heads,
    extract_layerwise_cls_attention, get_last_selfattention_reference,
    load_dino_vitb16,
)

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
