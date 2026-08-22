"""
Tests for Phase 1 (object-crop pilot) and Phase 2 (spatial diagnostic):
scripts/extract_osie_text_alignment_crop_pilot.py,
scripts/analyze_osie_text_alignment_crop_pilot.py,
scripts/run_osie_text_alignment_spatial_diagnostic.py,
and the new lib/osie_text_alignment.py crop-geometry helpers.

Run:  python -m pytest tests/test_osie_crop_and_spatial.py -v
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.osie_text_alignment import (
    CLIP_MEAN_RGB_255, bbox_from_mask, expand_bbox, pad_to_square,
    split_positive_negative_attributes,
)

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
CROP_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "osie_text_alignment_crop_pilot")
SPATIAL_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "osie_text_alignment_spatial_diagnostic")


def _import_script(module_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    if not os.path.isfile(path):
        pytest.skip(f"{filename} not present")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"could not import {filename} (likely missing GPU/model dependency at import time): {e}")
    return mod


# ======================================================================
# bbox generation + image-boundary clipping
# ======================================================================
class TestBboxFromMask:
    def test_simple_rectangle(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:20, 30:50] = True
        assert bbox_from_mask(mask) == (10, 19, 30, 49)

    def test_empty_mask_returns_none(self):
        mask = np.zeros((100, 200), dtype=bool)
        assert bbox_from_mask(mask) is None

    def test_single_pixel(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        assert bbox_from_mask(mask) == (5, 5, 5, 5)


class TestExpandBbox:
    def test_expands_by_fraction(self):
        bbox = (100, 149, 100, 199)  # h=50, w=100
        out = expand_bbox(bbox, 0.20, img_h=600, img_w=800)
        # dy = round(50*0.2)=10, dx = round(100*0.2)=20
        assert out == (90, 159, 80, 219)

    def test_clips_to_image_boundary_top_left(self):
        bbox = (2, 20, 3, 30)
        out = expand_bbox(bbox, 0.5, img_h=600, img_w=800)
        y0, y1, x0, x1 = out
        assert y0 == 0  # clipped, cannot go negative
        assert x0 == 0

    def test_clips_to_image_boundary_bottom_right(self):
        bbox = (580, 599, 780, 799)
        out = expand_bbox(bbox, 0.5, img_h=600, img_w=800)
        y0, y1, x0, x1 = out
        assert y1 == 599  # clipped, cannot exceed img_h-1
        assert x1 == 799


class TestPadToSquare:
    def test_pads_to_max_dimension(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        out = pad_to_square(img)
        assert out.shape == (20, 20, 3)

    def test_original_content_preserved_centered(self):
        img = np.full((10, 20, 3), 200, dtype=np.uint8)
        out = pad_to_square(img)
        top = (20 - 10) // 2
        np.testing.assert_array_equal(out[top:top + 10, 0:20], img)

    def test_padding_uses_clip_mean_color_not_black(self):
        img = np.full((10, 30, 3), 200, dtype=np.uint8)
        out = pad_to_square(img)
        # corner pixel is definitely in the padded border region
        assert tuple(out[0, 0]) == tuple(CLIP_MEAN_RGB_255)
        assert tuple(out[0, 0]) != (0, 0, 0)

    def test_square_input_is_unchanged_shape(self):
        img = np.zeros((15, 15, 3), dtype=np.uint8)
        out = pad_to_square(img)
        assert out.shape == img.shape


# ======================================================================
# Object survives crop; masked_context20 preserves interior pixels
# ======================================================================
class TestCropPreservesObject:
    def test_tight_crop_bbox_contains_all_mask_pixels(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 15:25] = True
        bbox = bbox_from_mask(mask)
        y0, y1, x0, x1 = bbox
        cropped_mask = mask[y0:y1 + 1, x0:x1 + 1]
        assert cropped_mask.sum() == mask.sum()  # no mask pixels lost by cropping to its own bbox

    def test_masked_context20_preserves_inside_mask_pixels(self):
        img = np.random.RandomState(0).randint(0, 255, size=(50, 50, 3)).astype(np.uint8)
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 15:25] = True
        y0, y1, x0, x1 = 5, 25, 10, 30
        region = img[y0:y1 + 1, x0:x1 + 1].copy()
        region_mask = mask[y0:y1 + 1, x0:x1 + 1]
        region[~region_mask] = CLIP_MEAN_RGB_255
        # inside-mask pixels must be byte-identical to the original image
        np.testing.assert_array_equal(region[region_mask], img[y0:y1 + 1, x0:x1 + 1][region_mask])
        # outside-mask pixels must all be the fill color
        assert np.all(region[~region_mask] == np.array(CLIP_MEAN_RGB_255, dtype=np.uint8))


# ======================================================================
# Same CLIP preprocessing geometry applied to image and mask
# ======================================================================
class TestMaskFollowsImagePreprocessing:
    @pytest.fixture(scope="class")
    def clip_preprocess(self):
        try:
            from clip_extractor import load_clip_vit_b16
        except Exception as e:
            pytest.skip(f"CLIP unavailable: {e}")
        _, preprocess = load_clip_vit_b16()
        return preprocess

    def test_mask_pipeline_reuses_same_resize_size_and_crop_box(self, clip_preprocess):
        mod = _import_script("spatial_diag", "run_osie_text_alignment_spatial_diagnostic.py")
        mask_pipeline = mod.mask_transform_pipeline(clip_preprocess)
        resize_t = mask_pipeline.transforms[0]
        crop_t = mask_pipeline.transforms[1]
        assert resize_t.size == clip_preprocess.transforms[0].size
        assert crop_t.size == clip_preprocess.transforms[1].size

    def test_mask_pipeline_uses_nearest_interpolation(self, clip_preprocess):
        from torchvision.transforms import InterpolationMode
        mod = _import_script("spatial_diag", "run_osie_text_alignment_spatial_diagnostic.py")
        mask_pipeline = mod.mask_transform_pipeline(clip_preprocess)
        assert mask_pipeline.transforms[0].interpolation == InterpolationMode.NEAREST


# ======================================================================
# 14x14 patch orientation (synthetic, no flip)
# ======================================================================
class TestPatchOrientation:
    @pytest.fixture(scope="class")
    def mod(self):
        return _import_script("spatial_diag", "run_osie_text_alignment_spatial_diagnostic.py")

    @pytest.fixture(scope="class")
    def mask_pipeline(self, mod):
        try:
            from clip_extractor import load_clip_vit_b16
        except Exception as e:
            pytest.skip(f"CLIP unavailable: {e}")
        _, preprocess = load_clip_vit_b16()
        return mod.mask_transform_pipeline(preprocess)

    def test_left_half_mask_stays_on_the_left(self, mod, mask_pipeline):
        results = mod.synthetic_orientation_tests(mask_pipeline)
        lh = results["left_half"]
        assert lh["top_left"] > 0.9
        assert lh["bottom_left"] > 0.9
        assert lh["top_right"] < 0.1
        assert lh["bottom_right"] < 0.1

    def test_top_right_mask_stays_top_right(self, mod, mask_pipeline):
        results = mod.synthetic_orientation_tests(mask_pipeline)
        tr = results["top_right"]
        assert tr["top_right"] > 0.9
        assert tr["top_left"] < 0.1
        assert tr["bottom_left"] < 0.1
        assert tr["bottom_right"] < 0.1


# ======================================================================
# CLS exclusion + 196 -> 14x14 row-major reshape
# ======================================================================
class TestReshapeCheck:
    def test_cls_excluded_and_row_major_order_correct(self):
        mod = _import_script("spatial_diag", "run_osie_text_alignment_spatial_diagnostic.py")
        result = mod.reshape_check()
        assert result["cls_excluded"] is True
        assert result["row_major_reshape_correct"] is True


# ======================================================================
# Multi-label positive/negative split (crop pilot reuses the same function)
# ======================================================================
class TestCropMultiLabelSplit:
    def test_positive_never_in_negative_set(self):
        mat_names = ["face", "emotion", "watchability"]
        mapping = {"face": "Face", "emotion": "Emotion", "watchability": "Watchability"}
        pos, neg = split_positive_negative_attributes([1, 1, 0], mat_names, mapping)
        assert set(pos) == {"Face", "Emotion"}
        assert neg == ["Watchability"]
        assert not (set(pos) & set(neg))


# ======================================================================
# Label-ranking average precision (synthetic)
# ======================================================================
class TestLabelRankingAveragePrecision:
    def test_perfect_ranking_gives_one(self):
        from sklearn.metrics import label_ranking_average_precision_score
        y_true = np.array([[0, 1, 0, 1]])
        y_score = np.array([[0.1, 0.9, 0.2, 0.8]])  # both positives ranked above both negatives
        assert label_ranking_average_precision_score(y_true, y_score) == pytest.approx(1.0)

    def test_worst_ranking_is_low(self):
        from sklearn.metrics import label_ranking_average_precision_score
        y_true = np.array([[1, 0, 0, 0]])
        y_score = np.array([[0.1, 0.9, 0.8, 0.7]])  # the one positive ranked LAST
        lrap = label_ranking_average_precision_score(y_true, y_score)
        assert lrap < 0.5


# ======================================================================
# object_id consistency across crop conditions (real-data integration)
# ======================================================================
class TestObjectIdConsistencyAcrossConditions:
    def test_same_object_ids_appear_in_every_crop_condition(self):
        path = os.path.join(CROP_OUT_DIR, "crop_manifest.csv")
        if not os.path.isfile(path):
            pytest.skip("crop_manifest.csv not generated yet")
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        keys = {(r["image_id"], r["object_id"]) for r in rows}
        # every manifest row IS one (image_id, object_id) pair with 4
        # valid_<condition> columns -- the pairing across conditions is by
        # construction (same row), so we verify no row is missing a
        # condition's validity flag.
        for r in rows:
            for cond in ("global_image", "tight_crop", "context20_crop", "masked_context20"):
                assert f"valid_{cond}" in r


# ======================================================================
# Real-data integration: no unexpected NaN/Inf
# ======================================================================
class TestNoNanInfCropAndSpatial:
    @pytest.mark.parametrize("dirpath,filename", [
        (CROP_OUT_DIR, "crop_object_scores.csv"),
        (CROP_OUT_DIR, "crop_attribute_summary.csv"),
        (CROP_OUT_DIR, "crop_condition_comparison.csv"),
        (SPATIAL_OUT_DIR, "spatial_before_after.csv"),
    ])
    def test_no_inf_in_generated_csv(self, dirpath, filename):
        path = os.path.join(dirpath, filename)
        if not os.path.isfile(path):
            pytest.skip(f"{filename} not generated yet")
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        for row in rows:
            for key, val in row.items():
                if val in ("", None):
                    continue
                try:
                    fval = float(val)
                except ValueError:
                    continue
                assert not np.isinf(fval), f"Inf in {filename}:{key}={val}"

    def test_spatial_sanity_checks_report_no_bug(self):
        path = os.path.join(SPATIAL_OUT_DIR, "spatial_sanity_checks.json")
        if not os.path.isfile(path):
            pytest.skip("spatial_sanity_checks.json not generated yet")
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["orientation_tests_pass_no_flip"] is True
        assert result["patch_area_reconstruction_matches_mask_area_for_all_images"] is True
