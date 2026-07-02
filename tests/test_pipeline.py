"""
Minimal tests for each module.
Run:  python -m pytest tests/test_pipeline.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config import Config
from schema import Fixation, GazeRecord, validate_record
from data_loader import load_gaze
from gaze_heatmap import fixations_to_heatmap
from metrics import nss, auc_judd, sauc
from vit_extractor import extract_attention_maps
from aggregation import compute_all_metrics, summarise_group_layer


@pytest.fixture
def config():
    return Config(seed=42, dummy_n_subjects_per_group=3, dummy_n_images=2, depth=4)


# ==== schema ====
class TestSchema:
    def test_valid_record(self):
        rec = GazeRecord(
            subject_id="S001", group="healthy", image_id="img_0001",
            fixations=[Fixation(100, 200)],
            coord_system="image_topleft_pixel", image_size=(720, 480))
        validate_record(rec)  # should not raise

    def test_invalid_group(self):
        rec = GazeRecord(
            subject_id="S001", group="unknown", image_id="img_0001",
            fixations=[Fixation(100, 200)],
            coord_system="image_topleft_pixel", image_size=(720, 480))
        with pytest.raises(ValueError, match="Invalid group"):
            validate_record(rec)

    def test_out_of_bounds(self):
        rec = GazeRecord(
            subject_id="S001", group="healthy", image_id="img_0001",
            fixations=[Fixation(800, 200)],
            coord_system="image_topleft_pixel", image_size=(720, 480))
        with pytest.raises(ValueError, match="out of bounds"):
            validate_record(rec)


# ==== data_loader ====
class TestDataLoader:
    def test_dummy_load(self, config):
        records = load_gaze("dummy", config)
        n_expected = (config.dummy_n_subjects_per_group
                      * len(config.groups)
                      * config.dummy_n_images)
        assert len(records) == n_expected
        groups = set(r.group for r in records)
        assert groups == {"healthy", "schizophrenia"}

    def test_unknown_source(self, config):
        with pytest.raises(ValueError):
            load_gaze("nonexistent", config)


# ==== gaze_heatmap ====
class TestHeatmap:
    def test_shape_and_sum(self):
        pts = np.array([[360, 240], [100, 100], [500, 300]])
        hmap = fixations_to_heatmap(pts, (720, 480), sigma=30)
        assert hmap.shape == (480, 720)
        assert abs(hmap.sum() - 1.0) < 1e-5

    def test_empty_fixations(self):
        pts = np.empty((0, 2))
        hmap = fixations_to_heatmap(pts, (720, 480), sigma=30)
        assert hmap.sum() == 0.0


# ==== metrics ====
class TestMetrics:
    def test_nss_perfect(self):
        """If fixations land on the peak, NSS should be high."""
        sal = np.zeros((100, 100), dtype=np.float64)
        sal[50, 50] = 1.0
        pts = np.array([[50, 50]])
        score = nss(sal, pts)
        assert score > 0

    def test_auc_perfect(self):
        sal = np.zeros((100, 100), dtype=np.float64)
        sal[50, 50] = 1.0
        pts = np.array([[50, 50]])
        score = auc_judd(sal, pts)
        assert score > 0.9

    def test_sauc_runs(self):
        rng = np.random.RandomState(0)
        sal = rng.rand(100, 100)
        pts = rng.rand(10, 2) * 99
        other = rng.rand(50, 2) * 99
        score = sauc(sal, pts, other)
        assert 0 <= score <= 1

    def test_nss_empty(self):
        sal = np.random.rand(100, 100)
        pts = np.empty((0, 2))
        assert np.isnan(nss(sal, pts))


# ==== vit_extractor (dummy) ====
class TestViTExtractor:
    def test_dummy_shapes(self, config):
        ids = {"img_0000": "", "img_0001": ""}
        maps = extract_attention_maps(ids, config, dummy=True)
        assert len(maps) == 2
        for k, v in maps.items():
            assert v.shape == (config.depth, config.heatmap_height, config.heatmap_width)
            # each layer sums to ~1
            for l in range(config.depth):
                assert abs(v[l].sum() - 1.0) < 1e-4


# ==== aggregation ====
class TestAggregation:
    def test_end_to_end_dummy(self, config):
        records = load_gaze("dummy", config)
        image_ids = sorted(set(r.image_id for r in records))
        attn = extract_attention_maps(
            {i: "" for i in image_ids}, config, dummy=True)
        results = compute_all_metrics(records, attn, config)
        assert len(results) > 0
        assert all(k in results[0] for k in
                   ["subject_id", "group", "layer", "nss", "auc_judd", "sauc"])

        summary = summarise_group_layer(results)
        assert len(summary) == len(config.groups) * config.depth
        for row in summary:
            assert "nss_mean" in row
            assert "sauc_mean" in row
