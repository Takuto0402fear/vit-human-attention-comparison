"""
Tests for Phase 1 (all-12-layer extension):
scripts/extract_osie_probe_all12_full700_features.py,
scripts/probe_osie_all12_full700.py, and the new
lib/osie_text_alignment_probe.py functions (save_fold_assignments_csv,
save_oof_predictions_csv, benjamini_hochberg_fdr, peak_layer_analysis).

Run:  python -m pytest tests/test_osie_probe_all12.py -v
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.osie_text_alignment_probe import (
    benjamini_hochberg_fdr, image_level_bootstrap_auprc_diff, peak_layer_analysis,
    run_linear_probe, save_fold_assignments_csv, save_oof_predictions_csv,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "osie_probe_all12_full700")


# ======================================================================
# L1-L12 hook shapes / CLS exclusion (real model, skip if unavailable)
# ======================================================================
class TestAllLayerHooks:
    @pytest.fixture(scope="class")
    def hidden_states(self):
        try:
            import torch
            from PIL import Image
            from clip_extractor import load_clip_vit_b16
            from lib.clip_vit_hidden import visual_forward_with_hidden_states
        except Exception as e:
            pytest.skip(f"CLIP stack unavailable: {e}")
        model, preprocess = load_clip_vit_b16()
        img_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "osie",
                                 "predicting-human-gaze-beyond-pixels", "data", "stimuli", "1001.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("sample stimulus not present")
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        with torch.inference_mode():
            hs, _ = visual_forward_with_hidden_states(
                model.visual, tensor.type(model.dtype), pos_embed=None, layers_zero_based=list(range(12)))
        return hs

    def test_all_12_layers_present_with_correct_shape(self, hidden_states):
        assert set(hidden_states.keys()) == set(range(12))
        for l in range(12):
            assert hidden_states[l].shape == (1, 197, 768)

    def test_cls_excluded_leaves_196_patches_every_layer(self, hidden_states):
        for l in range(12):
            patch_tokens = hidden_states[l][0, 1:]
            assert patch_tokens.shape[0] == 196

    def test_1_indexed_display_matches_0_indexed_internal(self):
        layers_display = list(range(1, 13))
        layers_zero_based = list(range(12))
        assert [d - 1 for d in layers_display] == layers_zero_based


# ======================================================================
# save_fold_assignments_csv / save_oof_predictions_csv (synthetic)
# ======================================================================
class TestFoldAndOofPersistence:
    def _make_oof_data(self, n_objects=12, n_layers=3, seed=0):
        rng = np.random.RandomState(seed)
        image_ids = np.array([f"img{i % 4}" for i in range(n_objects)])
        object_keys = np.array([f"img{i % 4}_{i}" for i in range(n_objects)])
        fold_ids = np.array([i % 3 for i in range(n_objects)])
        y_true = rng.randint(0, 2, size=n_objects)
        oof_data = {}
        for layer in range(1, n_layers + 1):
            oof_data[("Attr", "all_objects", layer)] = {
                "image_id": image_ids, "object_key": object_keys, "fold_id": fold_ids,
                "y_true": y_true, "y_proba": rng.rand(n_objects), "random_auprc": 0.5,
            }
        return oof_data

    def test_fold_assignments_identical_across_layers_saved_once(self, tmp_path):
        oof_data = self._make_oof_data()
        path = str(tmp_path / "fold_assignments.csv")
        n = save_fold_assignments_csv(oof_data, path)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n == 12  # one row per object (not x n_layers)

    def test_fold_assignments_raises_if_layers_disagree(self, tmp_path):
        oof_data = self._make_oof_data()
        # corrupt layer 2's fold assignment
        oof_data[("Attr", "all_objects", 2)]["fold_id"] = np.roll(
            oof_data[("Attr", "all_objects", 2)]["fold_id"], 1)
        path = str(tmp_path / "fold_assignments.csv")
        with pytest.raises(RuntimeError):
            save_fold_assignments_csv(oof_data, path)

    def test_oof_predictions_one_row_per_attribute_layer_object(self, tmp_path):
        oof_data = self._make_oof_data(n_objects=12, n_layers=3)
        path = str(tmp_path / "oof.csv")
        n = save_oof_predictions_csv(oof_data, path)
        assert n == 12 * 3
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 36
        assert set(int(r["layer"]) for r in rows) == {1, 2, 3}


# ======================================================================
# Benjamini-Hochberg FDR
# ======================================================================
class TestBenjaminiHochberg:
    def test_all_significant_pvalues_stay_significant(self):
        pvals = [0.001, 0.002, 0.003, 0.004]
        q = benjamini_hochberg_fdr(pvals)
        assert np.all(q < 0.05)

    def test_monotonic_nondecreasing_after_sorting(self):
        pvals = [0.5, 0.01, 0.3, 0.02, 0.9]
        q = benjamini_hochberg_fdr(pvals)
        order = np.argsort(pvals)
        assert np.all(np.diff(q[order]) >= -1e-12)

    def test_nan_pvalues_pass_through_and_excluded_from_family(self):
        pvals = [0.01, np.nan, 0.02]
        q = benjamini_hochberg_fdr(pvals)
        assert np.isnan(q[1])
        assert np.isfinite(q[0]) and np.isfinite(q[2])

    def test_known_reference_case(self):
        # classic textbook example: 5 p-values
        pvals = [0.01, 0.02, 0.03, 0.04, 0.20]
        q = benjamini_hochberg_fdr(pvals)
        # BH: q_i = p_i * m / rank; then enforce monotone from the top.
        # rank1: 0.01*5/1=0.05; rank2: 0.02*5/2=0.05; rank3: 0.03*5/3=0.05;
        # rank4: 0.04*5/4=0.05; rank5: 0.20*5/5=0.20
        np.testing.assert_allclose(sorted(q), [0.05, 0.05, 0.05, 0.05, 0.20], atol=1e-9)

    def test_reproducible_given_same_input(self):
        pvals = [0.3, 0.01, 0.04, 0.5]
        q1 = benjamini_hochberg_fdr(pvals)
        q2 = benjamini_hochberg_fdr(pvals)
        np.testing.assert_array_equal(q1, q2)


# ======================================================================
# Peak-layer detection with near-peak tie tolerance
# ======================================================================
class TestPeakLayerAnalysis:
    def test_single_clear_peak(self):
        auprc = {1: 0.2, 4: 0.3, 8: 0.9, 12: 0.5}
        # all other layers significantly below peak (CI excludes 0)
        diff_results = {
            (8, 1): {"ci95_lo": 0.5, "ci95_hi": 0.8},
            (8, 4): {"ci95_lo": 0.4, "ci95_hi": 0.7},
            (8, 12): {"ci95_lo": 0.2, "ci95_hi": 0.5},
        }
        result = peak_layer_analysis(auprc, diff_results, [1, 4, 8, 12])
        assert result["peak_layer"] == 8
        assert result["near_peak_label"] == "L8"
        assert result["layers_not_significantly_below_peak"] == [8]

    def test_near_peak_tie_produces_range_label(self):
        auprc = {7: 0.85, 8: 0.90, 9: 0.86}
        diff_results = {
            (8, 7): {"ci95_lo": -0.02, "ci95_hi": 0.10},  # CI includes 0 -- not significantly different
            (8, 9): {"ci95_lo": -0.03, "ci95_hi": 0.08},  # CI includes 0
        }
        result = peak_layer_analysis(auprc, diff_results, [7, 8, 9])
        assert result["peak_layer"] == 8
        assert result["near_peak_label"] == "L7-L9"
        assert result["layers_not_significantly_below_peak"] == [7, 8, 9]

    def test_non_contiguous_tier_lists_layers(self):
        auprc = {1: 0.5, 4: 0.51, 8: 0.9, 12: 0.5}
        diff_results = {
            (8, 1): {"ci95_lo": -0.01, "ci95_hi": 0.5},  # tied with peak
            (8, 4): {"ci95_lo": 0.1, "ci95_hi": 0.6},    # significantly below
            (8, 12): {"ci95_lo": -0.02, "ci95_hi": 0.5},  # tied with peak
        }
        result = peak_layer_analysis(auprc, diff_results, [1, 4, 8, 12])
        assert result["peak_layer"] == 8
        assert result["layers_not_significantly_below_peak"] == [1, 8, 12]
        assert result["near_peak_label"] == "L1,L8,L12"


# ======================================================================
# Real-data integration: reproducibility check + all-12-layer consistency
# ======================================================================
class TestRealAll12Integration:
    def test_reproducibility_check_passed(self):
        path = os.path.join(OUT_DIR, "extraction_reproducibility_check.json")
        if not os.path.isfile(path):
            pytest.skip("extraction_reproducibility_check.json not generated yet")
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        if result.get("skipped"):
            pytest.skip("reproducibility check was skipped (reference file absent)")
        assert result["passed"] is True

    def test_object_ids_consistent_across_layers_in_metadata(self):
        features_npz = os.path.join(OUT_DIR, "object_features_all12.npz")
        metadata_csv = os.path.join(OUT_DIR, "object_features_metadata.csv")
        if not (os.path.isfile(features_npz) and os.path.isfile(metadata_csv)):
            pytest.skip("Phase 1 all-12-layer outputs not generated yet")
        npz = np.load(features_npz, allow_pickle=False)
        object_keys = list(npz["object_keys"])
        for l in range(1, 13):
            assert f"raw_L{l}" in npz.files
            assert npz[f"raw_L{l}"].shape[0] == len(object_keys)
        with open(metadata_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(object_keys)
        for row, key in zip(rows, object_keys):
            assert f"{row['image_id']}_{row['object_id']}" == key

    @pytest.mark.parametrize("filename", [
        "probe_fold_scores.csv", "probe_summary.csv", "probe_layer_differences.csv",
        "probe_peak_layers.csv", "probe_permutation_baseline.csv",
    ])
    def test_no_inf_in_generated_csv(self, filename):
        path = os.path.join(OUT_DIR, filename)
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
