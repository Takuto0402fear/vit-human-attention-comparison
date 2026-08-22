"""
Tests for Phase 4: lib/osie_text_alignment_probe.py and
scripts/probe_osie_full700.py.

Run:  python -m pytest tests/test_osie_probe_full700.py -v
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.osie_text_alignment_probe import (
    compute_metrics, image_level_bootstrap_auprc_diff, make_pipeline, run_linear_probe,
)

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
FULL700_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "osie_probe_full700")
PILOT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "osie_text_alignment_pilot")


# ======================================================================
# 50-image and 700-image scripts use the identical implementation
# ======================================================================
class TestSharedImplementation:
    def test_both_scripts_call_the_same_lib_function(self):
        for filename in ("probe_osie_text_alignment.py", "probe_osie_full700.py"):
            path = os.path.join(SCRIPTS_DIR, filename)
            with open(path, encoding="utf-8") as f:
                src = f.read()
            assert "from lib.osie_text_alignment_probe import" in src
            assert "run_linear_probe(" in src
            # neither script re-implements its own StratifiedGroupKFold/Pipeline logic
            assert "StratifiedGroupKFold(" not in src
            assert "Pipeline([" not in src


# ======================================================================
# Synthetic end-to-end run of run_linear_probe (fast, tiny data)
# ======================================================================
class TestRunLinearProbeSynthetic:
    @pytest.fixture
    def tiny_npz_and_csv(self, tmp_path):
        rng = np.random.RandomState(0)
        n_images = 50
        objects_per_image = 3
        rows = []
        object_keys, raw_L4, raw_L8, raw_L12 = [], [], [], []
        for img_i in range(n_images):
            image_id = f"img{img_i:03d}"
            for obj_i in range(objects_per_image):
                is_pos = (img_i % 2 == 0)  # perfectly separable by image parity
                key = f"{image_id}_{obj_i}"
                object_keys.append(key)
                base = np.ones(8) * (5.0 if is_pos else -5.0)
                raw_L4.append(base + rng.normal(0, 0.1, size=8))
                raw_L8.append(base + rng.normal(0, 0.1, size=8))
                raw_L12.append(base + rng.normal(0, 0.1, size=8))
                rows.append({
                    "image_id": image_id, "object_id": obj_i,
                    "positive_attributes": "TestAttr" if is_pos else "",
                    "negative_attributes": "" if is_pos else "TestAttr",
                })

        npz_path = tmp_path / "object_features.npz"
        np.savez_compressed(npz_path, object_keys=np.array(object_keys),
                             raw_L4=np.array(raw_L4), raw_L8=np.array(raw_L8), raw_L12=np.array(raw_L12))
        csv_path = tmp_path / "objects_metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["image_id", "object_id", "positive_attributes", "negative_attributes"])
            w.writeheader()
            w.writerows(rows)
        return str(npz_path), str(csv_path)

    def test_perfectly_separable_data_gives_near_perfect_auprc(self, tiny_npz_and_csv, tmp_path):
        npz_path, csv_path = tiny_npz_and_csv
        out_dir = str(tmp_path / "out")
        fold_rows, summary_rows, perm_rows, insufficient, _ = run_linear_probe(
            npz_path, csv_path, out_dir, layers=(4, 8, 12), attributes=("TestAttr",),
            n_permutation_reps=3, seed=42, verbose=False)
        assert len(insufficient) == 0
        auprc_rows = [r for r in summary_rows if r["metric"] == "auprc"]
        assert all(r["mean"] > 0.9 for r in auprc_rows)

    def test_no_image_id_leaked_across_train_test(self, tiny_npz_and_csv, tmp_path):
        # run_linear_probe itself raises RuntimeError on any leak (verified
        # inline for every fold) -- this test just confirms it completes
        # without raising, i.e. the leak-check never fired.
        npz_path, csv_path = tiny_npz_and_csv
        out_dir = str(tmp_path / "out2")
        run_linear_probe(npz_path, csv_path, out_dir, layers=(4,), attributes=("TestAttr",),
                          n_permutation_reps=2, seed=42, verbose=False)

    def test_seed_reproducibility(self, tiny_npz_and_csv, tmp_path):
        npz_path, csv_path = tiny_npz_and_csv
        out_dir_a = str(tmp_path / "out_a")
        out_dir_b = str(tmp_path / "out_b")
        fold_a, *_ = run_linear_probe(npz_path, csv_path, out_dir_a, layers=(4,),
                                       attributes=("TestAttr",), n_permutation_reps=2, seed=42, verbose=False)
        fold_b, *_ = run_linear_probe(npz_path, csv_path, out_dir_b, layers=(4,),
                                       attributes=("TestAttr",), n_permutation_reps=2, seed=42, verbose=False)
        for ra, rb in zip(fold_a, fold_b):
            assert ra["auprc"] == rb["auprc"]
            assert ra["auroc"] == rb["auroc"]

    def test_no_nan_inf_in_fold_scores(self, tiny_npz_and_csv, tmp_path):
        npz_path, csv_path = tiny_npz_and_csv
        out_dir = str(tmp_path / "out_c")
        fold_rows, *_ = run_linear_probe(npz_path, csv_path, out_dir, layers=(4, 8, 12),
                                          attributes=("TestAttr",), n_permutation_reps=2, seed=42, verbose=False)
        for r in fold_rows:
            for key in ("auprc", "auroc", "balanced_accuracy"):
                assert np.isfinite(r[key])


# ======================================================================
# make_pipeline / compute_metrics
# ======================================================================
class TestPipelineAndMetrics:
    def test_pipeline_has_expected_hyperparameters(self):
        pipe = make_pipeline(seed=42)
        clf = pipe.named_steps["clf"]
        assert clf.penalty == "l2"
        assert clf.C == 1.0
        assert clf.class_weight == "balanced"
        assert clf.max_iter == 5000
        assert clf.random_state == 42

    def test_compute_metrics_single_class_returns_nan(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1])
        m = compute_metrics(y_true, y_pred, y_proba)
        assert np.isnan(m["auprc"])
        assert np.isnan(m["auroc"])


# ======================================================================
# Image-level bootstrap AUPRC difference
# ======================================================================
class TestLayerDifferenceBootstrap:
    def test_identical_layers_give_near_zero_diff(self):
        rng = np.random.RandomState(0)
        n = 200
        image_ids = np.array([f"img{i % 40}" for i in range(n)])
        y_true = (rng.rand(n) > 0.5).astype(int)
        y_proba = np.where(y_true == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
        oof_a = {"image_id": image_ids, "y_true": y_true, "y_proba": y_proba, "random_auprc": 0.5}
        oof_b = {"image_id": image_ids, "y_true": y_true, "y_proba": y_proba, "random_auprc": 0.5}
        result = image_level_bootstrap_auprc_diff(oof_a, oof_b, seed=42, n_boot=500)
        assert result["observed_diff"] == pytest.approx(0.0, abs=1e-9)
        assert result["ci95_lo"] <= 0.0 <= result["ci95_hi"]

    def test_mismatched_inputs_raise(self):
        oof_a = {"image_id": np.array(["a", "b"]), "y_true": np.array([0, 1]), "y_proba": np.array([0.1, 0.9])}
        oof_b = {"image_id": np.array(["a", "c"]), "y_true": np.array([0, 1]), "y_proba": np.array([0.1, 0.9])}
        with pytest.raises(AssertionError):
            image_level_bootstrap_auprc_diff(oof_a, oof_b, seed=42, n_boot=10)


# ======================================================================
# Real-data integration (skip cleanly if Phase 4 outputs not generated)
# ======================================================================
class TestRealFull700Integration:
    def test_full700_manifest_reuses_existing_700_image_list(self):
        manifest_path = os.path.join(FULL700_OUT_DIR, "full700_manifest.csv")
        source_path = os.path.join(os.path.dirname(__file__), "..", "outputs",
                                    "osie_attribute_grounding_full700", "full700_image_list.csv")
        if not (os.path.isfile(manifest_path) and os.path.isfile(source_path)):
            pytest.skip("Phase 4 outputs not generated yet")
        with open(manifest_path, encoding="utf-8") as f:
            reused = sorted(r["image_id"] for r in csv.DictReader(f))
        with open(source_path, encoding="utf-8") as f:
            source = sorted(r["image_id"] for r in csv.DictReader(f))
        assert reused == source
        assert len(reused) == 700

    def test_no_insufficient_data_for_any_attribute(self):
        path = os.path.join(FULL700_OUT_DIR, "probe_go_no_go.json")
        if not os.path.isfile(path):
            pytest.skip("probe_go_no_go.json not generated yet")
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["n_insufficient_data"] == 0

    @pytest.mark.parametrize("filename", [
        "probe_fold_scores.csv", "probe_summary.csv", "probe_permutation_baseline.csv",
        "probe_layer_differences.csv",
    ])
    def test_no_inf_in_generated_csv(self, filename):
        path = os.path.join(FULL700_OUT_DIR, filename)
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
