"""
Tests for Phase 3: lib/osie_lowlevel_probe.py and
scripts/probe_osie_lowlevel_all12_full700.py.

Run:  python -m pytest tests/test_osie_lowlevel_probe.py -v
"""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.osie_lowlevel_probe import (
    align_features_to_targets, compute_metrics, image_level_bootstrap_r2_diff,
    make_ridge_pipeline, run_ridge_probe,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "osie_lowlevel_probe_all12_full700")


# ======================================================================
# Ridge regression recovers a known linear relationship
# ======================================================================
class TestRidgeRecoversKnownRelationship:
    def test_perfect_linear_signal_gives_high_r2(self, tmp_path):
        rng = np.random.RandomState(42)
        n_images, per_image = 60, 3
        n = n_images * per_image
        object_keys, image_ids = [], []
        X = rng.normal(size=(n, 16))
        # target is a KNOWN linear function of features + small noise
        true_w = rng.normal(size=16)
        y = X @ true_w + rng.normal(scale=0.05, size=n)

        for i in range(n_images):
            for j in range(per_image):
                idx = i * per_image + j
                object_keys.append(f"img{i:03d}_{j}")
                image_ids.append(f"img{i:03d}")

        npz_path = tmp_path / "object_features_all12.npz"
        np.savez_compressed(npz_path, object_keys=np.array(object_keys),
                             raw_L1=X, raw_L4=X, raw_L8=X, raw_L12=X)

        csv_path = tmp_path / "lowlevel_object_targets.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            fieldnames = ["image_id", "object_id", "valid", "skip_reason", "test_target"]
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for i in range(n_images):
                for j in range(per_image):
                    idx = i * per_image + j
                    w.writerow({"image_id": f"img{i:03d}", "object_id": j, "valid": "True",
                                "skip_reason": "", "test_target": f"{y[idx]:.6f}"})

        out_dir = str(tmp_path / "out")
        fold_rows, summary_rows, oof_data = run_ridge_probe(
            str(npz_path), str(csv_path), out_dir, layers=(1, 4, 8, 12),
            targets=("test_target",), alpha=1.0, seed=42, collect_oof=True)
        r2_rows = [r for r in summary_rows if r["metric"] == "r2"]
        assert all(r["mean"] > 0.8 for r in r2_rows)

    def test_pure_noise_target_gives_low_r2(self, tmp_path):
        rng = np.random.RandomState(0)
        n_images, per_image = 60, 3
        n = n_images * per_image
        X = rng.normal(size=(n, 16))
        y = rng.normal(size=n)  # unrelated to X

        object_keys, image_ids = [], []
        for i in range(n_images):
            for j in range(per_image):
                object_keys.append(f"img{i:03d}_{j}")

        npz_path = tmp_path / "object_features_all12.npz"
        np.savez_compressed(npz_path, object_keys=np.array(object_keys), raw_L4=X)

        csv_path = tmp_path / "lowlevel_object_targets.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            fieldnames = ["image_id", "object_id", "valid", "skip_reason", "noise_target"]
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for i in range(n_images):
                for j in range(per_image):
                    idx = i * per_image + j
                    w.writerow({"image_id": f"img{i:03d}", "object_id": j, "valid": "True",
                                "skip_reason": "", "noise_target": f"{y[idx]:.6f}"})

        out_dir = str(tmp_path / "out2")
        _, summary_rows, _ = run_ridge_probe(
            str(npz_path), str(csv_path), out_dir, layers=(4,),
            targets=("noise_target",), alpha=1.0, seed=42, collect_oof=False)
        r2_row = next(r for r in summary_rows if r["metric"] == "r2")
        assert r2_row["mean"] < 0.3  # should not fit noise well out-of-fold


# ======================================================================
# align_features_to_targets
# ======================================================================
class TestAlignFeaturesToTargets:
    def test_correct_alignment(self):
        feature_keys = ["b", "a", "c"]
        target_keys = ["a", "b", "c"]
        idx = align_features_to_targets(feature_keys, target_keys)
        np.testing.assert_array_equal(idx, [1, 0, 2])

    def test_missing_key_raises(self):
        feature_keys = ["a", "b"]
        target_keys = ["a", "c"]
        with pytest.raises(RuntimeError):
            align_features_to_targets(feature_keys, target_keys)


# ======================================================================
# compute_metrics edge cases
# ======================================================================
class TestComputeMetrics:
    def test_constant_y_true_returns_nan(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([1.1, 0.9, 1.0])
        m = compute_metrics(y_true, y_pred)
        assert np.isnan(m["r2"])

    def test_perfect_prediction_gives_r2_one(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()
        m = compute_metrics(y_true, y_pred)
        assert m["r2"] == pytest.approx(1.0)
        assert m["mae"] == pytest.approx(0.0)


# ======================================================================
# Ridge pipeline hyperparameters
# ======================================================================
class TestRidgePipeline:
    def test_alpha_is_1_and_scaler_present(self):
        pipe = make_ridge_pipeline(alpha=1.0)
        assert pipe.named_steps["ridge"].alpha == 1.0
        assert "scaler" in pipe.named_steps


# ======================================================================
# Image-level bootstrap R2 difference
# ======================================================================
class TestBootstrapR2Diff:
    def test_identical_predictions_give_zero_diff(self):
        rng = np.random.RandomState(0)
        n = 100
        image_ids = np.array([f"img{i % 20}" for i in range(n)])
        y_true = rng.normal(size=n)
        y_pred = y_true + rng.normal(scale=0.1, size=n)
        oof_a = {"image_id": image_ids, "y_true": y_true, "y_pred": y_pred}
        oof_b = {"image_id": image_ids, "y_true": y_true, "y_pred": y_pred}
        result = image_level_bootstrap_r2_diff(oof_a, oof_b, seed=42, n_boot=500)
        assert result["observed_diff"] == pytest.approx(0.0, abs=1e-9)

    def test_mismatched_image_ids_raise(self):
        oof_a = {"image_id": np.array(["a", "b"]), "y_true": np.array([1.0, 2.0]), "y_pred": np.array([1.0, 2.0])}
        oof_b = {"image_id": np.array(["a", "c"]), "y_true": np.array([1.0, 2.0]), "y_pred": np.array([1.0, 2.0])}
        with pytest.raises(AssertionError):
            image_level_bootstrap_r2_diff(oof_a, oof_b, seed=42, n_boot=10)


# ======================================================================
# Real-data integration
# ======================================================================
class TestRealLowlevelProbeIntegration:
    @pytest.mark.parametrize("filename", [
        "lowlevel_fold_scores.csv", "lowlevel_summary.csv",
        "lowlevel_layer_differences.csv", "lowlevel_peak_layers.csv",
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

    def test_fold_assignments_group_safe(self):
        path = os.path.join(OUT_DIR, "fold_assignments.csv")
        if not os.path.isfile(path):
            pytest.skip("fold_assignments.csv not generated yet")
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_target = {}
        for r in rows:
            by_target.setdefault(r["target"], {}).setdefault(r["fold"], set()).add(r["object_key"].rsplit("_", 1)[0])
        for target, folds in by_target.items():
            all_images_seen = list(folds.values())
            for i in range(len(all_images_seen)):
                for j in range(i + 1, len(all_images_seen)):
                    assert not (all_images_seen[i] & all_images_seen[j]), \
                        f"image_id leaked across folds for target={target}"
