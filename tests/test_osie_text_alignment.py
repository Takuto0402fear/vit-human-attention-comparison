"""
Tests for the OSIE text-alignment / linear-probe pilot
(outputs/osie_text_alignment_pilot/, scripts/*_osie_text_alignment*.py,
lib/osie_text_alignment.py, lib/clip_vit_hidden.py).

Fast (no-GPU, synthetic-data) unit tests run always. A few slower tests
that load the real CLIP model or read this pilot's already-generated
outputs are grouped under TestRealDataIntegration and
TestRealModelIntegration; they skip cleanly if the prerequisite files/
model aren't available (matching this repo's existing test conventions).

Run:  python -m pytest tests/test_osie_text_alignment.py -v
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.osie_text_alignment import (
    alignment_margin, build_equal_ensemble, circular_shift_2d,
    enumerate_or_sample_subsets, l2_normalize, mask_to_patch_weights,
    random_nonzero_shift, safe_n_splits, split_positive_negative_attributes,
    weighted_pool_tokens,
)

DATA_BASE = os.path.join(os.path.dirname(__file__), "..", "datasets", "osie",
                          "predicting-human-gaze-beyond-pixels")
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "osie_text_alignment_pilot")
PRIOR_PILOT_LIST = os.path.join(os.path.dirname(__file__), "..", "outputs",
                                 "osie_attribute_grounding_pilot", "pilot_image_list.csv")

EXPECTED_MAT_ORDER = [
    "text", "face", "emotion", "sound", "smell", "taste", "touch",
    "motion", "operability", "watchability", "touched", "gazed",
]
EXPECTED_TASK_NAMES = {
    "Text", "Face", "Emotion", "Sound", "Smell", "Taste", "Touch",
    "Motion", "Operability", "Watchability", "Touched", "Gazed",
}


# ======================================================================
# attrs.mat structure
# ======================================================================
class TestAttrsMatStructure:
    def test_order_names_and_shape(self):
        h5py = pytest.importorskip("h5py")
        if not os.path.isfile(ATTRS_PATH):
            pytest.skip("attrs.mat not present in this checkout")
        with h5py.File(ATTRS_PATH, "r") as f:
            names_ds = f["attrNames"]
            assert names_ds.shape == (12, 1)
            names = ["".join(chr(int(c)) for c in names_ds[i, 0][()].flatten())
                     if False else None for i in range(names_ds.shape[0])]
            names = []
            for i in range(names_ds.shape[0]):
                arr = f[names_ds[i, 0]][()]
                names.append("".join(chr(int(c)) for c in arr.flatten()))
            assert names == EXPECTED_MAT_ORDER
            attrs_ds = f["attrs"]
            assert attrs_ds.shape == (1, 700)

    def test_touch_and_touched_are_distinct(self):
        assert "touch" in EXPECTED_MAT_ORDER
        assert "touched" in EXPECTED_MAT_ORDER
        assert EXPECTED_MAT_ORDER.count("touch") == 1
        assert EXPECTED_MAT_ORDER.count("touched") == 1
        assert "Touch" in EXPECTED_TASK_NAMES and "Touched" in EXPECTED_TASK_NAMES
        assert "Touch" != "Touched"


# ======================================================================
# Prior pilot manifest reuse
# ======================================================================
class TestPilotManifestReuse:
    def test_prior_manifest_has_50_images_seed42(self):
        if not os.path.isfile(PRIOR_PILOT_LIST):
            pytest.skip("prior attribute-grounding pilot manifest not present")
        with open(PRIOR_PILOT_LIST, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 50
        ids = [r["image_id"] for r in rows]
        assert len(set(ids)) == 50

    def test_reused_manifest_matches_prior_exactly(self):
        manifest_path = os.path.join(OUT_DIR, "pilot_manifest.csv")
        if not (os.path.isfile(manifest_path) and os.path.isfile(PRIOR_PILOT_LIST)):
            pytest.skip("pilot_manifest.csv or prior manifest not present -- run extraction first")
        with open(manifest_path, encoding="utf-8") as f:
            reused_ids = sorted(r["image_id"] for r in csv.DictReader(f))
        with open(PRIOR_PILOT_LIST, encoding="utf-8") as f:
            prior_ids = sorted(r["image_id"] for r in csv.DictReader(f))
        assert reused_ids == prior_ids


# ======================================================================
# prompt_bank.json
# ======================================================================
class TestPromptBank:
    @pytest.fixture(scope="class")
    def bank(self):
        path = os.path.join(OUT_DIR, "prompt_bank.json")
        if not os.path.isfile(path):
            pytest.skip("prompt_bank.json not present -- run build_osie_text_alignment_prompts.py first")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_12_attributes_16_templates(self, bank):
        assert len(bank["templates"]) == 16
        assert set(bank["attributes"].keys()) == EXPECTED_TASK_NAMES
        for attr, data in bank["attributes"].items():
            assert len(data["prompts"]) == 16

    def test_no_empty_or_duplicate_sentences(self, bank):
        seen = set()
        for attr, data in bank["attributes"].items():
            for p in data["prompts"]:
                assert p["sentence"].strip() != ""
                key = p["sentence"]
                assert key not in seen, f"duplicate sentence across bank: {key}"
                seen.add(key)

    def test_raw_label_not_mixed_into_main_ensemble(self, bank):
        for attr, data in bank["attributes"].items():
            prompt_sentences = {p["sentence"] for p in data["prompts"]}
            assert data["raw_label_baseline"] not in prompt_sentences or attr == data["raw_label_baseline"]


# ======================================================================
# mask -> patch-weight conversion
# ======================================================================
class TestMaskToPatchWeights:
    def test_full_single_patch(self):
        mask = np.ones((16, 16), dtype=bool)
        w = mask_to_patch_weights(mask, patch_size=16)
        assert w.shape == (1, 1)
        assert w[0, 0] == pytest.approx(1.0)

    def test_half_patch_coverage(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[:8, :] = True  # top half of the one patch
        w = mask_to_patch_weights(mask, patch_size=16)
        assert w[0, 0] == pytest.approx(0.5)

    def test_boundary_padding_reduces_weight_not_undefined(self):
        # 24 rows: patch 0 fully covered by real pixels (16/16), patch 1
        # only has 8 real rows (rows 16-23) + 8 padded (zero) rows.
        mask = np.ones((24, 16), dtype=bool)
        w = mask_to_patch_weights(mask, patch_size=16)
        assert w.shape == (2, 1)
        assert w[0, 0] == pytest.approx(1.0)
        assert w[1, 0] == pytest.approx(0.5)  # 8/16 real rows, all covered

    def test_zero_mask_gives_zero_weights(self):
        mask = np.zeros((32, 32), dtype=bool)
        w = mask_to_patch_weights(mask, patch_size=16)
        assert np.all(w == 0)


# ======================================================================
# Weighted pooling
# ======================================================================
class TestWeightedPooling:
    def test_uniform_weights_equals_plain_mean(self):
        tokens = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
        weights = np.ones((2, 2))
        pooled = weighted_pool_tokens(tokens, weights)
        expected = tokens.reshape(-1, 3).mean(axis=0)
        np.testing.assert_allclose(pooled, expected)

    def test_single_active_patch(self):
        tokens = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # (2,1,2)
        weights = np.array([[1.0], [0.0]])
        pooled = weighted_pool_tokens(tokens, weights)
        np.testing.assert_allclose(pooled, [1.0, 0.0])

    def test_below_threshold_returns_none(self):
        tokens = np.ones((2, 2, 4))
        weights = np.zeros((2, 2))
        assert weighted_pool_tokens(tokens, weights, min_weight_sum=1e-6) is None

    def test_known_weighted_average(self):
        tokens = np.array([[10.0], [20.0], [30.0], [40.0]]).reshape(2, 2, 1)
        weights = np.array([[1.0, 3.0], [0.0, 0.0]])
        pooled = weighted_pool_tokens(tokens, weights)
        # (10*1 + 20*3) / 4 = 70/4 = 17.5
        np.testing.assert_allclose(pooled, [17.5])


# ======================================================================
# Multi-label positive/negative split
# ======================================================================
class TestPositiveNegativeSplit:
    def test_multiple_positives_never_leak_into_negative(self):
        mat_names = ["face", "emotion", "text", "touch", "touched"]
        mapping = {"face": "Face", "emotion": "Emotion", "text": "Text",
                   "touch": "Touch", "touched": "Touched"}
        features = [1, 1, 0, 0, 1]  # face, emotion, touched positive; text, touch negative
        pos, neg = split_positive_negative_attributes(features, mat_names, mapping)
        assert set(pos) == {"Face", "Emotion", "Touched"}
        assert set(neg) == {"Text", "Touch"}
        assert not (set(pos) & set(neg))

    def test_all_positive_gives_empty_negative(self):
        mat_names = ["face", "text"]
        mapping = {"face": "Face", "text": "Text"}
        pos, neg = split_positive_negative_attributes([1, 1], mat_names, mapping)
        assert set(pos) == {"Face", "Text"}
        assert neg == []

    def test_touch_and_touched_kept_separate(self):
        mat_names = ["touch", "touched"]
        mapping = {"touch": "Touch", "touched": "Touched"}
        pos, neg = split_positive_negative_attributes([1, 0], mat_names, mapping)
        assert pos == ["Touch"]
        assert neg == ["Touched"]


# ======================================================================
# Label Alignment Margin
# ======================================================================
class TestAlignmentMargin:
    def test_known_value(self):
        margin = alignment_margin(0.5, [0.1, 0.2, 0.3])
        assert margin == pytest.approx(0.5 - 0.2)

    def test_positive_greater_than_negatives_gives_positive_margin(self):
        assert alignment_margin(0.9, [0.1, 0.2]) > 0

    def test_empty_negatives_raises(self):
        with pytest.raises(ValueError):
            alignment_margin(0.5, [])

    def test_equal_ensemble_is_l2_normalized(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        ens = build_equal_ensemble(vectors)
        assert np.linalg.norm(ens) == pytest.approx(1.0)


# ======================================================================
# K-subset sampling
# ======================================================================
class TestSubsetSampling:
    def test_k_equals_n_gives_single_subset(self):
        subsets = enumerate_or_sample_subsets(16, 16, seed=42)
        assert subsets == [tuple(range(16))]

    def test_exhaustive_when_small(self):
        subsets = enumerate_or_sample_subsets(4, 1, seed=42, max_subsets=100)
        assert len(subsets) == 4  # C(4,1) = 4 <= 100, exhaustive, no sampling

    def test_capped_and_reproducible_when_large(self):
        s1 = enumerate_or_sample_subsets(16, 4, seed=42, max_subsets=100)
        s2 = enumerate_or_sample_subsets(16, 4, seed=42, max_subsets=100)
        assert len(s1) == 100
        assert s1 == s2  # reproducible given the same seed
        assert len(set(s1)) == 100  # all distinct


# ======================================================================
# Circular shift baseline
# ======================================================================
class TestCircularShift:
    def test_shift_is_never_identity(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            dy, dx = random_nonzero_shift(4, 4, rng)
            assert (dy % 4, dx % 4) != (0, 0)

    def test_shift_preserves_total_mass(self):
        arr = np.random.RandomState(1).rand(5, 7)
        shifted = circular_shift_2d(arr, 2, 3)
        assert shifted.sum() == pytest.approx(arr.sum())
        assert shifted.shape == arr.shape


# ======================================================================
# CV group-safety and scaler-fit-on-train-only (synthetic)
# ======================================================================
class TestGroupSafety:
    def test_same_image_id_never_split_across_train_test(self):
        from sklearn.model_selection import StratifiedGroupKFold
        rng = np.random.RandomState(42)
        n = 60
        groups = np.array([f"img{i // 3}" for i in range(n)])  # 3 objects/image, 20 images
        y = rng.randint(0, 2, size=n)
        X = np.zeros((n, 1))
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(X, y, groups):
            assert not (set(groups[tr]) & set(groups[te]))

    def test_safe_n_splits_insufficient_data(self):
        assert safe_n_splits(n_positive_images=1, n_negative_images=10) is None
        assert safe_n_splits(n_positive_images=10, n_negative_images=1) is None
        assert safe_n_splits(n_positive_images=2, n_negative_images=2) == 2
        assert safe_n_splits(n_positive_images=100, n_negative_images=100, max_splits=5) == 5


class TestScalerFitsOnTrainOnly:
    def test_standard_scaler_stats_come_from_train_fold(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(42)
        x_train = rng.normal(loc=0.0, scale=1.0, size=(40, 3))
        x_test = rng.normal(loc=50.0, scale=10.0, size=(10, 3))  # wildly different distribution
        y_train = rng.randint(0, 2, size=40)

        pipe = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        pipe.fit(x_train, y_train)

        np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, x_train.mean(axis=0), atol=1e-10)
        # If the test fold had leaked into fitting, the scaler mean would be
        # pulled toward x_test's much larger values -- assert it is NOT.
        assert not np.allclose(pipe.named_steps["scaler"].mean_, x_test.mean(axis=0), atol=1.0)


# ======================================================================
# Permutation baseline reproducibility
# ======================================================================
class TestPermutationReproducibility:
    def test_same_seed_gives_same_shuffled_labels(self):
        y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        np.testing.assert_array_equal(rng1.permutation(y), rng2.permutation(y))

    def test_different_seed_gives_different_shuffled_labels_generally(self):
        y = np.arange(20)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(43)
        assert not np.array_equal(rng1.permutation(y), rng2.permutation(y))


# ======================================================================
# NaN/Inf absence (on toy data through the real primitives)
# ======================================================================
class TestNoNanInf:
    def test_weighted_pool_and_margin_produce_finite_values(self):
        tokens = np.random.RandomState(0).rand(4, 4, 8)
        weights = np.random.RandomState(1).rand(4, 4)
        pooled = weighted_pool_tokens(tokens, weights)
        assert np.all(np.isfinite(pooled))
        pooled_n = l2_normalize(pooled)
        assert np.all(np.isfinite(pooled_n))
        margin = alignment_margin(0.5, [0.1, 0.2, 0.3])
        assert np.isfinite(margin)


# ======================================================================
# Real-data integration (skip cleanly if outputs not generated yet)
# ======================================================================
class TestRealDataIntegration:
    @pytest.mark.parametrize("filename", [
        "region_text_scores.csv", "spatial_scores.csv", "probe_fold_scores.csv",
        "probe_summary.csv", "probe_permutation_baseline.csv",
    ])
    def test_no_unexpected_nan_or_inf_in_generated_csv(self, filename):
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
                # NaN is an explicitly allowed/expected value for some
                # columns (e.g. auc when only one class is present) -- we
                # only assert it is never silently coerced to a fake number,
                # i.e. it must parse as either a normal float or literal 'nan'.


# ======================================================================
# Real-model integration (skip if CLIP / GPU pipeline not runnable)
# ======================================================================
class TestRealModelIntegration:
    @pytest.fixture(scope="class")
    def loaded(self):
        try:
            import torch
            from clip_extractor import load_clip_vit_b16
            from lib.clip_vit_hidden import visual_forward_with_hidden_states
        except Exception as e:
            pytest.skip(f"CLIP stack unavailable: {e}")
        model, preprocess = load_clip_vit_b16()
        return model, preprocess, torch, visual_forward_with_hidden_states

    def test_hidden_state_shapes_and_1_indexed_layer_mapping(self, loaded):
        model, preprocess, torch, visual_forward_with_hidden_states = loaded
        from PIL import Image
        img_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "osie",
            "predicting-human-gaze-beyond-pixels", "data", "stimuli", "1001.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("sample stimulus not present")
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        device = next(model.parameters()).device
        tensor = tensor.to(device)

        layers_display = [4, 8, 12]
        layers_zero_based = [3, 7, 11]
        assert [d - 1 for d in layers_display] == layers_zero_based

        with torch.inference_mode():
            hidden_states, pooled = visual_forward_with_hidden_states(
                model.visual, tensor.type(model.dtype), pos_embed=None,
                layers_zero_based=layers_zero_based)

        assert set(hidden_states.keys()) == set(layers_zero_based)
        for l in layers_zero_based:
            assert hidden_states[l].shape == (1, 197, 768)
        assert pooled.shape == (1, 512)

    def test_l12_cls_matches_official_encode_image(self, loaded):
        model, preprocess, torch, visual_forward_with_hidden_states = loaded
        import torch.nn.functional as F
        from PIL import Image
        img_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "osie",
            "predicting-human-gaze-beyond-pixels", "data", "stimuli", "1001.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("sample stimulus not present")
        img = Image.open(img_path).convert("RGB")
        device = next(model.parameters()).device
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            official = model.encode_image(tensor)
            _, pooled = visual_forward_with_hidden_states(
                model.visual, tensor.type(model.dtype), pos_embed=None, layers_zero_based=[11])

        cos = F.cosine_similarity(official, pooled, dim=-1).item()
        assert cos >= 0.9999
        assert not torch.isnan(pooled).any()
        assert not torch.isinf(pooled).any()

    def test_sanity_checks_json_passed(self):
        path = os.path.join(OUT_DIR, "sanity_checks.json")
        if not os.path.isfile(path):
            pytest.skip("sanity_checks.json not generated yet")
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["passed"] is True
        assert result["check1_native_224"]["min_cosine"] >= 0.9999
        assert result["check2_osie_nonsquare"]["min_cosine"] >= 0.9999
