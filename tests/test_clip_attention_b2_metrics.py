"""
Unit tests for lib/clip_attention_b2.py -- [B-2] M-shape spatial-distribution
metrics and statistics. Pure numpy, no CLIP/torch needed for the metric
tests (TestCacheHealthAndLayerMapping additionally reads the existing
attn_cache/*.npz produced by scripts/run_osie_attribute_grounding_pilot.py
--all-images, and is skipped if that file is not present).

Run:  python -m pytest tests/test_clip_attention_b2_metrics.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.clip_attention_b2 import (
    DegenerateAttentionError,
    bh_fdr,
    bootstrap_ci_mean_diff,
    centroid,
    cohens_dz,
    coverage_for_mass,
    cosine_similarity,
    dataset_mean_cosine_for_permutation,
    dataset_mean_jsd_for_permutation,
    dataset_mean_pearson_for_permutation,
    derangement_seed,
    fisher_z,
    fisher_z_array,
    fisher_z_inverse,
    fixed_point_free_permutation,
    foreground_background_metrics,
    hoyer_sparsity,
    iter_derangements,
    jensen_shannon_divergence_normalized,
    max_patch_attention,
    monte_carlo_one_sided_p,
    normalize_distribution,
    normalized_center_distance,
    normalized_entropy,
    pearson_corr,
    refuse_if_exists,
    sign_flip_test,
    top_k_mass,
)

GH, GW = 38, 50
N_PATCHES = GH * GW  # 1900
CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "osie_attribute_grounding_full700",
    "attn_cache", "clip_vitb16_full700_L4L8L12_patchgrid.npz")


def uniform_map():
    return np.full(N_PATCHES, 1.0 / N_PATCHES, dtype=np.float64)


def one_patch_map(k=0):
    a = np.zeros(N_PATCHES, dtype=np.float64)
    a[k] = 1.0
    return a


def left_right_halves():
    """(38,50) grid split into an exact left/right half (25 cols each,
    38*25=950 patches each) -- disjoint, complementary indicator maps."""
    left = np.zeros((GH, GW), dtype=np.float64)
    left[:, :GW // 2] = 1.0
    right = np.zeros((GH, GW), dtype=np.float64)
    right[:, GW // 2:] = 1.0
    return left, right


# ==== normalization ====
class TestNormalize:
    def test_uniform_sums_to_one(self):
        p = normalize_distribution(np.full(N_PATCHES, 2.0))
        assert np.isclose(p.sum(), 1.0)
        assert np.allclose(p, 1.0 / N_PATCHES)

    def test_degenerate_raises(self):
        with pytest.raises(DegenerateAttentionError):
            normalize_distribution(np.zeros(N_PATCHES))


# ==== uniform distribution ====
class TestUniformDistribution:
    def test_entropy_is_one(self):
        p = uniform_map()
        assert np.isclose(normalized_entropy(p, N_PATCHES), 1.0, atol=1e-10)

    def test_hoyer_is_zero(self):
        assert np.isclose(hoyer_sparsity(uniform_map()), 0.0, atol=1e-10)

    def test_max_is_lower_bound_1_over_n(self):
        assert np.isclose(max_patch_attention(uniform_map()), 1.0 / N_PATCHES)

    def test_centroid_is_center(self):
        p = uniform_map()
        cx, cy = centroid(p, (GH, GW))
        assert np.isclose(cx, 0.5, atol=1e-10)
        assert np.isclose(cy, 0.5, atol=1e-10)

    def test_center_distance_is_zero(self):
        p = uniform_map()
        d = normalized_center_distance(centroid(p, (GH, GW)))
        assert np.isclose(d, 0.0, atol=1e-10)

    def test_pearson_undefined_for_uniform_map(self):
        p = uniform_map()
        with pytest.raises(ValueError):
            pearson_corr(p, p)


# ==== single patch (maximally concentrated) ====
class TestOnePatchMass:
    def test_entropy_is_zero(self):
        assert np.isclose(normalized_entropy(one_patch_map(), N_PATCHES), 0.0)

    def test_hoyer_is_one(self):
        assert np.isclose(hoyer_sparsity(one_patch_map()), 1.0, atol=1e-10)

    def test_max_is_one(self):
        assert np.isclose(max_patch_attention(one_patch_map()), 1.0)

    def test_top10pct_mass_is_one(self):
        assert np.isclose(top_k_mass(one_patch_map(), 190), 1.0)

    def test_coverage_for_mass_50_is_single_patch(self):
        count, area_frac = coverage_for_mass(one_patch_map(), 0.5)
        assert count == 1
        assert np.isclose(area_frac, 1.0 / N_PATCHES)


# ==== known synthetic centroid ====
class TestKnownCentroid:
    def test_top_left_quadrant(self):
        grid = np.zeros((GH, GW), dtype=np.float64)
        grid[:GH // 2, :GW // 2] = 1.0
        p = grid / grid.sum()
        cx, cy = centroid(p, (GH, GW))
        # mass uniform over rows [0, 19), cols [0, 25) -> midpoint of that sub-square
        assert np.isclose(cx, (GW // 2) / 2 / GW, atol=1e-9)
        assert np.isclose(cy, (GH // 2) / 2 / GH, atol=1e-9)
        assert cx < 0.5 and cy < 0.5


# ==== similarity: identical vs. disjoint maps ====
class TestSimilarityIdenticalVsDisjoint:
    def test_identical_nonuniform_map_pearson_cosine_jsd(self):
        rng = np.random.RandomState(0)
        a = rng.rand(N_PATCHES) + 0.01  # non-uniform, strictly positive
        assert np.isclose(pearson_corr(a, a), 1.0, atol=1e-10)
        assert np.isclose(cosine_similarity(a, a), 1.0, atol=1e-10)
        p = normalize_distribution(a)
        assert np.isclose(jensen_shannon_divergence_normalized(p, p), 0.0, atol=1e-10)

    def test_disjoint_left_right_halves(self):
        left, right = left_right_halves()
        assert np.isclose(pearson_corr(left, right), -1.0, atol=1e-9)
        assert np.isclose(cosine_similarity(left, right), 0.0, atol=1e-10)
        p, q = normalize_distribution(left), normalize_distribution(right)
        assert np.isclose(jensen_shannon_divergence_normalized(p, q), 1.0, atol=1e-9)

    def test_different_maps_pearson_not_near_one(self):
        rng = np.random.RandomState(1)
        a = rng.rand(N_PATCHES) + 0.01
        b = rng.rand(N_PATCHES) + 0.01
        r = pearson_corr(a, b)
        assert r < 0.5  # two independent random maps should not correlate strongly


# ==== JSD with zeros ====
class TestJSDWithZeros:
    def test_partial_overlap_no_nan_inf(self):
        rng = np.random.RandomState(2)
        a = np.zeros(N_PATCHES)
        a[:100] = rng.rand(100) + 0.01
        b = np.zeros(N_PATCHES)
        b[50:150] = rng.rand(100) + 0.01
        p, q = normalize_distribution(a), normalize_distribution(b)
        jsd = jensen_shannon_divergence_normalized(p, q)
        assert np.isfinite(jsd)
        assert 0.0 <= jsd <= 1.0 + 1e-9

    def test_jsd_bounded_0_1(self):
        left, right = left_right_halves()
        p, q = normalize_distribution(left), normalize_distribution(right)
        jsd = jensen_shannon_divergence_normalized(p, q)
        assert 0.0 <= jsd <= 1.0 + 1e-9


# ==== Fisher z round-trip ====
class TestFisherZ:
    def test_roundtrip(self):
        for r in (-0.9, -0.3, 0.0, 0.4, 0.85):
            z = fisher_z(r)
            assert np.isclose(fisher_z_inverse(z), r, atol=1e-8)


# ==== foreground / background ====
class TestForegroundBackground:
    def test_half_foreground_uniform_attention_gives_half_conditional_mass(self):
        raw = np.full((GH, GW), 2.0)
        coverage = np.zeros((GH, GW))
        coverage[:, :GW // 2] = 1.0  # exact left half, well above the degenerate threshold
        m = foreground_background_metrics(raw, coverage)
        assert np.isclose(m["foreground_conditional_mass"], 0.5, atol=1e-9)
        assert np.isclose(m["background_conditional_mass"], 0.5, atol=1e-9)
        assert np.isclose(m["foreground_enrichment"], 1.0, atol=1e-9)
        assert not m["is_degenerate"]

    def test_all_foreground_coverage_is_flagged_degenerate_on_background_side(self):
        raw = np.full((GH, GW), 2.0)
        coverage = np.ones((GH, GW))  # background_area_fraction == 0
        m = foreground_background_metrics(raw, coverage)
        assert np.isclose(m["foreground_conditional_mass"], 1.0)
        assert np.isclose(m["background_conditional_mass"], 0.0)
        assert m["is_degenerate"]  # background side has ~0 area -- must be excluded, not silently kept

    def test_near_zero_foreground_area_is_degenerate(self):
        raw = np.full((GH, GW), 1.0)
        coverage = np.zeros((GH, GW))
        coverage[0, 0] = 0.0001  # far below 1 patch of area
        m = foreground_background_metrics(raw, coverage)
        assert m["is_degenerate"]
        assert m["foreground_enrichment"] is None

    def test_complement_relationship(self):
        rng = np.random.RandomState(3)
        raw = rng.rand(GH, GW) + 0.01
        coverage = (rng.rand(GH, GW) > 0.5).astype(np.float64)
        m = foreground_background_metrics(raw, coverage)
        assert np.isclose(
            m["foreground_conditional_mass"] + m["background_conditional_mass"], 1.0)


# ==== stats: bootstrap, sign-flip, Cohen's dz, BH-FDR ====
class TestStats:
    def test_bootstrap_reproducible_with_same_seed(self):
        rng = np.random.RandomState(42)
        diff = rng.randn(50) + 0.3
        ci1 = bootstrap_ci_mean_diff(diff, seed=42, n_boot=2000)
        ci2 = bootstrap_ci_mean_diff(diff, seed=42, n_boot=2000)
        assert ci1 == ci2

    def test_sign_flip_reproducible_with_same_seed(self):
        rng = np.random.RandomState(42)
        diff = rng.randn(50) + 0.3
        r1 = sign_flip_test(diff, seed=42, n_reps=2000)
        r2 = sign_flip_test(diff, seed=42, n_reps=2000)
        assert r1 == r2

    def test_sign_flip_pvalue_never_zero(self):
        diff = np.full(20, 5.0)  # extreme, always-positive diff
        _, p = sign_flip_test(diff, seed=42, n_reps=500)
        assert p > 0.0
        assert np.isclose(p, 1.0 / 501, atol=1e-9)

    def test_cohens_dz_nan_for_zero_variance(self):
        assert np.isnan(cohens_dz(np.full(10, 1.0)))

    def test_bh_fdr_monotonic_and_bounded(self):
        pvals = np.array([0.001, 0.02, 0.03, 0.5, np.nan, 0.04])
        q = bh_fdr(pvals)
        assert np.isnan(q[4])
        valid = ~np.isnan(q)
        assert np.all(q[valid] >= pvals[valid] - 1e-12)
        assert np.all(q[valid] <= 1.0)


# ==== non-overwrite guard ====
class TestRefuseIfExists:
    def test_raises_when_file_exists(self, tmp_path):
        f = tmp_path / "out.csv"
        f.write_text("x")
        with pytest.raises(RuntimeError):
            refuse_if_exists([str(f)])

    def test_silent_when_absent(self, tmp_path):
        f = tmp_path / "does_not_exist.csv"
        refuse_if_exists([str(f)])  # must not raise


# ==== repeated-shuffle robustness: derangement generation ====
class TestDerangements:
    def test_no_fixed_points(self):
        for seed in (42, 1, 12345):
            perm = fixed_point_free_permutation(700, seed)
            assert not np.any(perm == np.arange(700))

    def test_all_700_images_appear_exactly_once(self):
        perm = fixed_point_free_permutation(700, 42)
        assert sorted(perm.tolist()) == list(range(700))

    def test_same_seed_reproduces_identical_sequence(self):
        seq1 = [p.copy() for p in iter_derangements(700, base_seed=42, n_deran=5)]
        seq2 = [p.copy() for p in iter_derangements(700, base_seed=42, n_deran=5)]
        for a, b in zip(seq1, seq2):
            assert np.array_equal(a, b)

    def test_different_base_seed_gives_different_sequence(self):
        seq1 = [p.copy() for p in iter_derangements(700, base_seed=42, n_deran=5)]
        seq2 = [p.copy() for p in iter_derangements(700, base_seed=43, n_deran=5)]
        assert not all(np.array_equal(a, b) for a, b in zip(seq1, seq2))

    def test_derangement_seed_deterministic_and_distinct(self):
        assert derangement_seed(42, 0) == derangement_seed(42, 0)
        assert derangement_seed(42, 0) != derangement_seed(42, 1)
        assert derangement_seed(42, 0) != derangement_seed(43, 0)


# ==== repeated-shuffle robustness: dataset-mean similarity + one-sided MC p ====
class TestRepeatedShuffleRobustness:
    def _synthetic_layers(self, n_images=60, n_patches=200, seed=7):
        """Layer A and layer B share a per-image 'signal' component, so
        same-image (identity permutation) similarity should be much
        higher than a shuffled (derangement) permutation's."""
        rng = np.random.RandomState(seed)
        signal = rng.rand(n_images, n_patches) + 0.05
        noise_a = rng.rand(n_images, n_patches) * 0.05
        noise_b = rng.rand(n_images, n_patches) * 0.05
        a = signal + noise_a
        b = signal + noise_b
        return a, b

    def test_same_image_similarity_exceeds_shuffled(self):
        a, b = self._synthetic_layers()
        n = a.shape[0]
        identity = np.arange(n)
        deran = fixed_point_free_permutation(n, seed=123)

        centered_a = a - a.mean(axis=1, keepdims=True)
        centered_b = b - b.mean(axis=1, keepdims=True)
        norms_a = np.linalg.norm(centered_a, axis=1)
        norms_b = np.linalg.norm(centered_b, axis=1)

        same_pearson, _ = dataset_mean_pearson_for_permutation(
            centered_a, norms_a, centered_b, norms_b, identity)
        shuf_pearson, _ = dataset_mean_pearson_for_permutation(
            centered_a, norms_a, centered_b, norms_b, deran)
        assert same_pearson > shuf_pearson

        raw_norms_a = np.linalg.norm(a, axis=1)
        raw_norms_b = np.linalg.norm(b, axis=1)
        same_cos, _ = dataset_mean_cosine_for_permutation(a, raw_norms_a, b, raw_norms_b, identity)
        shuf_cos, _ = dataset_mean_cosine_for_permutation(a, raw_norms_a, b, raw_norms_b, deran)
        assert same_cos > shuf_cos

        pa = a / a.sum(axis=1, keepdims=True)
        pb = b / b.sum(axis=1, keepdims=True)
        log_pa, log_pb = np.log(pa), np.log(pb)
        same_jsd, _ = dataset_mean_jsd_for_permutation(pa, log_pa, pb, log_pb, identity)
        shuf_jsd, _ = dataset_mean_jsd_for_permutation(pa, log_pa, pb, log_pb, deran)
        assert same_jsd < shuf_jsd  # JSD: LOWER means MORE similar

    def test_pearson_cosine_direction_greater_jsd_direction_less(self):
        # a null distribution centered at 0, observed clearly above it
        null = np.random.RandomState(0).randn(1000) * 0.01
        p_greater_low = monte_carlo_one_sided_p(0.5, null, direction="greater")
        p_greater_high_in_null = monte_carlo_one_sided_p(0.0, null, direction="greater")
        assert p_greater_low < p_greater_high_in_null  # far above null -> smaller p

        p_less_low = monte_carlo_one_sided_p(-0.5, null, direction="less")
        p_less_high_in_null = monte_carlo_one_sided_p(0.0, null, direction="less")
        assert p_less_low < p_less_high_in_null  # far below null -> smaller p

    def test_monte_carlo_p_plus_one_correction_never_zero(self):
        null = np.full(999, 100.0)  # observed will be "more extreme" than all 999
        p = monte_carlo_one_sided_p(1000.0, null, direction="greater")
        assert p > 0.0
        assert np.isclose(p, 1.0 / 1000, atol=1e-9)  # (1+0)/(1+999)

    def test_fisher_z_array_matches_scalar(self):
        vals = np.array([-0.5, 0.0, 0.3, 0.8])
        arr_result = fisher_z_array(vals)
        scalar_result = np.array([fisher_z(v) for v in vals])
        assert np.allclose(arr_result, scalar_result)


# ==== repeated-shuffle output non-overwrite (uses the shared guard) ====
class TestRepeatedShuffleOutputProtection:
    def test_new_output_files_refuse_overwrite(self, tmp_path):
        f = tmp_path / "layer_pair_similarity_repeated_shuffle.csv"
        f.write_text("x")
        with pytest.raises(RuntimeError):
            refuse_if_exists([str(f)])

    def test_does_not_touch_unrelated_existing_files(self, tmp_path):
        untouched = tmp_path / "report.md"
        untouched.write_text("original content")
        new_file = tmp_path / "report_revised.md"
        # refuse_if_exists only complains about paths actually passed to it
        refuse_if_exists([str(new_file)])
        assert untouched.read_text() == "original content"


# ==== cache health + layer-index / image-ID mapping (skipped if cache absent) ====
class TestCacheHealthAndLayerMapping:
    @pytest.fixture(scope="class")
    def cache(self):
        if not os.path.isfile(CACHE_PATH):
            pytest.skip(f"cache not found: {CACHE_PATH}")
        return np.load(CACHE_PATH, allow_pickle=False)

    def test_layer_display_to_zero_based_mapping(self, cache):
        display = cache["layers_display"].tolist()
        zero_based = cache["layers_zero_based"].tolist()
        assert display == [4, 8, 12]
        assert zero_based == [3, 7, 11]
        assert [d - 1 for d in display] == zero_based

    def test_shape_700_3_38_50(self, cache):
        assert cache["attn"].shape == (700, 3, GH, GW)
        assert cache["attn"].dtype == np.float32

    def test_700_unique_stems_no_duplicates(self, cache):
        stems = cache["stems"].tolist()
        assert len(stems) == 700
        assert len(set(stems)) == 700

    def test_stems_align_with_attn_first_axis(self, cache):
        assert cache["stems"].shape[0] == cache["attn"].shape[0]

    def test_nonnegative_finite(self, cache):
        attn = cache["attn"]
        assert np.isfinite(attn).all()
        assert (attn >= 0).all()

    def test_row_sums_below_one(self, cache):
        # raw CLS->patch mass excludes CLS-self-attention, so < 1 always
        sums = cache["attn"].sum(axis=(2, 3))
        assert (sums > 0).all()
        assert (sums < 1.0 + 1e-6).all()


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
