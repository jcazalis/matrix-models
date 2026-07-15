"""Tests for Monte Carlo estimators."""

from __future__ import annotations

import numpy as np
import pytest

from sym_contractions.bruteforce import exact_probability
from sym_contractions.estimator import (
    _numba_build_cyclic_perm,
    _numba_count_cycles_mc,
    _numba_mc_pairs_parallel,
    _numba_random_shuffle_into,
    numba_mc_all_conjugacy_pairs,
)
from sym_contractions.utils import enumerate_partitions

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNumbaMcAllConjugacyPairs:
    def test_small_batch(self):
        """Run numba_mc_all_conjugacy_pairs for n=m=3 and verify shape."""
        n, m = 3, 3
        parts_n = enumerate_partitions(n)
        parts_m = enumerate_partitions(m)
        tau = np.arange(n + m, dtype=np.int32)

        estimates, std_errors = numba_mc_all_conjugacy_pairs(
            tau,
            n,
            m,
            n_samples_per_pair=1000,
            partitions_n=parts_n,
            partitions_m=parts_m,
            progress=False,
            seed=0,
        )

        p_n = len(parts_n)  # p(3) = 3
        p_m = len(parts_m)
        max_k = n + m
        assert estimates.shape == (p_n, p_m, max_k + 1)
        assert std_errors.shape == (p_n, p_m, max_k + 1)
        # All estimates should be valid probabilities
        assert np.all(estimates >= 0.0)
        assert np.all(estimates <= 1.0)
        assert np.all(std_errors >= 0.0)
        # Probabilities over k should sum to ~1 for each pair
        assert np.allclose(np.sum(estimates, axis=-1), 1.0, atol=0.05)

    def test_batch_vs_exact(self):
        """Batch MC estimates should match exact values (for small n, m)."""
        n, m = 6, 5
        n_samples = 10000

        parts_n = enumerate_partitions(n)
        parts_m = enumerate_partitions(m)
        tau = np.random.default_rng(42).permutation(n + m).astype(np.int32)

        # Get batch estimates
        estimates_batch, se_batch = numba_mc_all_conjugacy_pairs(
            tau,
            n,
            m,
            n_samples_per_pair=n_samples,
            partitions_n=parts_n,
            partitions_m=parts_m,
            progress=False,
            seed=123,
        )

        # Compare against exact values since n, m are small.
        for i, ct_n_tuple in enumerate(parts_n):
            for j, ct_m_tuple in enumerate(parts_m):
                exact = exact_probability(tau.tolist(), n, m, ct_n_tuple, ct_m_tuple)

                # Check batch estimate is close to exact across all k
                for k in range(n + m + 1):
                    est_batch = float(estimates_batch[i, j, k])
                    se = float(se_batch[i, j, k])
                    exact_k = float(exact[k])

                    # Within 4 standard errors of exact value
                    assert abs(est_batch - exact_k) < 4 * se + 0.02, (
                        f"Pair ({ct_n_tuple}, {ct_m_tuple}), k={k}: "
                        f"batch_est={est_batch:.4f}, exact={exact_k:.4f}, se={se:.4f}"
                    )

    @pytest.mark.parametrize("seed", [0, 123])
    def test_reproducible_for_fixed_seed(self, seed):
        """Running twice with same seed should produce identical results."""
        n, m = 4, 4
        n_samples = 2000
        tau = np.random.default_rng(7).permutation(n + m).astype(np.int32)

        out1, se1 = numba_mc_all_conjugacy_pairs(
            tau,
            n,
            m,
            n_samples_per_pair=n_samples,
            progress=False,
            seed=seed,
        )
        out2, se2 = numba_mc_all_conjugacy_pairs(
            tau,
            n,
            m,
            n_samples_per_pair=n_samples,
            progress=False,
            seed=seed,
        )

        assert np.array_equal(out1, out2)
        assert np.array_equal(se1, se2)

    def test_progress_true_branch(self):
        """progress=True should use batched path and still return valid outputs."""
        n, m = 2, 2
        tau = np.arange(n + m, dtype=np.int32)

        estimates, std_errors = numba_mc_all_conjugacy_pairs(
            tau,
            n,
            m,
            n_samples_per_pair=400,
            progress=True,
            seed=11,
        )

        assert estimates.shape == (2, 2, n + m + 1)
        assert std_errors.shape == (2, 2, n + m + 1)
        assert np.allclose(estimates.sum(axis=-1), 1.0, atol=0.08)


class TestEstimatorNumbaHelpers:
    def test_random_shuffle_into_is_permutation(self):
        n = 8
        buf = np.empty(n, dtype=np.int32)
        _numba_random_shuffle_into.py_func(buf)  # type: ignore
        assert sorted(buf.tolist()) == list(range(n))

    def test_build_cyclic_perm_identity(self):
        shuffled = np.array([2, 0, 1], dtype=np.int32)
        cycle_type = np.array([1, 1, 1], dtype=np.int32)
        perm = np.empty(3, dtype=np.int32)
        _numba_build_cyclic_perm.py_func(shuffled, cycle_type, perm)  # type: ignore
        assert np.array_equal(perm, np.array([0, 1, 2], dtype=np.int32))

    def test_build_cyclic_perm_single_cycle(self):
        shuffled = np.array([0, 1, 2, 3], dtype=np.int32)
        cycle_type = np.array([4, 0, 0, 0], dtype=np.int32)
        perm = np.empty(4, dtype=np.int32)
        _numba_build_cyclic_perm.py_func(shuffled, cycle_type, perm)  # type: ignore
        assert np.array_equal(perm, np.array([1, 2, 3, 0], dtype=np.int32))

    def test_count_cycles_mc(self):
        perm = np.array([1, 0, 2, 3], dtype=np.int32)
        visited = np.zeros(4, dtype=np.bool_)
        assert _numba_count_cycles_mc.py_func(perm, visited) == 3  # type: ignore

    def test_mc_pairs_parallel_reproducible(self):
        n, m = 2, 2
        nm = n + m
        tau = np.arange(nm, dtype=np.int32)

        ct_n_array = np.array([[2, 0], [1, 1]], dtype=np.int32)
        ct_m_array = np.array([[2, 0], [1, 1]], dtype=np.int32)

        pair_is = np.array([0, 1], dtype=np.int32)
        pair_js = np.array([1, 0], dtype=np.int32)
        seeds = np.array([12345, 67890], dtype=np.int64)

        out1 = _numba_mc_pairs_parallel(
            tau,
            ct_n_array,
            ct_m_array,
            pair_is,
            pair_js,
            300,
            seeds,
        )
        out2 = _numba_mc_pairs_parallel(
            tau,
            ct_n_array,
            ct_m_array,
            pair_is,
            pair_js,
            300,
            seeds,
        )

        assert out1.shape == (2, nm + 1)
        assert np.allclose(out1.sum(axis=-1), 1.0, atol=1e-12)
        np.testing.assert_allclose(out1, out2, atol=0.0)

    def test_mc_pairs_parallel_py_func_path(self):
        """Exercise Python body of numba kernel for coverage accounting."""
        n, m = 2, 2
        nm = n + m
        tau = np.arange(nm, dtype=np.int32)

        ct_n_array = np.array([[2, 0], [1, 1]], dtype=np.int32)
        ct_m_array = np.array([[2, 0], [1, 1]], dtype=np.int32)

        pair_is = np.array([0], dtype=np.int32)
        pair_js = np.array([1], dtype=np.int32)
        seeds = np.array([54321], dtype=np.int64)

        out = _numba_mc_pairs_parallel.py_func(  # type: ignore
            tau,
            ct_n_array,
            ct_m_array,
            pair_is,
            pair_js,
            80,
            seeds,
        )
        assert out.shape == (1, nm + 1)
        assert np.isclose(out[0].sum(), 1.0)
