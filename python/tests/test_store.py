"""Tests for the probability storage infrastructure."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from sym_contractions.store import (
    ProbabilityStore,
    ProbabilityStoreCollection,
    TauEntry,
    compute_and_store,
)
from sym_contractions.woven import WovenData, WovenEntry, WovenGroup, tau_to_involution

# ---------------------------------------------------------------------------
# ProbabilityStore basics
# ---------------------------------------------------------------------------


class TestProbabilityStoreCreate:
    def test_empty_store(self):
        store = ProbabilityStore.create(3, 2)
        assert store.n == 3
        assert store.m == 2
        assert store.p_n == 3  # p(3) = 3
        assert store.p_m == 2  # p(2) = 2
        assert len(store.entries) == 0

    def test_partition_ordering(self):
        """Partitions should match enumerate_partitions."""
        from sym_contractions.utils import enumerate_partitions

        store = ProbabilityStore.create(4, 3)
        assert store.partitions_n == enumerate_partitions(4)
        assert store.partitions_m == enumerate_partitions(3)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_empty_round_trip(self):
        store = ProbabilityStore.create(3, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            store.save(path)
            loaded = ProbabilityStore.load(path)
        assert loaded.n == 3
        assert loaded.m == 2
        assert loaded.partitions_n == store.partitions_n
        assert loaded.partitions_m == store.partitions_m
        assert len(loaded.entries) == 0

    def test_with_entry_round_trip(self):
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        tau = (1, 0, 3, 4, 2)
        p_n, p_m = store.p_n, store.p_m
        nm = n + m

        entry = TauEntry(
            tau=tau,
            probabilities=np.random.rand(p_n, p_m, nm + 1),
            std_errors=np.random.rand(p_n, p_m, nm + 1),
            is_exact=np.ones((p_n, p_m), dtype=bool),
            n_samples=np.zeros((p_n, p_m), dtype=np.int32),
        )
        store.entries[tau] = entry

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            store.save(path)
            loaded = ProbabilityStore.load(path)

        assert tau in loaded.entries
        np.testing.assert_array_equal(
            loaded.entries[tau].probabilities, entry.probabilities
        )
        np.testing.assert_array_equal(loaded.entries[tau].std_errors, entry.std_errors)
        np.testing.assert_array_equal(loaded.entries[tau].is_exact, entry.is_exact)
        np.testing.assert_array_equal(loaded.entries[tau].n_samples, entry.n_samples)

    def test_multiple_taus_round_trip(self):
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        nm = n + m

        for tau in [(0, 1, 2, 3, 4), (1, 0, 2, 3, 4), (4, 3, 2, 1, 0)]:
            store.entries[tau] = TauEntry(
                tau=tau,
                probabilities=np.random.rand(store.p_n, store.p_m, nm + 1),
                std_errors=np.zeros((store.p_n, store.p_m, nm + 1)),
                is_exact=np.ones((store.p_n, store.p_m), dtype=bool),
                n_samples=np.zeros((store.p_n, store.p_m), dtype=np.int32),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            store.save(path)
            loaded = ProbabilityStore.load(path)

        assert set(loaded.entries.keys()) == set(store.entries.keys())
        for tau in store.entries:
            np.testing.assert_array_equal(
                loaded.entries[tau].probabilities,
                store.entries[tau].probabilities,
            )


# ---------------------------------------------------------------------------
# NumPy accessors
# ---------------------------------------------------------------------------


class TestNumpyAccessors:
    def test_get_numpy_probabilities(self):
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        tau = (0, 1, 2, 3, 4)
        probs = np.random.rand(store.p_n, store.p_m, n + m + 1)
        store.entries[tau] = TauEntry(
            tau=tau,
            probabilities=probs,
            std_errors=np.zeros_like(probs),
            is_exact=np.ones((store.p_n, store.p_m), dtype=bool),
            n_samples=np.zeros((store.p_n, store.p_m), dtype=np.int32),
        )
        numpy_probs = store.get_numpy_probabilities(tau)
        assert numpy_probs.dtype == np.float32
        assert numpy_probs.shape == (store.p_n, store.p_m, n + m + 1)
        np.testing.assert_allclose(numpy_probs, probs, atol=1e-6)

    def test_get_numpy_std_errors(self):
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        tau = (0, 1, 2, 3, 4)
        se = np.random.rand(store.p_n, store.p_m, n + m + 1) * 0.01
        store.entries[tau] = TauEntry(
            tau=tau,
            probabilities=np.random.rand(store.p_n, store.p_m, n + m + 1),
            std_errors=se,
            is_exact=np.zeros((store.p_n, store.p_m), dtype=bool),
            n_samples=np.full((store.p_n, store.p_m), 1000, dtype=np.int32),
        )
        numpy_se = store.get_numpy_std_errors(tau)
        assert numpy_se.dtype == np.float32
        np.testing.assert_allclose(numpy_se, se, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_and_store
# ---------------------------------------------------------------------------


class TestComputeAndStore:
    def test_compute_single_tau(self):
        """Compute probabilities for one τ from scratch."""
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        tau = (1, 0, 3, 4, 2)

        compute_and_store(store, [tau], progress=False)

        assert tau in store.entries
        entry = store.entries[tau]
        assert entry.probabilities.shape == (store.p_n, store.p_m, n + m + 1)
        assert entry.std_errors.shape == (store.p_n, store.p_m, n + m + 1)
        assert entry.is_exact.shape == (store.p_n, store.p_m)
        # n + m = 5 < 12 → all exact
        assert entry.is_exact.all()
        assert (entry.n_samples == 0).all()
        assert np.allclose(entry.std_errors, 0.0)
        # Probabilities should sum to 1 for each pair.
        assert np.allclose(entry.probabilities.sum(axis=-1), 1.0)

    def test_compute_multiple_taus(self):
        """Compute probabilities for several τ's."""
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        taus = [
            (0, 1, 2, 3, 4),  # identity
            (1, 0, 3, 4, 2),  # non-trivial
            (4, 3, 2, 1, 0),  # reversal
        ]
        compute_and_store(store, taus, progress=False)
        assert len(store.entries) == 3
        for tau in taus:
            assert tau in store.entries

    def test_skip_exact(self):
        """Already-exact entries are not recomputed."""
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        tau = (1, 0, 3, 4, 2)

        compute_and_store(store, [tau], progress=False)
        original_probs = store.entries[tau].probabilities.copy()

        # Run again — should skip entirely.
        compute_and_store(store, [tau], progress=False)
        np.testing.assert_array_equal(store.entries[tau].probabilities, original_probs)

    def test_skip_sufficient_mc(self):
        """MC entries with enough samples are not recomputed."""
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        tau = (1, 0, 3, 4, 2)

        # Manually insert an MC entry with 50000 samples.
        p_n, p_m = store.p_n, store.p_m
        nm = n + m
        store.entries[tau] = TauEntry(
            tau=tau,
            probabilities=np.ones((p_n, p_m, nm + 1)) / (nm + 1),
            std_errors=np.full((p_n, p_m, nm + 1), 0.01),
            is_exact=np.zeros((p_n, p_m), dtype=bool),
            n_samples=np.full((p_n, p_m), 50000, dtype=np.int32),
        )

        # Request only 10000 samples — should skip.
        compute_and_store(store, [tau], n_samples_per_pair=10000, progress=False)
        assert (store.entries[tau].n_samples == 50000).all()

    def test_recompute_insufficient_mc(self):
        """MC entries with too few samples are recomputed."""
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        tau = (1, 0, 3, 4, 2)

        # Insert MC entry with only 100 samples.
        p_n, p_m = store.p_n, store.p_m
        nm = n + m
        store.entries[tau] = TauEntry(
            tau=tau,
            probabilities=np.ones((p_n, p_m, nm + 1)) / (nm + 1),
            std_errors=np.full((p_n, p_m, nm + 1), 0.1),
            is_exact=np.zeros((p_n, p_m), dtype=bool),
            n_samples=np.full((p_n, p_m), 100, dtype=np.int32),
        )

        # Request 10000 samples — should recompute (and since n+m<12, exact).
        compute_and_store(store, [tau], n_samples_per_pair=10000, progress=False)
        # After recomputation with small n+m, everything should be exact.
        assert store.entries[tau].is_exact.all()

    def test_correctness_against_exact(self):
        """Stored probabilities match direct exact computation."""
        from sym_contractions.bruteforce import numba_exact_all_conjugacy_pairs

        n, m = 3, 3
        store = ProbabilityStore.create(n, m)
        tau = (2, 0, 1, 5, 3, 4)

        compute_and_store(store, [tau], progress=False)

        ref = numba_exact_all_conjugacy_pairs(
            list(tau),
            n,
            m,
            store.partitions_n,
            store.partitions_m,
            progress=False,
        )
        np.testing.assert_allclose(store.entries[tau].probabilities, ref, atol=1e-10)

    def test_save_load_after_compute(self):
        """Computed store survives a save/load cycle."""
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        taus = [(0, 1, 2, 3, 4), (1, 0, 3, 4, 2)]
        compute_and_store(store, taus, progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            store.save(path)
            loaded = ProbabilityStore.load(path)

        for tau in taus:
            np.testing.assert_array_equal(
                loaded.entries[tau].probabilities,
                store.entries[tau].probabilities,
            )
            np.testing.assert_array_equal(
                loaded.entries[tau].std_errors,
                store.entries[tau].std_errors,
            )
            np.testing.assert_array_equal(
                loaded.entries[tau].is_exact,
                store.entries[tau].is_exact,
            )

    def test_reproducible_with_same_seed(self):
        """Same base seed gives identical MC estimates across runs."""
        n, m = 6, 6
        tau = (2, 0, 1, 5, 3, 4, 11, 6, 7, 8, 9, 10)

        store_a = ProbabilityStore.create(n, m)
        store_b = ProbabilityStore.create(n, m)

        compute_and_store(
            store_a,
            [tau],
            n_samples_per_pair=3000,
            max_sequential_search_space=0,
            max_parallel_search_space=0,
            progress=False,
            seed=2026,
        )
        compute_and_store(
            store_b,
            [tau],
            n_samples_per_pair=3000,
            max_sequential_search_space=0,
            max_parallel_search_space=0,
            progress=False,
            seed=2026,
        )

        np.testing.assert_allclose(
            store_a.entries[tau].probabilities,
            store_b.entries[tau].probabilities,
            atol=0.0,
        )
        np.testing.assert_allclose(
            store_a.entries[tau].std_errors,
            store_b.entries[tau].std_errors,
            atol=0.0,
        )

    def test_tau_order_independent_seed_derivation(self):
        """Per-τ seeds do not depend on iteration order."""
        n, m = 6, 6
        taus = [
            (2, 0, 1, 5, 3, 4, 11, 6, 7, 8, 9, 10),
            (1, 2, 0, 4, 5, 3, 8, 9, 10, 11, 6, 7),
        ]

        store_a = ProbabilityStore.create(n, m)
        store_b = ProbabilityStore.create(n, m)

        compute_and_store(
            store_a,
            taus,
            n_samples_per_pair=3000,
            max_sequential_search_space=0,
            max_parallel_search_space=0,
            progress=False,
            seed=77,
        )
        compute_and_store(
            store_b,
            list(reversed(taus)),
            n_samples_per_pair=3000,
            max_sequential_search_space=0,
            max_parallel_search_space=0,
            progress=False,
            seed=77,
        )

        for tau in taus:
            np.testing.assert_allclose(
                store_a.entries[tau].probabilities,
                store_b.entries[tau].probabilities,
                atol=0.0,
            )


class TestSummary:
    def test_summary_output(self):
        n, m = 3, 2
        store = ProbabilityStore.create(n, m)
        tau = (0, 1, 2, 3, 4)
        compute_and_store(store, [tau], progress=False)
        s = store.summary()
        assert "n=3" in s
        assert "m=2" in s
        assert "exact=" in s
        assert "τ entries: 1" in s


# ---------------------------------------------------------------------------
# ProbabilityStoreCollection
# ---------------------------------------------------------------------------


def _make_woven_entry(tau: tuple[int, ...]) -> WovenEntry:
    return WovenEntry(
        involution=tuple(tau_to_involution(tau)),
        tau=tau,
        pairs_1indexed=[],
        coefficient_poly=[(1, 1)],
    )


class TestProbabilityStoreCollection:
    def test_create_empty(self):
        collection = ProbabilityStoreCollection.create(Lambda=3)
        assert collection.Lambda == 3
        assert collection.stores == {}

    def test_save_load_round_trip(self, tmp_path):
        collection = ProbabilityStoreCollection.create(Lambda=2)
        store = collection._ensure_store(0, 0)
        tau = ()
        store.entries[tau] = TauEntry(
            tau=tau,
            probabilities=np.array([[[1.0]]], dtype=np.float64),
            std_errors=np.array([[[0.0]]], dtype=np.float64),
            is_exact=np.array([[True]], dtype=bool),
            n_samples=np.array([[0]], dtype=np.int32),
        )

        path = tmp_path / "collection.npz"
        collection.save(path)
        loaded = ProbabilityStoreCollection.load(path)

        assert loaded.Lambda == 2
        assert (0, 0) in loaded.stores
        np.testing.assert_array_equal(
            loaded.stores[(0, 0)].entries[tau].probabilities,
            np.array([[[1.0]]], dtype=np.float64),
        )

    def test_compute_from_woven(self):
        collection = ProbabilityStoreCollection.create(Lambda=2)
        woven = WovenData(
            operators="XX",
            trace_permutation=(1, 0),
            Lambda=3,
            groups={
                (0, 0): WovenGroup(nL=0, nR=0, entries=[_make_woven_entry(())]),
                (1, 1): WovenGroup(nL=1, nR=1, entries=[_make_woven_entry((0, 1))]),
                (3, 0): WovenGroup(nL=3, nR=0, entries=[_make_woven_entry((0, 1, 2))]),
            },
        )

        collection.compute_from_woven(woven, progress=False)

        assert (0, 0) in collection.stores
        assert (1, 1) in collection.stores
        assert (3, 0) not in collection.stores  # clipped by Lambda=2
        assert () in collection.stores[(0, 0)].entries
        assert (0, 1) in collection.stores[(1, 1)].entries
