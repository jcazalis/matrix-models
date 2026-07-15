"""Tests for brute-force exact computation of cycle count probabilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sym_contractions.bruteforce import (
    _numba_count_cycles,
    _numba_cycle_type_key,
    _numba_enumerate_conjugacy_class_direct,
    _numba_enumerate_selected_classes,
    _numba_exact_probability_pair,
    _numba_exact_probability_pair_parallel,
    all_perms_with_cycle_type,
    compose_direct_product,
    conjugacy_class_size,
    count_cycles,
    enumerate_conjugacy_class,
    exact_all_conjugacy_pairs,
    exact_probability,
    get_cycle_type,
    numba_exact_all_conjugacy_pairs,
    numba_parallel_exact_all_conjugacy_pairs,
)
from sym_contractions.utils import enumerate_partitions


class TestConjugacyClassSize:
    def test_identity(self):
        """Identity permutation (1^n) has class size 1."""
        assert conjugacy_class_size((1, 1, 1)) == 1
        assert conjugacy_class_size((1,)) == 1

    def test_full_cycle(self):
        """Single n-cycle has class size (n-1)!."""
        for n in range(2, 7):
            ct = (n,)
            assert conjugacy_class_size(ct) == math.factorial(n - 1)

    def test_transpositions(self):
        """Transpositions (2, 1^{n-2}) in S_n: class size = n(n-1)/2."""
        for n in range(2, 7):
            ct = (2,) + (1,) * (n - 2)
            assert conjugacy_class_size(ct) == n * (n - 1) // 2

    def test_sum_over_classes_equals_factorial(self):
        """Sum of class sizes over all conjugacy classes = n!."""
        for n in range(1, 8):
            parts = enumerate_partitions(n)
            total = sum(conjugacy_class_size(p) for p in parts)
            assert total == math.factorial(n)


class TestGetCycleType:
    def test_identity(self):
        assert get_cycle_type([0, 1, 2]) == (1, 1, 1)

    def test_single_cycle(self):
        assert get_cycle_type([1, 2, 0]) == (3,)

    def test_two_cycles(self):
        assert get_cycle_type([1, 0, 3, 2]) == (2, 2)


class TestExactHelpers:
    def test_all_perms_with_cycle_type(self):
        perms = all_perms_with_cycle_type(3, (2, 1))
        assert len(perms) == conjugacy_class_size((2, 1))
        assert all(get_cycle_type(p) == (2, 1) for p in perms)

    def test_compose_direct_product(self):
        # n=2, m=2
        tau = [2, 3, 0, 1]
        sigma = [1, 0]
        nu = [0, 1]
        composed = compose_direct_product(tau, sigma, nu)
        assert composed == [3, 2, 0, 1]

    def test_count_cycles(self):
        assert count_cycles([0, 1, 2, 3]) == 4
        assert count_cycles([1, 0, 3, 2]) == 2


class TestExactProbability:
    def test_identity_tau(self):
        """τ = id: ℓ(σ × ν) = ℓ(σ) + ℓ(ν), deterministic for fixed cycle types."""
        n, m = 3, 3
        tau = list(range(n + m))
        ct_n = (2, 1)  # 2 cycles
        ct_m = (3,)  # 1 cycle
        probs = exact_probability(tau, n, m, ct_n, ct_m)
        # Should be 1.0 at k=3, 0.0 elsewhere
        assert probs[3] == pytest.approx(1.0)
        assert sum(probs) == pytest.approx(1.0)

    def test_probabilities_sum_to_one(self):
        """Probabilities must sum to 1 for any pair."""
        n, m = 3, 2
        tau = [1, 0, 3, 4, 2]
        ct_n = (2, 1)
        ct_m = (2,)
        probs = exact_probability(tau, n, m, ct_n, ct_m)
        assert sum(probs) == pytest.approx(1.0)

    def test_shape(self):
        n, m = 3, 2
        tau = list(range(n + m))
        probs = exact_probability(tau, n, m, (3,), (2,))
        assert probs.shape == (n + m + 1,)


class TestExactAllConjugacyPairs:
    def test_shape_and_normalization(self):
        n, m = 3, 3
        tau = list(range(n + m))
        parts_n = enumerate_partitions(n)
        parts_m = enumerate_partitions(m)
        result = exact_all_conjugacy_pairs(tau, n, m, parts_n, parts_m, progress=False)
        assert result.shape == (len(parts_n), len(parts_m), n + m + 1)
        # Each row should sum to 1
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_defaults_and_progress_branch(self):
        n, m = 2, 2
        tau = list(range(n + m))
        result = exact_all_conjugacy_pairs(tau, n, m, progress=True)
        assert result.shape == (2, 2, n + m + 1)
        assert np.allclose(result.sum(axis=-1), 1.0)


# ---------------------------------------------------------------------------
# Tests for direct construction of conjugacy classes
# ---------------------------------------------------------------------------


class TestEnumerateConjugacyClass:
    @pytest.mark.parametrize(
        ("n", "ct"),
        [
            (1, (1,)),
            (3, (3,)),
            (3, (1, 1, 1)),
            (4, (2, 2)),
            (4, (4,)),
            (5, (3, 2)),
            (5, (2, 2, 1)),
            (6, (3, 2, 1)),
        ],
    )
    def test_class_size_matches_formula(self, n, ct):
        """Number of generated permutations equals conjugacy_class_size."""
        perms = enumerate_conjugacy_class(n, ct)
        assert len(perms) == conjugacy_class_size(ct)

    @pytest.mark.parametrize(
        ("n", "ct"),
        [
            (3, (3,)),
            (4, (2, 2)),
            (5, (3, 1, 1)),
            (6, (3, 2, 1)),
        ],
    )
    def test_all_have_correct_cycle_type(self, n, ct):
        """Every generated permutation has the requested cycle type."""
        for perm in enumerate_conjugacy_class(n, ct):
            assert get_cycle_type(perm) == ct

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_no_duplicates(self, n):
        """All generated permutations are distinct."""
        for ct in enumerate_partitions(n):
            perms = enumerate_conjugacy_class(n, ct)
            as_tuples = [tuple(p) for p in perms]
            assert len(as_tuples) == len(set(as_tuples))

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
    def test_total_equals_factorial(self, n):
        """Total permutations across all classes equals n!."""
        total = sum(
            len(enumerate_conjugacy_class(n, ct)) for ct in enumerate_partitions(n)
        )
        assert total == math.factorial(n)

    def test_matches_brute_force(self):
        """Matches the O(n!) brute force for n=5."""
        from sym_contractions.bruteforce import all_perms_with_cycle_type

        n = 5
        for ct in enumerate_partitions(n):
            direct = {tuple(p) for p in enumerate_conjugacy_class(n, ct)}
            brute = {tuple(p) for p in all_perms_with_cycle_type(n, ct)}
            assert direct == brute


# ---------------------------------------------------------------------------
# Tests for Numba-accelerated conjugacy class enumeration
# ---------------------------------------------------------------------------


class TestNumbaEnumerateSelectedClassesAll:
    def test_empty_n(self):
        """n=0 returns empty arrays for each partition."""
        partitions = [(1,)]  # Doesn't matter, will return empty array
        result = _numba_enumerate_selected_classes(0, partitions, [0])
        assert set(result.keys()) == {0}
        assert result[0].shape == (1, 0)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_all_partitions_class_sizes(self, n):
        """Generated classes have correct sizes when all partitions provided."""
        partitions = enumerate_partitions(n)
        selected = list(range(len(partitions)))
        result = _numba_enumerate_selected_classes(n, partitions, selected)

        assert len(result) == len(partitions)
        for i, ct in enumerate(partitions):
            arr = result[i]
            expected_size = conjugacy_class_size(ct)
            assert arr.shape[0] == expected_size, (
                f"Partition {ct} at index {i}: "
                f"expected {expected_size} perms, got {arr.shape[0]}"
            )
            assert arr.shape[1] == n

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_all_partitions_correct_cycle_types(self, n):
        """Each permutation has the correct cycle type."""
        partitions = enumerate_partitions(n)
        selected = list(range(len(partitions)))
        result = _numba_enumerate_selected_classes(n, partitions, selected)

        for i, ct in enumerate(partitions):
            arr = result[i]
            for perm in arr:
                computed_ct = get_cycle_type(perm.tolist())
                assert computed_ct == ct

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_all_partitions_no_duplicates(self, n):
        """No duplicate permutations within each class."""
        partitions = enumerate_partitions(n)
        selected = list(range(len(partitions)))
        result = _numba_enumerate_selected_classes(n, partitions, selected)

        for i, ct in enumerate(partitions):
            arr = result[i]
            as_tuples = {tuple(perm) for perm in arr}
            assert len(as_tuples) == arr.shape[0], f"Duplicates found in class {ct}"

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_all_partitions_total_equals_factorial(self, n):
        """Total permutations across all classes equals n!."""
        partitions = enumerate_partitions(n)
        selected = list(range(len(partitions)))
        result = _numba_enumerate_selected_classes(n, partitions, selected)
        total = sum(arr.shape[0] for arr in result.values())
        assert total == math.factorial(n)

    def test_subset_of_partitions(self):
        """Works correctly when only a subset of partitions is provided."""
        n = 4
        # Only request two specific cycle types out of p(4)=5 partitions
        partitions = [(4,), (2, 2)]  # n-cycle and two 2-cycles
        result = _numba_enumerate_selected_classes(n, partitions, [0, 1])

        assert set(result.keys()) == {0, 1}
        # Check first class: (4,) has (4-1)! = 6 permutations
        assert result[0].shape == (6, 4)
        for perm in result[0]:
            assert get_cycle_type(perm.tolist()) == (4,)

        # Check second class: (2,2) has 3 permutations
        assert result[1].shape == (3, 4)
        for perm in result[1]:
            assert get_cycle_type(perm.tolist()) == (2, 2)

    def test_single_partition(self):
        """Works with a single partition."""
        n = 5
        partitions = [(3, 2)]  # Just one cycle type
        result = _numba_enumerate_selected_classes(n, partitions, [0])

        assert set(result.keys()) == {0}
        expected_size = conjugacy_class_size((3, 2))
        assert result[0].shape == (expected_size, 5)
        for perm in result[0]:
            assert get_cycle_type(perm.tolist()) == (3, 2)

    def test_identity_and_full_cycle(self):
        """Test extreme partitions: identity and full cycle."""
        n = 5
        partitions = [(1, 1, 1, 1, 1), (5,)]  # Identity and 5-cycle
        result = _numba_enumerate_selected_classes(n, partitions, [0, 1])

        assert set(result.keys()) == {0, 1}
        # Identity: only one permutation
        assert result[0].shape == (1, 5)
        assert np.array_equal(result[0][0], np.arange(5))

        # Full cycle: (n-1)! permutations
        assert result[1].shape == (math.factorial(4), 5)
        for perm in result[1]:
            assert get_cycle_type(perm.tolist()) == (5,)

    def test_ordering_preserved(self):
        """Output order matches input partition order."""
        n = 4
        # Provide partitions in non-standard order
        partitions = [(2, 2), (4,), (1, 1, 1, 1), (3, 1)]
        result = _numba_enumerate_selected_classes(n, partitions, [0, 1, 2, 3])

        assert len(result) == 4
        # Verify each position matches the requested partition
        for i, ct in enumerate(partitions):
            arr = result[i]
            for perm in arr:
                computed_ct = get_cycle_type(perm.tolist())
                assert computed_ct == ct, (
                    f"Position {i}: expected cycle type {ct}, "
                    f"got {computed_ct} for permutation {perm.tolist()}"
                )


class TestNumbaEnumerateSelectedClasses:
    def test_selected_classes_match_python_direct_subset(self):
        n = 6
        partitions = enumerate_partitions(n)
        selected = [0, 3, 7, 3]

        selected_classes = _numba_enumerate_selected_classes(n, partitions, selected)

        assert set(selected_classes.keys()) == {0, 3, 7}
        for idx in {0, 3, 7}:
            direct = np.asarray(enumerate_conjugacy_class(n, partitions[idx]))
            assert selected_classes[idx].shape == direct.shape
            assert {tuple(row) for row in selected_classes[idx]} == {
                tuple(row) for row in direct
            }

    def test_selected_empty_indices(self):
        out = _numba_enumerate_selected_classes(5, enumerate_partitions(5), [])
        assert out == {}


class TestNumbaDirectConjugacyClassConstructor:
    @pytest.mark.parametrize(
        ("n", "ct"),
        [
            (3, (3,)),
            (4, (2, 2)),
            (5, (3, 2)),
            (6, (3, 2, 1)),
        ],
    )
    def test_matches_python_direct_constructor(self, n, ct):
        class_size = conjugacy_class_size(ct)
        direct_numba = _numba_enumerate_conjugacy_class_direct(
            n,
            np.asarray(ct, dtype=np.int32),
            class_size,
        )
        direct_python = enumerate_conjugacy_class(n, ct)

        assert direct_numba.shape == (class_size, n)
        assert {tuple(row) for row in direct_numba} == {
            tuple(row) for row in direct_python
        }

    def test_identity_class(self):
        n = 6
        ct = (1, 1, 1, 1, 1, 1)
        class_size = conjugacy_class_size(ct)
        out = _numba_enumerate_conjugacy_class_direct(
            n,
            np.asarray(ct, dtype=np.int32),
            class_size,
        )
        assert out.shape == (1, n)
        assert np.array_equal(out[0], np.arange(n, dtype=np.int32))


class TestNumbaInternalPyFunc:
    def test_numba_count_cycles_compiled(self):
        perm = np.array([1, 0, 3, 2], dtype=np.int32)
        assert _numba_count_cycles(perm) == 2

    def test_numba_cycle_type_key_compiled(self):
        perm = np.array([1, 0, 2, 3], dtype=np.int32)
        key = _numba_cycle_type_key(perm)
        assert key.shape == (4,)
        assert np.array_equal(key, np.array([2, 1, 1, 0], dtype=np.int32))

    def test_numba_exact_probability_pair_py_func(self):
        n, m = 2, 2
        tau = np.array([0, 1, 2, 3], dtype=np.int32)
        classes_n = _numba_enumerate_selected_classes(n, [(1, 1)], [0])
        classes_m = _numba_enumerate_selected_classes(m, [(2,)], [0])
        out = _numba_exact_probability_pair.py_func(  # type: ignore
            tau,
            classes_n[0],
            classes_m[0],
            n + m,
        )
        assert out.shape == (n + m + 1,)
        assert np.isclose(out.sum(), 1.0)

    def test_numba_exact_probability_pair_parallel_py_func(self):
        n, m = 2, 2
        tau = np.array([0, 1, 2, 3], dtype=np.int32)
        classes_n = _numba_enumerate_selected_classes(n, [(1, 1)], [0])
        classes_m = _numba_enumerate_selected_classes(m, [(2,)], [0])
        out = _numba_exact_probability_pair_parallel.py_func(  # type: ignore
            tau,
            classes_n[0],
            classes_m[0],
            n + m,
        )
        assert out.shape == (n + m + 1,)
        assert np.isclose(out.sum(), 1.0)


class TestNumbaExactAllConjugacyPairs:
    def test_matches_pure_python(self):
        """Numba batch result matches pure-Python batch for n=m=3."""
        n, m = 3, 3
        tau_list = [1, 0, 3, 5, 2, 4]
        parts_n = enumerate_partitions(n)
        parts_m = enumerate_partitions(m)

        python_result = exact_all_conjugacy_pairs(
            tau_list, n, m, parts_n, parts_m, progress=False
        )
        numba_result = numba_exact_all_conjugacy_pairs(
            tau_list, n, m, parts_n, parts_m, progress=False
        )
        np.testing.assert_allclose(numba_result, python_result, atol=1e-10)

    def test_shape_and_normalization(self):
        """Output has correct shape and probabilities sum to 1."""
        n, m = 4, 3
        tau = list(range(n + m))
        parts_n = enumerate_partitions(n)
        parts_m = enumerate_partitions(m)
        result = numba_exact_all_conjugacy_pairs(
            tau, n, m, parts_n, parts_m, progress=False
        )
        assert result.shape == (len(parts_n), len(parts_m), n + m + 1)
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_defaults_and_progress_branch(self):
        n, m = 2, 2
        tau = np.arange(n + m, dtype=np.int32)
        result = numba_exact_all_conjugacy_pairs(tau, n, m, progress=True)
        assert result.shape == (2, 2, n + m + 1)
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_n0_edge_case(self):
        n, m = 0, 2
        tau = np.arange(n + m, dtype=np.int32)
        result = numba_exact_all_conjugacy_pairs(tau, n, m, progress=False)
        # p(0)=1 (empty partition), p(2)=2
        assert result.shape == (1, 2, n + m + 1)
        assert np.allclose(result.sum(axis=-1), 1.0)


class TestNumbaParallelExactAllConjugacyPairs:
    def test_matches_sequential_numba(self):
        """Parallel numba matches sequential numba for n=m=3."""
        n, m = 3, 3
        tau_list = [1, 0, 3, 5, 2, 4]
        parts_n = enumerate_partitions(n)
        parts_m = enumerate_partitions(m)

        seq_result = numba_exact_all_conjugacy_pairs(
            tau_list, n, m, parts_n, parts_m, progress=False
        )
        par_result = numba_parallel_exact_all_conjugacy_pairs(
            tau_list, n, m, parts_n, parts_m, progress=False
        )
        np.testing.assert_allclose(par_result, seq_result, atol=1e-10)

    def test_shape_and_normalization(self):
        """Output has correct shape and probabilities sum to 1."""
        n, m = 4, 3
        tau = list(range(n + m))
        parts_n = enumerate_partitions(n)
        parts_m = enumerate_partitions(m)
        result = numba_parallel_exact_all_conjugacy_pairs(
            tau, n, m, parts_n, parts_m, progress=False
        )
        assert result.shape == (len(parts_n), len(parts_m), n + m + 1)
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_defaults_and_progress_branch(self):
        n, m = 2, 2
        tau = np.arange(n + m, dtype=np.int32)
        result = numba_parallel_exact_all_conjugacy_pairs(tau, n, m, progress=True)
        assert result.shape == (2, 2, n + m + 1)
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_parallel_kernel_else_branch(self):
        # Force size_n < size_m so the alternate parallel branch executes.
        n, m = 2, 4
        tau = np.arange(n + m, dtype=np.int32)
        classes_n = _numba_enumerate_selected_classes(n, [(1, 1)], [0])
        classes_m = _numba_enumerate_selected_classes(m, [(2, 1, 1)], [0])
        out = _numba_exact_probability_pair_parallel(
            tau,
            classes_n[0],
            classes_m[0],
            n + m,
        )
        assert out.shape == (n + m + 1,)
        assert np.isclose(out.sum(), 1.0)
