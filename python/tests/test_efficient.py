"""Tests for the efficient contraction computation module.

Validates the coset-reduction + character-theory pipeline against the
brute-force Route 1 (exhaustive probability → einsum) pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sym_contractions.bruteforce import (
    exact_all_conjugacy_pairs,
)
from sym_contractions.character_tables import (
    CharacterTableData,
)
from sym_contractions.coset import (
    CosetReductionData,
    compute_coset_reduction_from_gap,
    load_gap_coset_data,
)
from sym_contractions.efficient import (
    compute_all_contractions_efficient,
    compute_contraction_efficient,
    compute_rep_dimensions,
    compute_s_polynomial,
)
from sym_contractions.woven import (
    WovenData,
    WovenEntry,
    WovenGroup,
    compute_contraction_coefficients,
    tau_to_pairs_1indexed,
)

# ======================================================================
# Fixtures — small character tables built in-memory
# ======================================================================

CT_DIR = None  # set in conftest or loaded inline below

# GAP coset data fixture
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
_COSET_REPS_JSON = _DATA_DIR / "test-data" / "coset_reps.json"
_GAP_DATA = load_gap_coset_data(_COSET_REPS_JSON) if _COSET_REPS_JSON.exists() else None
_CT_DIR = _DATA_DIR / "character_tables"


def _get_gap_data():
    """Return loaded GAP data, skip if unavailable."""
    if _GAP_DATA is None:
        pytest.skip(f"GAP data not found: {_COSET_REPS_JSON}")
    return _GAP_DATA


def _make_ct_s0() -> CharacterTableData:
    """S_0: trivial."""
    return {
        "n": 0,
        "cycle_types": [[]],
        "class_sizes": np.array([1], dtype=np.int32),
        "characters": np.array([[1]], dtype=np.int32),
    }


def _make_ct_s1() -> CharacterTableData:
    """S_1: one irrep, one class (identity)."""
    return {
        "n": 1,
        "cycle_types": [[1]],
        "class_sizes": np.array([1], dtype=np.int32),
        "characters": np.array([[1]], dtype=np.int32),
    }


def _make_ct_s2() -> CharacterTableData:
    """S_2 in reverse-lex order: [(2), (1,1)].

    Irreps: (2)=trivial, (1,1)=sign.
    Characters:
        (2) class: χ^{(2)}=1,  χ^{(1,1)}=-1
        (1,1) class: χ^{(2)}=1, χ^{(1,1)}=1
    """
    return {
        "n": 2,
        "cycle_types": [[2], [1, 1]],
        "class_sizes": np.array([1, 1], dtype=np.int32),
        "characters": np.array([[1, 1], [-1, 1]], dtype=np.int32),
    }


def _make_ct_s3() -> CharacterTableData:
    """S_3 in reverse-lex order: [(3), (2,1), (1,1,1)].

    Irreps (same ordering): (3)=trivial, (2,1)=standard, (1,1,1)=sign.
    """
    return {
        "n": 3,
        "cycle_types": [[3], [2, 1], [1, 1, 1]],
        "class_sizes": np.array([2, 3, 1], dtype=np.int32),
        "characters": np.array(
            [
                [1, 1, 1],  # trivial: all 1
                [-1, 0, 2],  # standard: chi(3-cycle)=-1, chi(2,1)=0, chi(id)=2
                [1, -1, 1],  # sign
            ],
            dtype=np.int32,
        ),
    }


def _make_trivial_woven_parallel_fixture() -> WovenData:
    """Create a small woven object that exercises tau-level parallelism."""
    taus = [(0, 1), (1, 0)]
    entries = [
        WovenEntry(
            involution=tuple(),
            tau=tau,
            pairs_1indexed=tau_to_pairs_1indexed(tau),
            coefficient_poly=[(idx + 1, 1)],
        )
        for idx, tau in enumerate(taus)
    ]
    return WovenData(
        operators="XX",
        trace_permutation=(1, 0),
        Lambda=2,
        groups={(0, 2): WovenGroup(nL=0, nR=2, entries=entries)},
        mass=1.0,
        is_even=True,
        is_hermitian=False,
    )


def _contraction_signature(entry):
    return (
        tuple(tuple(pair) for pair in entry.pairs_1indexed),
        entry.R,
        entry.S,
        tuple(np.asarray(entry.coefficients).tolist()),
    )


# ======================================================================
# Tests: helper functions
# ======================================================================


class TestComputeSPolynomial:
    """Tests for the normalized S-polynomial ŝ_R(j) = Σ_{μ: ℓ(μ)=j} (|C_μ|/n!) χ^R(μ)."""

    def test_s0(self):
        ct = _make_ct_s0()
        s = compute_s_polynomial(ct)
        assert s.shape == (1, 1)
        assert s[0, 0] == 1.0

    def test_s1(self):
        ct = _make_ct_s1()
        s = compute_s_polynomial(ct)
        assert s.shape == (1, 2)
        assert s[0, 0] == 0.0
        # |C_{(1,)}|/1! = 1, χ^triv((1,)) = 1
        assert s[0, 1] == 1.0

    def test_s2(self):
        ct = _make_ct_s2()
        s = compute_s_polynomial(ct)
        assert s.shape == (2, 3)
        # j=0: no partition of 2 has 0 parts → all zeros
        np.testing.assert_array_equal(s[:, 0], [0, 0])
        # j=1: only (2,) with |C_{(2,)}|/2!=1/2 → (1/2) × χ^R((2,))
        np.testing.assert_allclose(s[:, 1], [0.5, -0.5])
        # j=2: only (1,1) with |C_{(1,1)}|/2!=1/2 → (1/2) × χ^R((1,1))
        np.testing.assert_allclose(s[:, 2], [0.5, 0.5])

    def test_s3(self):
        ct = _make_ct_s3()
        s = compute_s_polynomial(ct)
        assert s.shape == (3, 4)
        # j=0: always 0
        np.testing.assert_array_equal(s[:, 0], [0, 0, 0])
        # j=1: (3,) with |C_{(3,)}|/3!=2/6=1/3: (1/3) × χ^R((3,)) = (1/3)×[1,-1,1]
        np.testing.assert_allclose(s[:, 1], [1 / 3, -1 / 3, 1 / 3])
        # j=2: (2,1) with |C_{(2,1)}|/3!=3/6=1/2: (1/2) × χ^R((2,1)) = (1/2)×[1,0,-1]
        np.testing.assert_allclose(s[:, 2], [0.5, 0, -0.5])
        # j=3: (1,1,1) with |C_{(1,1,1)}|/3!=1/6: (1/6) × χ^R((1,1,1)) = (1/6)×[1,2,1]
        np.testing.assert_allclose(s[:, 3], [1 / 6, 2 / 6, 1 / 6])

    def test_sum_over_j_gives_one(self):
        """Σ_j ŝ_trivial(j) = Σ_μ (|C_μ|/n!) = 1."""
        for n, ct in [(1, _make_ct_s1()), (2, _make_ct_s2()), (3, _make_ct_s3())]:
            s = compute_s_polynomial(ct)
            # trivial rep is at index 0 in reverse-lex (largest partition)
            assert s[0, :].sum() == pytest.approx(1.0)


class TestComputeRepDimensions:
    def test_s1(self):
        ct = _make_ct_s1()
        f = compute_rep_dimensions(ct)
        np.testing.assert_array_equal(f, [1])

    def test_s2(self):
        ct = _make_ct_s2()
        f = compute_rep_dimensions(ct)
        np.testing.assert_array_equal(f, [1, 1])

    def test_s3(self):
        ct = _make_ct_s3()
        f = compute_rep_dimensions(ct)
        np.testing.assert_array_equal(f, [1, 2, 1])

    def test_sum_of_squares_equals_factorial(self):
        """Σ_R (f^R)² = n!  (a basic identity for S_n)."""
        for n, ct in [(1, _make_ct_s1()), (2, _make_ct_s2()), (3, _make_ct_s3())]:
            f = compute_rep_dimensions(ct)
            assert int((f**2).sum()) == math.factorial(n)


# ======================================================================
# Tests: contraction coefficients vs brute force
# ======================================================================


class TestContractionEfficient:
    """Compare efficient contraction against the brute-force einsum pipeline."""

    @pytest.mark.parametrize(
        "tau, n, m",
        [
            ([1, 0], 1, 1),
            ([1, 2, 0], 1, 2),
            ([1, 2, 0], 2, 1),
        ],
    )
    def test_matches_bruteforce_small(self, tau, n, m):
        """Efficient coefficients match Route 1 (exact probs → einsum)."""
        make_ct = {0: _make_ct_s0, 1: _make_ct_s1, 2: _make_ct_s2, 3: _make_ct_s3}
        ct_n = make_ct[n]()
        ct_m = make_ct[m]()

        # Brute-force probability matrix
        probs = exact_all_conjugacy_pairs(tau, n, m, progress=False)
        expected = compute_contraction_coefficients(probs, ct_n, ct_m, n, m)

        # Efficient method
        gap_data = _get_gap_data()
        coset_data = None
        if n > 0 and m > 0:
            entry = gap_data.lookup(tau, n, m)
            if entry is None:
                pytest.skip(
                    f"(tau={tau}, n={n}, m={m}) not in test GAP data; "
                    "regenerate with prepare_coset_input.py + generate_coset_reps.g"
                )
            coset_data = compute_coset_reduction_from_gap(tau, n, m, gap_data)
        actual = compute_contraction_efficient(tau, n, m, ct_n, ct_m, coset_data)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    @pytest.mark.parametrize(
        "tau, n, m",
        [
            ([1, 0, 3, 2], 2, 2),
            ([2, 3, 0, 1], 2, 2),
            ([1, 2, 3, 0], 2, 2),
            ([3, 2, 1, 0], 2, 2),
            ([0, 1, 2, 3], 2, 2),
        ],
    )
    def test_matches_bruteforce_n2_m2(self, tau, n, m):
        """Larger set of τ's with n=m=2."""
        ct_n = _make_ct_s2()
        ct_m = _make_ct_s2()

        probs = exact_all_conjugacy_pairs(tau, n, m, progress=False)
        expected = compute_contraction_coefficients(probs, ct_n, ct_m, n, m)

        gap_data = _get_gap_data()
        entry = gap_data.lookup(tau, n, m)
        if entry is None:
            pytest.skip(
                f"(tau={tau}, n={n}, m={m}) not in test GAP data; "
                "regenerate with prepare_coset_input.py + generate_coset_reps.g"
            )
        coset_data = compute_coset_reduction_from_gap(tau, n, m, gap_data)
        actual = compute_contraction_efficient(tau, n, m, ct_n, ct_m, coset_data)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_matches_bruteforce_n3_m2(self):
        """Cross-check with n=3, m=2, a specific τ."""
        tau = [2, 3, 4, 0, 1]  # 5-cycle
        n, m = 3, 2
        ct_n = _make_ct_s3()
        ct_m = _make_ct_s2()

        probs = exact_all_conjugacy_pairs(tau, n, m, progress=False)
        expected = compute_contraction_coefficients(probs, ct_n, ct_m, n, m)

        gap_data = _get_gap_data()
        entry = gap_data.lookup(tau, n, m)
        if entry is None:
            pytest.skip(
                f"(tau={tau}, n={n}, m={m}) not in test GAP data; "
                "regenerate with prepare_coset_input.py + generate_coset_reps.g"
            )
        coset_data = compute_coset_reduction_from_gap(tau, n, m, gap_data)
        actual = compute_contraction_efficient(tau, n, m, ct_n, ct_m, coset_data)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


class TestComputeAllContractionsEfficientParallel:
    def test_parallel_matches_serial_for_trivial_sector(self):
        woven = _make_trivial_woven_parallel_fixture()

        serial = compute_all_contractions_efficient(
            woven,
            _CT_DIR,
            verbose=False,
            parallel=False,
        )
        parallel = compute_all_contractions_efficient(
            woven,
            _CT_DIR,
            verbose=False,
            parallel=True,
            max_workers=2,
        )

        serial_entries = sorted(
            _contraction_signature(entry) for entry in serial.entries
        )
        parallel_entries = sorted(
            _contraction_signature(entry) for entry in parallel.entries
        )
        assert parallel.label == serial.label
        assert parallel.Lambda == serial.Lambda
        assert parallel_entries == serial_entries


class TestContractionTrivial:
    """Tests for n=0, m=0 edge cases."""

    def test_n0_m0(self):
        ct0 = _make_ct_s0()
        result = compute_contraction_efficient((), 0, 0, ct0, ct0)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == pytest.approx(1.0)

    def test_n0_m2(self):
        """n=0: c_k((), S) = χ^S(τ) / (m! f^S) × S_S(k)."""
        tau = (1, 0)  # swap in S_2
        ct0 = _make_ct_s0()
        ct2 = _make_ct_s2()

        result = compute_contraction_efficient(tau, 0, 2, ct0, ct2)
        assert result.shape == (1, 2, 3)

        # Verify against brute-force
        probs = exact_all_conjugacy_pairs(list(tau), 0, 2, progress=False)
        expected = compute_contraction_coefficients(probs, ct0, ct2, 0, 2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_n2_m0(self):
        tau = (1, 0)  # swap in S_2
        ct2 = _make_ct_s2()
        ct0 = _make_ct_s0()

        result = compute_contraction_efficient(tau, 2, 0, ct2, ct0)
        assert result.shape == (2, 1, 3)

        probs = exact_all_conjugacy_pairs(list(tau), 2, 0, progress=False)
        expected = compute_contraction_coefficients(probs, ct2, ct0, 2, 0)
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestLeftVsRightReduction:
    """Verify that reducing on left or right gives the same answer."""

    @pytest.mark.parametrize(
        "tau, n, m",
        [
            ([1, 0, 3, 2], 2, 2),
            ([2, 3, 0, 1], 2, 2),
            ([1, 2, 3, 0], 2, 2),
        ],
    )
    def test_both_sides_agree(self, tau, n, m):
        ct_n = _make_ct_s2()
        ct_m = _make_ct_s2()

        gap_data = _get_gap_data()
        entry = gap_data.lookup(tuple(tau), n, m)
        if entry is None:
            pytest.skip(
                f"(tau={tau}, n={n}, m={m}) not in test GAP data; "
                "regenerate with prepare_coset_input.py + generate_coset_reps.g"
            )

        # Build left-side reduction
        left_data = entry["left"]
        import numpy as _np

        reps_left = _np.array(left_data["reps_0indexed"], dtype=_np.int32)
        # Invert reps (same logic as compute_coset_reduction_from_gap)
        if reps_left.size > 0:
            inv = _np.empty_like(reps_left)
            for i in range(reps_left.shape[0]):
                for j in range(reps_left.shape[1]):
                    inv[i, reps_left[i, j]] = j
            reps_left = inv
        data_left = CosetReductionData(
            tau=tuple(tau),
            n=n,
            m=m,
            reduced_side="left",
            coset_reps=reps_left,
            h_order=left_data["h_order"],
            num_reps=left_data["num_reps"],
        )

        # Build right-side reduction
        right_data = entry["right"]
        reps_right = _np.array(right_data["reps_0indexed"], dtype=_np.int32)
        if reps_right.size > 0:
            inv = _np.empty_like(reps_right)
            for i in range(reps_right.shape[0]):
                for j in range(reps_right.shape[1]):
                    inv[i, reps_right[i, j]] = j
            reps_right = inv
        data_right = CosetReductionData(
            tau=tuple(tau),
            n=n,
            m=m,
            reduced_side="right",
            coset_reps=reps_right,
            h_order=right_data["h_order"],
            num_reps=right_data["num_reps"],
        )

        result_left = compute_contraction_efficient(tau, n, m, ct_n, ct_m, data_left)
        result_right = compute_contraction_efficient(tau, n, m, ct_n, ct_m, data_right)

        np.testing.assert_allclose(result_left, result_right, atol=1e-12)


class TestContractionNormalization:
    """Sanity checks on the contraction coefficient values."""

    def test_trivial_rep_d0_coefficient(self):
        """For R = trivial, S = trivial, c_0 should be ≥ 0."""
        tau = [2, 3, 0, 1]  # present in test GAP data
        ct2 = _make_ct_s2()
        gap_data = _get_gap_data()
        coset_data = compute_coset_reduction_from_gap(tau, 2, 2, gap_data)
        result = compute_contraction_efficient(tau, 2, 2, ct2, ct2, coset_data)
        # Coefficients should sum (approximately) to 1 for the trivial (R,S)
        # since it's the trace-weighted average over all perms.
        trivial_coeffs = result[0, 0, :]
        assert trivial_coeffs.sum() == pytest.approx(1.0, abs=1e-12)

    def test_sign_rep_coefficients_finite(self):
        """For the sign × sign reps, coefficients should be finite."""
        tau = [2, 3, 0, 1]  # present in test GAP data
        ct2 = _make_ct_s2()
        gap_data = _get_gap_data()
        coset_data = compute_coset_reduction_from_gap(tau, 2, 2, gap_data)
        result = compute_contraction_efficient(tau, 2, 2, ct2, ct2, coset_data)
        sign_coeffs = result[-1, -1, :]
        # Coefficients for sign × sign should be finite and not NaN
        assert np.all(np.isfinite(sign_coeffs))
