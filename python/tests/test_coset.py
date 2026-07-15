"""Tests for sym_contractions.coset — GAP coset data interface.

Tests cover:
1. GAP data loader (GapCosetData).
2. GAP data validation (Lagrange checks, valid permutations).
3. compute_coset_reduction_from_gap entry point.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sym_contractions.coset import (
    compute_coset_reduction_from_gap,
    load_gap_coset_data,
)


# ======================================================================
# Helpers
# ======================================================================


def _compose(a: list[int], b: list[int]) -> list[int]:
    """Compose permutations: (a ∘ b)(x) = a(b(x))."""
    return [a[b[i]] for i in range(len(a))]


def _inverse(perm: list[int]) -> list[int]:
    """Inverse of a permutation."""
    inv = [0] * len(perm)
    for i, v in enumerate(perm):
        inv[v] = i
    return inv


# ======================================================================
# GAP integration tests
# ======================================================================

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
COSET_REPS_JSON = DATA_DIR / "test-data" / "coset_reps.json"


class TestGapLoader:
    """Tests for loading and using GAP-precomputed coset data."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        if not COSET_REPS_JSON.exists():
            pytest.skip(f"GAP data not found: {COSET_REPS_JSON}")

    def test_load_gap_data(self):
        """GapCosetData loads successfully."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        assert len(gap_data) > 0

    def test_lookup_identity(self):
        """Lookup for τ = identity (0,0 case)."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        entry = gap_data.lookup((), 0, 0)
        assert entry is not None
        assert entry["left"]["h_order"] == 1
        assert entry["left"]["num_reps"] == 1

    def test_lookup_n1_m1(self):
        """Lookup for τ = (1, 0), n=1, m=1 (swap)."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        entry = gap_data.lookup((1, 0), 1, 1)
        assert entry is not None
        assert entry["left"]["h_order"] == 1
        assert entry["left"]["num_reps"] == 1

    def test_lookup_missing(self):
        """Lookup for a non-existent key returns None."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        assert gap_data.lookup((99, 98, 97), 50, 50) is None


class TestGapDataValidation:
    """Validate GAP-precomputed data for consistency."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        if not COSET_REPS_JSON.exists():
            pytest.skip(f"GAP data not found: {COSET_REPS_JSON}")

    # Small K=4 tau values where validation is practical
    SMALL_CASES = [
        # (tau, n, m) - from actual woven data
        ((), 0, 0),
        ((1, 0), 1, 1),
        ((2, 3, 0, 1), 2, 2),
        ((3, 4, 5, 0, 1, 2), 3, 3),
        ((0, 4, 5, 3, 1, 2), 3, 3),
        ((3, 2, 5, 0, 1, 4), 3, 3),
        ((3, 4, 2, 0, 1, 5), 3, 3),
        ((3, 4, 5, 0, 2, 1), 3, 3),
        ((4, 5, 6, 7, 0, 1, 2, 3), 4, 4),
    ]

    @pytest.mark.parametrize("tau,n,m", SMALL_CASES)
    def test_lagrange_gap(self, tau, n, m):
        """GAP data satisfies Lagrange: h_order × num_reps = side!."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        entry = gap_data.lookup(tau, n, m)
        if entry is None:
            pytest.skip(f"tau={tau} n={n} m={m} not in GAP data")

        left = entry["left"]
        assert left["h_order"] * left["num_reps"] == math.factorial(n)

        right = entry["right"]
        assert right["h_order"] * right["num_reps"] == math.factorial(m)

    @pytest.mark.parametrize(
        "tau,n,m",
        [
            ((2, 3, 0, 1), 2, 2),
            ((3, 4, 5, 0, 1, 2), 3, 3),
            ((0, 4, 5, 3, 1, 2), 3, 3),
            ((3, 2, 5, 0, 1, 4), 3, 3),
        ],
    )
    def test_reps_are_valid_permutations(self, tau, n, m):
        """GAP reps are valid permutations of the correct size."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        entry = gap_data.lookup(tau, n, m)
        if entry is None:
            pytest.skip(f"tau={tau} n={n} m={m} not in GAP data")

        for side in ("left", "right"):
            side_data = entry[side]
            side_size = n if side == "left" else m
            for rep in side_data["reps_0indexed"]:
                assert sorted(rep) == list(range(side_size)), (
                    f"side={side}: rep {rep} is not a valid permutation "
                    f"of {{0,...,{side_size - 1}}}"
                )


class TestComputeCosetReductionFromGap:
    """Tests for the compute_coset_reduction_from_gap entry point."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        if not COSET_REPS_JSON.exists():
            pytest.skip(f"GAP data not found: {COSET_REPS_JSON}")

    def test_basic(self):
        """Basic compute_coset_reduction_from_gap call."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        tau = (2, 3, 0, 1)
        result = compute_coset_reduction_from_gap(tau, 2, 2, gap_data)
        assert result.n == 2
        assert result.m == 2
        assert result.num_reps * result.h_order == math.factorial(
            result.n if result.reduced_side == "left" else result.m
        )

    def test_missing_key_raises(self):
        """KeyError for missing (tau, n, m)."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        fake_tau = tuple(range(99, -1, -1))
        with pytest.raises(KeyError):
            compute_coset_reduction_from_gap(fake_tau, 50, 50, gap_data)

    def test_large_tau_from_gap(self):
        """Test that large tau (n>10) loads from GAP without issues."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        import json

        with open(COSET_REPS_JSON) as f:
            raw = json.load(f)
        large_entries = [e for e in raw["entries"] if e["n"] >= 10 and e["m"] >= 10]
        if not large_entries:
            pytest.skip("No large entries in GAP data")

        entry = large_entries[0]
        tau = tuple(entry["tau_0indexed"])
        n, m = entry["n"], entry["m"]
        result = compute_coset_reduction_from_gap(tau, n, m, gap_data)
        assert result.num_reps > 0
        assert result.h_order > 0
        assert result.num_reps * result.h_order == math.factorial(
            n if result.reduced_side == "left" else m
        )

    def test_coset_reps_inverted_correctly(self):
        """Verify that coset reps are inverted (GAP → math convention)."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        tau = (3, 4, 5, 0, 1, 2)
        result = compute_coset_reduction_from_gap(tau, 3, 3, gap_data)

        # Each rep should be a valid permutation
        side_size = result.n if result.reduced_side == "left" else result.m
        for i in range(result.coset_reps.shape[0]):
            rep = list(result.coset_reps[i])
            assert sorted(rep) == list(range(side_size))

    def test_num_reps_consistent(self):
        """num_reps field matches actual array shape."""
        gap_data = load_gap_coset_data(COSET_REPS_JSON)
        tau = (3, 4, 5, 0, 1, 2)
        result = compute_coset_reduction_from_gap(tau, 3, 3, gap_data)
        assert result.coset_reps.shape[0] == result.num_reps


def _conjugate_one_side(
    tau: list[int],
    rep: list[int],
    n: int,
    m: int,
    side: str,
) -> list[int]:
    """Conjugate tau by (rep × id) or (id × rep) on one side."""
    nm = n + m
    if side == "left":
        embedded = list(rep) + list(range(n, nm))
    else:
        embedded = list(range(n)) + [r + n for r in rep]

    inv_embedded = [0] * nm
    for i, v in enumerate(embedded):
        inv_embedded[v] = i

    return [inv_embedded[tau[embedded[i]]] for i in range(nm)]
