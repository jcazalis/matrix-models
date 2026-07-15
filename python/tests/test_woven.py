"""Tests for the woven contraction module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sym_contractions.store import ProbabilityStoreCollection
from sym_contractions.woven import (
    ContractionEntry,
    ContractionResult,
    WovenData,
    WovenEntry,
    WovenGroup,
    _try_rationalize,
    compute_all_contractions,
    compute_contraction_coefficients,
    export_for_mathematica,
    import_precomputed_contractions,
    involution_to_tau,
    load_woven_json,
    tau_to_involution,
    tau_to_pairs_1indexed,
)

# ---------------------------------------------------------------------------
# Shared paths and fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WC_DIR = PROJECT_ROOT / "data" / "processed" / "woven_contractions"
CT_DIR = PROJECT_ROOT / "data" / "processed" / "character_tables"
TEST_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "test-data"


@pytest.fixture
def wc_k2_path() -> Path:
    return TEST_DATA_DIR / "wc_K2_m1._Lambda6.json"


@pytest.fixture
def wc_k0_path() -> Path:
    return TEST_DATA_DIR / "wc_K0_m1._Lambda6.json"


@pytest.fixture
def wc_k4_path() -> Path:
    return TEST_DATA_DIR / "wc_K4_m1._Lambda6.json"


@pytest.fixture
def minimal_woven_json(tmp_path) -> Path:
    """Create a minimal woven JSON file that doesn't require real data files."""
    data = {
        "K": 2,
        "Lambda": 2,
        "description": "Test data",
        "woven_contractions": {
            "0_0": [
                {
                    "permutation": [],
                    "coefficient_poly_d": [[0, 1], [0, 1], [1, 2]],
                }
            ],
            "1_1": [
                {
                    "permutation": [3, 2, 1, 0],
                    "coefficient_poly_d": [[1, 1], [0, 1], [1, 2]],
                }
            ],
        },
    }
    filepath = tmp_path / "wc_K2_Lambda2.json"
    with open(filepath, "w") as f:
        json.dump(data, f)
    return filepath


# ======================================================================
# involution_to_tau
# ======================================================================


class TestInvolutionToTau:
    def test_empty(self):
        assert involution_to_tau([]) == ()

    def test_swap_n2(self):
        # Involution [3,2,1,0] on {0,1,2,3}: n=2, tau=(1,0)
        assert involution_to_tau([3, 2, 1, 0]) == (1, 0)

    def test_identity_n2(self):
        # Involution [2,3,0,1] on {0,1,2,3}: n=2, tau=(0,1)
        assert involution_to_tau([2, 3, 0, 1]) == (0, 1)

    def test_identity_n4(self):
        # Involution [6,7,4,5,2,3,0,1] on {0..7}: n=4, tau=(2,3,0,1)
        assert involution_to_tau([6, 7, 4, 5, 2, 3, 0, 1]) == (2, 3, 0, 1)

    def test_mixed_n4(self):
        # Involution [5,4,6,7,1,0,2,3]: n=4, tau=(1,0,2,3)
        assert involution_to_tau([5, 4, 6, 7, 1, 0, 2, 3]) == (1, 0, 2, 3)

    def test_output_is_valid_permutation(self):
        """Output must be a valid permutation of range(n)."""
        inv = [5, 4, 6, 7, 1, 0, 2, 3]
        tau = involution_to_tau(inv)
        n = len(inv) // 2
        assert sorted(tau) == list(range(n))

    def test_single_element(self):
        # n=1: involution [1,0] -> tau = (perm[0]-1,) = (0,) = identity in S_1
        assert involution_to_tau([1, 0]) == (0,)

    @pytest.mark.parametrize(
        "inv",
        [
            [3, 2, 1, 0],
            [2, 3, 0, 1],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [5, 4, 6, 7, 1, 0, 2, 3],
            [7, 6, 5, 4, 3, 2, 1, 0],
            [],
            [1, 0],
        ],
    )
    def test_involution_property(self, inv):
        """An involution applied twice should give the identity."""
        if len(inv) == 0:
            return
        reapplied = [inv[inv[i]] for i in range(len(inv))]
        assert reapplied == list(range(len(inv)))


# ======================================================================
# tau_to_pairs_1indexed
# ======================================================================


class TestTauToPairs1Indexed:
    def test_empty(self):
        assert tau_to_pairs_1indexed(()) == []

    def test_identity_n1(self):
        # n=1: tau=(0,) -> pairs=[[1, 2]]
        assert tau_to_pairs_1indexed((0,)) == [[1, 2]]

    def test_identity_n2(self):
        # n=2: tau=(0,1) -> [[1, 3], [2, 4]]
        assert tau_to_pairs_1indexed((0, 1)) == [[1, 3], [2, 4]]

    def test_swap_n2(self):
        # n=2: tau=(1,0) -> [[1, 4], [2, 3]]
        assert tau_to_pairs_1indexed((1, 0)) == [[1, 4], [2, 3]]

    def test_n4(self):
        # n=4: tau=(2,3,0,1)
        pairs = tau_to_pairs_1indexed((2, 3, 0, 1))
        assert pairs == [[1, 7], [2, 8], [3, 5], [4, 6]]

    def test_pair_ranges(self):
        """Lower indices ∈ {1..n}, upper indices ∈ {n+1..2n}."""
        for n in range(1, 6):
            tau = tuple(range(n))  # identity
            pairs = tau_to_pairs_1indexed(tau)
            lowers = [p[0] for p in pairs]
            uppers = [p[1] for p in pairs]
            assert lowers == list(range(1, n + 1))
            assert all(n + 1 <= u <= 2 * n for u in uppers)


# ======================================================================
# tau_to_involution (inverse of involution_to_tau)
# ======================================================================


class TestTauToInvolution:
    def test_empty(self):
        assert tau_to_involution(()) == []

    def test_swap_n2(self):
        assert tau_to_involution((1, 0)) == [3, 2, 1, 0]

    def test_identity_n2(self):
        assert tau_to_involution((0, 1)) == [2, 3, 0, 1]

    @pytest.mark.parametrize(
        "inv",
        [
            [],
            [1, 0],
            [3, 2, 1, 0],
            [2, 3, 0, 1],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [5, 4, 6, 7, 1, 0, 2, 3],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ],
    )
    def test_round_trip_involution_tau_involution(self, inv):
        """involution -> tau -> involution should be identity."""
        tau = involution_to_tau(inv)
        reconstructed = tau_to_involution(tau)
        assert reconstructed == inv

    @pytest.mark.parametrize(
        "tau",
        [
            (),
            (0,),
            (0, 1),
            (1, 0),
            (2, 3, 0, 1),
            (1, 0, 2, 3),
            (3, 2, 1, 0),
        ],
    )
    def test_round_trip_tau_involution_tau(self, tau):
        """tau -> involution -> tau should be identity."""
        inv = tau_to_involution(tau)
        reconstructed = involution_to_tau(inv)
        assert reconstructed == tau

    def test_output_is_involution(self):
        """tau_to_involution output must be an involution (self-inverse)."""
        tau = (2, 3, 0, 1)
        inv = tau_to_involution(tau)
        reapplied = [inv[inv[i]] for i in range(len(inv))]
        assert reapplied == list(range(len(inv)))


# ======================================================================
# load_woven_json
# ======================================================================


class TestLoadWovenJson:
    def test_load_minimal(self, minimal_woven_json):
        woven = load_woven_json(minimal_woven_json)
        assert woven.K == 2
        assert woven.Lambda == 2
        assert (0, 0) in woven.groups
        assert (1, 1) in woven.groups
        assert len(woven.groups) == 2

    def test_group_structure(self, minimal_woven_json):
        woven = load_woven_json(minimal_woven_json)
        g00 = woven.groups[(0, 0)]
        assert g00.nL == 0
        assert g00.nR == 0
        assert len(g00.entries) == 1
        assert g00.entries[0].tau == ()
        assert g00.entries[0].involution == ()
        assert g00.entries[0].pairs_1indexed == []

    def test_entry_tau_conversion(self, minimal_woven_json):
        woven = load_woven_json(minimal_woven_json)
        g11 = woven.groups[(1, 1)]
        entry = g11.entries[0]
        assert entry.involution == (3, 2, 1, 0)
        assert entry.tau == (1, 0)
        assert entry.pairs_1indexed == [[1, 4], [2, 3]]

    def test_coefficient_parsing(self, minimal_woven_json):
        woven = load_woven_json(minimal_woven_json)
        g00 = woven.groups[(0, 0)]
        # coeff should be [(0,1), (0,1), (1,2)] → 0 + 0*d + (1/2)*d^2
        assert g00.entries[0].coefficient_poly == [(0, 1), (0, 1), (1, 2)]

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K2_m1._Lambda6.json").exists(),
        reason="Real data files not available",
    )
    def test_load_real_k2(self, wc_k2_path):
        woven = load_woven_json(wc_k2_path)
        woven = woven.filter_by_max_excitations(2)
        assert woven.K == 2
        assert woven.Lambda == 2
        # K=2,Lambda=6 has groups: (0,0), (0,2), (1,1), (2,0), (2,2)
        expected_groups = {
            (0, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (2, 2),
        }
        assert set(woven.groups.keys()) == expected_groups

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K0_m1._Lambda6.json").exists(),
        reason="Real data files not available",
    )
    def test_load_real_k0(self, wc_k0_path):
        woven = load_woven_json(wc_k0_path)
        woven = woven.filter_by_max_excitations(0)
        assert woven.K == 0
        assert woven.Lambda == 0
        assert set(woven.groups.keys()) == {(0, 0)}
        assert len(woven.groups[(0, 0)].entries) == 1

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K4_m1._Lambda6.json").exists(),
        reason="Real data files not available",
    )
    def test_load_real_k4(self, wc_k4_path):
        woven = load_woven_json(wc_k4_path)
        assert woven.K == 4
        assert woven.Lambda == 6
        assert len(woven.groups) > 0
        # All taus must be valid permutations
        for (nL, nR), group in woven.groups.items():
            n = nL + nR
            for entry in group.entries:
                assert len(entry.tau) == n
                if n > 0:
                    assert sorted(entry.tau) == list(range(n))


# ======================================================================
# _try_rationalize
# ======================================================================


class TestTryRationalize:
    def test_integer(self):
        assert _try_rationalize(3.0) == (3, 1)

    def test_half(self):
        assert _try_rationalize(0.5) == (1, 2)

    def test_negative(self):
        assert _try_rationalize(-0.25) == (-1, 4)

    def test_zero(self):
        assert _try_rationalize(0.0) == (0, 1)

    def test_third(self):
        num, den = _try_rationalize(1.0 / 3.0)
        assert abs(num / den - 1 / 3) < 1e-10

    def test_irrational_approximation(self):
        """Should approximate irrational numbers within denominator bound."""
        num, den = _try_rationalize(np.pi, max_denom=1000)
        assert abs(num / den - np.pi) < 1e-3


# ======================================================================
# compute_contraction_coefficients
# ======================================================================


class TestComputeContractionCoefficients:
    def test_trivial_n0_m0(self):
        """n=0, m=0: single class, single rep, probs should give 1."""
        # S_0: 1 class, 1 rep, χ=1, weight=1
        ct_0 = {
            "n": 0,
            "cycle_types": [[]],
            "class_sizes": np.array([1], dtype=np.int32),
            "characters": np.array([[1]], dtype=np.int32),
        }
        # Probs for τ=() in S_0×S_0: ℓ=0 with probability 1
        # Shape (1, 1, 1)
        probs = np.array([[[1.0]]])
        result = compute_contraction_coefficients(probs, ct_0, ct_0, 0, 0)
        assert result.shape == (1, 1, 1)
        np.testing.assert_allclose(result[0, 0, 0], 1.0)

    def test_identity_tau_n1_m1(self):
        """For identity τ in S_1×S_1, contraction of id×id gives d."""
        # S_1: one class (1,), one rep, χ=1, class_size=1, weight=1/1!=1
        ct_1 = {
            "n": 1,
            "cycle_types": [[1]],
            "class_sizes": np.array([1], dtype=np.int32),
            "characters": np.array([[1]], dtype=np.int32),
        }
        # τ=identity=(0,1), σ=id in S_1, ν=id in S_1
        # τ∘(σ×ν) = id∘(id×id) = (0,1): 2 cycles → ℓ=2
        # But identity τ in S_1 is (0) with σ,ν each in S_1
        # Actually, n=1, m=1, so nm=2. τ=(0,1) means identity.
        # (σ×ν) with σ=(0), ν=(0) → (0,1).
        # τ∘(σ×ν) = (0,1)∘(0,1) = (0,1) → 2 fixed points → ℓ=2.
        # Probs shape (1, 1, 3) with probs[0,0,2] = 1.0
        probs = np.zeros((1, 1, 3))
        probs[0, 0, 2] = 1.0  # ℓ=2 with probability 1
        result = compute_contraction_coefficients(probs, ct_1, ct_1, 1, 1)
        # c_k(R=(1,), S=(1,)) = weight*χ*χ*P(ℓ=k)
        # weight = (1/1!)*(1/1!) = 1, χ=1
        # So c_2 = 1.0, all others = 0
        assert result.shape == (1, 1, 3)
        np.testing.assert_allclose(result[0, 0, 2], 1.0, atol=1e-12)
        np.testing.assert_allclose(result[0, 0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(result[0, 0, 1], 0.0, atol=1e-12)

    def test_shape(self):
        """Output shape should be (p_n, p_m, n+m+1)."""
        ct_3 = {
            "n": 3,
            "cycle_types": [[3], [2, 1], [1, 1, 1]],
            "class_sizes": np.array([2, 3, 1], dtype=np.int32),
            "characters": np.array([[1, 1, 1], [2, 0, -1], [1, -1, 1]], dtype=np.int32),
        }
        ct_2 = {
            "n": 2,
            "cycle_types": [[2], [1, 1]],
            "class_sizes": np.array([1, 1], dtype=np.int32),
            "characters": np.array([[1, 1], [1, -1]], dtype=np.int32),
        }
        probs = np.random.rand(3, 2, 6)  # (p_n=3, p_m=2, nm+1=6)
        result = compute_contraction_coefficients(probs, ct_3, ct_2, 3, 2)
        assert result.shape == (3, 2, 6)

    def test_uniform_probs_summing(self):
        """If probs uniform in k-space, result should still have correct shape
        and be deterministic."""
        ct_2 = {
            "n": 2,
            "cycle_types": [[2], [1, 1]],
            "class_sizes": np.array([1, 1], dtype=np.int32),
            "characters": np.array([[1, 1], [1, -1]], dtype=np.int32),
        }
        probs = np.ones((2, 2, 5)) / 5.0
        result = compute_contraction_coefficients(probs, ct_2, ct_2, 2, 2)
        assert result.shape == (2, 2, 5)
        # All k-entries should be equal for each (r, s) due to uniform probs
        for r in range(2):
            for s in range(2):
                np.testing.assert_allclose(
                    result[r, s, 0],
                    result[r, s, 1],
                    atol=1e-12,
                    err_msg=f"Non-uniform at r={r}, s={s}",
                )


# ======================================================================
# compute_all_contractions (integration tests)
# ======================================================================


class TestComputeAllContractions:
    def _build_collection_file(self, woven: WovenData, out_path: Path) -> Path:
        collection = ProbabilityStoreCollection.create(woven.Lambda)
        collection.compute_from_woven(woven, progress=False)
        collection.save(out_path)
        return out_path

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K0_m1._Lambda6.json").exists() or not CT_DIR.exists(),
        reason="Real data files not available",
    )
    def test_k0_trivial(self, wc_k0_path, tmp_path):
        """K=0, Lambda=6: only (0,0) group with τ=(), one (R,S) pair."""
        woven = load_woven_json(wc_k0_path)
        woven = woven.filter_by_max_excitations(0)
        store_path = self._build_collection_file(woven, tmp_path / "k0_collection.npz")
        result = compute_all_contractions(
            woven,
            CT_DIR,
            store_collection_path=store_path,
            verbose=False,
        )
        assert result.label == woven.label
        assert result.Lambda == 0
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry.R == ()
        assert entry.S == ()
        assert entry.pairs_1indexed == []
        # τ=() in S_0×S_0: ℓ=0 with prob 1 → c_0 = 1.0
        np.testing.assert_allclose(entry.coefficients[0], 1.0, atol=1e-10)

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K2_m1._Lambda6.json").exists() or not CT_DIR.exists(),
        reason="Real data files not available",
    )
    def test_k2_output_structure(self, wc_k2_path, tmp_path):
        """K=2, Lambda=6: check output structure is consistent."""
        woven = load_woven_json(wc_k2_path)
        store_path = self._build_collection_file(woven, tmp_path / "k2_collection.npz")
        result = compute_all_contractions(
            woven,
            CT_DIR,
            store_collection_path=store_path,
            verbose=False,
        )
        assert result.label == woven.label
        assert result.Lambda == 6
        assert len(result.entries) > 0

        for entry in result.entries:
            # We can at least check lengths are consistent
            assert len(entry.coefficients.shape) == 1

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K2_m1._Lambda6.json").exists() or not CT_DIR.exists(),
        reason="Real data files not available",
    )
    def test_k2_known_values(self, wc_k2_path, tmp_path):
        """Compare specific K=2 results against known Mathematica values.

        For K=2, the group (1,1) with τ=(1,0) (swap) has:
        - ComputeContractions[(1,), (1,), {pairs}] should give a polynomial in d.
        """
        woven = load_woven_json(wc_k2_path)
        store_path = self._build_collection_file(
            woven, tmp_path / "k2_collection_known.npz"
        )
        result = compute_all_contractions(
            woven,
            CT_DIR,
            store_collection_path=store_path,
            verbose=False,
        )

        # Find the entry for τ=(1,0), R=(1,), S=(1,)
        target = None
        for entry in result.entries:
            if entry.R == (1,) and entry.S == (1,):
                target = entry
                break

        assert target is not None, "Entry for R=(1,), S=(1,) not found"

        # From the JSON: (1,1) group, τ=(1,0), pairs=[[1,4],[2,3]]
        # Mathematica ComputeContractions for this gives d (coefficient of d^1 = 1)
        # because χ_{(1,)}(id)*χ_{(1,)}(id)/(1!*1!) * d^{ℓ(τ∘(id×id))}
        # τ=(1,0), σ×ν=(0,1), τ∘(σ×ν) = (1,0)∘(0,1) = (1,0) → 1 cycle → d^1
        np.testing.assert_allclose(target.coefficients[1], 1.0, atol=1e-10)

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K2_m1._Lambda6.json").exists() or not CT_DIR.exists(),
        reason="Real data files not available",
    )
    def test_k2_symmetry_0_2_vs_2_0(self, wc_k2_path, tmp_path):
        """Groups (0,2) and (2,0) should produce symmetric contraction results.

        For τ=(1,0): ComputeContractions[(), (2,)] should equal
        ComputeContractions[(2,), ()].
        """
        woven = load_woven_json(wc_k2_path)
        store_path = self._build_collection_file(
            woven, tmp_path / "k2_collection_symmetry.npz"
        )
        result = compute_all_contractions(
            woven,
            CT_DIR,
            store_collection_path=store_path,
            verbose=False,
        )

        # Find entries for the two symmetric cases
        entry_0_2 = None
        entry_2_0 = None
        for entry in result.entries:
            if entry.R == () and entry.S == (2,):
                entry_0_2 = entry
            if entry.R == (2,) and entry.S == ():
                entry_2_0 = entry

        assert entry_0_2 is not None
        assert entry_2_0 is not None

        # Check both are τ=(1,0) with equivalent but different pairs
        # The polynomial coefficients should match
        min_len = min(len(entry_0_2.coefficients), len(entry_2_0.coefficients))
        np.testing.assert_allclose(
            entry_0_2.coefficients[:min_len],
            entry_2_0.coefficients[:min_len],
            atol=1e-10,
        )


# ======================================================================
# export_for_mathematica / import_precomputed_contractions round-trip
# ======================================================================


class TestExportImport:
    def _make_sample_result(self) -> ContractionResult:
        """Create a small ContractionResult for testing."""
        entries = [
            ContractionEntry(
                pairs_1indexed=[[1, 4], [2, 3]],
                R=(1,),
                S=(1,),
                coefficients=np.array([0.0, 1.0]),
            ),
            ContractionEntry(
                pairs_1indexed=[[1, 7], [2, 8], [3, 5], [4, 6]],
                R=(2,),
                S=(2,),
                coefficients=np.array([0.0, 0.5, 0.5]),
            ),
            ContractionEntry(
                pairs_1indexed=[[1, 7], [2, 8], [3, 5], [4, 6]],
                R=(1, 1),
                S=(1, 1),
                coefficients=np.array([0.0, -0.5, 0.5]),
            ),
            # All-zero entry (should be skipped during export)
            ContractionEntry(
                pairs_1indexed=[[1, 7], [2, 8], [3, 5], [4, 6]],
                R=(2,),
                S=(1, 1),
                coefficients=np.array([0.0, 0.0, 0.0]),
            ),
        ]
        return ContractionResult(label="XX_p21", Lambda=2, entries=entries)

    def test_export_json_structure(self, tmp_path):
        result = self._make_sample_result()
        out = tmp_path / "test.json"
        export_for_mathematica(result, out)

        with open(out) as f:
            data = json.load(f)

        assert data["label"] == "XX_p21"
        assert data["Lambda"] == 2
        assert "precomputed_contractions" in data
        # All-zero entry should be skipped → 3 entries
        assert len(data["precomputed_contractions"]) == 3

    def test_export_trims_trailing_zeros(self, tmp_path):
        """Entries with trailing zero coefficients should be trimmed."""
        entries = [
            ContractionEntry(
                pairs_1indexed=[[1, 3], [2, 4]],
                R=(1,),
                S=(1,),
                coefficients=np.array([0.5, 1.0, 0.0, 0.0, 0.0]),
            ),
        ]
        result = ContractionResult(label="XX_p21", Lambda=2, entries=entries)
        out = tmp_path / "test.json"
        export_for_mathematica(result, out)

        with open(out) as f:
            data = json.load(f)

        exported = data["precomputed_contractions"][0]
        # Should have 2 coefficients, not 5
        assert len(exported["coefficients"]) == 2

    def test_export_rationalize(self, tmp_path):
        """With rationalize=True, coefficients should be exact rationals."""
        entries = [
            ContractionEntry(
                pairs_1indexed=[[1, 3], [2, 4]],
                R=(1,),
                S=(1,),
                coefficients=np.array([0.5, 0.25]),
            ),
        ]
        result = ContractionResult(label="XX_p21", Lambda=2, entries=entries)
        out = tmp_path / "test.json"
        export_for_mathematica(result, out, rationalize=True)

        with open(out) as f:
            data = json.load(f)

        coeffs = data["precomputed_contractions"][0]["coefficients"]
        assert coeffs[0] == [1, 2]  # 0.5 = 1/2
        assert coeffs[1] == [1, 4]  # 0.25 = 1/4

    def test_export_no_rationalize(self, tmp_path):
        """With rationalize=False, coefficients stored as [value, 1]."""
        entries = [
            ContractionEntry(
                pairs_1indexed=[[1, 3], [2, 4]],
                R=(1,),
                S=(1,),
                coefficients=np.array([0.5]),
            ),
        ]
        result = ContractionResult(label="XX_p21", Lambda=2, entries=entries)
        out = tmp_path / "test.json"
        export_for_mathematica(result, out, rationalize=False)

        with open(out) as f:
            data = json.load(f)

        coeffs = data["precomputed_contractions"][0]["coefficients"]
        assert coeffs[0] == [0.5, 1]

    def test_round_trip(self, tmp_path):
        """Export and re-import should preserve data."""
        result = self._make_sample_result()
        out = tmp_path / "test.json"
        export_for_mathematica(result, out)

        reimported = import_precomputed_contractions(out)

        # 3 non-zero entries
        assert len(reimported) == 3

        # Check a specific entry
        key = (((1, 4), (2, 3)), (1,), (1,))
        assert key in reimported
        np.testing.assert_allclose(reimported[key], [0.0, 1.0], atol=1e-12)

    def test_round_trip_preserves_rationals(self, tmp_path):
        """Export with rationalization and re-import should be numerically exact."""
        entries = [
            ContractionEntry(
                pairs_1indexed=[[1, 4], [2, 3]],
                R=(1,),
                S=(1,),
                coefficients=np.array([1 / 3, 1 / 6, -1 / 12]),
            ),
        ]
        result = ContractionResult(label="XX_p21", Lambda=2, entries=entries)
        out = tmp_path / "test.json"
        export_for_mathematica(result, out, rationalize=True)

        reimported = import_precomputed_contractions(out)
        key = (((1, 4), (2, 3)), (1,), (1,))
        np.testing.assert_allclose(reimported[key], [1 / 3, 1 / 6, -1 / 12], atol=1e-12)

    def test_export_creates_directory(self, tmp_path):
        """Export should create parent directories if they don't exist."""
        result = self._make_sample_result()
        out = tmp_path / "nested" / "deep" / "test.json"
        export_for_mathematica(result, out)
        assert out.exists()

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K2_Lambda6.json").exists() or not CT_DIR.exists(),
        reason="Real data files not available",
    )
    def test_full_pipeline_round_trip(self, wc_k2_path, tmp_path):
        """Full pipeline: load JSON → compute → export → reimport."""
        woven = load_woven_json(wc_k2_path)
        collection_path = tmp_path / "k2_collection_pipeline.npz"
        ProbabilityStoreCollection.create(woven.Lambda).compute_from_woven(
            woven, progress=False
        ).save(collection_path)
        result = compute_all_contractions(
            woven,
            CT_DIR,
            store_collection_path=collection_path,
            verbose=False,
        )

        out = tmp_path / "precomputed_K2.json"
        export_for_mathematica(result, out)
        reimported = import_precomputed_contractions(out)

        assert len(reimported) > 0

        # Every reimported entry should have a valid coefficient array
        for key, coeffs in reimported.items():
            pairs_key, R, S = key
            assert isinstance(pairs_key, tuple)
            assert isinstance(R, tuple)
            assert isinstance(S, tuple)
            assert isinstance(coeffs, np.ndarray)
            assert coeffs.dtype == np.float64


# ======================================================================
# Edge cases and consistency
# ======================================================================


class TestEdgeCases:
    def test_involution_to_tau_list_input(self):
        """Should accept lists, not just tuples."""
        assert involution_to_tau([3, 2, 1, 0]) == (1, 0)

    def test_involution_to_tau_numpy_input(self):
        """Should accept numpy arrays."""
        inv = np.array([3, 2, 1, 0])
        assert involution_to_tau(inv) == (1, 0)  # type: ignore

    def test_tau_consistency_with_mathematica_pairs(self):
        """Verify that our pairs format matches Mathematica's permutation0IndexedToPairs.

        In Mathematica:
            permutation0IndexedToPairs[{3, 2, 1, 0}]
            → Table[{i, perm[[i]] + 1}, {i, 1, n}] where n = Length[perm]/2
            = {{1, 4}, {2, 3}}  (1-indexed)

        Our conversion:
            inv = [3, 2, 1, 0]
            tau = involution_to_tau(inv) = (1, 0)
            pairs = tau_to_pairs_1indexed(tau) = [[1, 4], [2, 3]]  ✓
        """
        inv = [3, 2, 1, 0]  # n=2
        tau = involution_to_tau(inv)
        pairs = tau_to_pairs_1indexed(tau)
        # Mathematica's permutation0IndexedToPairs does:
        # for i in 1..n: {i, perm[i-1] + 1}
        n = len(inv) // 2
        mathematica_pairs = [[i + 1, inv[i] + 1] for i in range(n)]
        assert pairs == mathematica_pairs

    @pytest.mark.parametrize(
        "inv",
        [
            [2, 3, 0, 1],
            [3, 2, 1, 0],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [5, 4, 6, 7, 1, 0, 2, 3],
        ],
    )
    def test_pairs_match_mathematica_convention(self, inv):
        """For any involution, our pairs should match Mathematica's convention."""
        n = len(inv) // 2
        tau = involution_to_tau(inv)
        pairs = tau_to_pairs_1indexed(tau)
        mathematica_pairs = [[i + 1, inv[i] + 1] for i in range(n)]
        assert pairs == mathematica_pairs

    def test_woven_entry_dataclass(self):
        """WovenEntry should store all fields correctly."""
        entry = WovenEntry(
            involution=(3, 2, 1, 0),
            tau=(1, 0),
            pairs_1indexed=[[1, 4], [2, 3]],
            coefficient_poly=[(1, 1), (0, 1), (1, 2)],
        )
        assert entry.involution == (3, 2, 1, 0)
        assert entry.tau == (1, 0)
        assert entry.pairs_1indexed == [[1, 4], [2, 3]]
        assert entry.coefficient_poly == [(1, 1), (0, 1), (1, 2)]

    def test_woven_data_dataclass(self):
        """WovenData should store metadata correctly."""
        data = WovenData(
            operators="XXXX", trace_permutation=(1, 2, 3, 0), Lambda=12, groups={}
        )
        assert data.K == 4
        assert data.Lambda == 12
        assert len(data.groups) == 0

    def test_contraction_entry_dataclass(self):
        """ContractionEntry should store coefficients as ndarray."""
        entry = ContractionEntry(
            pairs_1indexed=[[1, 3], [2, 4]],
            R=(2,),
            S=(1, 1),
            coefficients=np.array([0.0, 0.5, 0.5]),
        )
        assert isinstance(entry.coefficients, np.ndarray)
        assert entry.coefficients.shape == (3,)

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K4_m1._Lambda6.json").exists(),
        reason="Real data files not available",
    )
    def test_filter_by_max_excitations(self, wc_k4_path: Path):
        """Filter should keep only groups with nL, nR <= max_excitations."""
        woven = load_woven_json(wc_k4_path)
        original_count = len(woven.groups)

        # Filter to max 4 excitations
        filtered = woven.filter_by_max_excitations(4)

        # Check all groups satisfy constraint
        for nL, nR in filtered.groups.keys():
            assert nL <= 4, f"nL={nL} exceeds max_excitations=4"
            assert nR <= 4, f"nR={nR} exceeds max_excitations=4"

        # Check Lambda is updated
        assert filtered.Lambda == 4

        # Check K is preserved
        assert filtered.K == woven.K

        # Check we have fewer groups
        assert len(filtered.groups) <= original_count

    def test_filter_by_max_excitations_empty(self):
        """Filter with max_excitations below all groups returns empty."""
        groups = {
            (2, 2): WovenGroup(nL=2, nR=2, entries=[]),
            (3, 3): WovenGroup(nL=3, nR=3, entries=[]),
        }
        woven = WovenData(
            operators="XXXX", trace_permutation=(1, 2, 3, 0), Lambda=10, groups=groups
        )

        # Filter to max 1 should remove all groups
        filtered = woven.filter_by_max_excitations(1)
        assert len(filtered.groups) == 0
        assert filtered.Lambda == 1

    @pytest.mark.skipif(
        not (TEST_DATA_DIR / "wc_K2_m1._Lambda6.json").exists(),
        reason="Real data files not available",
    )
    def test_filter_by_max_excitations_preserves_entries(self, wc_k2_path: Path):
        """Filter should preserve WovenEntry objects unchanged."""
        woven = load_woven_json(wc_k2_path)
        filtered = woven.filter_by_max_excitations(100)  # Keep all

        # Should be identical since all groups pass filter
        assert len(filtered.groups) == len(woven.groups)
        for key in woven.groups:
            original_entries = woven.groups[key].entries
            filtered_entries = filtered.groups[key].entries
            assert original_entries == filtered_entries
