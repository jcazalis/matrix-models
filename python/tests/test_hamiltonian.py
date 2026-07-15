"""Tests for the hamiltonian module (Observable / Hamiltonian API)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from sym_contractions.hamiltonian import (
    EvaluatedObservable,
    FreeHamiltonian,
    HamiltonianDense,
    HamiltonianSparse,
    Observable,
    ObservableDense,
    ObservableSparse,
    _horner_evaluate,
    _label_to_ops_spec,
    _label_to_woven_filename,
    _mass_to_filename_str,
    _resolve_coupling,
    normalization,
    partition_list,
)
from sym_contractions.woven import (
    ContractionEntry,
    ContractionResult,
    WovenData,
    WovenEntry,
    WovenGroup,
    import_precomputed_contractions,
    load_woven_json,
)

# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WC_DIR = PROJECT_ROOT / "data" / "processed" / "woven_contractions"


# ======================================================================
# partition_list
# ======================================================================


class TestPartitionList:
    def test_lambda_0(self):
        assert partition_list(0) == [()]

    def test_lambda_1(self):
        assert partition_list(1) == [(), (1,)]

    def test_lambda_2(self):
        assert partition_list(2) == [(), (1,), (2,), (1, 1)]

    def test_lambda_3(self):
        expected = [(), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1)]
        assert partition_list(3) == expected

    def test_lambda_4_length(self):
        # p(0)=1, p(1)=1, p(2)=2, p(3)=3, p(4)=5 → total 12
        assert len(partition_list(4)) == 12

    def test_partitions_sorted_by_excitation(self):
        """Partitions are grouped by excitation level (sum)."""
        basis = partition_list(6)
        excitations = [sum(R) for R in basis]
        assert excitations == sorted(excitations)


# ======================================================================
# normalization
# ======================================================================


class TestNormalization:
    def test_empty_partition(self):
        assert normalization((), 5) == 1.0

    def test_single_box(self):
        assert normalization((1,), 5) == 5.0
        assert normalization((1,), 10) == 10.0

    def test_partition_2(self):
        assert normalization((2,), 3) == 3 * 4

    def test_partition_11(self):
        assert normalization((1, 1), 3) == 3 * 2

    def test_partition_21(self):
        assert normalization((2, 1), 3) == 3 * 4 * 2

    def test_partition_111(self):
        assert normalization((1, 1, 1), 3) == 3 * 2 * 1

    def test_partition_22(self):
        assert normalization((2, 2), 3) == 3 * 4 * 2 * 3

    def test_positive_when_d_ge_num_rows(self):
        for R in partition_list(5):
            d = max(len(R), 1)
            assert normalization(R, d) > 0


# ======================================================================
# Fixtures: synthetic data for assembly tests
# ======================================================================


def _make_synthetic_K2_Lambda2() -> tuple[ContractionResult, WovenData]:
    """Create synthetic K=2, Lambda=2 data for testing.

    Basis: [(), (1,), (2,), (1,1)]

    WovenData has two groups:
    - (0, 0): one entry with pairs=[], coeff_poly = d²/2 → [0, 0, 1/2]
    - (1, 1): one entry with pairs=[[1,3],[2,4]] (identity tau),
              coeff_poly = 1 + d² → [1, 0, 1]

    ContractionResult: we make contraction entries for the (pairs, R, S)
    combinations that exist.
    """
    we_00 = WovenEntry(
        involution=(),
        tau=(),
        pairs_1indexed=[],
        coefficient_poly=[(0, 1), (0, 1), (1, 2)],  # d²/2
    )
    wg_00 = WovenGroup(nL=0, nR=0, entries=[we_00])

    we_11 = WovenEntry(
        involution=(1, 0),
        tau=(0,),
        pairs_1indexed=[[1, 2]],
        coefficient_poly=[(1, 1)],  # constant 1
    )
    wg_11 = WovenGroup(nL=1, nR=1, entries=[we_11])

    woven = WovenData(
        operators="XX",
        trace_permutation=(1, 0),
        Lambda=2,
        groups={(0, 0): wg_00, (1, 1): wg_11},
        mass=1,
        is_even=True,
    )

    entries: list[ContractionEntry] = []
    entries.append(
        ContractionEntry(
            pairs_1indexed=[],
            R=(),
            S=(),
            coefficients=np.array([1.0]),
        )
    )
    entries.append(
        ContractionEntry(
            pairs_1indexed=[[1, 2]],
            R=(1,),
            S=(1,),
            coefficients=np.array([0.0, 1.0]),  # d
        )
    )
    contraction = ContractionResult(label="XX_p21", Lambda=2, entries=entries)
    return contraction, woven


def _make_synthetic_complex_term() -> tuple[ContractionResult, WovenData]:
    """Create a minimal term with purely imaginary woven coefficients."""
    we_00 = WovenEntry(
        involution=(),
        tau=(),
        pairs_1indexed=[],
        coefficient_poly=[(0, 1, 1, 1)],  # +i
    )
    wg_00 = WovenGroup(nL=0, nR=0, entries=[we_00])

    woven = WovenData(
        operators="XXP",
        trace_permutation=(1, 2, 0),
        Lambda=1,
        groups={(0, 0): wg_00},
        mass=1,
        is_even=False,
        is_hermitian=False,
    )

    contraction = ContractionResult(
        label="XXP_p231",
        Lambda=1,
        entries=[
            ContractionEntry(
                pairs_1indexed=[],
                R=(),
                S=(),
                coefficients=np.array([1.0]),
            )
        ],
    )
    return contraction, woven


def _make_observable(label: str = "PP", n_basis: int = 4) -> ObservableDense:
    """Create a simple dense Observable for testing (PP-like)."""
    coeffs = np.zeros((2, n_basis, n_basis))
    coeffs[0, 0, 0] = 1.0  # constant term
    coeffs[1, 1, 1] = 2.0  # linear in d
    return ObservableDense(
        label=label,
        Lambda=2,
        reference_mass=1.0,
        basis=partition_list(2),
        unnorm_coeffs=coeffs,
        is_even=True,
        is_hermitian=True,
    )


# ======================================================================
# Observable.from_data
# ======================================================================


class TestObservableFromData:
    def test_synthetic_basis_size(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        assert obs.size == 4

    def test_synthetic_symmetry(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        for k in range(obs.unnorm_coeffs.shape[0]):
            m = obs.unnorm_coeffs[k]
            np.testing.assert_array_almost_equal(m, m.T)

    def test_synthetic_00_element(self):
        """H_int_unnorm[0,0] should be convolve([0,0,0.5], [1]) = [0,0,0.5]."""
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        assert obs.unnorm_coeffs[0, 0, 0] == pytest.approx(0.0)
        assert obs.unnorm_coeffs[1, 0, 0] == pytest.approx(0.0)
        assert obs.unnorm_coeffs[2, 0, 0] == pytest.approx(0.5)

    def test_synthetic_11_element(self):
        """H_int_unnorm[1,1] should be convolve([1], [0,1]) = [0,1]."""
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        assert obs.unnorm_coeffs[0, 1, 1] == pytest.approx(0.0)
        assert obs.unnorm_coeffs[1, 1, 1] == pytest.approx(1.0)

    def test_synthetic_missing_entries_are_zero(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        for k in range(obs.unnorm_coeffs.shape[0]):
            assert obs.unnorm_coeffs[k, 0, 2] == 0.0
            assert obs.unnorm_coeffs[k, 0, 3] == 0.0

    def test_label_and_metadata(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        assert obs.label == "XX_p21"
        assert obs.Lambda == 2
        assert obs.reference_mass == 1.0
        assert obs.is_even is True
        assert obs.is_hermitian is None  # not set in the fixture

    def test_K_property(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        assert obs.K == 2

    def test_filename_property(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        assert obs.filename == "XX_p21_Lambda2.npz"


# ======================================================================
# Observable evaluation
# ======================================================================


class TestObservableEvaluate:
    @pytest.fixture
    def obs(self) -> ObservableDense:
        contraction, woven = _make_synthetic_K2_Lambda2()
        return ObservableDense.from_data(contraction, woven)

    def test_returns_evaluated_observable(self, obs):
        result = obs.evaluate(d=5)
        assert isinstance(result, EvaluatedObservable)
        assert result.d_values == [5]
        assert result.mass == 1.0
        assert result.label == "XX_p21"
        assert result.Lambda == 2
        assert result.ground_state_only is False
        assert result.is_even is True
        assert result.is_hermitian is None

    def test_matrix_shape(self, obs):
        result = obs.evaluate(d=5)
        assert result.matrices[5].shape == (4, 4)

    def test_matrix_symmetric(self, obs):
        result = obs.evaluate(d=5)
        np.testing.assert_array_almost_equal(result.matrices[5], result.matrices[5].T)

    def test_d_filter_removes_tall_partitions(self, obs):
        result = obs.evaluate(d=1)
        assert len(result.bases[1]) == 3
        assert all(len(R) <= 1 for R in result.bases[1])

    def test_ground_state_filters_odd_excitations(self, obs):
        result = obs.evaluate(d=100, ground_state_only=True)
        assert len(result.bases[100]) == 3
        for R in result.bases[100]:
            assert sum(R) % 2 == 0

    def test_combined_filter(self, obs):
        result = obs.evaluate(d=1, ground_state_only=True)
        assert result.bases[1] == [(), (2,)]
        assert result.matrices[1].shape == (2, 2)

    def test_mass_override(self, obs):
        result = obs.evaluate(d=5, mass=2.0)
        assert result.mass == 2.0

    def test_batch_d_returns_dicts(self, obs):
        result = obs.evaluate(d=[1, 5])
        assert result.d_values == [1, 5]
        assert set(result.matrices.keys()) == {1, 5}
        assert set(result.bases.keys()) == {1, 5}
        assert result.matrices[1].shape == (3, 3)
        assert result.matrices[5].shape == (4, 4)

    def test_singleton_batch_matches_scalar(self, obs):
        scalar = obs.evaluate(d=5)
        batched = obs.evaluate(d=[5])
        np.testing.assert_array_almost_equal(scalar.matrices[5], batched.matrices[5])
        assert scalar.bases[5] == batched.bases[5]

    def test_complex_coefficients(self):
        contraction, woven = _make_synthetic_complex_term()
        obs = ObservableDense.from_data(contraction, woven)
        assert np.iscomplexobj(obs.unnorm_coeffs)
        result = obs.evaluate(d=3)
        assert np.iscomplexobj(result.matrices[3])


class TestObservableSparseEvaluate:
    @pytest.fixture
    def obs(self) -> ObservableSparse:
        contraction, woven = _make_synthetic_K2_Lambda2()
        return ObservableSparse.from_data(contraction, woven)

    def test_sparse_filename_property(self, obs):
        assert obs.filename == "XX_p21_Lambda2_sparse.npz"

    def test_sparse_evaluate_returns_csr(self, obs):
        result = obs.evaluate(d=5)
        assert sparse.isspmatrix_csr(result.matrices[5])

    def test_sparse_matches_dense(self, obs):
        contraction, woven = _make_synthetic_K2_Lambda2()
        dense_obs = ObservableDense.from_data(contraction, woven)
        sparse_eval = obs.evaluate(d=5).matrices[5].toarray()
        dense_eval = dense_obs.evaluate(d=5).matrices[5]
        np.testing.assert_array_almost_equal(sparse_eval, dense_eval)

    def test_sparse_parallel_matches_serial(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        serial = ObservableSparse.from_data(contraction, woven, parallel=False)
        parallel = ObservableSparse.from_data(
            contraction,
            woven,
            parallel=True,
            max_workers=2,
        )

        assert parallel.degree_count() == serial.degree_count()
        for serial_coeff, parallel_coeff in zip(
            serial.unnorm_coeffs,
            parallel.unnorm_coeffs,
            strict=True,
        ):
            np.testing.assert_array_almost_equal(
                parallel_coeff.toarray(), serial_coeff.toarray()
            )

    def test_sparse_handles_shuffled_sector_entries(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        shuffled = ContractionResult(
            label=contraction.label,
            Lambda=contraction.Lambda,
            entries=list(reversed(contraction.entries)),
        )

        ordered_obs = ObservableSparse.from_data(contraction, woven, parallel=False)
        shuffled_obs = ObservableSparse.from_data(shuffled, woven, parallel=False)

        for ordered_coeff, shuffled_coeff in zip(
            ordered_obs.unnorm_coeffs,
            shuffled_obs.unnorm_coeffs,
            strict=True,
        ):
            np.testing.assert_array_almost_equal(
                shuffled_coeff.toarray(), ordered_coeff.toarray()
            )


# =====================================================================
# EvaluatedObservable merge
# =====================================================================


class TestEvaluatedObservableMerge:
    def _make_eval(
        self,
        *,
        d_values: list[int],
        label: str = "XX_p21",
        mass: float = 1.0,
        Lambda: int = 2,
        ground_state_only: bool = False,
        is_even: bool | None = True,
        is_hermitian: bool | None = True,
    ) -> EvaluatedObservable:
        matrices = {d: np.eye(2) * d for d in d_values}
        bases = {d: [(), (1,)] for d in d_values}
        return EvaluatedObservable(
            matrices=matrices,
            bases=bases,
            d_values=list(d_values),
            mass=mass,
            label=label,
            Lambda=Lambda,
            ground_state_only=ground_state_only,
            is_even=is_even,
            is_hermitian=is_hermitian,
        )

    def test_merge_non_overlapping(self):
        a = self._make_eval(d_values=[3])
        b = self._make_eval(d_values=[5])
        merged = a.merge(b)
        assert merged.d_values == [3, 5]
        assert set(merged.matrices.keys()) == {3, 5}

    def test_merge_overlapping_identical_ok(self):
        a = self._make_eval(d_values=[3])
        b = self._make_eval(d_values=[3, 5])
        merged = a.merge(b)
        assert merged.d_values == [3, 5]
        np.testing.assert_array_almost_equal(merged.matrices[3], np.eye(2) * 3)

    def test_merge_rejects_different_lambda(self):
        a = self._make_eval(d_values=[3], Lambda=2)
        b = self._make_eval(d_values=[5], Lambda=3)
        with pytest.raises(ValueError, match="different Lambda"):
            a.merge(b)

    def test_merge_rejects_different_ground_state_only(self):
        a = self._make_eval(d_values=[3], ground_state_only=False)
        b = self._make_eval(d_values=[5], ground_state_only=True)
        with pytest.raises(ValueError, match="different ground_state_only"):
            a.merge(b)

    def test_merge_rejects_different_is_even(self):
        a = self._make_eval(d_values=[3], is_even=True)
        b = self._make_eval(d_values=[5], is_even=False)
        with pytest.raises(ValueError, match="different is_even"):
            a.merge(b)

    def test_merge_rejects_different_is_hermitian(self):
        a = self._make_eval(d_values=[3], is_hermitian=True)
        b = self._make_eval(d_values=[5], is_hermitian=False)
        with pytest.raises(ValueError, match="different is_hermitian"):
            a.merge(b)

    def test_merge_rejects_overlapping_matrix_mismatch(self):
        a = self._make_eval(d_values=[3])
        b = self._make_eval(d_values=[3])
        b.matrices[3] = np.eye(2) * 999
        with pytest.raises(ValueError, match="different matrix values"):
            a.merge(b)


class TestEvaluatedObservableExpectation:
    def _make_eval(self) -> EvaluatedObservable:
        op = np.array([[2.0, 1.0j], [-1.0j, 3.0]], dtype=np.complex128)
        return EvaluatedObservable(
            matrices={5: op},
            bases={5: [(), (1,)]},
            d_values=[5],
            mass=1.0,
            label="O",
            Lambda=2,
            ground_state_only=False,
            is_even=None,
            is_hermitian=True,
        )

    def test_pure_state_expectation(self):
        ev = self._make_eval()
        psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        val = ev.expectation_value(5, psi)
        assert val == pytest.approx(2.0 + 0.0j)
        assert isinstance(val, float)

    def test_mixed_state_expectation(self):
        ev = self._make_eval()
        rho = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex128)
        val = ev.expectation_value(5, rho)
        assert val == pytest.approx(2.75 + 0.0j)
        assert isinstance(val, float)

    def test_overlap_two_pure_states(self):
        ev = self._make_eval()
        psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        phi = np.array([0.0 + 0.0j, 1.0 + 0.0j])
        val = ev.expectation_value(5, psi, bra_state=phi)
        assert val == pytest.approx(-1.0j)
        assert isinstance(val, complex)

    def test_missing_d_raises(self):
        ev = self._make_eval()
        with pytest.raises(KeyError, match="no cached matrix"):
            ev.expectation_value(6, np.array([1.0, 0.0]))

    def test_incompatible_state_size_raises(self):
        ev = self._make_eval()
        with pytest.raises(ValueError, match="incompatible"):
            ev.expectation_value(5, np.array([1.0, 0.0, 0.0]))

    def test_overlap_with_matrix_state_raises(self):
        ev = self._make_eval()
        rho = np.eye(2)
        with pytest.raises(ValueError, match="1-D vector"):
            ev.expectation_value(5, rho, bra_state=np.array([1.0, 0.0]))


# ======================================================================
# Observable properties
# ======================================================================


class TestObservableProperties:
    def test_n_x_n_p_counts(self):
        obs = _make_observable("XPXP")
        assert obs.n_x == 2
        assert obs.n_p == 2

    def test_mass_exponent_pure_x(self):
        obs = _make_observable("XXXX_p2341")
        assert obs.mass_exponent == -2.0

    def test_mass_exponent_mixed(self):
        obs = _make_observable("XP_p21")
        assert obs.mass_exponent == 0.0

    def test_mass_exponent_pure_p(self):
        obs = _make_observable("PP_p21")
        assert obs.mass_exponent == 1.0

    def test_K_property(self):
        obs = _make_observable("XPXP_p2341")
        assert obs.K == 4

    def test_rescaled_coeffs_identity(self):
        coeffs = np.random.default_rng(42).random((3, 4, 4))
        obs = ObservableDense(
            label="XXXX_p2341",
            Lambda=2,
            reference_mass=0.5,
            basis=partition_list(2),
            unnorm_coeffs=coeffs,
        )
        result = obs.rescaled_coeffs(0.5)
        assert result is coeffs

    def test_rescaled_coeffs_pure_x(self):
        coeffs = np.ones((2, 4, 4))
        obs = ObservableDense(
            label="XXXX_p2341",
            Lambda=2,
            reference_mass=1.0,
            basis=partition_list(2),
            unnorm_coeffs=coeffs,
        )
        result = obs.rescaled_coeffs(2.0)
        expected = coeffs * (2.0 / 1.0) ** (-2.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rescaled_coeffs_balanced(self):
        coeffs = np.ones((2, 4, 4)) * 3.0
        obs = ObservableDense(
            label="XP_p21",
            Lambda=2,
            reference_mass=0.5,
            basis=partition_list(2),
            unnorm_coeffs=coeffs,
        )
        result = obs.rescaled_coeffs(42.0)
        np.testing.assert_array_almost_equal(result, coeffs)

    def test_max_degree_zero_polynomial(self):
        coeffs = np.zeros((5, 4, 4))
        obs = ObservableDense(
            label="XX_p21",
            Lambda=2,
            reference_mass=1.0,
            basis=partition_list(2),
            unnorm_coeffs=coeffs,
        )
        assert obs.max_degree == 0

    def test_max_degree_ignores_trailing_zero_dense_slices(self):
        coeffs = np.zeros((5, 4, 4))
        coeffs[2, 1, 1] = 1.0
        obs = ObservableDense(
            label="XX_p21",
            Lambda=2,
            reference_mass=1.0,
            basis=partition_list(2),
            unnorm_coeffs=coeffs,
        )
        assert obs.max_degree == 2

    def test_filtered_basis(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        fb = obs.filtered_basis(d=1)
        assert fb == [(), (1,), (2,)]


# ======================================================================
# Horner evaluation
# ======================================================================


class TestHornerEvaluation:
    def test_evaluate_matches_direct_summation(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)
        d = 7
        direct = np.zeros_like(obs.unnorm_coeffs[0])
        for k in range(obs.unnorm_coeffs.shape[0]):
            direct += d**k * obs.unnorm_coeffs[k]
        horner = _horner_evaluate(obs.unnorm_coeffs, d)
        np.testing.assert_array_almost_equal(horner, direct)

    def test_evaluate_with_indices(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)
        d = 7
        indices = np.array([0, 1])
        sub = _horner_evaluate(obs.unnorm_coeffs, d, indices)
        full = _horner_evaluate(obs.unnorm_coeffs, d)
        np.testing.assert_array_almost_equal(sub, full[:2, :2])

    def test_sparse_evaluate_matches_direct_summation(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableSparse.from_data(contraction, woven)
        d = 7
        direct = np.zeros((4, 4), dtype=np.float64)
        for k, coeff in enumerate(obs.unnorm_coeffs):
            direct += d**k * coeff.toarray()
        horner = _horner_evaluate(obs.unnorm_coeffs, d)
        assert sparse.isspmatrix_csr(horner)
        np.testing.assert_array_almost_equal(horner.toarray(), direct)


# ======================================================================
# Integration with real data files (skipped if files are missing)
# ======================================================================


def _has_real_data() -> bool:
    wc = WC_DIR / "wc_K4_Lambda14.json"
    pc = WC_DIR / "precomputed_K4_Lambda4.json"
    return wc.exists() and pc.exists()


@pytest.mark.skipif(not _has_real_data(), reason="Real data files not available")
class TestIntegrationRealData:
    @pytest.fixture
    def obs(self) -> Observable:
        wc_path = WC_DIR / "wc_K4_Lambda14.json"
        pc_path = WC_DIR / "precomputed_K4_Lambda4.json"

        woven = load_woven_json(wc_path)
        woven = woven.filter_by_max_excitations(4)

        precomputed = import_precomputed_contractions(pc_path)

        entries: list[ContractionEntry] = []
        for (pairs, R, S), coeffs in precomputed.items():
            entries.append(
                ContractionEntry(
                    pairs_1indexed=[list(p) for p in pairs],
                    R=R,
                    S=S,
                    coefficients=coeffs,
                )
            )
        contraction = ContractionResult(label=woven.label, Lambda=4, entries=entries)
        return ObservableDense.from_data(contraction, woven)

    def test_basis_size(self, obs):
        assert obs.size == 12

    def test_symmetry(self, obs):
        for k in range(obs.unnorm_coeffs.shape[0]):
            m = obs.unnorm_coeffs[k]
            np.testing.assert_array_almost_equal(
                m, m.T, err_msg=f"Asymmetric at degree {k}"
            )

    def test_evaluated_symmetry(self, obs):
        for d in [5, 10, 20]:
            result = obs.evaluate(d=d)
            np.testing.assert_array_almost_equal(
                result.matrices[d],
                result.matrices[d].T,
                err_msg=f"Asymmetric evaluation at d={d}",
            )

    def test_d_filtering(self, obs):
        result = obs.evaluate(d=2)
        for R in result.bases[2]:
            assert len(R) <= 2

    def test_ground_state_filtering(self, obs):
        result = obs.evaluate(d=10, ground_state_only=True)
        for R in result.bases[10]:
            assert sum(R) % 2 == 0

    def test_nonzero_entries(self, obs):
        result = obs.evaluate(d=10)
        assert np.any(result.matrices[10] != 0)

    def test_eigenvalues_are_real(self, obs):
        # Build a Hamiltonian to test eigenvalues
        ham = HamiltonianDense(
            observables=[obs],
            Lambda=obs.Lambda,
            default_mass=obs.reference_mass,
        )
        H = ham.evaluate(d=10, coupling={"XXXX_p2341": lambda d: 1.0 / d})
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(np.isreal(eigenvalues))


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_empty_woven(self):
        woven = WovenData(
            operators="XX", trace_permutation=(1, 0), Lambda=2, groups={}, mass=1
        )
        contraction = ContractionResult(label="XX_p21", Lambda=2, entries=[])
        obs = Observable.from_data(contraction, woven)
        assert obs.size == 4
        result = obs.evaluate(d=5)
        np.testing.assert_array_almost_equal(
            result.matrices[5], np.zeros_like(result.matrices[5])
        )

    def test_repr(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        s = repr(obs)
        assert "XX_p21" in s
        assert "Λ=2" in s
        assert "storage=dense" in s

    def test_empty_result_for_d_0(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = Observable.from_data(contraction, woven)
        result = obs.evaluate(d=0)
        assert len(result.bases[0]) == 1
        assert result.bases[0] == [()]


# ======================================================================
# Resolve coupling helper
# ======================================================================


class TestResolveCoupling:
    def test_none_returns_one(self):
        assert _resolve_coupling(None, 5.0, 1.0) == 1.0

    def test_float(self):
        assert _resolve_coupling(0.3, 5.0, 1.0) == 0.3

    def test_callable_one_arg(self):
        result = _resolve_coupling(lambda d: 1.0 / d, 5.0, 1.0)
        assert result == pytest.approx(0.2)

    def test_callable_two_args(self):
        result = _resolve_coupling(lambda d, m: d * m, 5.0, 2.0)
        assert result == pytest.approx(10.0)


# ======================================================================
# FreeHamiltonian
# ======================================================================


class TestFreeHamiltonian:
    def test_construction(self):
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        assert free.label == "free"
        assert free.Lambda == 2
        assert free.is_even is True
        assert free.is_hermitian is True
        assert free.K == 0
        assert free.n_x == 0
        assert free.n_p == 0

    def test_evaluate_diagonal(self):
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        result = free.evaluate(d=5)
        H = result.matrices[5]
        assert sparse.isspmatrix_csr(H)
        assert np.allclose(H.toarray(), np.diag(np.diag(H.toarray())))

    def test_evaluate_values(self):
        d = 5
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        result = free.evaluate(d=d)
        expected_diag = [d**2 / 2 + 0, d**2 / 2 + 1, d**2 / 2 + 2, d**2 / 2 + 2]
        np.testing.assert_array_almost_equal(
            result.matrices[d].diagonal(), expected_diag
        )

    def test_mass_scaling(self):
        d = 5
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        r1 = free.evaluate(d=d, mass=1.0)
        r2 = free.evaluate(d=d, mass=2.0)
        np.testing.assert_array_almost_equal(
            r2.matrices[d].toarray(),
            2.0 * r1.matrices[d].toarray(),
        )

    def test_filtering(self):
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        result = free.evaluate(d=1)
        assert len(result.bases[1]) == 3
        assert all(len(R) <= 1 for R in result.bases[1])

    def test_ground_state_filter(self):
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        result = free.evaluate(d=100, ground_state_only=True)
        for R in result.bases[100]:
            assert sum(R) % 2 == 0

    def test_save_raises(self):
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        with pytest.raises(NotImplementedError):
            free.save("/tmp/shouldnt_exist.npz")

    def test_basis_size(self):
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        assert free.size == 4

    def test_batch_d(self):
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        result = free.evaluate(d=[1, 5])
        assert result.d_values == [1, 5]
        assert result.matrices[1].shape == (3, 3)
        assert result.matrices[5].shape == (4, 4)

    def test_summary_includes_storage(self):
        free = FreeHamiltonian(Lambda=2, mass=1.0)
        assert "storage=sparse" in free.summary()


# ======================================================================
# Hamiltonian
# ======================================================================


class TestHamiltonian:
    @pytest.fixture
    def obs(self) -> ObservableDense:
        contraction, woven = _make_synthetic_K2_Lambda2()
        return ObservableDense.from_data(contraction, woven)

    @pytest.fixture
    def ham(self, obs) -> HamiltonianDense:
        return HamiltonianDense(
            observables=[obs],
            Lambda=obs.Lambda,
            default_mass=obs.reference_mass,
        )

    def test_basic_properties(self, ham):
        assert ham.Lambda == 2
        assert ham.size == 4
        assert ham.K == 2
        assert ham.mass == 1.0
        assert ham.labels == ["XX_p21"]

    def test_evaluate_returns_ndarray(self, ham):
        H = ham.evaluate(d=5)
        assert isinstance(H, np.ndarray)
        assert H.shape == (4, 4)

    def test_evaluate_symmetric(self, ham):
        H = ham.evaluate(d=5)
        np.testing.assert_array_almost_equal(H, H.T)

    def test_coupling_none_means_one(self, ham):
        r1 = ham.evaluate(d=5, coupling=None)
        r2 = ham.evaluate(d=5, coupling=1.0)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_coupling_scalar(self, ham, obs):
        H_full = ham.evaluate(d=5, coupling=1.0)
        H_half = ham.evaluate(d=5, coupling=0.5)
        # H_full = H_free + 1.0 * H_int
        # H_half = H_free + 0.5 * H_int
        H_free = FreeHamiltonian(2, 1.0).evaluate(d=5).matrices[5].toarray()
        H_int = H_full - H_free
        expected = H_free + 0.5 * H_int
        np.testing.assert_array_almost_equal(H_half, expected)

    def test_coupling_callable(self, ham):
        d = 5
        g1 = 0.3
        H_func = ham.evaluate(d=d, coupling=lambda d: g1 / d)
        H_scalar = ham.evaluate(d=d, coupling=g1 / d)
        np.testing.assert_array_almost_equal(H_func, H_scalar)

    def test_coupling_zero_gives_free_only(self, ham):
        d = 5
        H = ham.evaluate(d=d, coupling=0.0)
        H_free = FreeHamiltonian(2, 1.0).evaluate(d=d).matrices[d].toarray()
        np.testing.assert_array_almost_equal(H, H_free)

    def test_d_filter(self, ham):
        H = ham.evaluate(d=1)
        assert H.shape == (3, 3)

    def test_ground_state_filter(self, ham):
        H = ham.evaluate(d=100, ground_state_only=True)
        assert H.shape == (3, 3)

    def test_combined_filter(self, ham):
        H = ham.evaluate(d=1, ground_state_only=True)
        assert H.shape == (2, 2)

    def test_evaluate_reuses_cache_for_same_d(self, ham):
        obs = ham.observables[0]
        calls: list[list[int | float]] = []
        orig_evaluate = obs.evaluate

        def spy_evaluate(
            d,
            *,
            mass=None,
            ground_state_only=False,
        ):
            calls.append(list(np.atleast_1d(d)))
            return orig_evaluate(d, mass=mass, ground_state_only=ground_state_only)

        obs.evaluate = spy_evaluate  # type: ignore[method-assign]

        ham.evaluate(d=5)
        ham.evaluate(d=5)
        assert calls == [[5]]

    def test_precompute_populates_cache(self, ham):
        obs = ham.observables[0]
        calls: list[list[int | float]] = []
        orig_evaluate = obs.evaluate

        def spy_evaluate(
            d,
            *,
            mass=None,
            ground_state_only=False,
        ):
            calls.append(list(np.atleast_1d(d)))
            return orig_evaluate(d, mass=mass, ground_state_only=ground_state_only)

        obs.evaluate = spy_evaluate  # type: ignore[method-assign]

        ham.precompute([5, 6])
        ham.evaluate(d=5)
        ham.evaluate(d=6)
        assert calls == [[5, 6]]

    def test_energy_pure_state(self, ham):
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        energy = ham.energy(d=5, state=psi)
        H = ham.evaluate(d=5)
        expected = np.vdot(psi, H @ psi)
        assert energy == pytest.approx(expected)

    def test_energy_mixed_state(self, ham):
        rho = np.eye(4, dtype=np.complex128) / 4.0
        energy = ham.energy(d=5, state=rho)
        H = ham.evaluate(d=5)
        expected = np.trace(H @ rho)
        assert energy == pytest.approx(expected)

    def test_average_excitation_number_pure_state(self, ham):
        # Basis ordering for Lambda=2: [(), (1,), (2,), (1,1)]
        psi = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
        n_avg = ham.average_excitation_number(d=5, state=psi)
        assert n_avg == pytest.approx(2.0 + 0.0j)

    def test_average_excitation_number_mixed_state(self, ham):
        rho = np.diag([0.25, 0.25, 0.25, 0.25]).astype(np.complex128)
        n_avg = ham.average_excitation_number(d=5, state=rho)
        # average excitation = (0 + 1 + 2 + 2) / 4 = 1.25
        assert n_avg == pytest.approx(1.25 + 0.0j)

    def test_average_excitation_number_with_std(self, ham):
        rho = np.diag([0.25, 0.25, 0.25, 0.25]).astype(np.complex128)
        n_avg, n_std = ham.average_excitation_number(d=5, state=rho, return_std=True)
        assert n_avg == pytest.approx(1.25)
        # values are [0,1,2,2], mean=1.25, variance=0.6875
        assert n_std == pytest.approx(np.sqrt(0.6875))

    def test_average_excitation_number_with_std_rejects_overlap(self, ham):
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        phi = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
        with pytest.raises(ValueError, match="Standard deviation"):
            ham.average_excitation_number(
                d=5,
                state=psi,
                bra_state=phi,
                return_std=True,
            )


# ======================================================================
# Hamiltonian with multiple observables
# ======================================================================


class TestHamiltonianMultiObs:
    @pytest.fixture
    def ham(self) -> HamiltonianDense:
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs1 = ObservableDense.from_data(contraction, woven)
        obs2 = _make_observable("PP_p21")
        return HamiltonianDense(
            observables=[obs1, obs2],
            Lambda=2,
            default_mass=1.0,
        )

    def test_labels(self, ham):
        assert ham.labels == ["XX_p21", "PP_p21"]

    def test_K_max(self, ham):
        assert ham.K == 2

    def test_evaluate_returns_ndarray(self, ham):
        H = ham.evaluate(d=5)
        assert isinstance(H, np.ndarray)

    def test_per_obs_coupling_dict(self, ham):
        coupling = {"XX_p21": 0.5, "PP_p21": 2.0}
        H = ham.evaluate(d=5, coupling=coupling)
        # Verify: H = H_free + 0.5*O_XX + 2.0*O_PP
        H_free = FreeHamiltonian(2, 1.0).evaluate(d=5).matrices[5].toarray()
        O_XX = ham.observables[0].evaluate(d=5).matrices[5]
        O_PP = ham.observables[1].evaluate(d=5).matrices[5]
        expected = H_free + 0.5 * O_XX + 2.0 * O_PP
        np.testing.assert_array_almost_equal(H, expected)

    def test_per_obs_coupling_missing_key(self, ham):
        coupling = {"XX_p21": 0.5}  # PP not specified → 1.0
        H = ham.evaluate(d=5, coupling=coupling)
        H_free = FreeHamiltonian(2, 1.0).evaluate(d=5).matrices[5].toarray()
        O_XX = ham.observables[0].evaluate(d=5).matrices[5]
        O_PP = ham.observables[1].evaluate(d=5).matrices[5]
        expected = H_free + 0.5 * O_XX + 1.0 * O_PP
        np.testing.assert_array_almost_equal(H, expected)

    def test_repr(self, ham):
        s = repr(ham)
        assert "XX_p21" in s
        assert "PP_p21" in s
        assert "storage=dense" in s


class TestHamiltonianSummary:
    def test_single_observable_summary_includes_storage(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)
        ham = HamiltonianDense(
            observables=[obs],
            Lambda=obs.Lambda,
            default_mass=obs.reference_mass,
        )

        assert "storage=dense" in ham.summary()

    def test_sparse_summary_includes_storage(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableSparse.from_data(contraction, woven)
        ham = HamiltonianSparse(
            observables=[obs],
            Lambda=obs.Lambda,
            default_mass=obs.reference_mass,
        )

        assert "storage=sparse" in ham.summary()


# ======================================================================
# Mass rescaling in Hamiltonian evaluation
# ======================================================================


class TestMassRescaling:
    @pytest.fixture
    def ham(self) -> HamiltonianDense:
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)
        return HamiltonianDense(
            observables=[obs], Lambda=obs.Lambda, default_mass=obs.reference_mass
        )

    def test_default_mass_used(self, ham):
        H = ham.evaluate(d=5)
        assert H.shape == (4, 4)

    def test_mass_scales_free_hamiltonian(self, ham):
        d = 5
        H1 = ham.evaluate(d=d, mass=1.0, coupling=0.0)
        H2 = ham.evaluate(d=d, mass=2.0, coupling=0.0)
        np.testing.assert_array_almost_equal(H2, 2.0 * H1)

    def test_mass_rescales_interaction(self, ham):
        d = 5
        m0 = 1.0
        m1 = 2.0
        # Observable "XX_p21" has n_x=2, n_p=0, exponent=-1
        obs = ham.observables[0]
        assert obs.mass_exponent == -1.0

        O0 = obs.evaluate(d=d, mass=m0).matrices[d]
        O1 = obs.evaluate(d=d, mass=m1).matrices[d]
        expected_scale = (m1 / m0) ** obs.mass_exponent
        np.testing.assert_array_almost_equal(O1, expected_scale * O0)


# ======================================================================
# Free coupling
# ======================================================================


class TestFreeCoupling:
    @pytest.fixture
    def ham(self) -> HamiltonianDense:
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)
        return HamiltonianDense(
            observables=[obs], Lambda=obs.Lambda, default_mass=obs.reference_mass
        )

    def test_default_free_coupling_is_one(self, ham):
        assert ham.free_coupling == 1.0

    def test_free_coupling_scales_free_part(self, ham):
        d = 5
        H1 = ham.evaluate(d=d, coupling=0.0)
        H2 = ham.evaluate(d=d, coupling=0.0, free_coupling=0.5)
        np.testing.assert_array_almost_equal(H2, 0.5 * H1)

    def test_free_coupling_callable_d(self, ham):
        d = 5
        H_func = ham.evaluate(d=d, coupling=0.0, free_coupling=lambda d: 1.0 / d)
        H_scalar = ham.evaluate(d=d, coupling=0.0, free_coupling=1.0 / d)
        np.testing.assert_array_almost_equal(H_func, H_scalar)

    def test_free_coupling_callable_d_m(self, ham):
        d = 5
        m = 1.0
        H_func = ham.evaluate(
            d=d, mass=m, coupling=0.0, free_coupling=lambda d, m: d * m
        )
        H_scalar = ham.evaluate(d=d, mass=m, coupling=0.0, free_coupling=5.0)
        np.testing.assert_array_almost_equal(H_func, H_scalar)

    def test_stored_free_coupling(self):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)
        ham = HamiltonianDense(
            observables=[obs],
            Lambda=obs.Lambda,
            default_mass=obs.reference_mass,
            free_coupling=0.3,
        )
        H = ham.evaluate(d=5, coupling=0.0)
        H_ref = ham.evaluate(d=5, coupling=0.0, free_coupling=0.3)
        np.testing.assert_array_almost_equal(H, H_ref)


# ======================================================================
# Coupling signatures
# ======================================================================


class TestCouplingSignatures:
    @pytest.fixture
    def ham(self) -> HamiltonianDense:
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)
        return HamiltonianDense(
            observables=[obs], Lambda=obs.Lambda, default_mass=obs.reference_mass
        )

    def test_coupling_none_is_one(self, ham):
        r1 = ham.evaluate(d=5, coupling=None)
        r2 = ham.evaluate(d=5, coupling=1.0)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_coupling_float(self, ham):
        H = ham.evaluate(d=5, coupling=0.5)
        H_ref = ham.evaluate(d=5, coupling=1.0)
        H_free = FreeHamiltonian(2, 1.0).evaluate(d=5).matrices[5].toarray()
        expected = H_free + 0.5 * (H_ref - H_free)
        np.testing.assert_array_almost_equal(H, expected)

    def test_coupling_callable_d_only(self, ham):
        d = 5
        H_func = ham.evaluate(d=d, coupling=lambda d: 0.1 / d)
        H_scalar = ham.evaluate(d=d, coupling=0.1 / d)
        np.testing.assert_array_almost_equal(H_func, H_scalar)

    def test_coupling_callable_d_m(self, ham):
        d = 5
        m = 1.0
        H_func = ham.evaluate(d=d, mass=m, coupling=lambda d, m: 1.0 / (d * m))
        H_scalar = ham.evaluate(d=d, mass=m, coupling=1.0 / (d * m))
        np.testing.assert_array_almost_equal(H_func, H_scalar)

    def test_coupling_dict_single_obs(self, ham):
        d = 5
        H_dict = ham.evaluate(d=d, coupling={"XX_p21": 0.3})
        H_scalar = ham.evaluate(d=d, coupling=0.3)
        np.testing.assert_array_almost_equal(H_dict, H_scalar)


# ======================================================================
# Serialization (Observable save / load)
# ======================================================================


class TestSerialization:
    def test_round_trip(self, tmp_path):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)

        path = tmp_path / "obs.npz"
        obs.save(path)
        loaded = ObservableDense.load(path)

        assert loaded.label == obs.label
        assert loaded.Lambda == obs.Lambda
        assert loaded.reference_mass == obs.reference_mass
        assert loaded.is_even == obs.is_even
        assert loaded.is_hermitian == obs.is_hermitian
        assert loaded.basis == obs.basis
        np.testing.assert_array_equal(loaded.unnorm_coeffs, obs.unnorm_coeffs)

    def test_round_trip_evaluation_matches(self, tmp_path):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)

        path = tmp_path / "obs.npz"
        obs.save(path)
        loaded = ObservableDense.load(path)

        r_orig = obs.evaluate(d=5)
        r_load = loaded.evaluate(d=5)
        np.testing.assert_array_almost_equal(r_load.matrices[5], r_orig.matrices[5])

    def test_vacuum_partition_survives(self, tmp_path):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)
        assert () in obs.basis

        path = tmp_path / "obs.npz"
        obs.save(path)
        loaded = ObservableDense.load(path)
        assert () in loaded.basis
        assert loaded.basis == obs.basis

    def test_npz_extension_auto_appended(self, tmp_path):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)

        path = tmp_path / "obs_noext"
        obs.save(path)
        loaded = ObservableDense.load(path.with_suffix(".npz"))
        assert loaded.Lambda == obs.Lambda

    def test_creates_parent_directory(self, tmp_path):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableDense.from_data(contraction, woven)

        path = tmp_path / "sub" / "dir" / "obs.npz"
        obs.save(path)
        loaded = ObservableDense.load(path)
        assert loaded.Lambda == obs.Lambda


class TestSparseSerialization:
    def test_max_degree_ignores_trailing_zero_sparse_slices(self):
        basis = partition_list(2)
        coeffs = [
            sparse.csr_matrix((len(basis), len(basis)), dtype=np.float64)
            for _ in range(5)
        ]
        coeffs[2] = sparse.csr_matrix(([1.0], ([1], [1])), shape=(4, 4))
        obs = ObservableSparse(
            label="XX_p21",
            Lambda=2,
            reference_mass=1.0,
            basis=basis,
            unnorm_coeffs=coeffs,
        )

        assert obs.max_degree == 2

    def test_round_trip(self, tmp_path):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableSparse.from_data(contraction, woven)

        path = tmp_path / obs.filename
        obs.save(path)
        loaded = ObservableSparse.load(path)

        assert loaded.label == obs.label
        assert loaded.filename.endswith("_sparse.npz")
        assert loaded.basis == obs.basis
        for k in range(obs.degree_count()):
            np.testing.assert_array_equal(
                loaded.coeff_at_degree(k).toarray(),
                obs.coeff_at_degree(k).toarray(),
            )

    def test_round_trip_evaluation_matches_dense(self, tmp_path):
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableSparse.from_data(contraction, woven)
        path = tmp_path / obs.filename
        obs.save(path)
        loaded = ObservableSparse.load(path)
        result = loaded.evaluate(d=5)
        assert sparse.isspmatrix_csr(result.matrices[5])

    def test_round_trip_trims_trailing_zero_sparse_slices(self, tmp_path):
        basis = partition_list(2)
        coeffs = [
            sparse.csr_matrix((len(basis), len(basis)), dtype=np.float64)
            for _ in range(5)
        ]
        coeffs[2] = sparse.csr_matrix(([1.0], ([1], [1])), shape=(4, 4))
        obs = ObservableSparse(
            label="XX_p21",
            Lambda=2,
            reference_mass=1.0,
            basis=basis,
            unnorm_coeffs=coeffs,
        )

        path = tmp_path / obs.filename
        obs.save(path)
        loaded = ObservableSparse.load(path)

        assert loaded.max_degree == 2
        assert loaded.degree_count() == 3
        np.testing.assert_array_equal(
            loaded.coeff_at_degree(2).toarray(),
            coeffs[2].toarray(),
        )


class TestHamiltonianSparse:
    @pytest.fixture
    def ham(self) -> HamiltonianSparse:
        contraction, woven = _make_synthetic_K2_Lambda2()
        obs = ObservableSparse.from_data(contraction, woven)
        return HamiltonianSparse(
            observables=[obs],
            Lambda=obs.Lambda,
            default_mass=obs.reference_mass,
        )

    def test_evaluate_returns_sparse_matrix(self, ham):
        H = ham.evaluate(d=5)
        assert sparse.isspmatrix_csr(H)
        assert H.shape == (4, 4)

    def test_sparse_matches_dense(self, ham):
        contraction, woven = _make_synthetic_K2_Lambda2()
        dense_ham = HamiltonianDense(
            observables=[ObservableDense.from_data(contraction, woven)],
            Lambda=2,
            default_mass=1.0,
        )
        np.testing.assert_array_almost_equal(
            ham.evaluate(d=5).toarray(),
            dense_ham.evaluate(d=5),
        )


# ======================================================================
# Hamiltonian mismatched basis sizes
# ======================================================================


class TestHamiltonianValidation:
    def test_mismatched_basis_raises(self):
        obs = ObservableDense(
            label="XX_p21",
            Lambda=4,
            reference_mass=1.0,
            basis=partition_list(4),
            unnorm_coeffs=np.zeros((2, 12, 12)),
        )
        with pytest.raises(ValueError, match="basis size"):
            HamiltonianDense(observables=[obs], Lambda=2, default_mass=1.0)


# ======================================================================
# Label / filename helpers
# ======================================================================


class TestLabelToOpsSpec:
    def test_basic(self):
        assert _label_to_ops_spec("XX_p21") == "XX:21"

    def test_four_operators(self):
        assert _label_to_ops_spec("XXXX_p2341") == "XXXX:2341"

    def test_mixed_operators(self):
        assert _label_to_ops_spec("XPXP_p2341") == "XPXP:2341"

    def test_no_separator_raises(self):
        with pytest.raises(ValueError, match="missing '_p' separator"):
            _label_to_ops_spec("XXXX")


class TestMassToFilenameStr:
    def test_half(self):
        assert _mass_to_filename_str(0.5) == "0.5"

    def test_integer_one(self):
        assert _mass_to_filename_str(1) == "1."

    def test_integer_two(self):
        assert _mass_to_filename_str(2) == "2."

    def test_string_fraction(self):
        assert _mass_to_filename_str("1/2") == "0.5"

    def test_string_integer(self):
        assert _mass_to_filename_str("1") == "1."

    def test_non_integer_float(self):
        assert _mass_to_filename_str(0.25) == "0.25"


class TestLabelToWovenFilename:
    def test_production(self):
        fname = _label_to_woven_filename("XXXX_p2341", 0.5, 14)
        assert fname == "wc_op_XXXX_p2341_m0.5_Lambda14.json"

    def test_test_mode(self):
        fname = _label_to_woven_filename("XX_p21", 1, 6)
        assert fname == "wc_op_XX_p21_m1._Lambda6.json"

    def test_string_mass(self):
        fname = _label_to_woven_filename("XXXX_p2341", "1/2", 10)
        assert fname == "wc_op_XXXX_p2341_m0.5_Lambda10.json"


# ======================================================================
# build_observables (smoke test — empty labels)
# ======================================================================


class TestBuildObservables:
    def test_empty_labels_raises(self):
        from sym_contractions.hamiltonian import build_observables

        with pytest.raises(ValueError, match="must not be empty"):
            build_observables([])
