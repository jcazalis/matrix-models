"""Observable and Hamiltonian construction from woven contraction data.

Central class hierarchy:

- :class:`Observable` — a single traced monomial operator ``tr(M(X, P))``
  stored as a polynomial-in-*d* coefficient stack.  Can evaluate, save,
  and load independently.
- :class:`FreeHamiltonian` — subclass of :class:`Observable` for the free
  part ``H_free[R, R] = m (d²/2 + |R|)``, computed analytically.
- :class:`Hamiltonian` — a weighted sum of observables:

  .. math::

      H = c_\\text{free}\\,H_\\text{free}(m)
        + \\sum_i g_i(d, m)\\,O_i(d)

Typical workflow
----------------
>>> from sym_contractions.hamiltonian import Observable, Hamiltonian, build_observables
>>> observables = build_observables(["XXXX_p2341"], Lambda=6, mass=0.5)
>>> ham = Hamiltonian(observables, Lambda=6, mass=0.5)
>>> H = ham.evaluate(d=10, coupling=lambda d: 0.1 / d)
>>> eigenvalues = np.linalg.eigvalsh(H)

Mass rescaling
--------------
Each observable rescales analytically via :math:`m^{(n_P - n_X)/2}`.
"""

from __future__ import annotations

import dataclasses
import inspect
import multiprocessing as mp
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, cast

import numpy as np
from scipy import sparse

from sym_contractions.utils import enumerate_partitions
from sym_contractions.woven import ContractionEntry, ContractionResult, WovenData

CoeffStackDense = np.ndarray
CoeffStackSparse = list[sparse.csr_matrix]
CoeffStack = CoeffStackDense | CoeffStackSparse
EvaluatedMatrix = np.ndarray | sparse.csr_matrix
PairsKey = tuple[tuple[int, ...], ...]

_MAX_PARALLEL_WORKERS = 8

# ======================================================================
# Coefficient conversion helper
# ======================================================================


def _coeff_poly_to_array(poly: list[tuple[int, ...]]) -> np.ndarray:
    """Convert a woven coefficient polynomial to a NumPy array.

    Each element of *poly* is either a 2-tuple ``(numerator, denominator)``
    for a real rational coefficient, or a 4-tuple
    ``(re_num, re_den, im_num, im_den)`` for a complex rational coefficient
    (as produced by the v2 JSON schema).

    Returns a 1-D ``float64`` array when all imaginary parts are zero,
    otherwise a ``complex128`` array.
    """
    if not poly:
        return np.zeros(1, dtype=np.float64)

    first = poly[0]
    if len(first) == 2:
        # Legacy real-only format
        return np.array([num / den for num, den in poly], dtype=np.float64)

    # 4-tuple complex format
    reals = np.array([t[0] / t[1] for t in poly], dtype=np.float64)
    imags = np.array([t[2] / t[3] for t in poly], dtype=np.float64)

    if np.allclose(imags, 0, atol=1e-30):
        return reals
    return reals + 1j * imags


# ======================================================================
# Basis and normalization utilities
# ======================================================================


def partition_list(max_lambda: int) -> list[tuple[int, ...]]:
    """Build the full partition basis from excitation 0 to *max_lambda*.

    Equivalent to Mathematica's ``PartitionList[maxLambda]``.
    Partitions within each excitation level are in reverse lexicographic
    order.

    Parameters
    ----------
    max_lambda : int
        Maximum excitation level.

    Returns
    -------
    list[tuple[int, ...]]
        Ordered list of all integer partitions of 0 through *max_lambda*.

    Examples
    --------
    >>> partition_list(3)
    [(), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1)]
    """
    basis: list[tuple[int, ...]] = []
    for n in range(max_lambda + 1):
        basis.extend(enumerate_partitions(n))
    return basis


def normalization(R: tuple[int, ...], d: int | float) -> float:
    r"""Compute the normalization factor for a Young projector state.

    .. math::

        \text{Norm}(R, d) = \prod_{(i,j) \in R} (d + j - i)

    Box coordinates ``(i, j)`` are 1-indexed (row, column).
    Returns 1 for the empty partition (empty product).

    Parameters
    ----------
    R : tuple[int, ...]
        Integer partition in descending order.
    d : int or float
        Dimension parameter.

    Returns
    -------
    float
        Normalization factor.  Always positive when ``d >= len(R)``.

    Examples
    --------
    >>> normalization((), 5)
    1.0
    >>> normalization((1,), 5)
    5.0
    >>> normalization((2, 1), 3)
    24.0
    >>> normalization((1, 1, 1), 3)
    6.0
    """
    product = 1.0
    for i, r_i in enumerate(R, 1):
        for j in range(1, r_i + 1):
            product *= d + j - i
    return product


# ======================================================================
# Horner evaluation helper
# ======================================================================


def _horner_evaluate_dense(
    coeffs: np.ndarray,
    d: int | float,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Horner evaluation of a coefficient stack at *d*.

    Parameters
    ----------
    coeffs : np.ndarray
        Shape ``(max_degree + 1, size, size)``.
    d : int or float
        Dimension parameter.
    indices : np.ndarray or None
        Basis indices for sub-matrix extraction.

    Returns
    -------
    np.ndarray
        Evaluated matrix at *d*.
    """
    if indices is not None:
        coeffs = coeffs[np.ix_(np.arange(coeffs.shape[0]), indices, indices)]

    result = coeffs[-1].copy()
    for k in range(coeffs.shape[0] - 2, -1, -1):
        result = result * d + coeffs[k]
    return result


def _horner_evaluate_sparse(
    coeffs: list[sparse.csr_matrix],
    d: int | float,
    indices: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Horner evaluation of a sparse coefficient stack at *d*."""
    coeff_slices = coeffs
    if indices is not None:
        coeff_slices = [coeff[indices][:, indices].tocsr() for coeff in coeffs]

    result = coeff_slices[-1].copy()
    for k in range(len(coeff_slices) - 2, -1, -1):
        result = (result * d + coeff_slices[k]).tocsr()
    return result


def _horner_evaluate(
    coeffs: CoeffStack,
    d: int | float,
    indices: np.ndarray | None = None,
) -> EvaluatedMatrix:
    """Dispatch Horner evaluation based on dense/sparse coefficient storage."""
    if isinstance(coeffs, np.ndarray):
        return _horner_evaluate_dense(coeffs, d, indices)
    return _horner_evaluate_sparse(coeffs, d, indices)


def _empty_sparse_matrix(
    size: int, dtype: np.dtype | type[np.generic]
) -> sparse.csr_matrix:
    """Create an empty CSR matrix with square shape ``(size, size)``."""
    return sparse.csr_matrix((size, size), dtype=dtype)


def _to_dense_array(matrix: EvaluatedMatrix) -> np.ndarray:
    """Convert a dense or sparse matrix into a NumPy array."""
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix).toarray()
    return np.asarray(matrix)


def _to_csr_matrix(matrix: EvaluatedMatrix) -> sparse.csr_matrix:
    """Convert a dense or sparse matrix into CSR format."""
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix)
    return sparse.csr_matrix(matrix)


def _matrix_allclose(a: EvaluatedMatrix, b: EvaluatedMatrix) -> bool:
    """Return whether two dense or sparse matrices are numerically close."""
    if sparse.issparse(a) or sparse.issparse(b):
        left = _to_dense_array(a)
        right = _to_dense_array(b)
        return np.allclose(left, right)
    return np.allclose(np.asarray(a), np.asarray(b))


def _basis_to_array(basis: list[tuple[int, ...]]) -> np.ndarray:
    """Encode a partition basis as a zero-padded integer array."""
    max_parts = max((len(p) for p in basis), default=0)
    basis_arr = np.zeros((len(basis), max(max_parts, 1)), dtype=np.int32)
    for i, partition in enumerate(basis):
        for j, value in enumerate(partition):
            basis_arr[i, j] = value
    return basis_arr


def _basis_from_array(basis_arr: np.ndarray) -> list[tuple[int, ...]]:
    """Decode a zero-padded basis array into partition tuples."""
    basis: list[tuple[int, ...]] = []
    for row in basis_arr:
        nonzero = row[row > 0]
        if len(nonzero) == 0:
            basis.append(())
        else:
            basis.append(tuple(int(x) for x in nonzero))
    return basis


def _encode_optional_bool(value: bool | None) -> np.ndarray:
    """Encode ``None`` / ``bool`` metadata as ``int8``."""
    return np.array(-1 if value is None else int(value), dtype=np.int8)


def _decode_optional_bool(value: np.ndarray) -> bool | None:
    """Decode ``int8`` metadata into ``None`` / ``bool``."""
    raw = int(value)
    return None if raw == -1 else bool(raw)


def _observable_filename(label: str, Lambda: int, storage: str) -> str:
    """Return the canonical observable filename for the requested storage."""
    suffix = "_sparse" if storage == "sparse" else ""
    return f"{label}_Lambda{Lambda}{suffix}.npz"


def _sparse_dtype(coeffs: list[sparse.csr_matrix]) -> np.dtype:
    """Compute the combined dtype of sparse coefficient slices."""
    if not coeffs:
        return np.dtype(np.float64)
    dtype = coeffs[0].dtype
    for coeff in coeffs[1:]:
        dtype = np.result_type(dtype, coeff.dtype)
    return np.dtype(dtype)


def _normalize_sparse_matrix(
    matrix: sparse.csr_matrix,
    norm_sqrt: np.ndarray,
) -> sparse.csr_matrix:
    """Apply basis normalization to a sparse evaluated matrix."""
    inv_norm = 1.0 / norm_sqrt
    diag = sparse.diags(inv_norm, format="csr")
    return (diag @ matrix @ diag).tocsr()


def _symmetrize_sparse_matrix(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Symmetrize a sparse matrix while preserving complex conjugation."""
    upper = sparse.triu(matrix, format="csr")
    strict_upper = sparse.triu(matrix, k=1, format="csr")
    if np.issubdtype(matrix.dtype, np.complexfloating):
        return sparse.csr_matrix(upper + strict_upper.conjugate().transpose())
    return sparse.csr_matrix(upper + strict_upper.transpose())


def _sparse_nonzero_entry_count(coeffs: list[sparse.csr_matrix]) -> int:
    """Count matrix entries that are nonzero in at least one sparse degree slice."""
    support: sparse.csr_matrix | None = None
    for coeff in coeffs:
        if coeff.nnz == 0:
            continue
        mask = coeff.copy()
        mask.data = np.ones_like(mask.data, dtype=np.int8)
        support = mask if support is None else (support + mask).tocsr()
        assert support is not None
        support.sum_duplicates()
        support.data[:] = 1
    return 0 if support is None else int(support.nnz)


@dataclasses.dataclass(frozen=True)
class _SparseSectorTask:
    sector: tuple[int, int]
    entries: list[ContractionEntry]
    woven_polys: dict[PairsKey, np.ndarray]
    partition_to_idx: dict[tuple[int, ...], int]
    max_degree: int
    size: int


@dataclasses.dataclass(frozen=True)
class _SparseSectorResult:
    sector: tuple[int, int]
    coeffs: list[sparse.csr_matrix]
    n_contributions: int
    has_complex: bool


def _resolve_parallel_workers(
    task_count: int,
    max_workers: int | None,
) -> int:
    """Choose a conservative worker count for sparse assembly."""
    if task_count <= 1:
        return 1
    requested = max_workers if max_workers is not None else (os.cpu_count() or 1)
    return max(1, min(task_count, requested, _MAX_PARALLEL_WORKERS))


def _max_coefficient_degree(woven: WovenData) -> int:
    """Return the maximum observable degree implied by the woven data."""
    max_degree = 0
    for (_nL, nR), group in woven.groups.items():
        nL = group.nL
        for we in group.entries:
            woven_deg = max(len(we.coefficient_poly) - 1, 0)
            contr_deg = nL + nR
            max_degree = max(max_degree, woven_deg + contr_deg)
    return 1 if max_degree == 0 else max_degree


def _add_poly_inplace(acc: np.ndarray | None, poly: np.ndarray) -> np.ndarray:
    """Accumulate two coefficient arrays with dtype promotion as needed."""
    if acc is None:
        return poly.copy()

    dtype = np.result_type(acc.dtype, poly.dtype)
    if acc.dtype != dtype:
        acc = acc.astype(dtype, copy=True)
    if len(acc) < len(poly):
        grown = np.zeros(len(poly), dtype=dtype)
        grown[: len(acc)] = acc
        acc = grown
    acc[: len(poly)] += poly.astype(dtype, copy=False)
    return acc


def _build_woven_sector_polynomials(
    woven: WovenData,
) -> dict[tuple[int, int], dict[PairsKey, np.ndarray]]:
    """Pre-aggregate woven polynomials by excitation sector and pair key."""
    sector_polys: dict[tuple[int, int], dict[PairsKey, np.ndarray]] = {}
    for sector, group in sorted(woven.groups.items()):
        block_polys: dict[PairsKey, np.ndarray] = {}
        for entry in group.entries:
            pairs_key = tuple(tuple(pair) for pair in entry.pairs_1indexed)
            poly = _coeff_poly_to_array(entry.coefficient_poly)
            block_polys[pairs_key] = _add_poly_inplace(block_polys.get(pairs_key), poly)
        sector_polys[sector] = block_polys
    return sector_polys


def _entry_sector(entry: ContractionEntry) -> tuple[int, int]:
    """Return the excitation sector ``(nL, nR)`` for a contraction entry."""
    return sum(entry.R), sum(entry.S)


def _entries_are_sector_grouped(entries: list[ContractionEntry]) -> bool:
    """Return whether entries are already grouped contiguously by sector."""
    seen: set[tuple[int, int]] = set()
    current_sector: tuple[int, int] | None = None
    for entry in entries:
        sector = _entry_sector(entry)
        if sector != current_sector:
            if sector in seen:
                return False
            seen.add(sector)
            current_sector = sector
    return True


def _group_entries_by_sector(
    entries: list[ContractionEntry],
) -> list[tuple[tuple[int, int], list[ContractionEntry]]]:
    """Collect contraction entries by excitation sector in sorted order."""
    grouped: dict[tuple[int, int], list[ContractionEntry]] = {}
    for entry in entries:
        grouped.setdefault(_entry_sector(entry), []).append(entry)
    return [(sector, grouped[sector]) for sector in sorted(grouped)]


def _iter_sector_blocks(
    entries: list[ContractionEntry],
) -> list[tuple[tuple[int, int], list[ContractionEntry]]]:
    """Return contraction entries grouped by excitation sector."""
    if not entries:
        return []
    if not _entries_are_sector_grouped(entries):
        return _group_entries_by_sector(entries)

    blocks: list[tuple[tuple[int, int], list[ContractionEntry]]] = []
    current_sector = _entry_sector(entries[0])
    current_entries: list[ContractionEntry] = []
    for entry in entries:
        sector = _entry_sector(entry)
        if sector != current_sector:
            blocks.append((current_sector, current_entries))
            current_sector = sector
            current_entries = []
        current_entries.append(entry)
    blocks.append((current_sector, current_entries))
    return blocks


def _empty_sparse_coeff_stack(
    max_degree: int,
    size: int,
    dtype: np.dtype | type[np.generic],
) -> list[sparse.csr_matrix]:
    """Allocate an empty sparse coefficient stack."""
    return [_empty_sparse_matrix(size, dtype) for _ in range(max_degree + 1)]


def _promote_sparse_coeff_stack(
    coeffs: list[sparse.csr_matrix],
    dtype: np.dtype,
) -> list[sparse.csr_matrix]:
    """Promote a sparse coefficient stack to a wider dtype."""
    if np.dtype(_sparse_dtype(coeffs)) == np.dtype(dtype):
        return coeffs
    return [sparse.csr_matrix(coeff.astype(dtype)) for coeff in coeffs]


def _merge_sparse_coeff_stacks(
    target: list[sparse.csr_matrix],
    source: list[sparse.csr_matrix],
) -> None:
    """Add one sparse coefficient stack into another in place."""
    for degree, coeff in enumerate(source):
        if coeff.nnz == 0:
            continue
        target[degree] = (target[degree] + coeff).tocsr()
        target[degree].sum_duplicates()
        target[degree].eliminate_zeros()


def _assemble_sparse_sector(task: _SparseSectorTask) -> _SparseSectorResult:
    """Assemble one excitation-sector contribution for sparse observables."""
    rows: list[list[int]] = [[] for _ in range(task.max_degree + 1)]
    cols: list[list[int]] = [[] for _ in range(task.max_degree + 1)]
    data: list[list[float | complex]] = [[] for _ in range(task.max_degree + 1)]
    has_complex = False
    n_contributions = 0

    for entry in task.entries:
        pairs_key = tuple(tuple(pair) for pair in entry.pairs_1indexed)
        woven_poly = task.woven_polys.get(pairs_key)
        if woven_poly is None:
            continue

        c_poly = entry.coefficients
        if np.allclose(c_poly, 0, atol=1e-15):
            continue

        product = np.convolve(woven_poly, c_poly)
        if np.iscomplexobj(product):
            has_complex = True

        nz = np.flatnonzero(np.abs(product) > 1e-15)
        if len(nz) == 0:
            continue

        R_idx = task.partition_to_idx[entry.R]
        S_idx = task.partition_to_idx[entry.S]
        for degree in nz:
            rows[degree].append(R_idx)
            cols[degree].append(S_idx)
            data[degree].append(product[degree])
        n_contributions += 1

    dtype = np.complex128 if has_complex else np.float64
    coeffs = _empty_sparse_coeff_stack(task.max_degree, task.size, dtype)
    for degree in range(task.max_degree + 1):
        if not data[degree]:
            continue
        matrix = sparse.coo_matrix(
            (np.asarray(data[degree], dtype=dtype), (rows[degree], cols[degree])),
            shape=(task.size, task.size),
        )
        matrix.sum_duplicates()
        coeffs[degree] = sparse.csr_matrix(matrix)
        coeffs[degree].eliminate_zeros()

    return _SparseSectorResult(
        sector=task.sector,
        coeffs=coeffs,
        n_contributions=n_contributions,
        has_complex=has_complex,
    )


def _assemble_dense_coeffs(
    contraction: ContractionResult,
    woven: WovenData,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Assemble a dense coefficient stack from contraction and woven data."""
    Lambda = contraction.Lambda
    basis = partition_list(Lambda)
    size = len(basis)

    partition_to_idx: dict[tuple[int, ...], int] = {R: i for i, R in enumerate(basis)}
    contraction_index: dict[tuple, np.ndarray] = {}
    for entry in contraction.entries:
        pairs_key = tuple(tuple(p) for p in entry.pairs_1indexed)
        contraction_index[(pairs_key, entry.R, entry.S)] = entry.coefficients

    max_degree = _max_coefficient_degree(woven)

    h_coeffs = np.zeros((max_degree + 1, size, size), dtype=np.float64)
    n_contributions = 0

    for (nL, nR), group in sorted(woven.groups.items()):
        parts_nL = enumerate_partitions(nL)
        parts_nR = enumerate_partitions(nR)

        for we in group.entries:
            woven_poly = _coeff_poly_to_array(we.coefficient_poly)
            pairs_key = tuple(tuple(p) for p in we.pairs_1indexed)

            for R in parts_nL:
                R_idx = partition_to_idx[R]
                for S in parts_nR:
                    S_idx = partition_to_idx[S]
                    c_poly = contraction_index.get((pairs_key, R, S))
                    if c_poly is None or np.allclose(c_poly, 0, atol=1e-15):
                        continue

                    product = np.convolve(woven_poly, c_poly)
                    if np.iscomplexobj(product) and not np.iscomplexobj(h_coeffs):
                        h_coeffs = h_coeffs.astype(np.complex128)
                    h_coeffs[: len(product), R_idx, S_idx] += product
                    n_contributions += 1

        if verbose:
            print(f"  (nL={nL}, nR={nR}): {n_contributions} total contributions so far")

    if woven.is_hermitian is True:
        for k in range(h_coeffs.shape[0]):
            matrix = h_coeffs[k]
            upper = np.triu(matrix)
            if np.iscomplexobj(matrix):
                h_coeffs[k] = upper + np.triu(matrix, 1).conj().T
            else:
                h_coeffs[k] = upper + np.triu(matrix, 1).T

    if verbose:
        print(
            f"Observable assembled: label={contraction.label}, Λ={Lambda}, "
            f"basis_size={size}, max_degree={max_degree}, "
            f"contributions={n_contributions}"
        )

    return h_coeffs, basis


def _assemble_sparse_coeffs(
    contraction: ContractionResult,
    woven: WovenData,
    *,
    verbose: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
) -> tuple[list[sparse.csr_matrix], list[tuple[int, ...]]]:
    """Assemble a sparse CSR coefficient stack from contraction and woven data."""
    Lambda = contraction.Lambda
    basis = partition_list(Lambda)
    size = len(basis)

    partition_to_idx: dict[tuple[int, ...], int] = {R: i for i, R in enumerate(basis)}
    max_degree = _max_coefficient_degree(woven)
    sector_polys = _build_woven_sector_polynomials(woven)
    sector_blocks = [
        (sector, entries)
        for sector, entries in _iter_sector_blocks(contraction.entries)
        if sector in sector_polys
    ]

    coeffs = _empty_sparse_coeff_stack(max_degree, size, np.float64)
    has_complex = False
    n_contributions = 0

    worker_count = _resolve_parallel_workers(len(sector_blocks), max_workers)
    use_parallel = parallel and worker_count > 1 and bool(sector_blocks)
    if use_parallel and verbose:
        print(f"Sparse assembly parallelized over {worker_count} sector worker(s)")

    if use_parallel:
        tasks = [
            _SparseSectorTask(
                sector=sector,
                entries=entries,
                woven_polys=sector_polys[sector],
                partition_to_idx=partition_to_idx,
                max_degree=max_degree,
                size=size,
            )
            for sector, entries in sector_blocks
        ]
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            sector_results = executor.map(_assemble_sparse_sector, tasks)
            for result in sector_results:
                if result.has_complex and not has_complex:
                    coeffs = _promote_sparse_coeff_stack(
                        coeffs, np.dtype(np.complex128)
                    )
                    has_complex = True
                _merge_sparse_coeff_stacks(coeffs, result.coeffs)
                n_contributions += result.n_contributions
                if verbose:
                    nL, nR = result.sector
                    print(
                        f"  (nL={nL}, nR={nR}): {n_contributions} total contributions so far"
                    )
    else:
        for sector, entries in sector_blocks:
            result = _assemble_sparse_sector(
                _SparseSectorTask(
                    sector=sector,
                    entries=entries,
                    woven_polys=sector_polys[sector],
                    partition_to_idx=partition_to_idx,
                    max_degree=max_degree,
                    size=size,
                )
            )
            if result.has_complex and not has_complex:
                coeffs = _promote_sparse_coeff_stack(coeffs, np.dtype(np.complex128))
                has_complex = True
            _merge_sparse_coeff_stacks(coeffs, result.coeffs)
            n_contributions += result.n_contributions
            if verbose:
                nL, nR = result.sector
                print(
                    f"  (nL={nL}, nR={nR}): {n_contributions} total contributions so far"
                )

    if has_complex:
        coeffs = _promote_sparse_coeff_stack(coeffs, np.dtype(np.complex128))

    if woven.is_hermitian is True:
        coeffs = [_symmetrize_sparse_matrix(coeff) for coeff in coeffs]

    if verbose:
        print(
            f"Observable assembled: label={contraction.label}, Λ={Lambda}, "
            f"basis_size={size}, max_degree={max_degree}, "
            f"contributions={n_contributions}"
        )

    return coeffs, basis


# ======================================================================
# Basis filtering helper
# ======================================================================


def _compute_mask(
    basis: list[tuple[int, ...]],
    d: int | float,
    *,
    is_even: bool | None = None,
    ground_state_only: bool = False,
) -> np.ndarray:
    """Boolean mask selecting basis elements for a given *d*.

    Parameters
    ----------
    basis : list[tuple[int, ...]]
        Full partition basis.
    d : int or float
        Dimension parameter.
    is_even : bool or None
        Whether the observable/operator is even.
    ground_state_only : bool
        If True and ``is_even is True``, exclude odd-excitation states.
    """
    apply_parity_filter = ground_state_only and (is_even is True)
    mask = np.ones(len(basis), dtype=bool)
    for idx, R in enumerate(basis):
        if len(R) > d:
            mask[idx] = False
        if apply_parity_filter and sum(R) % 2 != 0:
            mask[idx] = False
    return mask


def _normalize_d_values(
    d: int | float | list[int | float] | tuple[int | float, ...] | np.ndarray,
) -> list[int | float]:
    """Normalize scalar/sequence ``d`` input to unique ordered scalar values."""
    if isinstance(d, (int, float, np.integer, np.floating)):
        values = [d]
    elif isinstance(d, np.ndarray):
        values = list(d.flatten())
    elif isinstance(d, (list, tuple)):
        values = list(d)
    else:
        raise TypeError(f"Invalid d input {d!r}; expected scalar or sequence")

    normalized: list[int | float] = []
    seen: set[int | float] = set()
    for value in values:
        if isinstance(value, np.generic):
            key = value.item()
        else:
            key = value
        if not isinstance(key, (int, float)):
            raise TypeError(f"Invalid d value {key!r}; expected int or float")
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    if not normalized:
        raise ValueError("d must contain at least one value")
    return normalized


# ======================================================================
# Data structures
# ======================================================================


@dataclasses.dataclass
class EvaluatedObservable:
    """Result of evaluating an observable at one or more *d* values.

    Attributes
    ----------
    matrices : dict[float | int, EvaluatedMatrix]
        Normalized operator matrices keyed by each ``d`` value.
    bases : dict[float | int, list[tuple[int, ...]]]
        Filtered partition bases keyed by each ``d`` value.
    d_values : list[float | int]
        Ordered list of evaluated ``d`` values.
    mass : float
        Mass parameter used for this evaluation.
    label : str
        Observable label.
    Lambda : int
        Cutoff used to construct the observable basis.
    ground_state_only : bool
        Whether odd-excitation states were filtered out.
    is_even : bool or None
        Observable parity metadata.
    is_hermitian : bool or None
        Observable Hermitianity metadata.

    Notes
    -----
    Dictionaries are always used, including scalar input ``d``.
    """

    matrices: dict[float | int, EvaluatedMatrix]
    bases: dict[float | int, list[tuple[int, ...]]]
    d_values: list[float | int]
    mass: float
    label: str
    Lambda: int
    ground_state_only: bool
    is_even: bool | None
    is_hermitian: bool | None

    def merge(self, other: EvaluatedObservable) -> EvaluatedObservable:
        """Merge two evaluations of the same observable over different ``d`` sets."""
        if self.label != other.label:
            raise ValueError(
                "Cannot merge EvaluatedObservable with different labels: "
                f"{self.label!r} vs {other.label!r}"
            )
        if not np.isclose(self.mass, other.mass):
            raise ValueError(
                "Cannot merge EvaluatedObservable with different masses: "
                f"{self.mass} vs {other.mass}"
            )
        if self.Lambda != other.Lambda:
            raise ValueError(
                "Cannot merge EvaluatedObservable with different Lambda: "
                f"{self.Lambda} vs {other.Lambda}"
            )
        if self.ground_state_only != other.ground_state_only:
            raise ValueError(
                "Cannot merge EvaluatedObservable with different ground_state_only: "
                f"{self.ground_state_only} vs {other.ground_state_only}"
            )
        if self.is_even != other.is_even:
            raise ValueError(
                "Cannot merge EvaluatedObservable with different is_even: "
                f"{self.is_even} vs {other.is_even}"
            )
        if self.is_hermitian != other.is_hermitian:
            raise ValueError(
                "Cannot merge EvaluatedObservable with different is_hermitian: "
                f"{self.is_hermitian} vs {other.is_hermitian}"
            )

        merged_matrices = dict(self.matrices)
        merged_bases = dict(self.bases)
        merged_d_values = list(self.d_values)

        for d_value in other.d_values:
            if d_value in merged_matrices:
                if merged_bases[d_value] != other.bases[d_value]:
                    raise ValueError(
                        "Cannot merge overlapping d with different basis for "
                        f"d={d_value!r}"
                    )
                if not _matrix_allclose(
                    merged_matrices[d_value], other.matrices[d_value]
                ):
                    raise ValueError(
                        "Cannot merge overlapping d with different matrix values for "
                        f"d={d_value!r}"
                    )
                continue

            merged_matrices[d_value] = other.matrices[d_value]
            merged_bases[d_value] = other.bases[d_value]
            merged_d_values.append(d_value)

        return EvaluatedObservable(
            matrices=merged_matrices,
            bases=merged_bases,
            d_values=merged_d_values,
            mass=self.mass,
            label=self.label,
            Lambda=self.Lambda,
            ground_state_only=self.ground_state_only,
            is_even=self.is_even,
            is_hermitian=self.is_hermitian,
        )

    def expectation_value(
        self,
        d: int | float,
        state: np.ndarray | list,
        bra_state: np.ndarray | list | None = None,
    ) -> float | complex:
        """Compute expectation value or overlap at cached ``d``.

        Parameters
        ----------
        d : int or float
            Cached dimension key.
        state : np.ndarray or list
            Ket state ``|psi>`` (1-D vector) or density matrix ``rho`` (2-D).
        bra_state : np.ndarray or list or None
            Optional bra state ``|phi>`` used to compute overlap
            ``<phi|O|psi>``. Only valid when both states are pure vectors.

        Returns
        -------
        float or complex
            ``<psi|O|psi>``, ``tr(O rho)``, or ``<phi|O|psi>``.
            For Hermitian observables, expectation values (without
            ``bra_state``) are returned as ``float``.
        """
        if d not in self.matrices:
            raise KeyError(
                f"EvaluatedObservable '{self.label}' has no cached matrix for d={d!r}"
            )

        op = self.matrices[d]
        assert op is not None
        shape = op.shape
        assert shape is not None
        n = shape[0]

        ket = np.asarray(state)

        if bra_state is not None:
            if ket.ndim != 1:
                raise ValueError(
                    "state must be a 1-D vector when bra_state is provided"
                )
            bra = np.asarray(bra_state)
            if bra.ndim != 1:
                raise ValueError("bra_state must be a 1-D vector")
            if ket.shape[0] != n or bra.shape[0] != n:
                raise ValueError(
                    "State sizes are incompatible with operator shape "
                    f"{op.shape}: got ket={ket.shape}, bra={bra.shape}"
                )
            return complex(np.vdot(bra, op @ ket))

        if ket.ndim == 1:
            if ket.shape[0] != n:
                raise ValueError(
                    "State vector size is incompatible with operator shape "
                    f"{op.shape}: got {ket.shape}"
                )
            value = complex(np.vdot(ket, op @ ket))
            if self.is_hermitian is True:
                if np.isclose(value.imag, 0.0, atol=1e-12, rtol=1e-12):
                    return float(value.real)
                return float(np.real(value))
            return value

        if ket.ndim == 2:
            if ket.shape != (n, n):
                raise ValueError(
                    "Density matrix shape is incompatible with operator shape "
                    f"{op.shape}: got {ket.shape}"
                )
            value = complex(np.trace(_to_dense_array(op @ ket)))
            if self.is_hermitian is True:
                if np.isclose(value.imag, 0.0, atol=1e-12, rtol=1e-12):
                    return float(value.real)
                return float(np.real(value))
            return value

        raise ValueError(
            "state must be a 1-D pure state vector or a 2-D density matrix"
        )


# ======================================================================
# Observable
# ======================================================================


@dataclasses.dataclass
class Observable:
    """An invariant observable in the singlet basis.

    Stores the matrix elements as a polynomial in *d* (coefficient stack).
    Can be evaluated at a specific *d* and mass, saved to disk, and loaded
    back.

    Attributes
    ----------
    label : str
        Compact operator string, e.g. ``"XXXX_p2341"`` for
        ``tr(X^4)`` with trace permutation ``(2,3,4,1)``.
    Lambda : int
        Maximum excitation level (cutoff).
    reference_mass : float
        Mass value at which the coefficients were computed.
    basis : list[tuple[int, ...]]
        Full partition basis (all partitions of 0 through Lambda).
    unnorm_coeffs : object
        Storage-specific coefficient stack. Dense observables use a
        3-D ``np.ndarray`` with shape ``(max_degree + 1, size, size)``.
        Sparse observables use a list of CSR matrices, one per degree.
    is_even : bool or None
        Whether the monomial has an even number of operators (if known).
    is_hermitian : bool or None
        Whether the monomial is Hermitian (if known).
    """

    label: str
    Lambda: int
    reference_mass: float
    basis: list[tuple[int, ...]]
    unnorm_coeffs: CoeffStack
    is_even: bool | None = None
    is_hermitian: bool | None = None

    # ------------------------------------------------------------------
    # Derived counts
    # ------------------------------------------------------------------

    @property
    def _operators_part(self) -> str:
        """Extract the operator string from the label.

        Labels have the form ``"{ops}_p{perm}"`` (e.g. ``"XXXX_p2341"``).
        Returns just the operators part (e.g. ``"XXXX"``).  Legacy labels
        without the ``_p`` suffix are returned unchanged.
        """
        if "_p" in self.label:
            return self.label.split("_p")[0]
        return self.label

    @property
    def n_x(self) -> int:
        """Number of X operators in the monomial."""
        return self._operators_part.upper().count("X")

    @property
    def n_p(self) -> int:
        """Number of P operators in the monomial."""
        return self._operators_part.upper().count("P")

    @property
    def K(self) -> int:
        """Total operator degree (number of X + P operators)."""
        return self.n_x + self.n_p

    @property
    def mass_exponent(self) -> float:
        r"""Exponent of *m* in the mass scaling prefactor.

        Each X contributes :math:`1/\sqrt{2m}` and each P contributes
        :math:`\sqrt{m/2}`, giving an overall factor
        :math:`m^{(n_P - n_X)/2}`.
        """
        return (self.n_p - self.n_x) / 2.0

    @property
    def max_degree(self) -> int:
        """Highest polynomial degree with a nonzero coefficient slice.

        Returns 0 for the identically zero polynomial.
        """
        nonzero_degrees = self.iter_nonzero_degrees()
        return max(nonzero_degrees, default=0)

    @property
    def size(self) -> int:
        """Number of basis states (before any filtering)."""
        return len(self.basis)

    @property
    def filename(self) -> str:
        """Canonical filename for saving this observable.

        Returns e.g. ``"XXXX_p2341_Lambda14.npz"``.
        """
        return _observable_filename(self.label, self.Lambda, self.storage_kind)

    @property
    def storage_kind(self) -> str:
        """Return the storage representation kind for this observable."""
        raise NotImplementedError

    def coeff_at_degree(self, k: int) -> np.ndarray | sparse.csr_matrix:
        """Return the coefficient slice multiplying ``d^k``."""
        raise NotImplementedError

    def coeff_entry(self, k: int, i: int, j: int) -> float | complex:
        """Return the ``(i, j)`` entry in the coefficient slice for degree ``k``."""
        raise NotImplementedError

    def degree_count(self) -> int:
        """Return the number of polynomial coefficient slices."""
        raise NotImplementedError

    def coeff_dtype(self) -> np.dtype:
        """Return the dtype of the coefficient stack."""
        raise NotImplementedError

    def nonzero_entry_count(self) -> int:
        """Count entries that are nonzero in at least one coefficient slice."""
        raise NotImplementedError

    def iter_nonzero_degrees(self) -> list[int]:
        """Return degrees whose coefficient slice is not identically zero."""
        raise NotImplementedError

    def _empty_evaluated_matrix(self, dtype: np.dtype) -> EvaluatedMatrix:
        """Create an empty evaluated matrix matching the storage representation."""
        raise NotImplementedError

    def _normalize_evaluated_matrix(
        self,
        matrix: EvaluatedMatrix,
        norm_sqrt: np.ndarray,
    ) -> EvaluatedMatrix:
        """Apply basis normalization to an evaluated matrix."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Mass rescaling
    # ------------------------------------------------------------------

    def rescaled_coeffs(self, target_mass: float) -> CoeffStack:
        """Return coefficient stack rescaled from *reference_mass* to *target_mass*.

        Parameters
        ----------
        target_mass : float
            Desired evaluation mass.

        Returns
        -------
        np.ndarray
            Same shape as ``unnorm_coeffs``, multiplied by
            ``(target_mass / reference_mass) ** mass_exponent``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filtered_basis(
        self,
        d: int | float,
        *,
        ground_state_only: bool = False,
    ) -> list[tuple[int, ...]]:
        """Return the basis after filtering for a given *d*.

        Parameters
        ----------
        d : int or float
            Dimension parameter.  Partitions with more than *d* rows are
            excluded.
        ground_state_only : bool
            If True, also exclude partitions with odd total excitation
            (only when the observable is even).

        Returns
        -------
        list[tuple[int, ...]]
        """
        mask = _compute_mask(
            self.basis, d, is_even=self.is_even, ground_state_only=ground_state_only
        )
        return [self.basis[i] for i in np.where(mask)[0]]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        d: int | float | list[int | float] | tuple[int | float, ...] | np.ndarray,
        *,
        mass: float | None = None,
        ground_state_only: bool = False,
    ) -> EvaluatedObservable:
        """Evaluate the normalized operator matrix at one or more *d* values.

        Parameters
        ----------
        d : int, float, list, tuple, or np.ndarray
            Dimension parameter(s).
        mass : float or None
            Evaluation mass.  If ``None``, uses ``self.reference_mass``.
        ground_state_only : bool
            Restrict to the even-excitation (ground-state) sector.

        Returns
        -------
        EvaluatedObservable
        """
        eval_mass = mass if mass is not None else self.reference_mass
        d_values = _normalize_d_values(d)
        coeffs = self.rescaled_coeffs(eval_mass)

        matrices: dict[float | int, EvaluatedMatrix] = {}
        bases: dict[float | int, list[tuple[int, ...]]] = {}

        for d_i in d_values:
            mask = _compute_mask(
                self.basis,
                d_i,
                is_even=self.is_even,
                ground_state_only=ground_state_only,
            )
            indices = np.where(mask)[0]
            filtered = [self.basis[i] for i in indices]
            bases[d_i] = filtered
            n = len(indices)

            if n == 0:
                matrices[d_i] = self._empty_evaluated_matrix(self.coeff_dtype())
                continue

            # Normalization
            norms = np.array([normalization(R, d_i) for R in filtered])
            norm_sqrt = np.sqrt(norms)

            # Horner evaluation with mass rescaling
            H_unnorm = _horner_evaluate(coeffs, d_i, indices)
            matrices[d_i] = self._normalize_evaluated_matrix(H_unnorm, norm_sqrt)

        return EvaluatedObservable(
            matrices=matrices,
            bases=bases,
            d_values=d_values,
            mass=eval_mass,
            label=self.label,
            Lambda=self.Lambda,
            ground_state_only=ground_state_only,
            is_even=self.is_even,
            is_hermitian=self.is_hermitian,
        )

    # ------------------------------------------------------------------
    # Construction from data
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        contraction: ContractionResult,
        woven: WovenData,
        *,
        storage: str = "dense",
        verbose: bool = False,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> Observable:
        r"""Build an Observable from precomputed contraction and woven data.

        For each matrix element ``(R, S)`` computes

        .. math::

            O_\text{unnorm}[R, S](d)
              = \sum_\tau \text{woven\_coeff}(\tau, d)
                        \;\text{contraction}(\tau, R, S, d)

        by convolving the woven and contraction polynomial coefficient
        arrays and accumulating into a 3-D coefficient stack.

        Parameters
        ----------
        contraction : ContractionResult
            Precomputed contraction coefficients.
        woven : WovenData
            Woven contraction data carrying the coefficient polynomials.
        verbose : bool
            Print progress messages.

        Returns
        -------
        Observable
        """
        if cls is Observable:
            target_cls = ObservableDense if storage == "dense" else ObservableSparse
            return target_cls.from_data(
                contraction,
                woven,
                verbose=verbose,
                parallel=parallel,
                max_workers=max_workers,
            )
        raise NotImplementedError(
            f"{cls.__name__}.from_data must be implemented by concrete subclasses"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the observable to a compressed ``.npz`` file.

        Parameters
        ----------
        path : str or Path
            Output file path.  A ``.npz`` extension is appended
            automatically by NumPy if not present.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> Observable:
        """Load an observable previously saved with :meth:`save`.

        Parameters
        ----------
        path : str or Path
            Path to the ``.npz`` file.

        Returns
        -------
        Observable
        """
        if cls is Observable:
            path = Path(path)
            if path.name.endswith("_sparse.npz"):
                return ObservableSparse.load(path)
            return ObservableDense.load(path)
        raise NotImplementedError(
            f"{cls.__name__}.load must be implemented by concrete subclasses"
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary string."""
        n_nonzero = self.nonzero_entry_count()
        return (
            f"Observable(label={self.label}, Λ={self.Lambda}, "
            f"storage={self.storage_kind}, "
            f"mass={self.reference_mass}, "
            f"basis_size={self.size}, "
            f"max_poly_degree={self.max_degree}, "
            f"nonzero_elements={n_nonzero})"
        )

    def __repr__(self) -> str:  # noqa: D105
        return self.summary()


# ======================================================================
# FreeHamiltonian
# ======================================================================


class FreeHamiltonian(Observable):
    r"""The free part of the Hamiltonian: ``H_free[R, R] = m (d²/2 + |R|)``.

    A subclass of :class:`Observable` that computes the free Hamiltonian
    analytically — no stored coefficient stack is needed.
    """

    def __init__(self, Lambda: int, mass: float) -> None:
        basis = partition_list(Lambda)
        # We pass a dummy 0-sized coefficient array; evaluate() is overridden.
        super().__init__(
            label="free",
            Lambda=Lambda,
            reference_mass=mass,
            basis=basis,
            unnorm_coeffs=np.empty((0, 0, 0)),
            is_even=True,
            is_hermitian=True,
        )

    @property
    def storage_kind(self) -> str:
        """FreeHamiltonian uses sparse evaluated matrices."""
        return "sparse"

    def coeff_at_degree(self, k: int) -> np.ndarray:
        raise NotImplementedError("FreeHamiltonian does not store coefficient slices")

    def coeff_entry(self, k: int, i: int, j: int) -> float | complex:
        raise NotImplementedError("FreeHamiltonian does not store coefficient slices")

    def degree_count(self) -> int:
        return 0

    def coeff_dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    def nonzero_entry_count(self) -> int:
        return self.size

    def iter_nonzero_degrees(self) -> list[int]:
        return []

    def _empty_evaluated_matrix(self, dtype: np.dtype) -> sparse.csr_matrix:
        return _empty_sparse_matrix(0, dtype)

    def _normalize_evaluated_matrix(
        self,
        matrix: EvaluatedMatrix,
        norm_sqrt: np.ndarray,
    ) -> sparse.csr_matrix:
        return _to_csr_matrix(matrix)

    @property
    def K(self) -> int:
        """Free Hamiltonian has no operator insertions."""
        return 0

    @property
    def n_x(self) -> int:
        return 0

    @property
    def n_p(self) -> int:
        return 0

    @property
    def mass_exponent(self) -> float:
        return 0.0

    @property
    def max_degree(self) -> int:
        return 2  # H_free is quadratic in d

    def rescaled_coeffs(self, target_mass: float) -> np.ndarray:
        """Not applicable for FreeHamiltonian."""
        raise NotImplementedError("FreeHamiltonian computes its matrix analytically.")

    def evaluate(
        self,
        d: int | float | list[int | float] | tuple[int | float, ...] | np.ndarray,
        *,
        mass: float | None = None,
        ground_state_only: bool = False,
    ) -> EvaluatedObservable:
        r"""Evaluate the free Hamiltonian at one or more *d* values.

        .. math::

            H_\text{free}[R, R] = m\,(d^2/2 + |R|)

        Parameters
        ----------
        d : int, float, list, tuple, or np.ndarray
            Dimension parameter(s).
        mass : float or None
            Evaluation mass.  If ``None``, uses ``self.reference_mass``.
        ground_state_only : bool
            Restrict to even-excitation states.

        Returns
        -------
        EvaluatedObservable
        """
        eval_mass = mass if mass is not None else self.reference_mass
        d_values = _normalize_d_values(d)

        matrices: dict[float | int, EvaluatedMatrix] = {}
        bases: dict[float | int, list[tuple[int, ...]]] = {}

        for d_i in d_values:
            mask = _compute_mask(
                self.basis,
                d_i,
                is_even=self.is_even,
                ground_state_only=ground_state_only,
            )
            indices = np.where(mask)[0]
            filtered = [self.basis[i] for i in indices]
            bases[d_i] = filtered
            n = len(indices)

            if n == 0:
                matrices[d_i] = _empty_sparse_matrix(0, np.float64)
                continue

            excitations = np.array([sum(R) for R in filtered], dtype=np.float64)
            diagonal = eval_mass * (d_i**2 / 2.0 + excitations)
            matrices[d_i] = sparse.csr_matrix(sparse.diags(diagonal, format="csr"))

        return EvaluatedObservable(
            matrices=matrices,
            bases=bases,
            d_values=d_values,
            mass=eval_mass,
            label=self.label,
            Lambda=self.Lambda,
            ground_state_only=ground_state_only,
            is_even=self.is_even,
            is_hermitian=self.is_hermitian,
        )

    def save(self, path: str | Path) -> None:
        """Not supported for FreeHamiltonian (reconstructible from Lambda + mass)."""
        raise NotImplementedError(
            "FreeHamiltonian does not need serialization; "
            "reconstruct via FreeHamiltonian(Lambda, mass)."
        )

    def summary(self) -> str:
        return (
            f"FreeHamiltonian(Λ={self.Lambda}, storage={self.storage_kind}, "
            f"mass={self.reference_mass}, "
            f"basis_size={self.size})"
        )


@dataclasses.dataclass(repr=False)
class ObservableDense(Observable):
    """Observable backed by a dense 3-D NumPy coefficient stack."""

    @property
    def storage_kind(self) -> str:
        return "dense"

    def coeff_at_degree(self, k: int) -> np.ndarray:
        coeffs = cast(np.ndarray, self.unnorm_coeffs)
        return coeffs[k]

    def coeff_entry(self, k: int, i: int, j: int) -> float | complex:
        coeffs = cast(np.ndarray, self.unnorm_coeffs)
        return coeffs[k, i, j]

    def degree_count(self) -> int:
        coeffs = cast(np.ndarray, self.unnorm_coeffs)
        return coeffs.shape[0]

    def coeff_dtype(self) -> np.dtype:
        coeffs = cast(np.ndarray, self.unnorm_coeffs)
        return np.dtype(coeffs.dtype)

    def nonzero_entry_count(self) -> int:
        coeffs = cast(np.ndarray, self.unnorm_coeffs)
        return int(np.count_nonzero(np.any(coeffs != 0, axis=0)))

    def iter_nonzero_degrees(self) -> list[int]:
        coeffs = cast(np.ndarray, self.unnorm_coeffs)
        return [k for k in range(self.degree_count()) if np.any(coeffs[k] != 0)]

    def _empty_evaluated_matrix(self, dtype: np.dtype) -> np.ndarray:
        return np.empty((0, 0), dtype=dtype)

    def _normalize_evaluated_matrix(
        self,
        matrix: EvaluatedMatrix,
        norm_sqrt: np.ndarray,
    ) -> np.ndarray:
        norm_outer = np.outer(norm_sqrt, norm_sqrt)
        return _to_dense_array(matrix) / norm_outer

    def rescaled_coeffs(self, target_mass: float) -> np.ndarray:
        coeffs = cast(np.ndarray, self.unnorm_coeffs)
        if target_mass == self.reference_mass:
            return coeffs
        ratio = target_mass / self.reference_mass
        return coeffs * (ratio**self.mass_exponent)

    @classmethod
    def from_data(
        cls,
        contraction: ContractionResult,
        woven: WovenData,
        *,
        storage: str = "dense",
        verbose: bool = False,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> ObservableDense:
        if storage != "dense":
            raise ValueError("ObservableDense.from_data requires storage='dense'")
        coeffs, basis = _assemble_dense_coeffs(contraction, woven, verbose=verbose)
        return cls(
            label=contraction.label,
            Lambda=contraction.Lambda,
            reference_mass=woven.mass,
            basis=basis,
            unnorm_coeffs=coeffs,
            is_even=woven.is_even,
            is_hermitian=woven.is_hermitian,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        coeffs = cast(np.ndarray, self.unnorm_coeffs)
        data: dict[str, object] = {
            "label": np.array(self.label),
            "meta_Lambda": np.array(self.Lambda, dtype=np.int32),
            "meta_reference_mass": np.array(self.reference_mass, dtype=np.float64),
            "meta_is_even": _encode_optional_bool(self.is_even),
            "meta_is_hermitian": _encode_optional_bool(self.is_hermitian),
            "basis": _basis_to_array(self.basis),
            "unnorm_coeffs": coeffs,
        }
        np.savez_compressed(path, **data)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> ObservableDense:
        path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            return cls(
                label=str(f["label"]),
                Lambda=int(f["meta_Lambda"]),
                reference_mass=float(f["meta_reference_mass"]),
                basis=_basis_from_array(f["basis"]),
                unnorm_coeffs=f["unnorm_coeffs"].copy(),
                is_even=_decode_optional_bool(f["meta_is_even"]),
                is_hermitian=_decode_optional_bool(f["meta_is_hermitian"]),
            )


@dataclasses.dataclass(repr=False)
class ObservableSparse(Observable):
    """Observable backed by a degree-indexed list of CSR coefficient slices."""

    @property
    def storage_kind(self) -> str:
        return "sparse"

    def coeff_at_degree(self, k: int) -> sparse.csr_matrix:
        coeffs = cast(list[sparse.csr_matrix], self.unnorm_coeffs)
        return coeffs[k]

    def coeff_entry(self, k: int, i: int, j: int) -> float | complex:
        coeffs = cast(list[sparse.csr_matrix], self.unnorm_coeffs)
        return coeffs[k][i, j]

    def degree_count(self) -> int:
        coeffs = cast(list[sparse.csr_matrix], self.unnorm_coeffs)
        return len(coeffs)

    def coeff_dtype(self) -> np.dtype:
        coeffs = cast(list[sparse.csr_matrix], self.unnorm_coeffs)
        return _sparse_dtype(coeffs)

    def nonzero_entry_count(self) -> int:
        coeffs = cast(list[sparse.csr_matrix], self.unnorm_coeffs)
        return _sparse_nonzero_entry_count(coeffs)

    def iter_nonzero_degrees(self) -> list[int]:
        coeffs = cast(list[sparse.csr_matrix], self.unnorm_coeffs)
        return [k for k, coeff in enumerate(coeffs) if coeff.nnz > 0]

    def _empty_evaluated_matrix(self, dtype: np.dtype) -> sparse.csr_matrix:
        return _empty_sparse_matrix(0, dtype)

    def _normalize_evaluated_matrix(
        self,
        matrix: EvaluatedMatrix,
        norm_sqrt: np.ndarray,
    ) -> sparse.csr_matrix:
        return _normalize_sparse_matrix(_to_csr_matrix(matrix), norm_sqrt)

    def rescaled_coeffs(self, target_mass: float) -> list[sparse.csr_matrix]:
        coeffs = cast(list[sparse.csr_matrix], self.unnorm_coeffs)
        if target_mass == self.reference_mass:
            return coeffs
        ratio = (target_mass / self.reference_mass) ** self.mass_exponent
        return [sparse.csr_matrix(coeff * ratio) for coeff in coeffs]

    @classmethod
    def from_data(
        cls,
        contraction: ContractionResult,
        woven: WovenData,
        *,
        storage: str = "sparse",
        verbose: bool = False,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> ObservableSparse:
        if storage != "sparse":
            raise ValueError("ObservableSparse.from_data requires storage='sparse'")
        coeffs, basis = _assemble_sparse_coeffs(
            contraction,
            woven,
            verbose=verbose,
            parallel=parallel,
            max_workers=max_workers,
        )
        return cls(
            label=contraction.label,
            Lambda=contraction.Lambda,
            reference_mass=woven.mass,
            basis=basis,
            unnorm_coeffs=coeffs,
            is_even=woven.is_even,
            is_hermitian=woven.is_hermitian,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        coeffs = cast(list[sparse.csr_matrix], self.unnorm_coeffs)
        data: dict[str, object] = {
            "label": np.array(self.label),
            "meta_Lambda": np.array(self.Lambda, dtype=np.int32),
            "meta_reference_mass": np.array(self.reference_mass, dtype=np.float64),
            "meta_is_even": _encode_optional_bool(self.is_even),
            "meta_is_hermitian": _encode_optional_bool(self.is_hermitian),
            "meta_storage_kind": np.array("sparse"),
            "meta_sparse_format": np.array("csr"),
            "meta_max_degree": np.array(self.max_degree, dtype=np.int32),
            "basis": _basis_to_array(self.basis),
            "meta_present_degrees": np.array(
                self.iter_nonzero_degrees(), dtype=np.int32
            ),
        }
        for degree in self.iter_nonzero_degrees():
            coeff = coeffs[degree]
            data[f"degree_{degree}_data"] = coeff.data
            data[f"degree_{degree}_indices"] = coeff.indices
            data[f"degree_{degree}_indptr"] = coeff.indptr
            data[f"degree_{degree}_shape"] = np.array(coeff.shape, dtype=np.int32)
        np.savez_compressed(path, **data)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> ObservableSparse:
        path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            basis = _basis_from_array(f["basis"])
            size = len(basis)
            max_degree = int(f["meta_max_degree"])
            present_degrees = [int(x) for x in f["meta_present_degrees"]]
            coeffs: list[sparse.csr_matrix] = [
                _empty_sparse_matrix(size, np.float64) for _ in range(max_degree + 1)
            ]
            for degree in present_degrees:
                coeffs[degree] = sparse.csr_matrix(
                    (
                        f[f"degree_{degree}_data"],
                        f[f"degree_{degree}_indices"],
                        f[f"degree_{degree}_indptr"],
                    ),
                    shape=tuple(int(x) for x in f[f"degree_{degree}_shape"]),
                )
            dtype = _sparse_dtype(coeffs)
            coeffs = [sparse.csr_matrix(coeff.astype(dtype)) for coeff in coeffs]
            return cls(
                label=str(f["label"]),
                Lambda=int(f["meta_Lambda"]),
                reference_mass=float(f["meta_reference_mass"]),
                basis=basis,
                unnorm_coeffs=coeffs,
                is_even=_decode_optional_bool(f["meta_is_even"]),
                is_hermitian=_decode_optional_bool(f["meta_is_hermitian"]),
            )


# ======================================================================
# Hamiltonian
# ======================================================================


@dataclasses.dataclass
class Hamiltonian:
    r"""A weighted sum of observables plus a free Hamiltonian.

    .. math::

        H = c_\text{free}\,H_\text{free}(m)
          + \sum_i g_i(d, m)\,O_i(d)

    Attributes
    ----------
    observables : list[Observable]
        Interaction observables (does *not* include the free part).
    Lambda : int
        Maximum excitation level.
    default_mass : float
        Default mass for evaluation (can be overridden per call).
    free_coupling : float
        Default coupling for the free Hamiltonian (default 1.0).
    """

    observables: list[Observable]
    Lambda: int
    default_mass: float
    free_coupling: float = 1.0
    _free_eval_cache: dict[tuple[float, bool], EvaluatedObservable] = dataclasses.field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _obs_eval_cache: dict[tuple[int, float, bool], EvaluatedObservable] = (
        dataclasses.field(
            default_factory=dict,
            init=False,
            repr=False,
        )
    )

    def __post_init__(self) -> None:
        basis_size = len(partition_list(self.Lambda))
        for obs in self.observables:
            if obs.size != basis_size:
                raise ValueError(
                    f"Observable '{obs.label}' has basis size "
                    f"{obs.size}, expected {basis_size} "
                    f"for Lambda={self.Lambda}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def basis(self) -> list[tuple[int, ...]]:
        """Full partition basis."""
        return partition_list(self.Lambda)

    @property
    def size(self) -> int:
        """Number of basis states (before any filtering)."""
        return len(self.basis)

    @property
    def K(self) -> int:
        """Maximum operator count across all observables."""
        return max((obs.K for obs in self.observables), default=0)

    @property
    def mass(self) -> float:
        """Alias for ``default_mass``."""
        return self.default_mass

    @mass.setter
    def mass(self, value: float) -> None:
        self.default_mass = value

    @property
    def labels(self) -> list[str]:
        """Labels of all observables."""
        return [obs.label for obs in self.observables]

    @property
    def storage_kind(self) -> str:
        """Return the evaluated matrix storage kind for this Hamiltonian."""
        raise NotImplementedError

    @property
    def is_even(self) -> bool | None:
        """Whether all observables are even.

        Returns ``None`` if any observable has unknown parity.
        """
        flags = [obs.is_even for obs in self.observables]
        if any(f is None for f in flags):
            return None
        return all(flags)

    @property
    def is_hermitian(self) -> bool | None:
        """Whether all observables are Hermitian.

        Returns ``None`` if any observable has unknown Hermitianity.
        """
        flags = [obs.is_hermitian for obs in self.observables]
        if any(f is None for f in flags):
            return None
        return all(flags)

    @property
    def max_degree(self) -> int:
        """Maximum polynomial degree across all observables."""
        if not self.observables:
            return 0
        return max(obs.max_degree for obs in self.observables)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filtered_basis(
        self,
        d: int | float,
        *,
        ground_state_only: bool = False,
    ) -> list[tuple[int, ...]]:
        """Return the basis after filtering for a given *d*.

        Parameters
        ----------
        d : int or float
            Dimension parameter.
        ground_state_only : bool
            If True, also exclude partitions with odd total excitation
            (only when all observables are even).
        """
        mask = _compute_mask(
            self.basis, d, is_even=self.is_even, ground_state_only=ground_state_only
        )
        return [self.basis[i] for i in np.where(mask)[0]]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _get_cached_free_evaluation(
        self,
        d_values: list[int | float],
        *,
        mass: float,
        ground_state_only: bool,
    ) -> EvaluatedObservable:
        """Return cached free-Hamiltonian evaluations, computing only missing d values."""
        cache_key = (float(mass), ground_state_only)
        cached = self._free_eval_cache.get(cache_key)
        missing = (
            d_values
            if cached is None
            else [d for d in d_values if d not in cached.matrices]
        )

        if not missing:
            assert cached is not None
            return cached

        free = FreeHamiltonian(self.Lambda, mass)
        new_eval = free.evaluate(
            missing,
            mass=mass,
            ground_state_only=ground_state_only,
        )
        merged = new_eval if cached is None else cached.merge(new_eval)
        self._free_eval_cache[cache_key] = merged
        return merged

    def _get_cached_observable_evaluation(
        self,
        obs: Observable,
        d_values: list[int | float],
        *,
        mass: float,
        ground_state_only: bool,
    ) -> EvaluatedObservable:
        """Return cached observable evaluations, computing only missing d values."""
        cache_key = (id(obs), float(mass), ground_state_only)
        cached = self._obs_eval_cache.get(cache_key)
        missing = (
            d_values
            if cached is None
            else [d for d in d_values if d not in cached.matrices]
        )

        if not missing:
            assert cached is not None
            return cached

        new_eval = obs.evaluate(
            missing,
            mass=mass,
            ground_state_only=ground_state_only,
        )
        merged = new_eval if cached is None else cached.merge(new_eval)
        self._obs_eval_cache[cache_key] = merged
        return merged

    def precompute(
        self,
        d: int | float | list[int | float] | tuple[int | float, ...] | np.ndarray,
        *,
        mass: float | None = None,
        ground_state_only: bool = False,
    ) -> None:
        """Populate internal caches for free/interaction observables at one or more d values."""
        eval_mass = mass if mass is not None else self.default_mass
        d_values = _normalize_d_values(d)
        self._get_cached_free_evaluation(
            d_values,
            mass=eval_mass,
            ground_state_only=ground_state_only,
        )
        for obs in self.observables:
            self._get_cached_observable_evaluation(
                obs,
                d_values,
                mass=eval_mass,
                ground_state_only=ground_state_only,
            )

    def energy(
        self,
        d: int | float,
        state: np.ndarray | list,
        *,
        mass: float | None = None,
        coupling: float | Callable | dict[str, float | Callable] | None = None,
        free_coupling: float | Callable | None = None,
        ground_state_only: bool = False,
        bra_state: np.ndarray | list | None = None,
    ) -> complex:
        """Compute energy expectation/overlap from the evaluated Hamiltonian matrix."""
        H = self.evaluate(
            d,
            mass=mass,
            coupling=coupling,
            free_coupling=free_coupling,
            ground_state_only=ground_state_only,
        )
        eval_mass = mass if mass is not None else self.default_mass
        eval_h = EvaluatedObservable(
            matrices={d: H},
            bases={d: self.filtered_basis(d, ground_state_only=ground_state_only)},
            d_values=[d],
            mass=eval_mass,
            label="hamiltonian",
            Lambda=self.Lambda,
            ground_state_only=ground_state_only,
            is_even=self.is_even,
            is_hermitian=self.is_hermitian,
        )
        return eval_h.expectation_value(d, state, bra_state=bra_state)

    def average_excitation_number(
        self,
        d: int | float,
        state: np.ndarray | list,
        *,
        ground_state_only: bool = False,
        bra_state: np.ndarray | list | None = None,
        return_std: bool = False,
    ) -> float | complex | tuple[float | complex, float]:
        """Compute ``<N>`` (or overlap) and optionally its standard deviation.

        ``N`` is defined on the filtered basis by ``N[R, R] = |R|``.

        Parameters
        ----------
        d : int or float
            Dimension key for cached evaluations.
        state : np.ndarray or list
            Pure state vector or density matrix.
        ground_state_only : bool
            Restrict to even-excitation states.
        bra_state : np.ndarray or list or None
            Optional bra state to compute overlap ``<phi|N|psi>``.
        return_std : bool
            If True, also return standard deviation
            ``sqrt(<(N-<N>)^2>)``. This requires expectation mode
            (``bra_state is None``).
        """
        if return_std and bra_state is not None:
            raise ValueError(
                "Standard deviation is only defined for expectation values "
                "(bra_state must be None)."
            )

        d_key: int | float = d
        free_eval = self._get_cached_free_evaluation(
            [d_key],
            mass=1.0,
            ground_state_only=ground_state_only,
        )
        if d_key not in free_eval.matrices:
            raise KeyError(f"FreeHamiltonian evaluation missing key for d={d_key}")

        free_matrix = _to_dense_array(free_eval.matrices[d_key])
        excit_diag = np.real(np.diag(free_matrix) - (float(d) ** 2) / 2.0)
        number_op = np.diag(excit_diag)

        number_eval = EvaluatedObservable(
            matrices={d_key: number_op},
            bases={d_key: free_eval.bases[d_key]},
            d_values=[d_key],
            mass=1.0,
            label="excitation_number",
            Lambda=self.Lambda,
            ground_state_only=ground_state_only,
            is_even=True,
            is_hermitian=True,
        )

        mean = number_eval.expectation_value(d_key, state, bra_state=bra_state)
        if not return_std:
            return mean

        mean_scalar = float(np.real(mean))
        centered = number_op - mean_scalar * np.eye(number_op.shape[0])
        centered_sq = centered @ centered

        variance_eval = EvaluatedObservable(
            matrices={d_key: centered_sq},
            bases={d_key: free_eval.bases[d_key]},
            d_values=[d_key],
            mass=1.0,
            label="excitation_number_centered_squared",
            Lambda=self.Lambda,
            ground_state_only=ground_state_only,
            is_even=True,
            is_hermitian=True,
        )
        variance = float(np.real(variance_eval.expectation_value(d_key, state)))
        std = float(np.sqrt(max(variance, 0.0)))
        return mean, std

    def evaluate(
        self,
        d: int | float,
        *,
        mass: float | None = None,
        coupling: float | Callable | dict[str, float | Callable] | None = None,
        free_coupling: float | Callable | None = None,
        ground_state_only: bool = False,
    ) -> EvaluatedMatrix:
        r"""Evaluate the full Hamiltonian at a specific *d*.

        .. math::

            H = c_\text{free}\,H_\text{free}(m)
              + \sum_i g_i(d, m)\,O_i(d)

        Parameters
        ----------
        d : int or float
            Dimension parameter (typically a positive integer).
        mass : float or None
            Evaluation mass.  If ``None``, uses ``self.default_mass``.
        coupling : float, callable, dict, or None
            Coupling specification for observables.

            - ``None`` → 1.0 for all observables.
            - ``float`` → same constant for all observables.
            - ``callable`` → applied to all observables.  Signature is
              either ``g(d)`` or ``g(d, m)`` (auto-detected).
            - ``dict[str, ...]`` → per-observable, keyed by label.

        free_coupling : float, callable, or None
            Coupling for the free Hamiltonian.  If ``None``, uses
            ``self.free_coupling`` (default 1.0).
        ground_state_only : bool
            Restrict to the even-excitation (ground-state) sector.

        Returns
        -------
        np.ndarray or scipy.sparse.csr_matrix
            Full Hamiltonian matrix, shape ``(n, n)``.
        """
        raise NotImplementedError

    def _initialize_accumulator(
        self,
        free_matrix: EvaluatedMatrix,
        c_free: float,
    ) -> EvaluatedMatrix:
        """Initialize the Hamiltonian accumulator from the free part."""
        raise NotImplementedError

    def _add_weighted_matrix(
        self,
        accumulator: EvaluatedMatrix,
        weight: float,
        matrix: EvaluatedMatrix,
    ) -> EvaluatedMatrix:
        """Add a weighted observable matrix to the accumulator."""
        raise NotImplementedError

    def _empty_matrix(self) -> EvaluatedMatrix:
        """Return the empty Hamiltonian matrix for this storage kind."""
        raise NotImplementedError

    def _evaluate_impl(
        self,
        d: int | float,
        *,
        mass: float | None = None,
        coupling: float | Callable | dict[str, float | Callable] | None = None,
        free_coupling: float | Callable | None = None,
        ground_state_only: bool = False,
    ) -> EvaluatedMatrix:
        """Shared evaluation implementation for dense and sparse Hamiltonians."""
        if not isinstance(d, (int, float, np.integer, np.floating)):
            raise TypeError(
                "Hamiltonian.evaluate currently supports scalar d only; "
                "use Observable.evaluate for batched d values."
            )
        d_scalar = float(d)
        eval_mass = mass if mass is not None else self.default_mass
        d_key: int | float = d

        # 1. Build or fetch free Hamiltonian from cache
        free_result = self._get_cached_free_evaluation(
            [d_key],
            mass=eval_mass,
            ground_state_only=ground_state_only,
        )
        if d_key not in free_result.matrices:
            raise KeyError(f"FreeHamiltonian evaluation missing key for d={d_key}")
        free_matrix = free_result.matrices[d_key]
        assert free_matrix is not None
        shape = free_matrix.shape
        assert shape is not None
        n = shape[0]

        if n == 0:
            return self._empty_matrix()

        # Resolve free coupling
        if free_coupling is None:
            c_free = self.free_coupling
            if callable(c_free):
                c_free = _resolve_coupling(c_free, d_scalar, eval_mass)
        else:
            c_free = _resolve_coupling(free_coupling, d_scalar, eval_mass)

        H = self._initialize_accumulator(free_matrix, float(c_free))

        # 2. Evaluate each observable (cache-backed, missing d only)
        for obs in self.observables:
            obs_result = self._get_cached_observable_evaluation(
                obs,
                [d_key],
                mass=eval_mass,
                ground_state_only=ground_state_only,
            )
            if d_key not in obs_result.matrices:
                raise KeyError(
                    f"Observable '{obs.label}' evaluation missing key for d={d_key}"
                )
            obs_matrix = obs_result.matrices[d_key]
            g = _resolve_coupling(
                _get_obs_coupling(coupling, obs.label), d_scalar, eval_mass
            )
            H = self._add_weighted_matrix(H, g, obs_matrix)

        return H

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary string."""
        labels = self.labels
        if len(self.observables) == 1:
            obs = self.observables[0]
            n_nonzero = obs.nonzero_entry_count()
            return (
                f"Hamiltonian(label={obs.label}, Λ={self.Lambda}, "
                f"storage={self.storage_kind}, "
                f"mass={self.default_mass}, "
                f"basis_size={self.size}, "
                f"max_poly_degree={obs.max_degree}, "
                f"nonzero_elements={n_nonzero})"
            )
        return (
            f"Hamiltonian(observables={labels}, Λ={self.Lambda}, "
            f"storage={self.storage_kind}, "
            f"mass={self.default_mass}, "
            f"basis_size={self.size}, "
            f"n_observables={len(self.observables)})"
        )

    def __repr__(self) -> str:  # noqa: D105
        return self.summary()


@dataclasses.dataclass(repr=False)
class HamiltonianDense(Hamiltonian):
    """Hamiltonian whose evaluated matrices are dense NumPy arrays."""

    @property
    def storage_kind(self) -> str:
        return "dense"

    def _initialize_accumulator(
        self,
        free_matrix: EvaluatedMatrix,
        c_free: float,
    ) -> np.ndarray:
        matrix = _to_dense_array(free_matrix)
        return c_free * matrix

    def _add_weighted_matrix(
        self,
        accumulator: EvaluatedMatrix,
        weight: float,
        matrix: EvaluatedMatrix,
    ) -> np.ndarray:
        dense_acc = _to_dense_array(accumulator)
        dense_matrix = _to_dense_array(matrix)
        if np.iscomplexobj(dense_matrix) and not np.iscomplexobj(dense_acc):
            dense_acc = dense_acc.astype(np.complex128)
        return dense_acc + weight * dense_matrix

    def _empty_matrix(self) -> np.ndarray:
        return np.empty((0, 0), dtype=np.float64)

    def evaluate(
        self,
        d: int | float,
        *,
        mass: float | None = None,
        coupling: float | Callable | dict[str, float | Callable] | None = None,
        free_coupling: float | Callable | None = None,
        ground_state_only: bool = False,
    ) -> np.ndarray:
        return np.asarray(
            self._evaluate_impl(
                d,
                mass=mass,
                coupling=coupling,
                free_coupling=free_coupling,
                ground_state_only=ground_state_only,
            )
        )


@dataclasses.dataclass(repr=False)
class HamiltonianSparse(Hamiltonian):
    """Hamiltonian whose evaluated matrices are sparse CSR matrices."""

    @property
    def storage_kind(self) -> str:
        return "sparse"

    def _initialize_accumulator(
        self,
        free_matrix: EvaluatedMatrix,
        c_free: float,
    ) -> sparse.csr_matrix:
        matrix = _to_csr_matrix(free_matrix)
        return (c_free * matrix).tocsr()

    def _add_weighted_matrix(
        self,
        accumulator: EvaluatedMatrix,
        weight: float,
        matrix: EvaluatedMatrix,
    ) -> sparse.csr_matrix:
        acc = _to_csr_matrix(accumulator)
        rhs = _to_csr_matrix(matrix)
        return (acc + weight * rhs).tocsr()

    def _empty_matrix(self) -> sparse.csr_matrix:
        return _empty_sparse_matrix(0, np.float64)

    def evaluate(
        self,
        d: int | float,
        *,
        mass: float | None = None,
        coupling: float | Callable | dict[str, float | Callable] | None = None,
        free_coupling: float | Callable | None = None,
        ground_state_only: bool = False,
    ) -> sparse.csr_matrix:
        return _to_csr_matrix(
            self._evaluate_impl(
                d,
                mass=mass,
                coupling=coupling,
                free_coupling=free_coupling,
                ground_state_only=ground_state_only,
            )
        )


# ======================================================================
# Coupling resolution helpers
# ======================================================================


def _resolve_coupling(
    spec: float | Callable | None,
    d: float,
    m: float,
) -> float:
    """Resolve a coupling specification to a float value.

    Callables may accept either ``(d,)`` or ``(d, m)``; the arity is
    auto-detected via :func:`inspect.signature`.
    """
    if spec is None:
        return 1.0
    if not callable(spec):
        return float(spec)
    # Auto-detect arity
    try:
        sig = inspect.signature(spec)
        n_params = len(
            [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
        )
    except (ValueError, TypeError):
        n_params = 1
    if n_params >= 2:
        return float(spec(d, m))
    return float(spec(d))


def _get_obs_coupling(
    coupling: float | Callable | dict | None,
    label: str,
) -> float | Callable | None:
    """Extract the coupling for a specific observable from a coupling spec."""
    if isinstance(coupling, dict):
        return coupling.get(label, 1.0)
    return coupling


# ======================================================================
# Label / filename helpers
# ======================================================================


def _label_to_ops_spec(label: str) -> str:
    """Convert a Python label to the ``--ops`` format used by ``generate_data.sh``.

    ``"XX_p21"`` → ``"XX:21"``
    ``"XXXX_p2341"`` → ``"XXXX:2341"``
    """
    if "_p" not in label:
        raise ValueError(
            f"Label '{label}' does not match the expected '{'{ops}_p{perm}'}' "
            f"format (missing '_p' separator)"
        )
    ops, perm = label.split("_p", 1)
    return f"{ops}:{perm}"


def _mass_to_filename_str(mass: float | str) -> str:
    """Format a mass value the way Mathematica's ``N[mass]`` does.

    Mathematica's ``ToString[N[mass]]`` produces:

    - ``"0.5"`` for ``1/2``
    - ``"1."`` for ``1``
    - ``"2."`` for ``2``
    """
    if isinstance(mass, str):
        if "/" in mass:
            num, den = mass.split("/", 1)
            mass = float(num) / float(den)
        else:
            mass = float(mass)

    s = str(float(mass))
    if s.endswith(".0"):
        s = s[:-1]
    return s


def _label_to_woven_filename(label: str, mass: float | str, Lambda: int) -> str:
    """Build the expected woven-contraction JSON filename for a label.

    ``"XX_p21", mass=1, Lambda=6`` → ``"wc_op_XX_p21_m1._Lambda6.json"``
    """
    ops, perm = label.split("_p", 1)
    m_str = _mass_to_filename_str(mass)
    return f"wc_op_{ops}_p{perm}_m{m_str}_Lambda{Lambda}.json"


# ======================================================================
# Pipeline: build observables
# ======================================================================


def _build_observable_from_woven(
    woven: WovenData,
    ct_dir: str | Path,
    gap_coset_path: str | Path | None = None,
    *,
    storage: str = "dense",
    verbose: bool = True,
    parallel: bool = False,
    max_workers: int | None = None,
) -> Observable:
    """Build a single Observable from woven data via efficient contractions."""
    from sym_contractions.efficient import compute_all_contractions_efficient

    # Exploit Hermiticity: skip lower-diagonal excitation groups
    w = (
        woven.filter_upper_diagonal_excitations()
        if woven.is_hermitian is True
        else woven
    )

    contractions = compute_all_contractions_efficient(
        w,
        ct_dir,
        gap_coset_path,
        verbose=verbose,
        parallel=parallel,
        max_workers=max_workers,
    )
    return Observable.from_data(
        contractions,
        woven,
        storage=storage,
        verbose=verbose,
        parallel=parallel,
        max_workers=max_workers,
    )


def build_observables(
    labels: list[str],
    *,
    Lambda: int = 14,
    mass: float | str = 0.5,
    clean: bool = False,
    verbose: bool = True,
    save: bool = True,
    storage: str = "dense",
    parallel: bool = False,
    max_workers: int | None = None,
) -> list[Observable]:
    r"""Run the full data-generation pipeline and return Observable objects.

    For each label:

    1. Check ``OBSERVABLE_DIR`` for a saved ``.npz`` file.  If found,
       load and rescale the mass accordingly.
    2. Otherwise, run ``generate_data.sh``, compute contractions, and
       build the Observable from scratch.
    3. If *save* is ``True``, save each newly computed Observable.

    Parameters
    ----------
    labels : list[str]
        Operator labels in Python format, e.g. ``["XXXX_p2341"]``.
    Lambda : int
        Cutoff parameter.
    mass : float or str
        Mass parameter.  Can be a string fraction (``"1/2"``).
    clean : bool
        If ``True``, run ``clean_data.sh`` before generating data.
    verbose : bool
        Print progress messages.
    save : bool
        If ``True``, save each newly computed observable to
        ``OBSERVABLE_DIR``.
    storage : str
        Storage format for the resulting Observables.  Either ``"dense"`` or
        ``"sparse"``.  This controls the type of the returned Observables
        (``ObservableDense`` vs ``ObservableSparse``) and the format of their evaluated matrices.
    parallel : bool
        Whether to parallelize the efficient contraction route and, for
        sparse storage only, the sparse assembly path.
    max_workers : int or None
        Optional upper bound on worker processes used when ``parallel`` is enabled.

    Returns
    -------
    list[Observable]
    """
    if not labels:
        raise ValueError("labels list must not be empty")
    if storage not in {"dense", "sparse"}:
        raise ValueError("storage must be either 'dense' or 'sparse'")

    from sym_contractions import (
        CHARACTER_TABLE_DIR,
        DATA_ROOT,
        OBSERVABLE_DIR,
        PROJECT_ROOT,
    )

    # Check which observables are already saved
    result: list[Observable] = []
    labels_to_compute: list[str] = []
    observable_cls = ObservableDense if storage == "dense" else ObservableSparse

    for label in labels:
        expected_filename = _observable_filename(label, Lambda, storage)
        saved_path = OBSERVABLE_DIR / expected_filename
        if saved_path.exists():
            if verbose:
                print(f"Loading saved observable: {saved_path}")
            obs = observable_cls.load(saved_path)
            result.append(obs)
        else:
            labels_to_compute.append(label)
            result.append(None)  # type: ignore[arg-type]  # placeholder

    if labels_to_compute:
        # Run data generation for missing observables
        data_dir = PROJECT_ROOT / "data"
        clean_script = data_dir / "clean_data.sh"
        gen_script = data_dir / "generate_data.sh"

        woven_dir = DATA_ROOT / "woven_contractions"
        coset_path = DATA_ROOT / "coset_reps" / "coset_reps.json"

        if clean:
            if verbose:
                print("Running clean_data.sh ...")
            subprocess.run(
                ["bash", str(clean_script)],
                check=True,
                capture_output=not verbose,
            )

        # Generate data
        ops_csv = ",".join(_label_to_ops_spec(lbl) for lbl in labels_to_compute)
        mass_str = str(mass)

        cmd: list[str] = [
            "bash",
            str(gen_script),
            "--lambda",
            str(Lambda),
            "--ops",
            ops_csv,
            "--mass",
            mass_str,
        ]

        if verbose:
            print(f"Running generate_data.sh with: {' '.join(cmd[2:])}")
        subprocess.run(cmd, check=True, capture_output=not verbose)

        # Build observables from woven data
        from sym_contractions.woven import load_woven_json

        for label in labels_to_compute:
            fname = _label_to_woven_filename(label, mass, Lambda)
            woven_path = woven_dir / fname
            if not woven_path.exists():
                raise FileNotFoundError(
                    f"Expected woven JSON not found after generation: {woven_path}"
                )

            if not coset_path.exists():
                raise FileNotFoundError(
                    f"Coset representatives file not found: {coset_path}"
                )

            woven = load_woven_json(woven_path)

            if verbose:
                print(
                    f"\nComputing observable '{label}' "
                    f"(is_hermitian={woven.is_hermitian})"
                )

            obs = _build_observable_from_woven(
                woven,
                CHARACTER_TABLE_DIR,
                coset_path,
                storage=storage,
                verbose=verbose,
                parallel=parallel,
                max_workers=max_workers,
            )

            if save:
                save_path = OBSERVABLE_DIR / obs.filename
                if verbose:
                    print(f"Saving observable: {save_path}")
                obs.save(save_path)

            # Fill placeholder
            idx = labels.index(label)
            result[idx] = obs

    return result
