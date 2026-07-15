"""Efficient contraction computation via coset reduction + character theory.

Replaces the brute-force O(n!×m!) probability computation with an
algebraic approach combining two ingredients:

1. **Coset reduction** on one side (left or right) replaces the
   exhaustive sum over a symmetric group with a sum over coset
   representatives of S_n/H_n (or S_m/H_m).

2. **Character theory** (connection coefficients + orthogonality of
   characters) eliminates the sum over the OTHER side entirely,
   replacing it with character values and the "S-polynomial".

For left-side reduction the formula is:

.. math::

    c_k(R, S) = \\frac{|H_n|}{(n!)^2\\, m!\\, f_S}
      \\sum_{\\alpha \\vdash n} |C_\\alpha|\\, \\chi_R(\\alpha)
      \\sum_{g \\in S_n/H_n}
        \\chi^S(\\rho_{g,\\alpha})\\,
        S_S(k - c_{F,g,\\alpha})

where:

* g ranges over coset representatives of S_n / H_n.
* σ_g = g ∘ σ₀ ∘ g⁻¹  (σ₀ is a canonical representative of class α).
* τ′ = τ ∘ (σ_g × id).
* c_{F,g,α} = number of first-block-only cycles of τ′.
* ρ_{g,α} ∈ S_m = effective ("return map") permutation on the second block.
* S_S(j)   = Σ_{μ ⊢ m, ℓ(μ)=j} χ^S(μ).
* f_S      = dim(S) = χ^S(identity class).

An analogous formula holds for right-side reduction.
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sym_contractions._numba_kernels import (
    batch_fused_left,
    batch_fused_right,
    build_padded_cycle_types,
)
from sym_contractions.bruteforce import get_cycle_type
from sym_contractions.character_tables import (
    CharacterTableData,
    compute_class_fraction,
    load_character_table,
)
from sym_contractions.coset import (
    CosetReductionData,
    GapCosetData,
    compute_coset_reduction_from_gap,
    load_gap_coset_data,
)
from sym_contractions.utils import canonical_representative, enumerate_partitions
from sym_contractions.woven import (
    ContractionEntry,
    ContractionResult,
    WovenData,
    WovenGroup,
    tau_to_pairs_1indexed,
)

_MAX_PARALLEL_WORKERS = 8
_WORKER_CT_DIR: Path | None = None
_WORKER_GAP_COSET_PATH: Path | None = None
_WORKER_CT_CACHE: dict[int, CharacterTableData] = {}
_WORKER_GAP_DATA: GapCosetData | None = None


@dataclasses.dataclass(frozen=True)
class _ContractionTask:
    nL: int
    nR: int
    tau: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _ContractionTaskResult:
    task: _ContractionTask
    coeffs_all: np.ndarray
    reduced_side: str | None
    num_reps: int | None


# ======================================================================
# Helper functions
# ======================================================================


def compute_s_polynomial(ct: CharacterTableData) -> np.ndarray:
    r"""Compute the *normalized* class-size-weighted S-polynomial.

    .. math::

        \hat s_R(j) = \sum_{\substack{\mu \vdash n \\ \ell(\mu) = j}}
          \frac{|C_\mu|}{n!}\;\chi^R(\mu).

    Using the class fraction :math:`|C_\mu|/n!` instead of the raw
    class size avoids integer overflow for large *n* (n > 20).

    Parameters
    ----------
    ct : CharacterTableData
        Character table for S_n.

    Returns
    -------
    np.ndarray
        Shape ``(p, n + 1)`` where ``p = len(ct['cycle_types'])``.
        Entry ``[r, j]`` is :math:`\hat s_{R_r}(j)`.
    """
    n = ct["n"]
    p = len(ct["cycle_types"])

    if n == 0:
        # S_0: one irrep (trivial), one partition (empty) with ℓ(()) = 0.
        result = np.zeros((1, 1), dtype=np.float64)
        result[0, 0] = 1.0
        return result

    result = np.zeros((p, n + 1), dtype=np.float64)
    chars = ct["characters"].astype(np.float64)  # (p, p)

    for i, mu in enumerate(ct["cycle_types"]):
        num_parts = len(mu)  # ℓ(μ)
        class_frac = compute_class_fraction(mu)  # |C_μ| / n!
        result[:, num_parts] += class_frac * chars[:, i]

    return result


def compute_rep_dimensions(ct: CharacterTableData) -> np.ndarray:
    r"""Get the dimension *f^R* = χ^R(identity) for every irrep.

    In the reverse-lex ordered character table the identity class
    (cycle type ``(1, 1, …, 1)``) is the **last** column.

    Parameters
    ----------
    ct : CharacterTableData

    Returns
    -------
    np.ndarray
        Shape ``(p,)`` with ``p = len(ct['cycle_types'])``.
    """
    return ct["characters"][:, -1].astype(np.float64)


def _build_ct_index(cycle_types: list[list[int]]) -> dict[tuple[int, ...], int]:
    """Build a dict mapping cycle-type tuples to their index.

    O(1) lookup replacing the O(p) linear scan of
    :func:`find_cycle_type_index` on cold paths.
    """
    return {tuple(ct): i for i, ct in enumerate(cycle_types)}


def _load_ct(n: int, ct_dir: Path) -> CharacterTableData:
    """Load or synthesise a character table for S_n."""
    ct_path = ct_dir / f"ct_{n}.json"
    if ct_path.exists():
        return load_character_table(ct_path, safe_int=True)
    if n == 0:
        return {
            "n": 0,
            "cycle_types": [[]],
            "class_sizes": np.array([1], dtype=np.int32),
            "characters": np.array([[1]], dtype=np.int32),
        }
    raise FileNotFoundError(f"Character table not found: {ct_path}")


def _init_contraction_worker(ct_dir: str, gap_coset_path: str | None) -> None:
    """Initialize per-process caches for contraction workers."""
    global _WORKER_CT_DIR, _WORKER_GAP_COSET_PATH, _WORKER_CT_CACHE, _WORKER_GAP_DATA

    _WORKER_CT_DIR = Path(ct_dir)
    _WORKER_GAP_COSET_PATH = None if gap_coset_path is None else Path(gap_coset_path)
    _WORKER_CT_CACHE = {}
    _WORKER_GAP_DATA = None


def _load_worker_ct(n: int) -> CharacterTableData:
    """Load a character table from the worker-local cache."""
    if _WORKER_CT_DIR is None:
        raise RuntimeError("Contraction worker was not initialized")
    if n not in _WORKER_CT_CACHE:
        _WORKER_CT_CACHE[n] = _load_ct(n, _WORKER_CT_DIR)
    return _WORKER_CT_CACHE[n]


def _load_worker_gap_data() -> GapCosetData | None:
    """Load GAP coset data lazily inside a worker process."""
    global _WORKER_GAP_DATA

    if _WORKER_GAP_COSET_PATH is None:
        return None
    if _WORKER_GAP_DATA is None:
        _WORKER_GAP_DATA = load_gap_coset_data(_WORKER_GAP_COSET_PATH)
    return _WORKER_GAP_DATA


def _resolve_parallel_workers(
    task_count: int,
    max_workers: int | None,
) -> int:
    """Choose a conservative worker count for CPU-bound contractions."""
    if task_count <= 1:
        return 1
    requested = max_workers if max_workers is not None else (os.cpu_count() or 1)
    return max(1, min(task_count, requested, _MAX_PARALLEL_WORKERS))


def _require_gap_data(
    gap_data: GapCosetData | None,
    *,
    n: int,
    m: int,
) -> GapCosetData:
    """Validate that GAP coset data is available for non-trivial sectors."""
    if gap_data is None:
        raise ValueError(
            "GAP precomputed coset data is required. "
            "Load it with load_gap_coset_data() and pass as gap_data="
        )
    return gap_data


def _compute_tau_result(
    tau: tuple[int, ...],
    n: int,
    m: int,
    ct_n: CharacterTableData,
    ct_m: CharacterTableData,
    gap_data: GapCosetData | None,
) -> _ContractionTaskResult:
    """Compute the contraction tensor for a single ``(nL, nR, tau)`` task."""
    coset_data: CosetReductionData | None = None
    reduced_side: str | None = None
    num_reps: int | None = None
    if n > 0 and m > 0:
        resolved_gap_data = _require_gap_data(gap_data, n=n, m=m)
        coset_data = compute_coset_reduction_from_gap(tau, n, m, resolved_gap_data)
        reduced_side = coset_data.reduced_side
        num_reps = coset_data.num_reps

    coeffs_all = compute_contraction_efficient(tau, n, m, ct_n, ct_m, coset_data)
    return _ContractionTaskResult(
        task=_ContractionTask(nL=n, nR=m, tau=tau),
        coeffs_all=coeffs_all,
        reduced_side=reduced_side,
        num_reps=num_reps,
    )


def _compute_tau_result_worker(task: _ContractionTask) -> _ContractionTaskResult:
    """Worker entry point for one tau task."""
    ct_n = _load_worker_ct(task.nL)
    ct_m = _load_worker_ct(task.nR)
    gap_data = _load_worker_gap_data()
    return _compute_tau_result(task.tau, task.nL, task.nR, ct_n, ct_m, gap_data)


# ======================================================================
# Core computation — left / right / trivial
# ======================================================================


def _contraction_reduce_left(
    tau: tuple[int, ...],
    n: int,
    m: int,
    ct_n: CharacterTableData,
    ct_m: CharacterTableData,
    coset_data: CosetReductionData,
) -> np.ndarray:
    r"""Contraction coefficients via left-side coset reduction.

    Uses a two-phase approach:

    1. **Batch kernel** — compute all ``(c_F, j_rho)`` values for every
       ``(alpha, g)`` pair in a single Numba call.
    2. **Grouped vectorized accumulation** — group pairs by their
       ``(c_F, j_rho)`` outcome, aggregate class-size weights, then
       accumulate via matrix–vector products and broadcast adds.

    Returns shape ``(p_n, p_m, n+m+1)``.
    """
    nm = n + m
    p_n = len(ct_n["cycle_types"])
    p_m = len(ct_m["cycle_types"])

    s_poly = compute_s_polynomial(ct_m)  # (p_m, m+1)  — normalized
    f_m = compute_rep_dimensions(ct_m)  # (p_m,)

    chars_n = ct_n["characters"].astype(np.float64)  # (p_n, p_n)
    chars_m = ct_m["characters"].astype(np.float64)  # (p_m, p_m)

    # Class fractions |C_α| / n!  (overflow-safe)
    class_fracs_n = np.array(
        [compute_class_fraction(ct_n["cycle_types"][i]) for i in range(p_n)],
        dtype=np.float64,
    )  # (p_n,)

    reps = coset_data.coset_reps  # (num_reps, n)

    # --- Phase 1: batch Numba kernel ---
    # Build canonical representatives for every conjugacy class
    canonical_arr = np.array(
        [
            canonical_representative(tuple(ct_n["cycle_types"][i]), n)
            for i in range(p_n)
        ],
        dtype=np.int32,
    )  # (p_n, n)
    tau_arr = np.asarray(tau, dtype=np.int32)
    ct_m_padded, ct_m_lengths = build_padded_cycle_types(ct_m["cycle_types"])

    c_F_all, j_rho_all = batch_fused_left(
        tau_arr, canonical_arr, reps, n, m, ct_m_padded, ct_m_lengths, p_m
    )  # both (p_n, num_reps)

    if np.any(j_rho_all < 0):
        raise ValueError(f"Cycle type not found in S_{m} character table")

    # --- Phase 2: group by (c_F, j_rho) and accumulate vectorised ---
    # Encode (c_F, j_rho) as a single key: c_F * p_m + j_rho
    # Then for each unique key, aggregate the class-size weight per alpha.
    groups: dict[int, np.ndarray] = {}
    num_reps = reps.shape[0]
    for alpha_idx in range(p_n):
        cf = class_fracs_n[alpha_idx]
        for g_idx in range(num_reps):
            key = int(c_F_all[alpha_idx, g_idx]) * p_m + int(
                j_rho_all[alpha_idx, g_idx]
            )
            if key not in groups:
                groups[key] = np.zeros(p_n, dtype=np.float64)
            groups[key][alpha_idx] += cf

    result = np.zeros((p_n, p_m, nm + 1), dtype=np.float64)

    for key, weight in groups.items():
        c_F = key // p_m
        j_rho = key % p_m

        # agg_R[r] = \sum_alpha weight[alpha] * chi_R[r, alpha]
        agg_R = chars_n @ weight  # (p_n,)

        # free-side contribution: chi_S[s, j_rho] / f_S[s] * S_S[s, j]
        chi_s_scaled = chars_m[:, j_rho] / f_m  # (p_m,)
        jmax = min(m, nm - c_F)
        free_block = chi_s_scaled[:, None] * s_poly[:, : jmax + 1]  # (p_m, jmax+1)

        # Broadcast accumulate: result[r, s, c_F..c_F+jmax] += agg_R[r] * free[s,j]
        result[:, :, c_F : c_F + jmax + 1] += (
            agg_R[:, None, None] * free_block[None, :, :]
        )

    prefactor = 1.0 / coset_data.num_reps
    result *= prefactor
    return result


def _contraction_reduce_right(
    tau: tuple[int, ...],
    n: int,
    m: int,
    ct_n: CharacterTableData,
    ct_m: CharacterTableData,
    coset_data: CosetReductionData,
) -> np.ndarray:
    r"""Contraction coefficients via right-side coset reduction.

    Analogous to :func:`_contraction_reduce_left` but reduces
    on the right (S_m) side.

    Returns shape ``(p_n, p_m, n+m+1)``.
    """
    nm = n + m
    p_n = len(ct_n["cycle_types"])
    p_m = len(ct_m["cycle_types"])

    s_poly = compute_s_polynomial(ct_n)  # (p_n, n+1)  — normalized
    f_n = compute_rep_dimensions(ct_n)  # (p_n,)

    chars_n = ct_n["characters"].astype(np.float64)  # (p_n, p_n)
    chars_m = ct_m["characters"].astype(np.float64)  # (p_m, p_m)

    # Class fractions |C_β| / m!  (overflow-safe)
    class_fracs_m = np.array(
        [compute_class_fraction(ct_m["cycle_types"][i]) for i in range(p_m)],
        dtype=np.float64,
    )  # (p_m,)

    reps = coset_data.coset_reps  # (num_reps, m)

    # --- Phase 1: batch Numba kernel ---
    canonical_arr = np.array(
        [
            canonical_representative(tuple(ct_m["cycle_types"][i]), m)
            for i in range(p_m)
        ],
        dtype=np.int32,
    )  # (p_m, m)
    tau_arr = np.asarray(tau, dtype=np.int32)
    ct_n_padded, ct_n_lengths = build_padded_cycle_types(ct_n["cycle_types"])

    c_F_all, j_rho_all = batch_fused_right(
        tau_arr, canonical_arr, reps, n, m, ct_n_padded, ct_n_lengths, p_n
    )  # both (p_m, num_reps)

    if np.any(j_rho_all < 0):
        raise ValueError(f"Cycle type not found in S_{n} character table")

    # --- Phase 2: group by (c_F, j_rho) and accumulate vectorised ---
    groups: dict[int, np.ndarray] = {}
    num_reps = reps.shape[0]
    for beta_idx in range(p_m):
        cf = class_fracs_m[beta_idx]
        for g_idx in range(num_reps):
            key = int(c_F_all[beta_idx, g_idx]) * p_n + int(j_rho_all[beta_idx, g_idx])
            if key not in groups:
                groups[key] = np.zeros(p_m, dtype=np.float64)
            groups[key][beta_idx] += cf

    result = np.zeros((p_n, p_m, nm + 1), dtype=np.float64)

    for key, weight in groups.items():
        c_F = key // p_n
        j_rho = key % p_n

        # agg_S[s] = \sum_beta weight[beta] * chi_S[s, beta]
        agg_S = chars_m @ weight  # (p_m,)

        # free-side contribution: chi_R[r, j_rho] / f_R[r] * S_R[r, j]
        chi_r_scaled = chars_n[:, j_rho] / f_n  # (p_n,)
        jmax = min(n, nm - c_F)
        free_block = chi_r_scaled[:, None] * s_poly[:, : jmax + 1]  # (p_n, jmax+1)

        # result[r, s, c_F..c_F+jmax] += free[r, j] * agg_S[s]
        result[:, :, c_F : c_F + jmax + 1] += (
            free_block[:, None, :] * agg_S[None, :, None]
        )

    prefactor = 1.0 / coset_data.num_reps
    result *= prefactor
    return result


def _contraction_trivial(
    tau: tuple[int, ...],
    n: int,
    m: int,
    ct_n: CharacterTableData,
    ct_m: CharacterTableData,
) -> np.ndarray:
    r"""Contraction coefficients when n = 0 or m = 0.

    When n = 0, using the normalized S-polynomial
    :math:`\hat s_S(k) = \sum_{\mu:\ell(\mu)=k} (|C_\mu|/m!)\,\chi^S(\mu)`:

    .. math::

        c_k((), S) = \frac{\chi^S(\tau)}{f_S}\, \hat s_S(k)

    Symmetrically for m = 0.

    Returns shape ``(p_n, p_m, n+m+1)``.
    """
    if n == 0 and m == 0:
        # Only one partition pair: ((), ()).  τ = () has 0 cycles.
        return np.ones((1, 1, 1), dtype=np.float64)

    if n == 0:
        p_m = len(ct_m["cycle_types"])
        s_poly = compute_s_polynomial(ct_m)  # (p_m, m+1)
        f_m = compute_rep_dimensions(ct_m)  # (p_m,)
        tau_ct = get_cycle_type(list(tau))
        ct_index = _build_ct_index(ct_m["cycle_types"])
        j_tau = ct_index.get(tuple(tau_ct))
        if j_tau is None:
            raise ValueError(f"Cycle type {tau_ct} not found in S_{m} character table")
        chars_m = ct_m["characters"].astype(np.float64)
        chi_tau = chars_m[:, j_tau]  # (p_m,)

        result = np.zeros((1, p_m, m + 1), dtype=np.float64)
        for s in range(p_m):
            result[0, s, :] = chi_tau[s] / f_m[s] * s_poly[s, :]
        return result

    # m == 0
    p_n = len(ct_n["cycle_types"])
    s_poly = compute_s_polynomial(ct_n)  # (p_n, n+1)
    f_n = compute_rep_dimensions(ct_n)  # (p_n,)
    tau_ct = get_cycle_type(list(tau))
    ct_index = _build_ct_index(ct_n["cycle_types"])
    j_tau = ct_index.get(tuple(tau_ct))
    if j_tau is None:
        raise ValueError(f"Cycle type {tau_ct} not found in S_{n} character table")
    chars_n = ct_n["characters"].astype(np.float64)
    chi_tau = chars_n[:, j_tau]  # (p_n,)

    result = np.zeros((p_n, 1, n + 1), dtype=np.float64)
    for r in range(p_n):
        result[r, 0, :] = chi_tau[r] / f_n[r] * s_poly[r, :]
    return result


# ======================================================================
# Public API
# ======================================================================


def compute_contraction_efficient(
    tau: tuple[int, ...] | list[int],
    n: int,
    m: int,
    ct_n: CharacterTableData,
    ct_m: CharacterTableData,
    coset_data: CosetReductionData | None = None,
) -> np.ndarray:
    """Compute contraction coefficients for a fixed τ.

    Dispatches to left or right coset reduction based on which side
    has fewer coset representatives.  For the trivial cases ``n = 0``
    or ``m = 0`` the result is computed algebraically without coset
    data.

    Parameters
    ----------
    tau : sequence of int
        Permutation in S_{n+m}, 0-indexed.
    n, m : int
        Block sizes.
    ct_n, ct_m : CharacterTableData
        Character tables for S_n and S_m.
    coset_data : CosetReductionData or None
        Precomputed coset reduction data.  Required when both n > 0
        and m > 0.

    Returns
    -------
    np.ndarray
        Shape ``(p_n, p_m, n+m+1)`` where ``result[r, s, k]`` is the
        coefficient of ``d^k`` for the contraction ``(R_r, S_s)``.
    """
    tau = tuple(tau)

    if n == 0 or m == 0:
        return _contraction_trivial(tau, n, m, ct_n, ct_m)

    if coset_data is None:
        raise ValueError("coset_data is required when n > 0 and m > 0")

    if coset_data.reduced_side == "left":
        return _contraction_reduce_left(tau, n, m, ct_n, ct_m, coset_data)
    else:
        return _contraction_reduce_right(tau, n, m, ct_n, ct_m, coset_data)


def compute_all_contractions_efficient(
    woven: WovenData,
    ct_dir: str | Path,
    gap_coset_path: str | Path | None = None,
    *,
    verbose: bool = True,
    parallel: bool = False,
    max_workers: int | None = None,
) -> ContractionResult:
    """Compute all contraction coefficients from woven data.

    Drop-in replacement for :func:`~sym_contractions.woven.compute_all_contractions`
    that uses coset reduction + character theory instead of stored
    probability matrices.

    Parameters
    ----------
    woven : WovenData
        Loaded woven contraction data.
    ct_dir : str or Path
        Directory containing ``ct_{n}.json`` character table files.
    gap_coset_path : str, Path or None
        Path to the GAP-generated ``coset_reps.json``.  If ``None``,
        coset representatives are computed via brute-force enumeration
        (practical for small *n*, *m*).
    verbose : bool
        Print progress messages.
    parallel : bool
        Whether to parallelize independent ``(nL, nR, tau)`` tasks
        using process-based workers.
    max_workers : int or None
        Optional upper bound on the number of worker processes used when
        ``parallel`` is enabled.

    Returns
    -------
    ContractionResult
    """
    ct_dir = Path(ct_dir)
    gap_coset_path = None if gap_coset_path is None else Path(gap_coset_path)
    gap_data: GapCosetData | None = None
    if gap_coset_path is not None:
        gap_data = load_gap_coset_data(gap_coset_path)

    group_specs: list[tuple[int, int, WovenGroup, list[tuple[int, ...]]]] = []
    total_tau_tasks = 0
    for (nL, nR), group in sorted(woven.groups.items()):
        tau_list = sorted({entry.tau for entry in group.entries})
        total_tau_tasks += len(tau_list)
        group_specs.append((nL, nR, group, tau_list))

    worker_count = _resolve_parallel_workers(total_tau_tasks, max_workers)
    use_parallel = parallel and worker_count > 1
    executor: ProcessPoolExecutor | None = None
    if use_parallel:
        mp_context = mp.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_init_contraction_worker,
            initargs=(
                str(ct_dir),
                None if gap_coset_path is None else str(gap_coset_path),
            ),
        )

    all_entries: list[ContractionEntry] = []
    try:
        for nL, nR, group, tau_list in group_specs:
            n, m = nL, nR
            if verbose:
                print(
                    f"[efficient] Processing (nL={nL}, nR={nR}): "
                    f"{len(group.entries)} woven entries"
                )
                print(f"  {len(tau_list)} unique tau(s), n={n}, m={m}")
                if use_parallel and tau_list:
                    print(
                        f"  Parallel tau dispatch enabled "
                        f"({min(len(tau_list), worker_count)} worker(s))"
                    )

            partitions_n = enumerate_partitions(n)
            partitions_m = enumerate_partitions(m)

            if verbose:
                print(
                    f"  Character tables loaded: "
                    f"p(n)={len(partitions_n)}, p(m)={len(partitions_m)}"
                )

            if executor is None:
                ct_n = _load_ct(n, ct_dir)
                ct_m = _load_ct(m, ct_dir)
                tau_results = (
                    _compute_tau_result(tau, n, m, ct_n, ct_m, gap_data)
                    for tau in tau_list
                )
            else:
                tasks = [_ContractionTask(nL=n, nR=m, tau=tau) for tau in tau_list]
                tau_results = executor.map(_compute_tau_result_worker, tasks)

            for result in tau_results:
                if (
                    verbose
                    and result.reduced_side is not None
                    and result.num_reps is not None
                ):
                    print(
                        f"    tau={list(result.task.tau)}: "
                        f"side={result.reduced_side}, "
                        f"reps={result.num_reps}"
                    )

                pairs = tau_to_pairs_1indexed(result.task.tau)
                for r_idx, R in enumerate(partitions_n):
                    for s_idx, S in enumerate(partitions_m):
                        all_entries.append(
                            ContractionEntry(
                                pairs_1indexed=pairs,
                                R=R,
                                S=S,
                                coefficients=result.coeffs_all[r_idx, s_idx],
                            )
                        )

            if verbose:
                print(f"  Done: {len(all_entries)} total entries so far")
                print("=" * 40)
    finally:
        if executor is not None:
            executor.shutdown()

    return ContractionResult(
        label=woven.label, Lambda=woven.Lambda, entries=all_entries
    )
