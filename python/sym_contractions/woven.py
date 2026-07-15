"""Woven contraction computation and Mathematica export.

Loads woven contraction data from JSON (exported by Mathematica's
``ExportWovenContractions``), computes contractions via cycle-count
probabilities and character tables, and exports results for
Mathematica's ``PrecomputedContractions`` format.

Typical workflow
----------------
>>> from sym_contractions.woven import (
...     load_woven_json, compute_all_contractions, export_for_mathematica
... )
>>> woven = load_woven_json("data/processed/woven_contractions/wc_K4_Lambda12.json")
>>> results = compute_all_contractions(
...     woven,
...     ct_dir="data/processed/character_tables",
...     store_collection_path="data/probability_stores/probability_store_collection.npz",
... )
>>> export_for_mathematica(results, "output/precomputed_K4.json")
"""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np

from sym_contractions.character_tables import (
    CharacterTableData,
    get_class_weights,
    load_character_table,
)
from sym_contractions.store import ProbabilityStoreCollection
from sym_contractions.utils import enumerate_partitions

# ======================================================================
# Data structures
# ======================================================================


@dataclasses.dataclass
class WovenEntry:
    """A single woven contraction entry from the JSON file.

    Attributes
    ----------
    involution : tuple[int, ...]
        0-indexed involution permutation on 2*(nL+nR) indices as stored
        in the JSON.
    tau : tuple[int, ...]
        Derived permutation in S_{nL+nR} for ``compute_probabilities``.
        ``tau[i] = involution[i] - (nL+nR)`` for ``i < nL+nR``.
    pairs_1indexed : list[list[int]]
        1-indexed contraction pairs for Mathematica.
        ``[[i+1, tau[i]+n+1] for i in range(n)]``.
    coefficient_poly : list[tuple[int, ...]]
        Polynomial coefficients from degree 0 upward.

        - 2-element tuples ``(numerator, denominator)`` for real coefficients.
        - 4-element tuples ``(re_num, re_den, im_num, im_den)`` for complex.
    """

    involution: tuple[int, ...]
    tau: tuple[int, ...]
    pairs_1indexed: list[list[int]]
    coefficient_poly: list[tuple[int, ...]]


@dataclasses.dataclass
class WovenGroup:
    """All woven entries for a fixed (nL, nR) pair."""

    nL: int
    nR: int
    entries: list[WovenEntry]


@dataclasses.dataclass
class WovenData:
    """Complete woven contraction data from a JSON file.

    Attributes
    ----------
    operators : str
        Operator string (e.g. ``"XXXX"``, ``"XPXP"``).
    trace_permutation : tuple[int, ...]
        0-indexed trace permutation.
    Lambda : int
        Maximum excitation level.
    groups : dict[tuple[int, int], WovenGroup]
        Woven entries grouped by ``(nL, nR)``.
    mass : float
        Mass parameter used when generating the woven contractions
        (e.g. ``0.5`` for ``m = 1/2``).
    is_even : bool or None
        Whether the monomial is even under parity (if known).
    is_hermitian : bool or None
        Whether the monomial is Hermitian (if known).
    """

    operators: str
    trace_permutation: tuple[int, ...]
    Lambda: int
    groups: dict[tuple[int, int], WovenGroup]
    mass: float = 0.5
    is_even: bool | None = None
    is_hermitian: bool | None = None

    @property
    def label(self) -> str:
        """Monomial label combining operators and permutation.

        Format: ``"{ops}_p{perm_1indexed}"``
        e.g. ``"XXXX_p2341"`` for cyclic ``tr(X^4)``.
        """
        perm_1idx = "".join(str(p + 1) for p in self.trace_permutation)
        return f"{self.operators}_p{perm_1idx}"

    @property
    def K(self) -> int:
        """Operator count (length of operators string).

        Provided for convenience; equivalent to ``len(self.operators)``.
        """
        return len(self.operators)

    def filter_by_max_excitations(self, max_excitations: int) -> "WovenData":
        """Filter groups to keep only states within max excitation level.

        Parameters
        ----------
        max_excitations : int
            Maximum number of excitations allowed for both nL and nR.

        Returns
        -------
        WovenData
            New WovenData object with filtered groups where both
            nL <= max_excitations and nR <= max_excitations.

        Examples
        --------
        >>> woven = load_woven_json("wc_op_XXXX_p2341_m0.5_Lambda12.json")
        >>> woven_small = woven.filter_by_max_excitations(4)
        >>> max(max(k) for k in woven_small.groups.keys())
        4
        """
        filtered_groups = {
            k: v
            for k, v in self.groups.items()
            if k[0] <= max_excitations and k[1] <= max_excitations
        }
        return WovenData(
            operators=self.operators,
            trace_permutation=self.trace_permutation,
            Lambda=max_excitations,
            groups=filtered_groups,
            mass=self.mass,
            is_even=self.is_even,
            is_hermitian=self.is_hermitian,
        )

    def filter_upper_diagonal_excitations(self) -> "WovenData":
        """Filter groups to keep only upper-diagonal excitation pairs.

        Retains only groups where ``nL <= nR``, exploiting Hermicity to reduce required compute.

        Returns
        -------
        WovenData
            New WovenData object with filtered groups where nL <= nR.

        Examples
        --------
        >>> woven = load_woven_json("wc_op_XXXX_p2341_m0.5_Lambda12.json")
        >>> woven_upper = woven.filter_upper_diagonal_excitations()
        >>> all(k[0] <= k[1] for k in woven_upper.groups.keys())
        True
        """
        filtered_groups = {k: v for k, v in self.groups.items() if k[0] <= k[1]}
        return WovenData(
            operators=self.operators,
            trace_permutation=self.trace_permutation,
            Lambda=self.Lambda,
            groups=filtered_groups,
            mass=self.mass,
            is_even=self.is_even,
            is_hermitian=self.is_hermitian,
        )


@dataclasses.dataclass
class ContractionResult:
    """Computed contraction coefficients for all (tau, R, S) triples.

    Attributes
    ----------
    label : str
        Monomial label (e.g. ``"XXXX_p2341"``).
    Lambda : int
        Maximum excitation level.
    entries : list[ContractionEntry]
        One entry per (tau, R, S) triple.
    """

    label: str
    Lambda: int
    entries: list[ContractionEntry]


@dataclasses.dataclass
class ContractionEntry:
    """Contraction coefficients for one (tau, R, S) triple.

    Attributes
    ----------
    pairs_1indexed : list[list[int]]
        1-indexed Mathematica contraction pairs.
    R : tuple[int, ...]
        Left partition (Young diagram).
    S : tuple[int, ...]
        Right partition (Young diagram).
    coefficients : np.ndarray
        Coefficient of d^k at index k, shape ``(nL+nR+1,)``.
    """

    pairs_1indexed: list[list[int]]
    R: tuple[int, ...]
    S: tuple[int, ...]
    coefficients: np.ndarray


# ======================================================================
# Conversion utilities
# ======================================================================


def involution_to_tau(perm: Sequence[int]) -> tuple[int, ...]:
    """Convert a 0-indexed involution on 2n indices to τ ∈ S_n.

    The JSON stores contraction patterns as involutions on
    ``{0, ..., 2n-1}`` where n = nL + nR.  The first n entries
    map lower indices to upper indices: ``perm[i] ∈ {n, ..., 2n-1}``
    for ``i < n``.  The permutation τ is recovered as
    ``tau[i] = perm[i] - n``.

    Parameters
    ----------
    perm : sequence of int
        0-indexed involution of length 2n.

    Returns
    -------
    tuple[int, ...]
        τ ∈ S_n as a 0-indexed tuple.

    Examples
    --------
    >>> involution_to_tau([3, 2, 1, 0])
    (1, 0)
    >>> involution_to_tau([6, 7, 4, 5, 2, 3, 0, 1])
    (2, 3, 0, 1)
    >>> involution_to_tau([])
    ()
    """
    if len(perm) == 0:
        return ()
    n = len(perm) // 2
    return tuple(perm[i] - n for i in range(n))


def tau_to_pairs_1indexed(tau: tuple[int, ...]) -> list[list[int]]:
    """Convert τ ∈ S_n (0-indexed) to 1-indexed Mathematica pairs.

    Each pair ``[i+1, tau[i]+n+1]`` connects lower index ``i+1``
    to upper index ``tau[i]+n+1`` in the rank-(n,n) tensor
    contraction notation used by Mathematica's
    ``ContractPermutationTensor``.

    Parameters
    ----------
    tau : tuple[int, ...]
        0-indexed permutation in S_n.

    Returns
    -------
    list[list[int]]
        1-indexed pairs for Mathematica.

    Examples
    --------
    >>> tau_to_pairs_1indexed((1, 0))
    [[1, 4], [2, 3]]
    >>> tau_to_pairs_1indexed((0, 1))
    [[1, 3], [2, 4]]
    >>> tau_to_pairs_1indexed(())
    []
    """
    n = len(tau)
    return [[i + 1, tau[i] + n + 1] for i in range(n)]


def tau_to_involution(tau: tuple[int, ...]) -> list[int]:
    """Convert τ ∈ S_n (0-indexed) back to a 0-indexed involution on 2n.

    Inverse of ``involution_to_tau``.

    Parameters
    ----------
    tau : tuple[int, ...]
        0-indexed permutation in S_n.

    Returns
    -------
    list[int]
        0-indexed involution of length 2n.
    """
    n = len(tau)
    perm = [0] * (2 * n)
    for i in range(n):
        perm[i] = tau[i] + n
        perm[tau[i] + n] = i
    return perm


# ======================================================================
# JSON loading
# ======================================================================


def load_woven_json(filepath: str | Path) -> WovenData:
    """Load woven contraction data from Mathematica-exported JSON.

    Supports both the new schema (``operators`` + ``trace_permutation``
    fields) and legacy schema (``K`` field, implies all-X operators with
    cyclic permutation).

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file.

    Returns
    -------
    WovenData
        Parsed woven contraction data with tau permutations derived
        from the stored involutions.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        raw = json.load(f)

    Lambda = raw["Lambda"]
    mass_raw = raw.get("mass", [1, 2])  # default [1, 2] = 1/2
    mass = mass_raw[0] / mass_raw[1]

    # Detect schema version
    if "operators" in raw:
        # New schema
        operators = raw["operators"]
        perm_0 = tuple(raw["trace_permutation"])
    elif "K" in raw:
        # Legacy schema: K → all-X operators with cyclic permutation
        K = raw["K"]
        operators = "X" * K
        if K == 0:
            perm_0 = ()
        else:
            # Cyclic perm 0-indexed: [1, 2, ..., K-1, 0]
            perm_0 = tuple(list(range(1, K)) + [0])
    else:
        raise ValueError(f"JSON file {filepath} has neither 'operators' nor 'K' field")

    wc = raw["woven_contractions"]

    groups: dict[tuple[int, int], WovenGroup] = {}
    for key_str, entry_list in wc.items():
        parts = key_str.split("_")
        nL, nR = int(parts[0]), int(parts[1])

        entries: list[WovenEntry] = []
        for entry in entry_list:
            inv = tuple(entry["permutation"])
            tau = involution_to_tau(inv)
            pairs = tau_to_pairs_1indexed(tau)
            raw_coeffs = entry["coefficient_poly_d"]
            # Detect coefficient format: 2-tuple (real) or 4-tuple (complex)
            coeff: list[tuple[int, int] | tuple[int, int, int, int]] = []
            for c in raw_coeffs:
                if len(c) == 2:
                    coeff.append((c[0], c[1]))
                elif len(c) == 4:
                    coeff.append((c[0], c[1], c[2], c[3]))
                else:
                    raise ValueError(
                        f"Unexpected coefficient format: {c} (expected 2 or 4 elements)"
                    )
            entries.append(
                WovenEntry(
                    involution=inv,
                    tau=tau,
                    pairs_1indexed=pairs,
                    coefficient_poly=coeff,
                )
            )

        groups[(nL, nR)] = WovenGroup(nL=nL, nR=nR, entries=entries)

    # Optional metadata fields (absent in legacy files)
    is_even = raw.get("is_even")
    is_hermitian = raw.get("is_hermitian")

    return WovenData(
        operators=operators,
        trace_permutation=perm_0,
        Lambda=Lambda,
        groups=groups,
        mass=mass,
        is_even=is_even,
        is_hermitian=is_hermitian,
    )


# ======================================================================
# Contraction computation
# ======================================================================


def _load_or_create_trivial_ct(n: int, ct_dir: Path) -> CharacterTableData:
    """Load character table for S_n, handling n=0 with ct_0.json or fallback."""
    ct_path = ct_dir / f"ct_{n}.json"
    if ct_path.exists():
        return load_character_table(ct_path)
    if n == 0:
        # Trivial S_0: one representation (empty partition), one class, χ=1
        return {
            "n": 0,
            "cycle_types": [[]],
            "class_sizes": np.array([1], dtype=np.int32),
            "characters": np.array([[1]], dtype=np.int32),
        }
    raise FileNotFoundError(f"Character table not found: {ct_path}")


def compute_contraction_coefficients(
    probs: np.ndarray,
    char_table_n: CharacterTableData,
    char_table_m: CharacterTableData,
    n: int,
    m: int,
) -> np.ndarray:
    """Compute contraction coefficients for all (R, S) representation pairs.

    For a fixed τ, computes:

    .. math::

        c_k(R, S) = \\sum_{i,j} \\frac{|C_i|}{n!} \\frac{|C_j|}{m!}
                     \\chi_R(C_i) \\chi_S(C_j) P(\\ell = k \\mid \\tau, C_i, C_j)

    Parameters
    ----------
    probs : np.ndarray
        Cycle-count probabilities, shape ``(p_n, p_m, n+m+1)``.
    char_table_n : CharacterTableData
        Character table for S_n from ``load_character_table``.
    char_table_m : CharacterTableData
        Character table for S_m from ``load_character_table``.
    n, m : int
        Sizes of the symmetric groups.

    Returns
    -------
    np.ndarray
        Shape ``(p_n, p_m, n+m+1)`` where ``result[r, s, k]`` is the
        coefficient of ``d^k`` in ``ComputeContractions[R_r, S_s, {τ}]``.
    """
    # Weights: |C_i| / n!  (overflow-safe via cycle_types)
    weights_n = np.asarray(get_class_weights(char_table_n["cycle_types"], n))
    weights_m = np.asarray(get_class_weights(char_table_m["cycle_types"], m))

    # Characters: shape (p_n, num_classes_n) and (p_m, num_classes_m)
    chars_n = np.asarray(char_table_n["characters"], dtype=np.float64)
    chars_m = np.asarray(char_table_m["characters"], dtype=np.float64)

    # Weighted probs: weights_n[i] * weights_m[j] * probs[i, j, k]
    # probs has shape (p_n_classes, p_m_classes, nm+1)
    weighted_probs = (
        probs
        * weights_n[:, np.newaxis, np.newaxis]
        * weights_m[np.newaxis, :, np.newaxis]
    )

    # Contract with characters:
    # result[r, s, k] = Σ_i Σ_j chars_n[r, i] * chars_m[s, j] * weighted_probs[i, j, k]
    result = np.einsum("ri, sj, ijk -> rsk", chars_n, chars_m, weighted_probs)

    return result


def compute_all_contractions(
    woven: WovenData,
    ct_dir: str | Path,
    store_collection_path: str | Path,
    *,
    verbose: bool = True,
) -> ContractionResult:
    """Compute all contractions from woven data.

    For each ``(nL, nR)`` group in the woven data:

    1. Collects all unique τ permutations.
    2. Computes cycle-count probabilities via ``compute_and_store``.
    3. Loads character tables for ``S_{nL}`` and ``S_{nR}``.
    4. Computes contraction coefficients for all ``(R, S)`` pairs.

    Parameters
    ----------
    woven : WovenData
        Loaded woven contraction data.
    ct_dir : str or Path
        Directory containing character table JSON files (``ct_{n}.json``).
    store_collection_path : str or Path
        Path to a ``.npz`` file loadable by
        ``ProbabilityStoreCollection.load`` containing precomputed
        probability stores indexed by ``(n, m)``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    ContractionResult
        Computed contractions for all ``(τ, R, S)`` triples.
    """
    ct_dir = Path(ct_dir)
    store_collection_path = Path(store_collection_path)
    store_collection = ProbabilityStoreCollection.load(store_collection_path)
    all_entries: list[ContractionEntry] = []

    for (nL, nR), group in sorted(woven.groups.items()):
        n, m = nL, nR

        if verbose:
            print(f"Processing (nL={nL}, nR={nR}): {len(group.entries)} woven entries")

        # Collect unique taus for this (n, m) group
        unique_taus: dict[tuple[int, ...], list[WovenEntry]] = {}
        for entry in group.entries:
            unique_taus.setdefault(entry.tau, []).append(entry)

        tau_list = list(unique_taus.keys())

        if verbose:
            print(f"  {len(tau_list)} unique tau(s), n={n}, m={m}")

        # ---- Load precomputed probabilities for this (n,m) group ----
        group_store = store_collection.stores.get((n, m))
        if group_store is None:
            raise KeyError(
                f"Missing ProbabilityStore for (n={n}, m={m}) in {store_collection_path}"
            )

        # ---- Load character tables ----
        ct_n = _load_or_create_trivial_ct(n, ct_dir)
        ct_m = _load_or_create_trivial_ct(m, ct_dir)

        partitions_n = enumerate_partitions(n)
        partitions_m = enumerate_partitions(m)

        if verbose:
            print(
                f"  Character tables loaded: p(n)={len(partitions_n)}, p(m)={len(partitions_m)}"
            )

        # ---- Compute contraction coefficients for each tau ----
        for tau in tau_list:
            tau_entry = group_store.entries.get(tau)
            if tau_entry is None:
                raise KeyError(
                    f"Missing tau={list(tau)} for (n={n}, m={m}) in {store_collection_path}"
                )
            probs = np.asarray(tau_entry.probabilities)

            # Shape (p_n, p_m, nm+1)
            coeffs_all = compute_contraction_coefficients(probs, ct_n, ct_m, n, m)

            # Generate entries for all (R, S) pairs
            pairs = tau_to_pairs_1indexed(tau)
            for r_idx, R in enumerate(partitions_n):
                for s_idx, S in enumerate(partitions_m):
                    all_entries.append(
                        ContractionEntry(
                            pairs_1indexed=pairs,
                            R=R,
                            S=S,
                            coefficients=coeffs_all[r_idx, s_idx],
                        )
                    )

        if verbose:
            print(f"  Done: {len(all_entries)} total entries so far")
            print("=" * 40)

    return ContractionResult(
        label=woven.label, Lambda=woven.Lambda, entries=all_entries
    )


# ======================================================================
# Export for Mathematica
# ======================================================================


def _try_rationalize(x: float, max_denom: int = 10_000_000) -> tuple[int, int]:
    """Attempt to convert a float to an exact rational (num, den).

    Uses ``fractions.Fraction`` with a denominator limit. Falls back
    to a high-precision float representation if rationalization fails.

    Parameters
    ----------
    x : float
        Value to rationalize.
    max_denom : int
        Maximum denominator for rationalization.

    Returns
    -------
    tuple[int, int]
        ``(numerator, denominator)`` pair.
    """
    frac = Fraction(x).limit_denominator(max_denom)
    return (frac.numerator, frac.denominator)


def export_for_mathematica(
    result: ContractionResult,
    output_path: str | Path,
    *,
    rationalize: bool = True,
    max_denom: int = 10_000_000,
) -> None:
    """Export computed contractions as JSON for Mathematica import.

    Produces a JSON file that Mathematica can import to reconstruct
    a ``PrecomputedContractions`` association with keys
    ``{pairs, R, S}`` and polynomial values in ``d``.

    Parameters
    ----------
    result : ContractionResult
        Computed contractions from ``compute_all_contractions``.
    output_path : str or Path
        Output JSON file path.
    rationalize : bool
        If True, attempt to convert floating-point coefficients to
        exact rational ``[numerator, denominator]`` pairs. If False,
        store as ``[float_value, 1]``.
    max_denom : int
        Maximum denominator for rationalization (only used when
        ``rationalize=True``).

    Notes
    -----
    The output JSON has the structure::

        {
            "K": 4,
            "Lambda": 12,
            "description": "...",
            "precomputed_contractions": [
                {
                    "pairs": [[1, 5], [2, 6], [3, 7], [4, 8]],
                    "R": [2, 2],
                    "S": [3, 1],
                    "coefficients": [[0, 1], [1, 4], ...]
                },
                ...
            ]
        }

    Mathematica import example::

        json = Import["file.json", "RawJSON"];
        precomputed = Association[
          ({#["pairs"], #["R"], #["S"]} ->
            Sum[#["coefficients"][[k, 1]] / #["coefficients"][[k, 2]]
                * d^(k-1), {k, Length[#["coefficients"]]}]) & /@
          json["precomputed_contractions"]
        ];
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries_json = []
    for entry in result.entries:
        coeffs = entry.coefficients

        # Skip entries where all coefficients are zero
        if np.allclose(coeffs, 0, atol=1e-15):
            continue

        # Trim trailing zeros
        last_nonzero = len(coeffs) - 1
        while last_nonzero > 0 and abs(coeffs[last_nonzero]) < 1e-15:
            last_nonzero -= 1
        trimmed = coeffs[: last_nonzero + 1]

        if rationalize:
            coeff_pairs = [list(_try_rationalize(float(c), max_denom)) for c in trimmed]
        else:
            coeff_pairs = [[float(c), 1] for c in trimmed]

        entries_json.append(
            {
                "pairs": entry.pairs_1indexed,
                "R": list(entry.R),
                "S": list(entry.S),
                "coefficients": coeff_pairs,
            }
        )

    data = {
        "label": result.label,
        "Lambda": result.Lambda,
        "description": (
            "Precomputed contractions ComputeContractions[R, S, {pairs}] "
            "for Mathematica's PrecomputedContractions option. "
            "Coefficients are polynomials in d stored as [numerator, denominator] "
            "rational pairs, from degree 0 upward. "
            "Generated by sym_contractions.woven."
        ),
        "precomputed_contractions": entries_json,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(entries_json)} contraction entries to {output_path}")


def import_precomputed_contractions(
    filepath: str | Path,
) -> dict[tuple, np.ndarray]:
    """Import previously exported contraction JSON back into Python.

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file exported by ``export_for_mathematica``.

    Returns
    -------
    dict
        Mapping ``(pairs_tuple, R, S) -> np.ndarray`` of coefficients.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        raw = json.load(f)

    result: dict[tuple, np.ndarray] = {}
    for entry in raw["precomputed_contractions"]:
        pairs_key = tuple(tuple(p) for p in entry["pairs"])
        R = tuple(entry["R"])
        S = tuple(entry["S"])
        coeffs = np.array(
            [c[0] / c[1] for c in entry["coefficients"]], dtype=np.float64
        )
        result[(pairs_key, R, S)] = coeffs

    return result
