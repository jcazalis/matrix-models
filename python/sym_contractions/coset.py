"""Coset reduction data structures and GAP precomputed data interface.

Loads coset representatives produced by GAP (``generate_coset_reps.g``)
and wraps them in a ``CosetReductionData`` object consumed by the
efficient contraction pipeline (``efficient.py``).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CosetReductionData:
    """Complete coset reduction data for a given τ and (n, m).

    Attributes:
        tau: The permutation τ ∈ S_{n+m}.
        n: First block size.
        m: Second block size.
        reduced_side: Which side has fewer coset representatives.
        coset_reps: shape (num_reps, side_size), int32.
            Each row is a permutation of {0, …, side_size-1} representing
            a right coset representative g such that Sn = ⊔ Hn · g.
        h_order: Order of the invariance subgroup H on the reduced side.
        num_reps: Number of coset representatives.
    """

    tau: tuple[int, ...]
    n: int
    m: int
    reduced_side: Literal["left", "right"]
    coset_reps: np.ndarray
    h_order: int
    num_reps: int


# ---------------------------------------------------------------------------
# GAP precomputed data: loader and lookup
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GapCosetData:
    """Precomputed coset data from a GAP output JSON.

    Keyed by (tau_0indexed, n, m) tuples for fast lookup.
    """

    _entries: dict[tuple[tuple[int, ...], int, int], dict]

    @staticmethod
    def load(filepath: str | Path) -> "GapCosetData":
        """Load GAP-computed coset representatives from a JSON file.

        The JSON is produced by ``generate_coset_reps.g`` and has the
        structure::

            { "entries": [ { "tau_0indexed": [...], "n": ..., "m": ...,
                              "left": {"h_order":..., "num_reps":..., "reps_0indexed":[...]},
                              "right": {...} }, ... ] }

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file.

        Returns
        -------
        GapCosetData
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            raw = json.load(f)

        entries: dict[tuple[tuple[int, ...], int, int], dict] = {}
        for entry in raw["entries"]:
            key = (tuple(entry["tau_0indexed"]), entry["n"], entry["m"])
            entries[key] = entry

        return GapCosetData(_entries=entries)

    def lookup(self, tau: Sequence[int], n: int, m: int) -> dict | None:
        """Look up precomputed coset data for a specific (tau, n, m).

        Returns None if not found in the precomputed data.
        """
        key = (tuple(tau), n, m)
        return self._entries.get(key)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple[tuple[int, ...], int, int]) -> bool:
        return key in self._entries


def load_gap_coset_data(filepath: str | Path) -> GapCosetData:
    """Load GAP-computed coset data from JSON.

    Convenience wrapper around ``GapCosetData.load``.

    Parameters
    ----------
    filepath : str or Path
        Path to the GAP output JSON file.

    Returns
    -------
    GapCosetData
        Loaded coset data with O(1) lookup by (tau, n, m).
    """
    return GapCosetData.load(filepath)


def compute_coset_reduction_from_gap(
    tau: Sequence[int],
    n: int,
    m: int,
    gap_data: GapCosetData,
) -> CosetReductionData:
    """Compute coset reduction using precomputed GAP data.

    Uses the efficiently computed coset representatives from GAP.

    Parameters
    ----------
    tau : sequence of int
        0-indexed permutation of {0, …, n+m-1}.
    n : int
        First block size.
    m : int
        Second block size.
    gap_data : GapCosetData
        Precomputed GAP coset data (from ``load_gap_coset_data``).

    Returns
    -------
    CosetReductionData

    Raises
    ------
    KeyError
        If the (tau, n, m) triple is not in the precomputed data.
    """
    tau_tuple = tuple(int(x) for x in tau)
    assert len(tau_tuple) == n + m

    entry = gap_data.lookup(tau_tuple, n, m)
    if entry is None:
        raise KeyError(
            f"(tau={tau_tuple}, n={n}, m={m}) not found in GAP precomputed data. "
            f"Run prepare_coset_input.py and generate_coset_reps.g first."
        )

    # Choose the side with fewer coset representatives
    left_data = entry["left"]
    right_data = entry["right"]

    if left_data["num_reps"] <= right_data["num_reps"]:
        reduced_side: Literal["left", "right"] = "left"
        side_data = left_data
    else:
        reduced_side = "right"
        side_data = right_data

    reps = np.array(side_data["reps_0indexed"], dtype=np.int32)
    if reps.ndim == 1 and reps.size == 0:
        side_size = n if reduced_side == "left" else m
        reps = (
            reps.reshape(1, side_size) if side_size == 0 else reps.reshape(0, side_size)
        )

    # Invert each representative: GAP's RightTransversal(S_n, H_n) returns
    # elements t such that S_n = ⊔ H_n·t in GAP multiplication (left-to-right
    # application), which corresponds to LEFT cosets tH_n in standard math
    # (right-to-left composition).  Our contraction formula requires RIGHT
    # coset representatives g such that S_n = ⊔ H_n·g (standard math), i.e.
    # orbits under g ↦ hg.  The conversion is g = t⁻¹.
    if reps.size > 0:
        inv_reps = np.empty_like(reps)
        for i in range(reps.shape[0]):
            for j in range(reps.shape[1]):
                inv_reps[i, reps[i, j]] = j
        reps = inv_reps

    # Validate Lagrange's theorem (log-domain to avoid factorial overflow)
    side_size = n if reduced_side == "left" else m
    log_lhs = math.log(side_data["h_order"]) + math.log(side_data["num_reps"])
    log_rhs = sum(math.log(k) for k in range(1, side_size + 1))
    assert abs(log_lhs - log_rhs) < 1e-6, (
        f"Lagrange check failed: log({side_data['h_order']}) + "
        f"log({side_data['num_reps']}) = {log_lhs:.6f} ≠ "
        f"log({side_size}!) = {log_rhs:.6f}"
    )

    return CosetReductionData(
        tau=tau_tuple,
        n=n,
        m=m,
        reduced_side=reduced_side,
        coset_reps=reps,
        h_order=side_data["h_order"],
        num_reps=side_data["num_reps"],
    )
