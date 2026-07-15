"""Persistent storage for cycle count probability results.

Stores probabilities indexed by (τ, n, m) with metadata tracking
whether each conjugacy-class pair was computed exactly or via MC
(and if so, how many samples were used), plus empirical standard
deviations.

Designed for downstream NumPy scalar products with character tables.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from sym_contractions.utils import enumerate_partitions

if TYPE_CHECKING:
    from sym_contractions.woven import WovenData

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TauEntry:
    """Results for a single fixed τ permutation.

    Attributes:
        tau: The permutation as a tuple of ints.
        probabilities: shape ``(p_n, p_m, n+m+1)`` float64.
            ``result[i, j, k]`` is P(ℓ(τ ∘ (σ × ν)) = k) for
            conjugacy pair (i, j).
        std_errors: shape ``(p_n, p_m, n+m+1)`` float64.
            Empirical standard deviations of the MC estimate.
            Zero for exactly-computed pairs.
        is_exact: shape ``(p_n, p_m)`` bool.
            True where the pair was computed exactly (Numba).
        n_samples: shape ``(p_n, p_m)`` int32.
            Number of MC samples used. Zero for exact pairs.
    """

    tau: tuple[int, ...]
    probabilities: np.ndarray
    std_errors: np.ndarray
    is_exact: np.ndarray
    n_samples: np.ndarray


@dataclasses.dataclass
class ProbabilityStore:
    """Stores cycle-count probabilities for a fixed (n, m) across τ's.

    The partition ordering is canonical (``enumerate_partitions``).
    All arrays share the same ``(p_n, p_m, n+m+1)`` layout so they
    can be stacked into NumPy arrays for downstream character-table
    contractions.

    Attributes:
        n: Size of σ's symmetric group S_n.
        m: Size of ν's symmetric group S_m.
        partitions_n: Ordered partitions of n.
        partitions_m: Ordered partitions of m.
        entries: Map from τ (as tuple) to ``TauEntry``.
    """

    n: int
    m: int
    partitions_n: list[tuple[int, ...]]
    partitions_m: list[tuple[int, ...]]
    entries: dict[tuple[int, ...], TauEntry] = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, n: int, m: int) -> ProbabilityStore:
        """Create an empty store for a given (n, m)."""
        return cls(
            n=n,
            m=m,
            partitions_n=enumerate_partitions(n),
            partitions_m=enumerate_partitions(m),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def p_n(self) -> int:
        return len(self.partitions_n)

    @property
    def p_m(self) -> int:
        return len(self.partitions_m)

    @property
    def nm(self) -> int:
        return self.n + self.m

    @property
    def tau_keys(self) -> list[tuple[int, ...]]:
        """Return all stored τ keys in insertion order."""
        return list(self.entries.keys())

    def get_numpy_probabilities(self, tau: tuple[int, ...]) -> np.ndarray:
        """Return probabilities for τ as a NumPy float32 array.

        Shape ``(p_n, p_m, n+m+1)``, ready for ``np.einsum`` with
        character-table arrays.
        """
        return np.asarray(self.entries[tau].probabilities, dtype=np.float32)

    def get_numpy_std_errors(self, tau: tuple[int, ...]) -> np.ndarray:
        """Return std errors for τ as a NumPy float32 array.

        Shape ``(p_n, p_m, n+m+1)``; zero for exact pairs.
        """
        return np.asarray(self.entries[tau].std_errors, dtype=np.float32)

    # ------------------------------------------------------------------
    # Serialisation (NPZ)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the store to an ``.npz`` file.

        Layout inside the archive::

            meta_n, meta_m          — scalars
            partitions_n            — (p_n, n) int32, zero-padded
            partitions_m            — (p_m, m) int32, zero-padded
            tau_keys                — (T, n+m) int32
            probabilities           — (T, p_n, p_m, n+m+1) float64
            std_errors              — (T, p_n, p_m, n+m+1) float64
            is_exact                — (T, p_n, p_m) bool
            n_samples               — (T, p_n, p_m) int32
        """
        path = Path(path)
        taus = list(self.entries.keys())
        T = len(taus)

        # Pad partitions to rectangular arrays.
        def _pad_partitions(parts: list[tuple[int, ...]], size: int) -> np.ndarray:
            rows = [list(p) + [0] * (size - len(p)) for p in parts]
            return np.array(rows, dtype=np.int32)

        data: dict[str, np.ndarray] = {
            "meta_n": np.array(self.n, dtype=np.int32),
            "meta_m": np.array(self.m, dtype=np.int32),
            "partitions_n": _pad_partitions(self.partitions_n, self.n),
            "partitions_m": _pad_partitions(self.partitions_m, self.m),
        }

        if T > 0:
            data["tau_keys"] = np.array(taus, dtype=np.int32)
            data["probabilities"] = np.stack(
                [self.entries[t].probabilities for t in taus]
            )
            data["std_errors"] = np.stack([self.entries[t].std_errors for t in taus])
            data["is_exact"] = np.stack([self.entries[t].is_exact for t in taus])
            data["n_samples"] = np.stack([self.entries[t].n_samples for t in taus])
        else:
            nm1 = self.nm + 1
            data["tau_keys"] = np.empty((0, self.nm), dtype=np.int32)
            data["probabilities"] = np.empty(
                (0, self.p_n, self.p_m, nm1), dtype=np.float64
            )
            data["std_errors"] = np.empty(
                (0, self.p_n, self.p_m, nm1), dtype=np.float64
            )
            data["is_exact"] = np.empty((0, self.p_n, self.p_m), dtype=bool)
            data["n_samples"] = np.empty((0, self.p_n, self.p_m), dtype=np.int32)

        np.savez_compressed(path, **data)  # type: ignore

    @classmethod
    def load(cls, path: str | Path) -> ProbabilityStore:
        """Load a store from an ``.npz`` file."""
        path = Path(path)
        with np.load(path) as f:
            n = int(f["meta_n"])
            m = int(f["meta_m"])
            parts_n_arr = f["partitions_n"]  # (p_n, n)
            parts_m_arr = f["partitions_m"]  # (p_m, m)
            tau_keys_arr = f["tau_keys"]  # (T, n+m)
            probs_arr = f["probabilities"]  # (T, p_n, p_m, n+m+1)
            se_arr = f["std_errors"]  # (T, p_n, p_m, n+m+1)
            exact_arr = f["is_exact"]  # (T, p_n, p_m)
            ns_arr = f["n_samples"]  # (T, p_n, p_m)

        # Special handling for n=0 or m=0 partition (the empty partition).
        def _unpad_partitions(arr: np.ndarray) -> list[tuple[int, ...]]:
            result = []
            for row in arr:
                nonzero = row[row > 0]
                if len(nonzero) == 0:
                    result.append(())
                else:
                    result.append(tuple(int(x) for x in nonzero))
            return result

        partitions_n = _unpad_partitions(parts_n_arr)
        partitions_m = _unpad_partitions(parts_m_arr)

        entries: dict[tuple[int, ...], TauEntry] = {}
        for idx in range(tau_keys_arr.shape[0]):
            tau_key = tuple(int(x) for x in tau_keys_arr[idx])
            entries[tau_key] = TauEntry(
                tau=tau_key,
                probabilities=probs_arr[idx].copy(),
                std_errors=se_arr[idx].copy(),
                is_exact=exact_arr[idx].copy(),
                n_samples=ns_arr[idx].copy(),
            )

        return cls(
            n=n,
            m=m,
            partitions_n=partitions_n,
            partitions_m=partitions_m,
            entries=entries,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, tau: tuple[int, ...] | None = None) -> str:
        """Return a human-readable summary of the store contents."""
        lines = [
            f"ProbabilityStore(n={self.n}, m={self.m})",
            f"  partitions: p(n)={self.p_n}, p(m)={self.p_m}, "
            f"pairs={self.p_n * self.p_m}",
            f"  τ entries: {len(self.entries)}",
        ]
        keys = [tau] if tau is not None else list(self.entries.keys())
        for t in keys:
            entry = self.entries[t]
            n_exact = int(entry.is_exact.sum())
            total = self.p_n * self.p_m
            n_mc = total - n_exact
            if n_mc > 0:
                mc_samples = entry.n_samples[~entry.is_exact]
                min_s, max_s = int(mc_samples.min()), int(mc_samples.max())
                mc_str = f", MC: {n_mc} pairs ({min_s}–{max_s} samples)"
            else:
                mc_str = ""
            lines.append(f"  τ={list(t)}: exact={n_exact}/{total}{mc_str}")
        return "\n".join(lines)


@dataclasses.dataclass
class ProbabilityStoreCollection:
    """Collection of ``ProbabilityStore`` objects for many ``(n, m)`` pairs.

    Attributes:
        Lambda: Maximum allowed excitation on each side.
        stores: Mapping ``(n, m) -> ProbabilityStore``.
    """

    Lambda: int
    stores: dict[tuple[int, int], ProbabilityStore] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def create(cls, Lambda: int) -> ProbabilityStoreCollection:
        """Create an empty collection with fixed maximum excitation ``Lambda``."""
        return cls(Lambda=Lambda)

    def _ensure_store(self, n: int, m: int) -> ProbabilityStore:
        key = (n, m)
        if key not in self.stores:
            self.stores[key] = ProbabilityStore.create(n, m)
        return self.stores[key]

    def compute_from_woven(
        self,
        woven: WovenData,
        *,
        n_samples_per_pair: int = 10_000,
        max_sequential_search_space: int = 200_000,
        max_parallel_search_space: int = 1_000_000_000,
        progress: bool = True,
        seed: int = 0,
    ) -> ProbabilityStoreCollection:
        """Compute and store all probabilities required by a ``WovenData`` object.

        If ``woven.Lambda`` exceeds this collection's ``Lambda``, the woven
        data is filtered via ``filter_by_max_excitations`` before computing.
        """
        effective_woven = woven
        if effective_woven.Lambda > self.Lambda:
            effective_woven = effective_woven.filter_by_max_excitations(self.Lambda)

        grouped_taus: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        for (n, m), group in effective_woven.groups.items():
            if n > self.Lambda or m > self.Lambda:
                continue
            tau_set = grouped_taus.setdefault((n, m), set())
            for entry in group.entries:
                tau_set.add(tuple(int(x) for x in entry.tau))

        for (n, m), tau_set in sorted(grouped_taus.items()):
            if not tau_set:
                continue
            store = self._ensure_store(n, m)
            compute_and_store(
                store,
                sorted(tau_set),
                n_samples_per_pair=n_samples_per_pair,
                max_sequential_search_space=max_sequential_search_space,
                max_parallel_search_space=max_parallel_search_space,
                progress=progress,
                seed=seed,
            )

        return self

    def save(self, path: str | Path) -> None:
        """Save the collection to one ``.npz`` file.

        The per-store payload reuses the same schema as
        :meth:`ProbabilityStore.save` with ``nm_{n}_{m}_`` prefixes.
        """
        path = Path(path)

        def _pad_partitions(parts: list[tuple[int, ...]], size: int) -> np.ndarray:
            rows = [list(p) + [0] * (size - len(p)) for p in parts]
            return np.array(rows, dtype=np.int32)

        nm_keys = sorted(self.stores.keys())
        data: dict[str, np.ndarray] = {
            "meta_lambda": np.array(self.Lambda, dtype=np.int32),
            "nm_keys": np.array(nm_keys, dtype=np.int32)
            if nm_keys
            else np.empty((0, 2), dtype=np.int32),
        }

        for n, m in nm_keys:
            store = self.stores[(n, m)]
            prefix = f"nm_{n}_{m}"
            taus = list(store.entries.keys())
            T = len(taus)

            data[f"{prefix}_partitions_n"] = _pad_partitions(store.partitions_n, n)
            data[f"{prefix}_partitions_m"] = _pad_partitions(store.partitions_m, m)

            if T > 0:
                data[f"{prefix}_tau_keys"] = np.array(taus, dtype=np.int32)
                data[f"{prefix}_probabilities"] = np.stack(
                    [store.entries[t].probabilities for t in taus]
                )
                data[f"{prefix}_std_errors"] = np.stack(
                    [store.entries[t].std_errors for t in taus]
                )
                data[f"{prefix}_is_exact"] = np.stack(
                    [store.entries[t].is_exact for t in taus]
                )
                data[f"{prefix}_n_samples"] = np.stack(
                    [store.entries[t].n_samples for t in taus]
                )
            else:
                nm1 = store.nm + 1
                data[f"{prefix}_tau_keys"] = np.empty((0, store.nm), dtype=np.int32)
                data[f"{prefix}_probabilities"] = np.empty(
                    (0, store.p_n, store.p_m, nm1), dtype=np.float64
                )
                data[f"{prefix}_std_errors"] = np.empty(
                    (0, store.p_n, store.p_m, nm1), dtype=np.float64
                )
                data[f"{prefix}_is_exact"] = np.empty(
                    (0, store.p_n, store.p_m), dtype=bool
                )
                data[f"{prefix}_n_samples"] = np.empty(
                    (0, store.p_n, store.p_m), dtype=np.int32
                )

        np.savez_compressed(path, **data)  # type: ignore

    @classmethod
    def load(cls, path: str | Path) -> ProbabilityStoreCollection:
        """Load a collection saved by :meth:`save`."""
        path = Path(path)

        def _unpad_partitions(arr: np.ndarray) -> list[tuple[int, ...]]:
            result: list[tuple[int, ...]] = []
            for row in arr:
                nonzero = row[row > 0]
                if len(nonzero) == 0:
                    result.append(())
                else:
                    result.append(tuple(int(x) for x in nonzero))
            return result

        stores: dict[tuple[int, int], ProbabilityStore] = {}

        with np.load(path) as f:
            Lambda = int(f["meta_lambda"])
            nm_keys_arr = f["nm_keys"]

            for row in nm_keys_arr:
                n, m = int(row[0]), int(row[1])
                prefix = f"nm_{n}_{m}"

                parts_n_arr = f[f"{prefix}_partitions_n"]
                parts_m_arr = f[f"{prefix}_partitions_m"]
                tau_keys_arr = f[f"{prefix}_tau_keys"]
                probs_arr = f[f"{prefix}_probabilities"]
                se_arr = f[f"{prefix}_std_errors"]
                exact_arr = f[f"{prefix}_is_exact"]
                ns_arr = f[f"{prefix}_n_samples"]

                entries: dict[tuple[int, ...], TauEntry] = {}
                for idx in range(tau_keys_arr.shape[0]):
                    tau_key = tuple(int(x) for x in tau_keys_arr[idx])
                    entries[tau_key] = TauEntry(
                        tau=tau_key,
                        probabilities=probs_arr[idx].copy(),
                        std_errors=se_arr[idx].copy(),
                        is_exact=exact_arr[idx].copy(),
                        n_samples=ns_arr[idx].copy(),
                    )

                stores[(n, m)] = ProbabilityStore(
                    n=n,
                    m=m,
                    partitions_n=_unpad_partitions(parts_n_arr),
                    partitions_m=_unpad_partitions(parts_m_arr),
                    entries=entries,
                )

        return cls(Lambda=Lambda, stores=stores)


# ---------------------------------------------------------------------------
# Batch computation with skip logic
# ---------------------------------------------------------------------------


def compute_and_store(
    store: ProbabilityStore,
    taus: Sequence[tuple[int, ...] | list[int]],
    n_samples_per_pair: int = 10_000,
    max_sequential_search_space: int = 200_000,
    max_parallel_search_space: int = 1_000_000_000,
    progress: bool = True,
    seed: int = 0,
) -> ProbabilityStore:
    """Compute probabilities for a list of τ's and merge into the store.

    For each τ and each conjugacy-class pair (i, j):

    - **Skip** if the pair is already exactly computed.
    - **Skip** if the pair was MC-estimated with ≥ ``n_samples_per_pair``
      samples.
    - **Recompute** otherwise.

    When all pairs of a τ can be skipped, that τ is skipped entirely.

    Args:
        store: Existing store to update in-place.
        taus: List of τ permutations (as tuples or lists of ints).
        n_samples_per_pair: MC samples per pair for new computations.
        max_sequential_search_space: Per-pair threshold for sequential
            Numba (passed to ``compute_probabilities``).
        max_parallel_search_space: Per-pair threshold for parallel
            Numba (passed to ``compute_probabilities``).
        progress: Show progress bar.
        seed: Base seed used to derive deterministic per-τ MC seeds.

    Returns:
        The same ``store`` object, mutated with new/updated entries.
    """

    # Lazy import: hybrid module is archived but still callable if present.
    from sym_contractions.archives.hybrid import compute_probabilities

    def _derive_tau_seed(base_seed: int, tau_key: tuple[int, ...]) -> int:
        """Derive a deterministic 31-bit seed from a base seed and τ.

        Stable across runs and independent of τ iteration order.
        """
        x = (base_seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        for value in tau_key:
            v = int(value) & 0xFFFFFFFFFFFFFFFF
            x ^= v + 0x9E3779B97F4A7C15 + ((x << 6) & 0xFFFFFFFFFFFFFFFF) + (x >> 2)
            x &= 0xFFFFFFFFFFFFFFFF
        return int(x & 0x7FFFFFFF)

    n, m = store.n, store.m
    nm = n + m
    partitions_n = store.partitions_n
    partitions_m = store.partitions_m
    p_n, p_m = store.p_n, store.p_m

    for raw_tau in taus:
        tau_key = tuple(int(x) for x in raw_tau)
        assert len(tau_key) == nm, f"τ has length {len(tau_key)}, expected {nm}"

        # Check what already exists for this τ.
        existing = store.entries.get(tau_key)

        if existing is not None:
            # Determine which pairs need recomputation.
            needs_update = ~existing.is_exact & (
                existing.n_samples < n_samples_per_pair
            )
            if not needs_update.any():
                continue  # everything is already good
        else:
            needs_update = np.ones((p_n, p_m), dtype=bool)

        pairs_to_update = [(int(i), int(j)) for i, j in np.argwhere(needs_update)]

        # Run compute_probabilities only for pairs that need update.
        tau_arr = np.array(tau_key, dtype=np.int32)

        probs, std_errors, pair_methods = compute_probabilities(
            tau_arr,
            n,
            m,
            partitions_n=partitions_n,
            partitions_m=partitions_m,
            pair_indices=pairs_to_update,
            n_samples_per_pair=n_samples_per_pair,
            max_sequential_search_space=max_sequential_search_space,
            max_parallel_search_space=max_parallel_search_space,
            progress=progress,
            mc_seed=_derive_tau_seed(seed, tau_key),
        )

        # Build is_exact and n_samples from per-pair method labels.
        new_is_exact = pair_methods == "exact"
        new_n_samples = np.where(pair_methods == "monte_carlo", n_samples_per_pair, 0)
        new_n_samples = new_n_samples.astype(np.int32)

        if std_errors is None:
            new_std_errors = np.zeros((p_n, p_m, nm + 1), dtype=np.float64)
        else:
            new_std_errors = np.asarray(std_errors, dtype=np.float64)

        # Merge with existing entry.
        if existing is not None:
            for i in range(p_n):
                for j in range(p_m):
                    if not needs_update[i, j]:
                        # Keep the existing (better) result.
                        probs[i, j] = existing.probabilities[i, j]
                        new_std_errors[i, j] = existing.std_errors[i, j]
                        new_is_exact[i, j] = existing.is_exact[i, j]
                        new_n_samples[i, j] = existing.n_samples[i, j]

        store.entries[tau_key] = TauEntry(
            tau=tau_key,
            probabilities=np.asarray(probs, dtype=np.float64),
            std_errors=new_std_errors,
            is_exact=new_is_exact,
            n_samples=new_n_samples,
        )

    return store
