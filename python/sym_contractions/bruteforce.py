"""Exact computation of cycle count probabilities.

Provides exhaustive enumeration over conjugacy classes for small
permutation groups using pure Python, and Numba backends.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations, permutations

import numba
import numpy as np
from tqdm import tqdm

from sym_contractions.utils import enumerate_partitions

# ---------------------------------------------------------------------------
# Conjugacy class sizes
# ---------------------------------------------------------------------------


def conjugacy_class_size(partition: tuple[int, ...]) -> int:
    """Compute the size of the conjugacy class with cycle type ``partition``.

    For a partition λ = (λ₁, λ₂, …) of n the conjugacy class in Sₙ has
    size::

        n! / (∏ᵢ λᵢ · ∏ⱼ mⱼ!)

    where mⱼ is the multiplicity of part j in λ.

    Args:
        partition: Cycle type as a tuple of positive ints (descending).

    Returns:
        Number of permutations with this cycle type.

    Examples:
        >>> conjugacy_class_size((3,))  # 3-cycles in S_3
        2
        >>> conjugacy_class_size((1, 1, 1))  # identity in S_3
        1
    """
    n = sum(partition)
    denom = 1
    for part in partition:
        denom *= part
    for count in Counter(partition).values():
        denom *= math.factorial(count)
    return math.factorial(n) // denom


# ---------------------------------------------------------------------------
# Direct construction of conjugacy classes
# ---------------------------------------------------------------------------


def enumerate_conjugacy_class(n: int, cycle_type: tuple[int, ...]) -> list[list[int]]:
    """Generate all permutations of {0, …, n-1} with a given cycle type.

    Uses direct construction: for each cycle, fix the smallest unused
    element as cycle leader, choose remaining elements via combinations,
    and permute them within the cycle. This is O(|C_λ|) — much faster
    than filtering all n! permutations.

    Args:
        n: Permutation size.
        cycle_type: Target cycle type as a descending tuple, e.g. (3, 2, 2).

    Returns:
        List of permutations (each a list of ints).

    Examples:
        >>> len(enumerate_conjugacy_class(4, (2, 2)))  # 3 permutations
        3
        >>> len(enumerate_conjugacy_class(4, (4,)))  # (4-1)! = 6
        6
    """
    results: list[list[int]] = []

    def _build(
        remaining_lengths: list[int],
        available: list[int],
        perm: list[int],
    ) -> None:
        if not remaining_lengths:
            results.append(perm[:])
            return

        # Leader is always the smallest available element.
        leader = available[0]
        rest = available[1:]

        # Try each *distinct* remaining cycle length for this leader.
        # This avoids overcounting when several cycles share the same length.
        seen_lengths: set[int] = set()
        for idx, length in enumerate(remaining_lengths):
            if length in seen_lengths:
                continue
            seen_lengths.add(length)

            new_remaining = remaining_lengths[:idx] + remaining_lengths[idx + 1 :]

            if length == 1:
                # Fixed point — no choice to make.
                _build(new_remaining, rest, perm)
            else:
                # Choose (length - 1) elements from rest for this cycle.
                for chosen in combinations(rest, length - 1):
                    remaining_available = [x for x in rest if x not in chosen]
                    # Permute chosen elements within the cycle.
                    for ordered in permutations(chosen):
                        new_perm = perm[:]
                        elems = [leader, *ordered]
                        for i in range(len(elems)):
                            new_perm[elems[i]] = elems[(i + 1) % len(elems)]
                        _build(new_remaining, remaining_available, new_perm)

    _build(list(cycle_type), list(range(n)), list(range(n)))
    return results


# ---------------------------------------------------------------------------
# Pure-Python helpers for exact enumeration
# ---------------------------------------------------------------------------


def get_cycle_type(perm: list[int]) -> tuple[int, ...]:
    """Compute the cycle type of a permutation.

    Args:
        perm: 0-indexed permutation as a list, where ``perm[i] = σ(i)``.

    Returns:
        Cycle type as a tuple of cycle lengths in descending order.

    Examples:
        >>> get_cycle_type([1, 2, 0])
        (3,)
        >>> get_cycle_type([0, 1, 2])
        (1, 1, 1)
    """
    n = len(perm)
    visited = [False] * n
    lengths: list[int] = []
    for i in range(n):
        if visited[i]:
            continue
        length = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def all_perms_with_cycle_type(n: int, ct: tuple[int, ...]) -> list[list[int]]:
    """Enumerate all permutations of {0, …, n-1} with a given cycle type.

    Warning: this is O(n!) — only for small n.

    Args:
        n: Permutation size.
        ct: Target cycle type (descending tuple).

    Returns:
        List of permutations (each a list of ints).
    """
    result: list[list[int]] = []
    for p in permutations(range(n)):
        if get_cycle_type(list(p)) == ct:
            result.append(list(p))
    return result


def compose_direct_product(
    tau: list[int], sigma: list[int], nu: list[int]
) -> list[int]:
    """Compute τ ∘ (σ × ν) in pure Python.

    σ acts on {0, …, n-1}, ν acts on {n, …, n+m-1}.

    Args:
        tau: Permutation in S_{n+m}.
        sigma: Permutation in S_n.
        nu: Permutation in S_m.

    Returns:
        The composed permutation as a list.
    """
    n = len(sigma)
    embedded = sigma + [x + n for x in nu]
    return [tau[embedded[i]] for i in range(len(tau))]


def count_cycles(perm: list[int]) -> int:
    """Count the number of cycles in a permutation.

    Args:
        perm: 0-indexed permutation as a list.

    Returns:
        Number of disjoint cycles.
    """
    return len(get_cycle_type(perm))


# ---------------------------------------------------------------------------
# Exact probability computation
# ---------------------------------------------------------------------------


def exact_probability(
    tau: list[int],
    n: int,
    m: int,
    ct_n: tuple[int, ...],
    ct_m: tuple[int, ...],
) -> np.ndarray:
    """Compute exact P(ℓ(τ ∘ (σ × ν)) = k) by exhaustive enumeration.

    Args:
        tau: Fixed permutation in S_{n+m}, as a list.
        n: Size of σ's symmetric group.
        m: Size of ν's symmetric group.
        ct_n: Cycle type for σ.
        ct_m: Cycle type for ν.

    Returns:
        NumPy array of shape (n+m+1,) where entry k is the exact
        probability that the composed permutation has k cycles.
    """
    perms_n = all_perms_with_cycle_type(n, ct_n)
    perms_m = all_perms_with_cycle_type(m, ct_m)
    max_k = n + m
    counts = np.zeros(max_k + 1, dtype=np.float64)
    total = 0
    for sigma in perms_n:
        for nu in perms_m:
            composed = compose_direct_product(tau, sigma, nu)
            k = count_cycles(composed)
            counts[k] += 1
            total += 1
    if total > 0:
        counts /= total
    return counts


def exact_all_conjugacy_pairs(
    tau: list[int],
    n: int,
    m: int,
    partitions_n: list[tuple[int, ...]] | None = None,
    partitions_m: list[tuple[int, ...]] | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Compute exact probabilities for all conjugacy class pairs.

    Args:
        tau: Fixed permutation in S_{n+m}, as a list.
        n: Size of σ's symmetric group.
        m: Size of ν's symmetric group.
        partitions_n: Partitions of n (default: enumerate automatically).
        partitions_m: Partitions of m (default: enumerate automatically).
        progress: Show progress bar.

    Returns:
        NumPy array of shape (p_n, p_m, n+m+1) where result[i, j, k]
        is the exact probability for conjugacy pair (i, j) at cycle count k.
    """
    if partitions_n is None:
        partitions_n = enumerate_partitions(n)
    if partitions_m is None:
        partitions_m = enumerate_partitions(m)

    p_n = len(partitions_n)
    p_m = len(partitions_m)
    max_k = n + m
    result = np.zeros((p_n, p_m, max_k + 1), dtype=np.float64)

    pairs = [(i, j) for i in range(p_n) for j in range(p_m)]
    if progress:
        pairs = tqdm(pairs, desc="Exact computation", unit="pair")

    for i, j in pairs:
        result[i, j] = exact_probability(tau, n, m, partitions_n[i], partitions_m[j])

    return result


# ---------------------------------------------------------------------------
# Numba-accelerated exact computation
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _numba_count_cycles(perm: np.ndarray) -> int:
    """Count cycles in a permutation (numba-compiled)."""
    n = perm.shape[0]
    visited = np.zeros(n, dtype=numba.boolean)  # type: ignore
    count = 0
    for i in range(n):
        if visited[i]:
            continue
        count += 1
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
    return count


@numba.njit(cache=False)
def _numba_cycle_type_key(perm: np.ndarray) -> np.ndarray:
    """Compute sorted-descending cycle lengths of a permutation.

    Returns an int32 array of length n, zero-padded, with cycle lengths
    sorted in descending order. Two permutations have the same cycle type
    iff their keys are element-wise equal.
    """
    n = perm.shape[0]
    visited = np.zeros(n, dtype=numba.boolean)  # type: ignore
    lengths = np.zeros(n, dtype=np.int32)
    idx = 0
    for i in range(n):
        if visited[i]:
            continue
        length = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            length += 1
        lengths[idx] = length
        idx += 1
    # Sort used cycle lengths in-place (descending).
    # Avoid np.sort(...)[::-1] here because negative-stride slicing can
    # trigger ndarray adaptation issues in some Numba/Python builds.
    for i in range(1, idx):
        current = lengths[i]
        j = i - 1
        while j >= 0 and lengths[j] < current:
            lengths[j + 1] = lengths[j]
            j -= 1
        lengths[j + 1] = current
    return lengths


@numba.njit(cache=True)
def _numba_next_combination(indices: np.ndarray, n: int, k: int) -> bool:
    """Advance ``indices`` to the next k-combination of ``range(n)``.

    Args:
        indices: Current combination indices in ascending order.
        n: Universe size.
        k: Combination size.

    Returns:
        True if advanced, False if already at last combination.
    """
    i = k - 1
    while i >= 0 and indices[i] == n - k + i:
        i -= 1
    if i < 0:
        return False
    indices[i] += 1
    for j in range(i + 1, k):
        indices[j] = indices[j - 1] + 1
    return True


@numba.njit(cache=True)
def _numba_direct_build_conjugacy_class(
    remaining_lengths: np.ndarray,
    rem_len: int,
    available: np.ndarray,
    avail_len: int,
    perm: np.ndarray,
    out: np.ndarray,
    out_idx: np.ndarray,
) -> None:
    """Recursive direct constructor for one conjugacy class (numba-compiled)."""
    if rem_len == 0:
        out[out_idx[0], :] = perm
        out_idx[0] += 1
        return

    leader = available[0]
    rest_len = avail_len - 1
    rest = np.empty(rest_len, dtype=np.int32)
    for i in range(rest_len):
        rest[i] = available[i + 1]

    seen_lengths = np.empty(rem_len, dtype=np.int32)
    seen_len = 0

    for idx in range(rem_len):
        length = remaining_lengths[idx]

        duplicate = False
        for s in range(seen_len):
            if seen_lengths[s] == length:
                duplicate = True
                break
        if duplicate:
            continue
        seen_lengths[seen_len] = length
        seen_len += 1

        new_remaining = np.empty(rem_len - 1, dtype=np.int32)
        wr = 0
        for j in range(rem_len):
            if j != idx:
                new_remaining[wr] = remaining_lengths[j]
                wr += 1

        if length == 1:
            _numba_direct_build_conjugacy_class(
                new_remaining,
                rem_len - 1,
                rest,
                rest_len,
                perm,
                out,
                out_idx,
            )
            continue

        k = length - 1
        comb_idx = np.empty(k, dtype=np.int32)
        for j in range(k):
            comb_idx[j] = j

        while True:
            chosen = np.empty(k, dtype=np.int32)
            chosen_mask = np.zeros(rest_len, dtype=numba.boolean)  # type: ignore
            for j in range(k):
                ci = comb_idx[j]
                chosen[j] = rest[ci]
                chosen_mask[ci] = True

            remaining_available = np.empty(rest_len - k, dtype=np.int32)
            ra = 0
            for j in range(rest_len):
                if not chosen_mask[j]:
                    remaining_available[ra] = rest[j]
                    ra += 1

            ordered = chosen.copy()
            c = np.zeros(k, dtype=np.int32)

            while True:
                new_perm = perm.copy()
                prev = leader
                for j in range(k):
                    nxt = ordered[j]
                    new_perm[prev] = nxt
                    prev = nxt
                new_perm[prev] = leader

                _numba_direct_build_conjugacy_class(
                    new_remaining,
                    rem_len - 1,
                    remaining_available,
                    rest_len - k,
                    new_perm,
                    out,
                    out_idx,
                )

                i2 = 0
                advanced_perm = False
                while i2 < k:
                    if c[i2] < i2:
                        if i2 % 2 == 0:
                            ordered[0], ordered[i2] = ordered[i2], ordered[0]
                        else:
                            ordered[c[i2]], ordered[i2] = ordered[i2], ordered[c[i2]]
                        c[i2] += 1
                        i2 = 0
                        advanced_perm = True
                        break
                    c[i2] = 0
                    i2 += 1

                if not advanced_perm:
                    break

            if not _numba_next_combination(comb_idx, rest_len, k):
                break


@numba.njit(cache=True)
def _numba_enumerate_conjugacy_class_direct(
    n: int,
    cycle_type: np.ndarray,
    class_size: int,
) -> np.ndarray:
    """Enumerate one conjugacy class with direct construction in numba."""
    out = np.empty((class_size, n), dtype=np.int32)
    if n == 0:
        return out

    rem_len = cycle_type.shape[0]
    remaining = np.empty(rem_len, dtype=np.int32)
    for i in range(rem_len):
        remaining[i] = cycle_type[i]

    available = np.arange(n, dtype=np.int32)
    perm = np.arange(n, dtype=np.int32)
    out_idx = np.zeros(1, dtype=np.int64)

    _numba_direct_build_conjugacy_class(
        remaining,
        rem_len,
        available,
        n,
        perm,
        out,
        out_idx,
    )
    return out


def _numba_enumerate_selected_classes(
    n: int,
    partitions: list[tuple[int, ...]],
    selected_indices: list[int],
) -> dict[int, np.ndarray]:
    """Enumerate only selected conjugacy classes of S_n.

    This helper is intended for hybrid exact/MC workflows where only a
    subset of conjugacy classes is needed for exact computation.

    Args:
        n: Permutation size.
        partitions: List of cycle types (descending tuples).
        selected_indices: Indices into ``partitions`` to enumerate.

    Returns:
        Dict mapping partition index -> int32 array of shape
        (class_size, n) containing all permutations in that class.
    """
    unique_indices = sorted(set(selected_indices))
    if not unique_indices:
        return {}

    if n == 0:
        return {idx: np.empty((1, 0), dtype=np.int32) for idx in unique_indices}

    classes: dict[int, np.ndarray] = {}
    for idx in unique_indices:
        cycle_type = np.asarray(partitions[idx], dtype=np.int32)
        class_size = conjugacy_class_size(partitions[idx])
        classes[idx] = _numba_enumerate_conjugacy_class_direct(
            n,
            cycle_type,
            class_size,
        )
    return classes


@numba.njit(cache=True)
def _numba_exact_probability_pair(
    tau: np.ndarray,
    perms_n: np.ndarray,
    perms_m: np.ndarray,
    max_k: int,
) -> np.ndarray:
    """Compute exact probabilities for one conjugacy pair (numba-compiled).

    Args:
        tau: Permutation in S_{n+m}, shape (n+m,).
        perms_n: All σ in the class, shape (size_n, n).
        perms_m: All ν in the class, shape (size_m, m).
        max_k: n + m (maximum possible cycle count).

    Returns:
        float64 array of shape (max_k+1,) with probabilities.
    """
    n = perms_n.shape[1]
    size_n = perms_n.shape[0]
    size_m = perms_m.shape[0]
    nm = tau.shape[0]
    counts = np.zeros(max_k + 1, dtype=np.float64)
    total = size_n * size_m

    embedded = np.empty(nm, dtype=np.int32)
    composed = np.empty(nm, dtype=np.int32)

    for si in range(size_n):
        # Build embedded = sigma ++ (nu + n)
        for x in range(n):
            embedded[x] = perms_n[si, x]
        for sj in range(size_m):
            for x in range(perms_m.shape[1]):
                embedded[n + x] = perms_m[sj, x] + n
            # composed = tau[embedded]
            for x in range(nm):
                composed[x] = tau[embedded[x]]
            k = _numba_count_cycles(composed)
            counts[k] += 1.0

    for k in range(max_k + 1):
        counts[k] /= total
    return counts


def numba_exact_all_conjugacy_pairs(
    tau: np.ndarray | list[int],
    n: int,
    m: int,
    partitions_n: list[tuple[int, ...]] | None = None,
    partitions_m: list[tuple[int, ...]] | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Compute exact probabilities for all conjugacy pairs using Numba.

    Enumerates each conjugacy class via a single-pass Numba-compiled
    permutation generator, then performs composition and cycle counting
    in Numba-compiled inner loops.

    Args:
        tau: Fixed permutation in S_{n+m}, as list or numpy array.
        n: Size of σ's symmetric group.
        m: Size of ν's symmetric group.
        partitions_n: Partitions of n (default: enumerate automatically).
        partitions_m: Partitions of m (default: enumerate automatically).
        progress: Show progress bar.

    Returns:
        NumPy array of shape (p_n, p_m, n+m+1) where result[i, j, k]
        is the exact probability for conjugacy pair (i, j) at cycle count k.
    """
    if partitions_n is None:
        partitions_n = enumerate_partitions(n)
    if partitions_m is None:
        partitions_m = enumerate_partitions(m)

    tau_arr = np.asarray(tau, dtype=np.int32)

    # Pre-enumerate all conjugacy classes using selected-class constructor.
    all_indices_n = list(range(len(partitions_n)))
    all_indices_m = list(range(len(partitions_m)))
    classes_n = _numba_enumerate_selected_classes(n, partitions_n, all_indices_n)
    classes_m = _numba_enumerate_selected_classes(m, partitions_m, all_indices_m)

    p_n = len(partitions_n)
    p_m = len(partitions_m)
    max_k = n + m
    result = np.zeros((p_n, p_m, max_k + 1), dtype=np.float64)

    pairs = [(i, j) for i in range(p_n) for j in range(p_m)]
    if progress:
        pairs = tqdm(pairs, desc="Numba exact computation", unit="pair")

    for i, j in pairs:
        result[i, j] = _numba_exact_probability_pair(
            tau_arr,
            classes_n[i],
            classes_m[j],
            max_k,
        )

    return result


@numba.njit(parallel=True, cache=True)
def _numba_exact_probability_pair_parallel(
    tau: np.ndarray,
    perms_n: np.ndarray,
    perms_m: np.ndarray,
    max_k: int,
) -> np.ndarray:
    """Compute exact probabilities for one conjugacy pair (parallel numba).

    Parallelises the outer loop over the larger conjugacy class (σ or ν)
    using ``numba.prange``, with thread-local accumulation to avoid contention.
    This adaptive strategy improves load balancing when class sizes differ.

    Args:
        tau: Permutation in S_{n+m}, shape (n+m,).
        perms_n: All σ in the class, shape (size_n, n).
        perms_m: All ν in the class, shape (size_m, m).
        max_k: n + m (maximum possible cycle count).

    Returns:
        float64 array of shape (max_k+1,) with probabilities.
    """
    n = perms_n.shape[1]
    size_n = perms_n.shape[0]
    size_m = perms_m.shape[0]
    nm = tau.shape[0]
    total = size_n * size_m

    counts = np.zeros(max_k + 1, dtype=np.float64)

    # Parallelize over the larger dimension for better load balancing.
    if size_n >= size_m:
        # Thread-local counts: shape (size_n, max_k+1).
        local_counts = np.zeros((size_n, max_k + 1), dtype=np.float64)

        for si in numba.prange(size_n):
            embedded = np.empty(nm, dtype=np.int32)
            composed = np.empty(nm, dtype=np.int32)
            for x in range(n):
                embedded[x] = perms_n[si, x]
            for sj in range(size_m):
                for x in range(perms_m.shape[1]):
                    embedded[n + x] = perms_m[sj, x] + n
                for x in range(nm):
                    composed[x] = tau[embedded[x]]
                k = _numba_count_cycles(composed)
                local_counts[si, k] += 1.0

        # Reduce across threads.
        for si in range(size_n):
            for k in range(max_k + 1):
                counts[k] += local_counts[si, k]
    else:
        # Thread-local counts: shape (size_m, max_k+1).
        local_counts = np.zeros((size_m, max_k + 1), dtype=np.float64)

        for sj in numba.prange(size_m):
            embedded = np.empty(nm, dtype=np.int32)
            composed = np.empty(nm, dtype=np.int32)
            for x in range(perms_m.shape[1]):
                embedded[n + x] = perms_m[sj, x] + n
            for si in range(size_n):
                for x in range(n):
                    embedded[x] = perms_n[si, x]
                for x in range(nm):
                    composed[x] = tau[embedded[x]]
                k = _numba_count_cycles(composed)
                local_counts[sj, k] += 1.0

        # Reduce across threads.
        for sj in range(size_m):
            for k in range(max_k + 1):
                counts[k] += local_counts[sj, k]

    for k in range(max_k + 1):
        counts[k] /= total
    return counts


def numba_parallel_exact_all_conjugacy_pairs(
    tau: np.ndarray | list[int],
    n: int,
    m: int,
    partitions_n: list[tuple[int, ...]] | None = None,
    partitions_m: list[tuple[int, ...]] | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Compute exact probabilities for all conjugacy pairs using parallel Numba.

    Same as ``numba_exact_all_conjugacy_pairs`` but uses ``numba.prange``
    to parallelise the inner Cartesian-product loop across CPU cores.

    Args:
        tau: Fixed permutation in S_{n+m}, as list or numpy array.
        n: Size of σ's symmetric group.
        m: Size of ν's symmetric group.
        partitions_n: Partitions of n (default: enumerate automatically).
        partitions_m: Partitions of m (default: enumerate automatically).
        progress: Show progress bar.

    Returns:
        NumPy array of shape (p_n, p_m, n+m+1) where result[i, j, k]
        is the exact probability for conjugacy pair (i, j) at cycle count k.
    """
    if partitions_n is None:
        partitions_n = enumerate_partitions(n)
    if partitions_m is None:
        partitions_m = enumerate_partitions(m)

    tau_arr = np.asarray(tau, dtype=np.int32)

    all_indices_n = list(range(len(partitions_n)))
    all_indices_m = list(range(len(partitions_m)))
    classes_n = _numba_enumerate_selected_classes(n, partitions_n, all_indices_n)
    classes_m = _numba_enumerate_selected_classes(m, partitions_m, all_indices_m)

    p_n = len(partitions_n)
    p_m = len(partitions_m)
    max_k = n + m
    result = np.zeros((p_n, p_m, max_k + 1), dtype=np.float64)

    pairs = [(i, j) for i in range(p_n) for j in range(p_m)]
    if progress:
        pairs = tqdm(pairs, desc="Numba parallel computation", unit="pair")

    for i, j in pairs:
        result[i, j] = _numba_exact_probability_pair_parallel(
            tau_arr,
            classes_n[i],
            classes_m[j],
            max_k,
        )

    return result


if __name__ == "__main__":
    n = 7
    m = 7

    rng = np.random.default_rng(0)
    tau = rng.permutation(n + m).astype(np.int32)

    partitions_n = enumerate_partitions(n)
    partitions_m = enumerate_partitions(m)

    all_indices_n = list(range(len(partitions_n)))
    all_indices_m = list(range(len(partitions_m)))
    classes_n = _numba_enumerate_selected_classes(n, partitions_n, all_indices_n)
    classes_m = _numba_enumerate_selected_classes(m, partitions_m, all_indices_m)

    p_n = len(classes_n)
    p_m = len(classes_m)

    pairs = [(i, j) for i in range(p_n) for j in range(p_m)]
    pairs = tqdm(pairs, desc="Numba parallel computation", unit=" pairs")

    for i, j in pairs:
        _numba_exact_probability_pair_parallel(tau, classes_n[i], classes_m[j], n + m)
