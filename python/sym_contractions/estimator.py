"""Monte Carlo estimators for cycle count probabilities (Numba backend).

Provides Numba-based estimation of P(ℓ(τ ∘ (σ × ν)) = k)
over all conjugacy-class pairs.
"""

from __future__ import annotations

import os

import numba
import numpy as np
from tqdm import tqdm

# =========================================================================
# Numba Monte Carlo estimation
# =========================================================================


@numba.njit(cache=True)
def _numba_random_shuffle_into(buf):
    """Fisher-Yates shuffle: fill *buf* with a random permutation of [0..n-1]."""
    n = buf.shape[0]
    for i in range(n):
        buf[i] = np.int32(i)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        buf[i], buf[j] = buf[j], buf[i]


@numba.njit(cache=True)
def _numba_build_cyclic_perm(shuffled, cycle_type, perm):
    """Build a permutation with a given cycle type from a shuffled array.

    The cycle type is zero-padded (trailing zeros mark the end).
    """
    n = shuffled.shape[0]
    pos = 0
    for c in range(n):
        length = cycle_type[c]
        if length == 0:
            break
        for i in range(length - 1):
            perm[shuffled[pos + i]] = shuffled[pos + i + 1]
        perm[shuffled[pos + length - 1]] = shuffled[pos]
        pos += length


@numba.njit(cache=True)
def _numba_count_cycles_mc(perm, visited):
    """Count cycles using a pre-allocated *visited* buffer."""
    nm = perm.shape[0]
    for i in range(nm):
        visited[i] = False
    count = 0
    for i in range(nm):
        if visited[i]:
            continue
        count += 1
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
    return count


@numba.njit(parallel=True, cache=True)
def _numba_mc_pairs_parallel(
    tau,
    ct_n_array,
    ct_m_array,
    pair_is,
    pair_js,
    n_samples,
    seeds,
):
    """MC estimation for a batch of conjugacy pairs, parallel over pairs.

    Args:
        tau: int32 array, shape (nm,).
        ct_n_array: int32 array, shape (p_n, n) — padded cycle types.
        ct_m_array: int32 array, shape (p_m, m) — padded cycle types.
        pair_is: int32 array, shape (n_batch,) — row indices into ct_n.
        pair_js: int32 array, shape (n_batch,) — col indices into ct_m.
        n_samples: int — MC samples per pair.
        seeds: int64 array, shape (n_batch,) — unique seed per pair.

    Returns:
        float64 array, shape (n_batch, nm+1) — probability estimates.
    """
    n = ct_n_array.shape[1]
    m = ct_m_array.shape[1]
    nm = n + m
    n_batch = pair_is.shape[0]

    result = np.zeros((n_batch, nm + 1), dtype=np.float64)

    for p in numba.prange(n_batch):
        np.random.seed(seeds[p])

        i_idx = pair_is[p]
        j_idx = pair_js[p]

        counts = np.zeros(nm + 1, dtype=np.float64)

        # Thread-local work buffers
        shuffle_n = np.empty(n, dtype=np.int32)
        sigma = np.empty(n, dtype=np.int32)
        shuffle_m = np.empty(m, dtype=np.int32)
        nu = np.empty(m, dtype=np.int32)
        embedded = np.empty(nm, dtype=np.int32)
        composed = np.empty(nm, dtype=np.int32)
        visited = np.zeros(nm, dtype=np.bool_)

        for _s in range(n_samples):
            _numba_random_shuffle_into(shuffle_n)
            _numba_build_cyclic_perm(shuffle_n, ct_n_array[i_idx], sigma)

            _numba_random_shuffle_into(shuffle_m)
            _numba_build_cyclic_perm(shuffle_m, ct_m_array[j_idx], nu)

            # embedded = sigma ⊕ (nu + n)
            for x in range(n):
                embedded[x] = sigma[x]
            for x in range(m):
                embedded[n + x] = nu[x] + n

            # composed = tau[embedded]
            for x in range(nm):
                composed[x] = tau[embedded[x]]

            k = _numba_count_cycles_mc(composed, visited)
            counts[k] += 1.0

        for k in range(nm + 1):
            result[p, k] = counts[k] / n_samples

    return result


def numba_mc_all_conjugacy_pairs(
    tau: np.ndarray,
    n: int,
    m: int,
    n_samples_per_pair: int,
    partitions_n: list[tuple[int, ...]] | None = None,
    partitions_m: list[tuple[int, ...]] | None = None,
    progress: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate P(ℓ(τ ∘ (σ × ν)) = k) for all conjugacy pairs using Numba MC.

    Uses ``numba.prange`` to parallelise across conjugacy-class pairs.
    Each pair runs *n_samples_per_pair* sequential MC samples; pairs are
    distributed across all CPU cores.

    Args:
        tau: Fixed permutation in S_{n+m}.
        n: Size of σ's symmetric group.
        m: Size of ν's symmetric group.
        n_samples_per_pair: MC samples per conjugacy class pair.
        partitions_n: Partitions of n (default: enumerate automatically).
        partitions_m: Partitions of m (default: enumerate automatically).
        progress: Show tqdm progress bar (pairs are processed in batches).
        seed: Random seed for reproducibility.

    Returns:
        (estimates, std_errors): Both numpy float64, shape (p_n, p_m, n+m+1).
    """
    from sym_contractions.utils import enumerate_partitions

    if partitions_n is None:
        partitions_n = enumerate_partitions(n)
    if partitions_m is None:
        partitions_m = enumerate_partitions(m)

    tau_arr = np.asarray(tau, dtype=np.int32)
    p_n = len(partitions_n)
    p_m = len(partitions_m)
    nm = n + m
    n_pairs = p_n * p_m

    # Build zero-padded cycle type arrays (numpy).
    ct_n = np.zeros((p_n, n), dtype=np.int32)
    for i, part in enumerate(partitions_n):
        for k, v in enumerate(part):
            ct_n[i, k] = v

    ct_m = np.zeros((p_m, m), dtype=np.int32)
    for j, part in enumerate(partitions_m):
        for k, v in enumerate(part):
            ct_m[j, k] = v

    # Flat pair-index arrays.
    pair_is = np.empty(n_pairs, dtype=np.int32)
    pair_js = np.empty(n_pairs, dtype=np.int32)
    for i in range(p_n):
        for j in range(p_m):
            pair_is[i * p_m + j] = i
            pair_js[i * p_m + j] = j

    # Unique seeds per pair.
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_pairs, dtype=np.int64)

    if not progress:
        flat_result = _numba_mc_pairs_parallel(
            tau_arr, ct_n, ct_m, pair_is, pair_js, n_samples_per_pair, seeds
        )
        result = flat_result.reshape(p_n, p_m, nm + 1)
        std_errors = np.sqrt(np.abs(result * (1.0 - result)) / n_samples_per_pair)
        return result, std_errors

    # Process in batches so tqdm can report progress.
    n_cores = os.cpu_count() or 8
    batch_size = max(n_cores, min(n_pairs, n_pairs // 20 + 1))
    flat_result = np.zeros((n_pairs, nm + 1), dtype=np.float64)

    n_batches = (n_pairs + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="Numba MC (parallel)", unit="batch"):
        start = b * batch_size
        end = min(start + batch_size, n_pairs)
        flat_result[start:end] = _numba_mc_pairs_parallel(
            tau_arr,
            ct_n,
            ct_m,
            pair_is[start:end],
            pair_js[start:end],
            n_samples_per_pair,
            seeds[start:end],
        )

    result = flat_result.reshape(p_n, p_m, nm + 1)
    std_errors = np.sqrt(np.abs(result * (1.0 - result)) / n_samples_per_pair)
    return result, std_errors
