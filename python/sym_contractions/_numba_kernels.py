"""Numba-compiled kernels for the efficient contraction inner loops.

These fused kernels replace the per-iteration Python call chain of
conjugate → compose → extract → cycle_type → index_lookup with a
single compiled function, eliminating interpreter overhead and
intermediate list/array conversions.
"""

from __future__ import annotations

import numba
import numpy as np

# ======================================================================
# Helper: padded cycle-type table for Numba lookups
# ======================================================================


def build_padded_cycle_types(
    cycle_types: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a zero-padded 2D array and a lengths vector from cycle types.

    Parameters
    ----------
    cycle_types : list[list[int]]
        Cycle types (partitions) in descending order, e.g.
        ``[[3], [2, 1], [1, 1, 1]]``.

    Returns
    -------
    padded : np.ndarray, shape (p, max_parts), int32
        Zero-padded cycle type entries.
    lengths : np.ndarray, shape (p,), int32
        Number of parts for each cycle type.
    """
    p = len(cycle_types)
    if p == 0:
        return np.empty((0, 0), dtype=np.int32), np.empty(0, dtype=np.int32)
    max_parts = max(len(ct) for ct in cycle_types)
    if max_parts == 0:
        max_parts = 1  # handle S_0 edge case
    padded = np.zeros((p, max_parts), dtype=np.int32)
    lengths = np.zeros(p, dtype=np.int32)
    for i, ct in enumerate(cycle_types):
        lengths[i] = len(ct)
        for j, val in enumerate(ct):
            padded[i, j] = val
    return padded, lengths


# ======================================================================
# Fused kernel — left-side coset reduction
# ======================================================================


@numba.njit(cache=True)
def fused_inner_left(
    tau: np.ndarray,
    sigma_0: np.ndarray,
    g: np.ndarray,
    n: int,
    m: int,
    ct_padded: np.ndarray,
    ct_lengths: np.ndarray,
    num_types: int,
) -> tuple[int, int]:
    """Fused inner-loop kernel for left-side coset reduction.

    Performs in one compiled pass:

    1. Conjugation: sigma_g = g . sigma_0 . g^{-1}
    2. Block composition: tau' = tau . (sigma_g x id_m)
    3. Effective-perm extraction: (c_F, rho) from tau'
    4. Cycle-type computation of rho (sorted descending)
    5. Index lookup in the padded cycle-type table

    Parameters
    ----------
    tau : int32 array, shape (n+m,)
    sigma_0 : int32 array, shape (n,)
    g : int32 array, shape (n,)
    n, m : int
    ct_padded : int32 array, shape (num_types, max_parts)
    ct_lengths : int32 array, shape (num_types,)
    num_types : int

    Returns
    -------
    c_F : int
    j_rho : int
        Index of rho's cycle type in the table, or -1 if not found.
    """
    nm = n + m

    # 1. Conjugate: sigma_g = g . sigma_0 . g^{-1}
    g_inv = np.empty(n, dtype=np.int32)
    for i in range(n):
        g_inv[g[i]] = i
    sigma_g = np.empty(n, dtype=np.int32)
    for i in range(n):
        sigma_g[i] = g[sigma_0[g_inv[i]]]

    # 2. Compose: tau_prime = tau . (sigma_g x id_m)
    tau_prime = np.empty(nm, dtype=np.int32)
    for i in range(n):
        tau_prime[i] = tau[sigma_g[i]]
    for i in range(n, nm):
        tau_prime[i] = tau[i]

    # 3. Extract effective perm — fixed block = {0,...,n-1}
    visited = np.zeros(nm, dtype=numba.boolean)
    c_F = 0
    for start in range(n):
        if visited[start]:
            continue
        j = start
        stays = True
        while not visited[j]:
            visited[j] = True
            if j >= n:
                stays = False
            j = tau_prime[j]
        if stays:
            c_F += 1

    # Return map rho on free block {n,...,n+m-1}
    rho = np.empty(m, dtype=np.int32)
    for idx in range(m):
        pos = tau_prime[n + idx]
        while not (n <= pos < nm):
            pos = tau_prime[pos]
        rho[idx] = pos - n

    # 4. Cycle type of rho (insertion sort descending)
    visited_rho = np.zeros(m, dtype=numba.boolean)
    lengths = np.zeros(m, dtype=np.int32)
    count = 0
    for i in range(m):
        if visited_rho[i]:
            continue
        length = 0
        j = i
        while not visited_rho[j]:
            visited_rho[j] = True
            j = rho[j]
            length += 1
        lengths[count] = length
        count += 1
    for i in range(1, count):
        current = lengths[i]
        j = i - 1
        while j >= 0 and lengths[j] < current:
            lengths[j + 1] = lengths[j]
            j -= 1
        lengths[j + 1] = current

    # 5. Find index in padded cycle-type table
    j_rho = -1
    for t in range(num_types):
        if ct_lengths[t] != count:
            continue
        match = True
        for k in range(count):
            if ct_padded[t, k] != lengths[k]:
                match = False
                break
        if match:
            j_rho = t
            break

    return c_F, j_rho


# ======================================================================
# Fused kernel — right-side coset reduction
# ======================================================================


@numba.njit(cache=True)
def fused_inner_right(
    tau: np.ndarray,
    nu_0: np.ndarray,
    g: np.ndarray,
    n: int,
    m: int,
    ct_padded: np.ndarray,
    ct_lengths: np.ndarray,
    num_types: int,
) -> tuple[int, int]:
    """Fused inner-loop kernel for right-side coset reduction.

    Same as fused_inner_left but for the right side:
    - Conjugates nu_0 in S_m instead of sigma_0 in S_n.
    - Composes tau . (id_n x nu_g).
    - Fixed block = {n,...,n+m-1}, free block = {0,...,n-1}.
    - Looks up rho's cycle type in the S_n table.

    Parameters
    ----------
    tau : int32 array, shape (n+m,)
    nu_0 : int32 array, shape (m,)
    g : int32 array, shape (m,)
    n, m : int
    ct_padded : int32 array, shape (num_types, max_parts)
    ct_lengths : int32 array, shape (num_types,)
    num_types : int

    Returns
    -------
    c_F : int
    j_rho : int
    """
    nm = n + m

    # 1. Conjugate: nu_g = g . nu_0 . g^{-1}
    g_inv = np.empty(m, dtype=np.int32)
    for i in range(m):
        g_inv[g[i]] = i
    nu_g = np.empty(m, dtype=np.int32)
    for i in range(m):
        nu_g[i] = g[nu_0[g_inv[i]]]

    # 2. Compose: tau_prime = tau . (id_n x nu_g)
    tau_prime = np.empty(nm, dtype=np.int32)
    for i in range(n):
        tau_prime[i] = tau[i]
    for j in range(m):
        tau_prime[n + j] = tau[nu_g[j] + n]

    # 3. Extract effective perm — fixed block = {n,...,n+m-1}
    visited = np.zeros(nm, dtype=numba.boolean)
    c_F = 0
    for start in range(n, nm):
        if visited[start]:
            continue
        j = start
        stays = True
        while not visited[j]:
            visited[j] = True
            if j < n:
                stays = False
            j = tau_prime[j]
        if stays:
            c_F += 1

    # Return map rho on free block {0,...,n-1}
    rho = np.empty(n, dtype=np.int32)
    for idx in range(n):
        pos = tau_prime[idx]
        while not (0 <= pos < n):
            pos = tau_prime[pos]
        rho[idx] = pos

    # 4. Cycle type of rho (insertion sort descending)
    visited_rho = np.zeros(n, dtype=numba.boolean)
    lengths = np.zeros(n, dtype=np.int32)
    count = 0
    for i in range(n):
        if visited_rho[i]:
            continue
        length = 0
        j = i
        while not visited_rho[j]:
            visited_rho[j] = True
            j = rho[j]
            length += 1
        lengths[count] = length
        count += 1
    for i in range(1, count):
        current = lengths[i]
        j = i - 1
        while j >= 0 and lengths[j] < current:
            lengths[j + 1] = lengths[j]
            j -= 1
        lengths[j + 1] = current

    # 5. Find index in padded cycle-type table
    j_rho = -1
    for t in range(num_types):
        if ct_lengths[t] != count:
            continue
        match = True
        for k in range(count):
            if ct_padded[t, k] != lengths[k]:
                match = False
                break
        if match:
            j_rho = t
            break

    return c_F, j_rho


# ======================================================================
# Batch kernels — process all (class, coset_rep) pairs at once
# ======================================================================


@numba.njit(cache=True)
def batch_fused_left(
    tau: np.ndarray,
    canonical_reps: np.ndarray,
    coset_reps: np.ndarray,
    n: int,
    m: int,
    ct_padded: np.ndarray,
    ct_lengths: np.ndarray,
    num_types: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch-process all (alpha, g) pairs for left reduction.

    Parameters
    ----------
    tau : int32 array, shape (n+m,)
    canonical_reps : int32 array, shape (p_n, n)
    coset_reps : int32 array, shape (num_reps, n)
    n, m : int
    ct_padded, ct_lengths : padded cycle-type table for S_m
    num_types : int

    Returns
    -------
    c_F_out : int32 array, shape (p_n, num_reps)
    j_rho_out : int32 array, shape (p_n, num_reps)
    """
    p_n = canonical_reps.shape[0]
    num_reps = coset_reps.shape[0]
    c_F_out = np.empty((p_n, num_reps), dtype=np.int32)
    j_rho_out = np.empty((p_n, num_reps), dtype=np.int32)
    for alpha_idx in range(p_n):
        for g_idx in range(num_reps):
            c_F, j_rho = fused_inner_left(
                tau,
                canonical_reps[alpha_idx],
                coset_reps[g_idx],
                n,
                m,
                ct_padded,
                ct_lengths,
                num_types,
            )
            c_F_out[alpha_idx, g_idx] = c_F
            j_rho_out[alpha_idx, g_idx] = j_rho
    return c_F_out, j_rho_out


@numba.njit(cache=True)
def batch_fused_right(
    tau: np.ndarray,
    canonical_reps: np.ndarray,
    coset_reps: np.ndarray,
    n: int,
    m: int,
    ct_padded: np.ndarray,
    ct_lengths: np.ndarray,
    num_types: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch-process all (beta, g) pairs for right reduction.

    Parameters
    ----------
    tau : int32 array, shape (n+m,)
    canonical_reps : int32 array, shape (p_m, m)
    coset_reps : int32 array, shape (num_reps, m)
    n, m : int
    ct_padded, ct_lengths : padded cycle-type table for S_n
    num_types : int

    Returns
    -------
    c_F_out : int32 array, shape (p_m, num_reps)
    j_rho_out : int32 array, shape (p_m, num_reps)
    """
    p_m = canonical_reps.shape[0]
    num_reps = coset_reps.shape[0]
    c_F_out = np.empty((p_m, num_reps), dtype=np.int32)
    j_rho_out = np.empty((p_m, num_reps), dtype=np.int32)
    for beta_idx in range(p_m):
        for g_idx in range(num_reps):
            c_F, j_rho = fused_inner_right(
                tau,
                canonical_reps[beta_idx],
                coset_reps[g_idx],
                n,
                m,
                ct_padded,
                ct_lengths,
                num_types,
            )
            c_F_out[beta_idx, g_idx] = c_F
            j_rho_out[beta_idx, g_idx] = j_rho
    return c_F_out, j_rho_out
