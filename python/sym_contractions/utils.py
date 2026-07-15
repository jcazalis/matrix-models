"""Partition enumeration and permutation utilities.

Partitions are returned in reverse lexicographic order (descending parts),
matching the standard mathematical convention.
"""

from __future__ import annotations

import numpy as np


def enumerate_partitions(n: int) -> list[tuple[int, ...]]:
    """Generate all integer partitions of n in reverse lexicographic order.

    Each partition is a tuple of positive integers in descending order
    that sum to n. For example, partitions of 4:
        (4,), (3,1), (2,2), (2,1,1), (1,1,1,1)

    Args:
        n: Positive integer to partition.

    Returns:
        List of partitions, each a tuple of descending positive integers.
    """
    if n == 0:
        return [()]

    result: list[tuple[int, ...]] = []
    # Stack-based iterative generation in reverse lexicographic order.
    # Each stack entry is (remaining_sum, max_part, parts_so_far).
    stack: list[tuple[int, int, list[int]]] = [(n, n, [])]

    while stack:
        remaining, max_part, parts = stack.pop()
        if remaining == 0:
            result.append(tuple(parts))
            continue
        # Iterate from min(remaining, max_part) down to 1 and push in
        # reverse so that larger parts are popped first (reverse lex).
        for k in range(1, min(remaining, max_part) + 1):
            stack.append((remaining - k, k, parts + [k]))

    return result


def partitions_to_padded_array(partitions: list[tuple[int, ...]], n: int) -> np.ndarray:
    """Convert a list of partitions to a zero-padded NumPy int32 array.

    Each partition is padded with zeros on the right to length n. This is
    the format consumed by vmap'd sampling functions.

    Example:
        partitions = [(3, 2), (2, 2, 1)]
        partitions_to_padded_array(partitions, 5)

        Returns:
            [[3, 2, 0, 0, 0],
             [2, 2, 1, 0, 0]]

    Args:
        partitions: List of partitions (tuples of descending positive ints).
        n: Partition size; each row is padded to this length.

    Returns:
        NumPy int32 array of shape (len(partitions), n).
    """
    rows = []
    for p in partitions:
        row = list(p) + [0] * (n - len(p))
        rows.append(row)
    return np.array(rows, dtype=np.int32)


def canonical_representative(cycle_type: tuple[int, ...], n: int) -> list[int]:
    """Build a canonical representative permutation for a given cycle type.

    Places cycles consecutively: the first c₁ elements form the first
    cycle, the next c₂ elements form the second, etc.

    Args:
        cycle_type: Descending tuple, e.g. (3, 2, 1).
        n: Permutation size (must equal sum of cycle_type).

    Returns:
        0-indexed permutation as a list.

    Examples:
        >>> canonical_representative((3, 2, 1), 6)
        [1, 2, 0, 4, 3, 5]
        >>> canonical_representative((2, 2), 4)
        [1, 0, 3, 2]
    """
    assert sum(cycle_type) == n
    perm = list(range(n))
    pos = 0
    for length in cycle_type:
        for i in range(length - 1):
            perm[pos + i] = pos + i + 1
        perm[pos + length - 1] = pos
        pos += length
    return perm
