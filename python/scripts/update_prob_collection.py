"""Update a ProbabilityStoreCollection from a woven-contractions JSON file.

Workflow
--------
1. Load woven JSON (Mathematica export).
2. Load an existing ProbabilityStoreCollection file if present,
   otherwise create a new one.
3. Compute and merge all required probabilities.
4. Save back to the same collection file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sym_contractions.bruteforce import conjugacy_class_size
from sym_contractions.store import (
    ProbabilityStore,
    ProbabilityStoreCollection,
    compute_and_store,
)
from sym_contractions.woven import WovenData, load_woven_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "data" / "processed"
DEFAULT_WOVEN_JSON = DATA_ROOT / "woven_contractions" / "wc_K4_Lambda14.json"
DEFAULT_STORE_FILE = (
    DATA_ROOT / "probability_stores" / "probability_store_collection.npz"
)


def _count_tau_entries(collection: ProbabilityStoreCollection) -> int:
    return sum(len(store.entries) for store in collection.stores.values())


def _prepare_grouped_taus(
    collection: ProbabilityStoreCollection,
    woven: WovenData,
) -> tuple[WovenData, list[tuple[tuple[int, int], list[tuple[int, ...]]]]]:
    effective_woven = woven
    # Keep only entries for n <= m
    effective_woven = effective_woven.filter_upper_diagonal_excitations()

    # Filter excitations higher than Lambda
    if effective_woven.Lambda > collection.Lambda:
        effective_woven = effective_woven.filter_by_max_excitations(collection.Lambda)

    grouped: list[tuple[tuple[int, int], list[tuple[int, ...]]]] = []
    for (n, m), group in sorted(effective_woven.groups.items()):
        if n > collection.Lambda or m > collection.Lambda:
            continue
        tau_set = {tuple(int(x) for x in entry.tau) for entry in group.entries}
        if tau_set:
            grouped.append(((n, m), sorted(tau_set)))

    return effective_woven, grouped


def _estimate_group_workload(
    store: ProbabilityStore,
    tau_list: list[tuple[int, ...]],
    n_samples_per_pair: int,
    max_sequential_search_space: int,
    max_parallel_search_space: int,
) -> dict[str, int]:
    """Estimate the actual work `compute_and_store` will perform for one group."""
    sizes_n = [conjugacy_class_size(ct) for ct in store.partitions_n]
    sizes_m = [conjugacy_class_size(ct) for ct in store.partitions_m]

    exact_pairs = 0
    mc_pairs = 0
    exact_elements = 0
    mc_samples = 0

    for tau_key in tau_list:
        existing = store.entries.get(tau_key)
        if existing is not None:
            needs_update = ~existing.is_exact & (
                existing.n_samples < n_samples_per_pair
            )
            if not needs_update.any():
                continue
        else:
            needs_update = np.ones((store.p_n, store.p_m), dtype=bool)

        for i, j in np.argwhere(needs_update):
            ss = sizes_n[int(i)] * sizes_m[int(j)]
            if ss <= max_parallel_search_space:
                exact_pairs += 1
                exact_elements += ss
            else:
                mc_pairs += 1
                mc_samples += n_samples_per_pair

    return {
        "exact_pairs": exact_pairs,
        "mc_pairs": mc_pairs,
        "exact_elements": exact_elements,
        "mc_samples": mc_samples,
    }


def _group_std_error_stats(
    store: ProbabilityStore,
    tau_list: list[tuple[int, ...]],
) -> dict[str, float] | None:
    values: list[np.ndarray] = []
    for tau in tau_list:
        entry = store.entries.get(tau)
        if entry is None:
            continue
        mc_mask = ~entry.is_exact
        if mc_mask.any():
            values.append(entry.std_errors[mc_mask].reshape(-1))

    if not values:
        return None

    vec = np.concatenate(values)
    return {
        "mean": float(np.mean(vec)),
        "median": float(np.median(vec)),
        "p95": float(np.percentile(vec, 95.0)),
        "max": float(np.max(vec)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load woven contractions, update a ProbabilityStoreCollection, "
            "and save it back to disk."
        )
    )
    parser.add_argument(
        "--woven-json",
        type=Path,
        default=DEFAULT_WOVEN_JSON,
        help=f"Path to woven contraction JSON (default: {DEFAULT_WOVEN_JSON})",
    )
    parser.add_argument(
        "--store-file",
        type=Path,
        default=DEFAULT_STORE_FILE,
        help=f"Path to ProbabilityStoreCollection NPZ (default: {DEFAULT_STORE_FILE})",
    )
    parser.add_argument(
        "--lambda-max",
        type=int,
        default=None,
        help=(
            "Lambda to use when creating a new collection. Default: woven JSON Lambda."
        ),
    )
    parser.add_argument("--n-samples-per-pair", type=int, default=10_000)
    parser.add_argument("--max-sequential-search-space", type=int, default=200_000)
    parser.add_argument("--max-parallel-search-space", type=int, default=1_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    woven_path = args.woven_json
    store_path = args.store_file

    woven = load_woven_json(woven_path)
    lambda_max = woven.Lambda if args.lambda_max is None else args.lambda_max

    if lambda_max > woven.Lambda:
        raise ValueError(
            f"Invalid --lambda-max: {lambda_max} exceeds woven Lambda={woven.Lambda}"
        )

    if store_path.exists():
        collection = ProbabilityStoreCollection.load(store_path)
        # Update the value of Lambda max if required
        collection.Lambda = (
            lambda_max if collection.Lambda < lambda_max else collection.Lambda
        )
        print(
            f"Loaded existing collection: {store_path} "
            f"(Lambda={collection.Lambda}, stores={len(collection.stores)})"
        )
    else:
        collection = ProbabilityStoreCollection.create(lambda_max)
        print(
            f"Created new collection: Lambda={collection.Lambda} "
            f"(from woven Lambda={woven.Lambda})"
        )

    before_stores = len(collection.stores)
    before_taus = _count_tau_entries(collection)

    effective_woven, grouped_taus = _prepare_grouped_taus(collection, woven)
    total_groups = len(grouped_taus)
    total_unique_taus = sum(len(taus) for _, taus in grouped_taus)

    total_exact_pairs = 0
    total_mc_pairs = 0
    total_exact_elements = 0
    total_mc_samples = 0
    total_elapsed = 0.0
    per_group_std: list[tuple[tuple[int, int], dict[str, float] | None]] = []

    print(
        "Workload: "
        f"woven groups={len(woven.groups)}"
        f" -> effective groups={len(effective_woven.groups)}, "
        f"compute groups={total_groups}, unique taus={total_unique_taus}"
    )

    # Create the folder for saving if needed
    store_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, ((n, m), tau_list) in enumerate(grouped_taus, start=1):
        key = (n, m)
        if key not in collection.stores:
            collection.stores[key] = ProbabilityStore.create(n, m)
        store = collection.stores[key]
        before_group_taus = len(store.entries)

        print(
            f"[{idx}/{total_groups}] (n,m)=({n},{m}) "
            f"processing {len(tau_list)} tau(s); "
            f"existing entries={before_group_taus}"
        )

        workload = _estimate_group_workload(
            store,
            tau_list,
            n_samples_per_pair=args.n_samples_per_pair,
            max_sequential_search_space=args.max_sequential_search_space,
            max_parallel_search_space=args.max_parallel_search_space,
        )
        print(
            f"[{idx}/{total_groups}] workload: "
            f"exact pairs={workload['exact_pairs']}, "
            f"mc pairs={workload['mc_pairs']}, "
            f"exact elements={workload['exact_elements']:,d}, "
            f"mc samples={workload['mc_samples']:,d}"
        )

        t0 = time.perf_counter()

        # Compute the tau iteratively
        for tau in tau_list:
            compute_and_store(
                store,
                [tau],
                n_samples_per_pair=args.n_samples_per_pair,
                max_sequential_search_space=args.max_sequential_search_space,
                max_parallel_search_space=args.max_parallel_search_space,
                progress=not args.no_progress,
                seed=args.seed,
            )

            # Save the collection after every tau has been computed
            collection.save(store_path)

        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed

        total_exact_pairs += workload["exact_pairs"]
        total_mc_pairs += workload["mc_pairs"]
        total_exact_elements += workload["exact_elements"]
        total_mc_samples += workload["mc_samples"]

        safe_elapsed = max(elapsed, 1e-12)
        mc_speed = workload["mc_samples"] / safe_elapsed
        exact_speed = workload["exact_elements"] / safe_elapsed

        std_stats = _group_std_error_stats(store, tau_list)
        per_group_std.append(((n, m), std_stats))

        after_group_taus = len(store.entries)
        print(
            f"[{idx}/{total_groups}] (n,m)=({n},{m}) done; "
            f"entries {before_group_taus} -> {after_group_taus}; "
            f"elapsed={elapsed:.2f}s; "
            f"MC speed={mc_speed:,.1f} samples/s; "
            f"Exact speed={exact_speed:,.1f} elements/s"
        )
        if std_stats is None:
            print(f"[{idx}/{total_groups}] std-error stats: no MC estimates")
        else:
            print(
                f"[{idx}/{total_groups}] std-error stats: "
                f"mean={std_stats['mean']:.3e}, "
                f"median={std_stats['median']:.3e}, "
                f"p95={std_stats['p95']:.3e}, "
                f"max={std_stats['max']:.3e}"
            )
        print("=" * 60)

    after_stores = len(collection.stores)
    after_taus = _count_tau_entries(collection)

    store_path.parent.mkdir(parents=True, exist_ok=True)
    collection.save(store_path)

    print(f"Saved collection to: {store_path}")
    print(
        "Summary: "
        f"stores {before_stores} -> {after_stores}, "
        f"tau entries {before_taus} -> {after_taus}"
    )
    safe_total_elapsed = max(total_elapsed, 1e-12)
    print("Computation report:")
    print(
        f"  exact computations: {total_exact_pairs} pair(s) "
        f"| elements generated: {total_exact_elements} "
        f"| speed: {total_exact_elements / safe_total_elapsed:,.1f} elements/s"
    )
    print(
        f"  MC estimations: {total_mc_pairs} pair(s) "
        f"| samples: {total_mc_samples} "
        f"| speed: {total_mc_samples / safe_total_elapsed:,.1f} samples/s"
    )


if __name__ == "__main__":
    main()
