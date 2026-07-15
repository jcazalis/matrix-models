"""Benchmark serial vs parallel speedups for efficient contractions.

This script benchmarks the two CPU-heavy stages that now expose
optional process-based parallelism:

1. ``compute_all_contractions_efficient``
2. ``_assemble_sparse_coeffs``

It reports per-stage wall times and speedups for serial vs parallel
execution on the same woven input.

Examples
--------
Benchmark both stages for an existing woven file:

    uv run python scripts/benchmark_parallel.py \
        --label XXXX_p2143 --lambda 18 --mass 1/2 \
        --repeats 3 --max-workers 4

Generate missing data first:

    uv run python scripts/benchmark_parallel.py \
        --label XXXX_p2143 --lambda 18 --mass 1/2 \
        --generate-missing

Benchmark only sparse assembly from a specific woven JSON:

    uv run python scripts/benchmark_parallel.py \
        --woven-path ../data/processed/woven_contractions/wc_K4_Lambda18.json \
        --assembly-only --repeats 5
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from sym_contractions import (
    CHARACTER_TABLE_DIR,
    DATA_ROOT,
    PROJECT_ROOT,
    load_woven_json,
)
from sym_contractions.efficient import compute_all_contractions_efficient
from sym_contractions.hamiltonian import (
    _assemble_sparse_coeffs,
    _label_to_ops_spec,
    _label_to_woven_filename,
    partition_list,
)
from sym_contractions.woven import ContractionResult, WovenData


@dataclass(frozen=True)
class BenchmarkStats:
    stage: str
    mode: str
    repeats: int
    workers: int
    times_seconds: list[float]
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    stdev_seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark serial vs parallel efficient contractions and sparse assembly."
    )
    parser.add_argument("--label", help="Observable label, e.g. XXXX_p2143")
    parser.add_argument("--lambda", dest="Lambda", type=int, help="Excitation cutoff")
    parser.add_argument(
        "--mass",
        default="0.5",
        help="Mass parameter used to locate or generate woven data (default: 0.5)",
    )
    parser.add_argument(
        "--woven-path",
        type=Path,
        help="Path to a woven JSON file. Overrides --label/--lambda/--mass lookup.",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Run data/generate_data.sh if the woven JSON is missing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions for each mode (default: 3)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional upper bound for parallel workers.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Disable warmup runs before timing.",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Benchmark only compute_all_contractions_efficient.",
    )
    parser.add_argument(
        "--assembly-only",
        action="store_true",
        help="Benchmark only _assemble_sparse_coeffs.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a JSON benchmark report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward verbose output to the benchmarked functions.",
    )
    args = parser.parse_args()

    if args.compute_only and args.assembly_only:
        raise ValueError("--compute-only and --assembly-only are mutually exclusive")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.max_workers is not None and args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1 when provided")
    if args.woven_path is None and (args.label is None or args.Lambda is None):
        raise ValueError("Provide --woven-path or both --label and --lambda")

    return args


def _maybe_generate_woven(args: argparse.Namespace, woven_path: Path) -> None:
    """Generate woven data if requested and the target file is missing."""
    if woven_path.exists() or not args.generate_missing:
        return
    if args.label is None or args.Lambda is None:
        raise FileNotFoundError(f"Woven JSON not found: {woven_path}")

    data_dir = PROJECT_ROOT / "data"
    gen_script = data_dir / "generate_data.sh"
    cmd = [
        "bash",
        str(gen_script),
        "--lambda",
        str(args.Lambda),
        "--ops",
        _label_to_ops_spec(args.label),
        "--mass",
        str(args.mass),
    ]
    print(f"Generating woven data with: {' '.join(cmd[2:])}")
    subprocess.run(cmd, check=True)


def _resolve_woven_path(args: argparse.Namespace) -> Path:
    """Resolve the woven JSON path from CLI arguments."""
    if args.woven_path is not None:
        return args.woven_path.resolve()

    assert args.label is not None
    assert args.Lambda is not None
    woven_name = _label_to_woven_filename(args.label, args.mass, args.Lambda)
    return (DATA_ROOT / "woven_contractions" / woven_name).resolve()


def _load_benchmark_inputs(
    args: argparse.Namespace,
) -> tuple[WovenData, WovenData, Path]:
    """Load woven input and the filtered contraction workload."""
    woven_path = _resolve_woven_path(args)
    _maybe_generate_woven(args, woven_path)
    if not woven_path.exists():
        raise FileNotFoundError(f"Woven JSON not found: {woven_path}")

    coset_path = DATA_ROOT / "coset_reps" / "coset_reps.json"
    if not coset_path.exists():
        raise FileNotFoundError(f"Coset representatives file not found: {coset_path}")

    woven = load_woven_json(woven_path)
    compute_woven = (
        woven.filter_upper_diagonal_excitations()
        if woven.is_hermitian is True
        else woven
    )
    return woven, compute_woven, coset_path


def _summarize_times(
    stage: str,
    mode: str,
    workers: int,
    times_seconds: list[float],
) -> BenchmarkStats:
    """Build summary statistics for one benchmark stage."""
    mean_seconds = statistics.fmean(times_seconds)
    median_seconds = statistics.median(times_seconds)
    stdev_seconds = statistics.stdev(times_seconds) if len(times_seconds) > 1 else 0.0
    return BenchmarkStats(
        stage=stage,
        mode=mode,
        repeats=len(times_seconds),
        workers=workers,
        times_seconds=times_seconds,
        mean_seconds=mean_seconds,
        median_seconds=median_seconds,
        min_seconds=min(times_seconds),
        max_seconds=max(times_seconds),
        stdev_seconds=stdev_seconds,
    )


def _time_callable(
    stage: str,
    mode: str,
    workers: int,
    repeats: int,
    callback: Callable[[], Any],
) -> tuple[BenchmarkStats, Any]:
    """Run one callback repeatedly and collect wall-time statistics."""
    times_seconds: list[float] = []
    last_result: Any = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        last_result = callback()
        times_seconds.append(time.perf_counter() - start)
    return _summarize_times(stage, mode, workers, times_seconds), last_result


def _format_speedup(
    serial_stats: BenchmarkStats, parallel_stats: BenchmarkStats
) -> str:
    """Format a speedup value from two benchmark summaries."""
    speedup = serial_stats.mean_seconds / parallel_stats.mean_seconds
    return f"{speedup:.2f}x"


def _print_stage_report(
    serial_stats: BenchmarkStats,
    parallel_stats: BenchmarkStats,
) -> None:
    """Print a compact two-line benchmark report for one stage."""
    print(f"\n{serial_stats.stage}:")
    print(
        "  serial   "
        f"mean={serial_stats.mean_seconds:.3f}s "
        f"median={serial_stats.median_seconds:.3f}s "
        f"min={serial_stats.min_seconds:.3f}s "
        f"max={serial_stats.max_seconds:.3f}s"
    )
    print(
        "  parallel "
        f"mean={parallel_stats.mean_seconds:.3f}s "
        f"median={parallel_stats.median_seconds:.3f}s "
        f"min={parallel_stats.min_seconds:.3f}s "
        f"max={parallel_stats.max_seconds:.3f}s "
        f"workers={parallel_stats.workers} "
        f"speedup={_format_speedup(serial_stats, parallel_stats)}"
    )


def _warmup_compute(
    compute_woven: WovenData,
    coset_path: Path,
    *,
    max_workers: int | None,
    verbose: bool,
) -> None:
    """Warm up contraction kernels and caches before timing."""
    print("Warming up compute_all_contractions_efficient ...")
    compute_all_contractions_efficient(
        compute_woven,
        CHARACTER_TABLE_DIR,
        coset_path,
        verbose=verbose,
        parallel=False,
        max_workers=max_workers,
    )
    compute_all_contractions_efficient(
        compute_woven,
        CHARACTER_TABLE_DIR,
        coset_path,
        verbose=verbose,
        parallel=True,
        max_workers=max_workers,
    )


def _warmup_assembly(
    contraction: ContractionResult,
    woven: WovenData,
    *,
    max_workers: int | None,
    verbose: bool,
) -> None:
    """Warm up sparse assembly before timing."""
    print("Warming up _assemble_sparse_coeffs ...")
    _assemble_sparse_coeffs(
        contraction,
        woven,
        verbose=verbose,
        parallel=False,
        max_workers=max_workers,
    )
    _assemble_sparse_coeffs(
        contraction,
        woven,
        verbose=verbose,
        parallel=True,
        max_workers=max_workers,
    )


def main() -> None:
    args = _parse_args()
    woven, compute_woven, coset_path = _load_benchmark_inputs(args)

    print("Benchmark configuration:")
    print(f"  woven_label={woven.label}")
    print(f"  Lambda={woven.Lambda}")
    print(f"  mass={woven.mass}")
    print(f"  total_sectors={len(woven.groups)}")
    print(f"  compute_sectors={len(compute_woven.groups)}")
    print(f"  basis_size={len(partition_list(woven.Lambda))}")
    if args.max_workers is not None:
        print(f"  max_workers={args.max_workers}")

    benchmark_compute = not args.assembly_only
    benchmark_assembly = not args.compute_only

    report: dict[str, Any] = {
        "woven_label": woven.label,
        "Lambda": woven.Lambda,
        "mass": woven.mass,
        "total_sectors": len(woven.groups),
        "compute_sectors": len(compute_woven.groups),
        "basis_size": len(partition_list(woven.Lambda)),
        "max_workers": args.max_workers,
        "repeats": args.repeats,
        "results": {},
    }

    contraction_for_assembly: ContractionResult | None = None

    if benchmark_compute:
        if not args.skip_warmup:
            _warmup_compute(
                compute_woven,
                coset_path,
                max_workers=args.max_workers,
                verbose=args.verbose,
            )

        serial_compute, serial_contractions = _time_callable(
            stage="compute_all_contractions_efficient",
            mode="serial",
            workers=1,
            repeats=args.repeats,
            callback=lambda: compute_all_contractions_efficient(
                compute_woven,
                CHARACTER_TABLE_DIR,
                coset_path,
                verbose=args.verbose,
                parallel=False,
                max_workers=args.max_workers,
            ),
        )
        parallel_compute, parallel_contractions = _time_callable(
            stage="compute_all_contractions_efficient",
            mode="parallel",
            workers=args.max_workers or 0,
            repeats=args.repeats,
            callback=lambda: compute_all_contractions_efficient(
                compute_woven,
                CHARACTER_TABLE_DIR,
                coset_path,
                verbose=args.verbose,
                parallel=True,
                max_workers=args.max_workers,
            ),
        )
        _print_stage_report(serial_compute, parallel_compute)
        report["results"]["compute_all_contractions_efficient"] = {
            "serial": asdict(serial_compute),
            "parallel": asdict(parallel_compute),
            "speedup": serial_compute.mean_seconds / parallel_compute.mean_seconds,
            "entry_count": len(serial_contractions.entries),
        }
        contraction_for_assembly = serial_contractions

        if len(serial_contractions.entries) != len(parallel_contractions.entries):
            raise RuntimeError(
                "Serial and parallel contraction benchmarks produced different entry counts"
            )

    if benchmark_assembly:
        if contraction_for_assembly is None:
            print("Preparing contraction input for sparse assembly benchmark ...")
            contraction_for_assembly = compute_all_contractions_efficient(
                compute_woven,
                CHARACTER_TABLE_DIR,
                coset_path,
                verbose=args.verbose,
                parallel=False,
                max_workers=args.max_workers,
            )

        if not args.skip_warmup:
            _warmup_assembly(
                contraction_for_assembly,
                woven,
                max_workers=args.max_workers,
                verbose=args.verbose,
            )

        serial_assembly, _ = _time_callable(
            stage="_assemble_sparse_coeffs",
            mode="serial",
            workers=1,
            repeats=args.repeats,
            callback=lambda: _assemble_sparse_coeffs(
                contraction_for_assembly,
                woven,
                verbose=args.verbose,
                parallel=False,
                max_workers=args.max_workers,
            ),
        )
        parallel_assembly, _ = _time_callable(
            stage="_assemble_sparse_coeffs",
            mode="parallel",
            workers=args.max_workers or 0,
            repeats=args.repeats,
            callback=lambda: _assemble_sparse_coeffs(
                contraction_for_assembly,
                woven,
                verbose=args.verbose,
                parallel=True,
                max_workers=args.max_workers,
            ),
        )
        _print_stage_report(serial_assembly, parallel_assembly)
        report["results"]["_assemble_sparse_coeffs"] = {
            "serial": asdict(serial_assembly),
            "parallel": asdict(parallel_assembly),
            "speedup": serial_assembly.mean_seconds / parallel_assembly.mean_seconds,
            "entry_count": len(contraction_for_assembly.entries),
        }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
