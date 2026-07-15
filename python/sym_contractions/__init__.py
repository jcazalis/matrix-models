"""Permutation group contraction computations for matrix models."""

from pathlib import Path as _Path

# ======================================================================
# Default data paths (relative to the repository root)
# ======================================================================
from sym_contractions.bruteforce import (
    conjugacy_class_size,
    enumerate_conjugacy_class,
    exact_all_conjugacy_pairs,
    exact_probability,
    numba_exact_all_conjugacy_pairs,
    numba_parallel_exact_all_conjugacy_pairs,
)
from sym_contractions.character_tables import (
    compute_class_fraction,
    find_cycle_type_index,
    get_class_weights,
    load_character_table,
    load_character_tables_range,
    partition_to_cycle_type_key,
)
from sym_contractions.efficient import (
    compute_all_contractions_efficient,
    compute_contraction_efficient,
    compute_rep_dimensions,
    compute_s_polynomial,
)
from sym_contractions.estimator import (
    numba_mc_all_conjugacy_pairs,
)
from sym_contractions.hamiltonian import (
    EvaluatedObservable,
    FreeHamiltonian,
    Hamiltonian,
    HamiltonianDense,
    HamiltonianSparse,
    Observable,
    ObservableDense,
    ObservableSparse,
    build_observables,
    normalization,
    partition_list,
)
from sym_contractions.store import (
    ProbabilityStore,
    ProbabilityStoreCollection,
    TauEntry,
    compute_and_store,
)
from sym_contractions.utils import enumerate_partitions, partitions_to_padded_array
from sym_contractions.woven import (
    ContractionEntry,
    ContractionResult,
    WovenData,
    WovenEntry,
    WovenGroup,
    compute_all_contractions,
    compute_contraction_coefficients,
    export_for_mathematica,
    import_precomputed_contractions,
    involution_to_tau,
    load_woven_json,
    tau_to_involution,
    tau_to_pairs_1indexed,
)

PROJECT_ROOT = _Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
CHARACTER_TABLE_DIR = DATA_ROOT / "character_tables"
WOVEN_CONTRACTIONS_DIR = DATA_ROOT / "woven_contractions"
PROBABILITY_STORE_DIR = DATA_ROOT / "probability_stores"
HAMILTONIAN_DIR = DATA_ROOT / "hamiltonians"  # kept for backward compat
OBSERVABLE_DIR = DATA_ROOT / "observables"

__all__ = [
    "numba_mc_all_conjugacy_pairs",
    "enumerate_partitions",
    "partitions_to_padded_array",
    "conjugacy_class_size",
    "enumerate_conjugacy_class",
    "exact_probability",
    "exact_all_conjugacy_pairs",
    "numba_exact_all_conjugacy_pairs",
    "numba_parallel_exact_all_conjugacy_pairs",
    "ProbabilityStore",
    "ProbabilityStoreCollection",
    "TauEntry",
    "compute_and_store",
    "load_character_table",
    "load_character_tables_range",
    "compute_class_fraction",
    "get_class_weights",
    "partition_to_cycle_type_key",
    "find_cycle_type_index",
    "load_woven_json",
    "WovenEntry",
    "WovenGroup",
    "WovenData",
    "ContractionEntry",
    "ContractionResult",
    "involution_to_tau",
    "tau_to_pairs_1indexed",
    "tau_to_involution",
    "compute_contraction_coefficients",
    "compute_all_contractions",
    "export_for_mathematica",
    "import_precomputed_contractions",
    "EvaluatedObservable",
    "FreeHamiltonian",
    "Hamiltonian",
    "HamiltonianDense",
    "HamiltonianSparse",
    "Observable",
    "ObservableDense",
    "ObservableSparse",
    "normalization",
    "partition_list",
    "build_observables",
    "compute_all_contractions_efficient",
    "compute_contraction_efficient",
    "compute_rep_dimensions",
    "compute_s_polynomial",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "CHARACTER_TABLE_DIR",
    "WOVEN_CONTRACTIONS_DIR",
    "PROBABILITY_STORE_DIR",
    "HAMILTONIAN_DIR",
    "OBSERVABLE_DIR",
]
