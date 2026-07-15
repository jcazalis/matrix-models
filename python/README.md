# Python — Symmetric Contractions Library

## Directory Structure

```
python/
├── sym_contractions/             # Core library
│   ├── __init__.py               # Public API re-exports and default paths
│   ├── woven.py                  # Woven contraction loading and computation
│   ├── bruteforce.py             # Brute-force enumeration over conjugacy classes
│   ├── efficient.py              # Efficient coset-reduction + character-theory path
│   ├── estimator.py              # Numba-based Monte Carlo estimation
│   ├── coset.py                  # Coset reduction data loading from GAP
│   ├── store.py                  # Persistent probability storage (NPZ)
│   ├── hamiltonian.py            # Observable and Hamiltonian construction
│   ├── character_tables.py       # Character table loading from GAP JSON
│   ├── utils.py                  # Partition enumeration and permutation utilities
│   └── _numba_kernels.py         # Numba-compiled inner-loop kernels
├── scripts/                      # CLI scripts
│   ├── benchmark_parallel.py     # Serial vs parallel speed benchmarks
│   ├── prepare_coset_input.py    # Generate coset input for GAP from woven JSON
│   ├── prepare_observables.py    # Build and save observables for a Lambda range
│   └── update_prob_collection.py # Update ProbabilityStoreCollection from woven JSON
├── tests/                        # Unit tests (pytest)
│   ├── test_bruteforce.py
│   ├── test_character_tables.py
│   ├── test_coset.py
│   ├── test_efficient.py
│   ├── test_estimator.py
│   ├── test_hamiltonian.py
│   ├── test_store.py
│   └── test_woven.py
├── notebooks/                    # Demo and tutorial notebooks
│   ├── 00_demo.ipynb
│   ├── 00_template.ipynb
│   ├── 01_analyze-hamiltonian.ipynb
│   └── 02_compare-dense-sparse.ipynb
├── pyproject.toml
└── README.md
```

## Setup

```bash
cd python
uv sync
```

## Core Library (`sym_contractions/`)

### Woven Contractions (`woven.py`)

Loading, computation, and export of woven contraction data from Mathematica-generated JSON.

**Key exports**: `load_woven_json()`, `compute_all_contractions()`, `compute_contraction_coefficients()`, `import_precomputed_contractions()`, `export_for_mathematica()`, `involution_to_tau()`, `tau_to_involution()`, `WovenEntry`, `WovenGroup`, `ContractionEntry`, `ContractionResult`, `WovenData`.

### Brute-force Computation (`bruteforce.py`)

Exact enumeration over conjugacy classes for validation. Naive algorithm: $O(n!)$ per conjugacy class.

**Key exports**: `conjugacy_class_size()`, `enumerate_conjugacy_class()`, `exact_probability()`, `exact_all_conjugacy_pairs()`, `numba_exact_all_conjugacy_pairs()`, `numba_parallel_exact_all_conjugacy_pairs()`.

### Efficient Contraction Path (`efficient.py`)

The coset-reduction + character-theory pipeline. Uses `ProcessPoolExecutor` for parallelism (up to 8 workers). Scales to $\Lambda = 22+$, validated against brute force for small sizes.

**Key exports**: `compute_contraction_efficient()`, `compute_all_contractions_efficient()`, `compute_rep_dimensions()`, `compute_s_polynomial()`.

### Monte Carlo Estimation (`estimator.py`)

Numba-compiled Monte Carlo estimation of cycle-count probabilities. Used when brute force is intractable. Tracks sample counts and standard deviations.

**Key exports**: `numba_mc_all_conjugacy_pairs()`.

### Hamiltonian Assembly (`hamiltonian.py`)

Builds observables and Hamiltonians from woven contraction data. Supports dense (`numpy.ndarray`) and sparse (`scipy.sparse.csr_matrix`) coefficient stacks. Mass rescaling: $m^{(n_P - n_X)/2}$.

**Key exports**: `Observable`, `ObservableDense`, `ObservableSparse`, `FreeHamiltonian`, `HamiltonianDense`, `HamiltonianSparse`, `EvaluatedObservable`, `build_observables()`, `normalization()`, `partition_list()`.

### Probability Storage (`store.py`)

Persistent NPZ-backed storage for cycle-count probabilities. Tracks whether each conjugacy pair was computed exactly or via Monte Carlo.

**Key exports**: `ProbabilityStore`, `ProbabilityStoreCollection`, `TauEntry`, `compute_and_store()`.

### Character Tables (`character_tables.py`)

Loading and utility functions for GAP-generated character tables. Reorders from GAP's lexicographic to Python's reverse-lexicographic partition ordering.

**Key exports**: `load_character_table()`, `load_character_tables_range()`, `get_class_weights()`, `compute_class_fraction()`, `find_cycle_type_index()`, `partition_to_cycle_type_key()`.

### Coset Data (`coset.py`)

Data structures and loaders for GAP-generated coset reduction data.

**Key exports**: `CosetReductionData`, `GapCosetData`, `load_gap_coset_data()`, `compute_coset_reduction_from_gap()`.

### Utilities (`utils.py`)

Partition enumeration and permutation helpers.

**Key exports**: `enumerate_partitions()`, `partitions_to_padded_array()`, `canonical_representative()`.

## Default Data Paths

The package defines default paths in `__init__.py` relative to the repository root:

| Variable | Path |
|---|---|
| `PROJECT_ROOT` | `twobraner/` |
| `DATA_ROOT` | `twobraner/data/processed/` |
| `CHARACTER_TABLE_DIR` | `data/processed/character_tables/` |
| `WOVEN_CONTRACTIONS_DIR` | `data/processed/woven_contractions/` |
| `PROBABILITY_STORE_DIR` | `data/processed/probability_stores/` |
| `HAMILTONIAN_DIR` | `data/processed/hamiltonians/` |
| `OBSERVABLE_DIR` | `data/processed/observables/` |

## Tests

```bash
uv run pytest
```

Some tests require GAP-generated data. Generate test data first:

```bash
cd ../data && ./generate_data.sh --test
```

Test files:

| File | What it tests |
|---|---|
| `test_bruteforce.py` | Brute-force exact computation, sum-over-classes = $n!$ validation |
| `test_character_tables.py` | Character table loading and reordering |
| `test_coset.py` | GAP coset data loading, Lagrange checks |
| `test_efficient.py` | Efficient path vs brute-force for small sizes |
| `test_estimator.py` | MC estimation, batch shapes, MC vs exact comparison |
| `test_hamiltonian.py` | Observable/Hamiltonian API, serialization, mass rescaling |
| `test_store.py` | ProbabilityStore save/load, partition ordering |
| `test_woven.py` | Woven contraction loading, conversion utilities, Mathematica export |

## Benchmarks

To compare serial vs parallel speed for the efficient contraction path and
the sparse assembly path, run:

```bash
uv run python scripts/benchmark_parallel.py \
 --label XXXX_p2341 --lambda 18 --mass 1/2 \
 --repeats 3 --max-workers 4
```

Useful options:

- `--generate-missing`: run `data/generate_data.sh` if the woven JSON is missing.
- `--compute-only`: benchmark only `compute_all_contractions_efficient`.
- `--assembly-only`: benchmark only `_assemble_sparse_coeffs`.
- `--json-output path/to/report.json`: write the timing report as JSON.
- `--woven-path path/to/wc.json`: use a specific woven JSON file.

## Scaling Analysis

Scaling analysis is available through the notebooks in `exploration/`, particularly `00_scaling-analysis.ipynb`. The CSV and HDF5 outputs from that notebook are also stored in `exploration/`. For an interactive dashboard of the scaling analysis results, use the plotly/dash dashboard available in the notebook.

## Notebooks

### Demo & Tutorial (`notebooks/`)

- `00_demo.ipynb` — Demo notebook for the `sym_contractions` library
- `01_analyze-hamiltonian.ipynb` — Hamiltonian analysis walkthrough
- `02_compare-dense-sparse.ipynb` — Dense vs sparse comparison

### Exploration (`exploration/`)

- `00_explo.ipynb` — Exploratory analysis
- `00_scaling-analysis.ipynb` — Scaling analysis (spectra, runs, states)
- `01_fermion-mapping.ipynb` — Fermion mapping exploration
- `02_energy-thermal.ipynb` — Energy / thermal analysis
- `03_size_cosets.ipynb` — Coset size analysis
