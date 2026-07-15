# Data Directory

This directory contains all generated data used by both the Mathematica and Python sides of the project. Data is produced by a multi-step pipeline combining GAP scripts, a Mathematica script, and a Python script.

## Usage Across Projects

- **`../mathematica/`**: Reads `processed/conjugacy_classes/`, `processed/double_cosets/`, and `processed/woven_contractions/`.
- **`../python/`**: Reads `processed/character_tables/`, `processed/woven_contractions/`, `processed/coset_reps/`, `processed/probability_stores/`, `processed/hamiltonians/`, and `processed/observables/`.

## Scripts

### Main Pipeline

- **`generate_data.sh`**: Master script that runs the full 6-step pipeline (see below).
- **`clean_data.sh`**: Deletes all generated data files while preserving the folder structure.

### GAP Scripts (`gap_scripts/`)

These scripts are called by `generate_data.sh` and should not be run directly without pre-setting their input variables.

- `main.g`: Helper functions used by all other GAP scripts (`PrintInfo()`, `ToMathematicaString()`, etc.).
- `generate_ssct.g`: Generates character tables and conjugacy class elements in Mathematica format (`ssct_n.txt`). Requires `nmax`. Skipped in production mode.
- `generate_ct.g`: Generates character tables in Python/JSON format (`ct_n.json`). Requires `nmax`.
- `generate_dc.g`: Generates double coset representatives and sizes in Mathematica format (`dc_n{n}_p{p}_q{q}.txt`). Requires `nmax`, `max_offset`, and `odd_difference` (boolean).
- `generate_coset_reps.g`: Reads `_coset_input_work.g` (produced by Python step 5) and generates `coset_reps.json`. Uses helper functions from `coset_reps.g`.
- `coset_reps.g`: Core coset computation library (`ComputeCosetReps()`, `ProcessCosetBatch()`). Uses GAP's `Centralizer`, `ActionHomomorphism`, `RightTransversal`.
- `demo_ct.g`, `demo_dc.g`, `demo_dc_bis.g`, `demo_ssct.g`: Example scripts demonstrating individual functions.

### Mathematica Script (`../mathematica/`)

- `generate_wc.wls`: Computes woven contractions and writes `wc_op_{ops}_p{perm}_m{mass}_Lambda{Lambda}.json`. Called by `generate_data.sh` via `wolframscript`.

### Python Scripts (`../python/scripts/`)

- `prepare_coset_input.py`: Reads woven JSON files and generates the GAP input file `_coset_input_work.g` listing all `(τ, n, m)` triples.
- `prepare_observables.py`: Builds and saves observables for a range of Lambda values.
- `update_prob_collection.py`: Updates a `ProbabilityStoreCollection` from a woven JSON file.
- `benchmark_parallel.py`: Benchmarks serial vs parallel speed for efficient contractions and sparse assembly.

## Pipeline

`generate_data.sh` runs the following six steps in order:

| Step | Tool | Output |
|------|------|--------|
| 1 | GAP `generate_ssct.g` | `processed/conjugacy_classes/ssct_n.txt` |
| 2 | GAP `generate_ct.g` | `processed/character_tables/ct_n.json` |
| 3 | GAP `generate_dc.g` | `processed/double_cosets/dc_n{n}_p{p}_q{q}.txt` |
| 4 | Mathematica `generate_wc.wls` (with ops, perm, Lambda, mass) | `processed/woven_contractions/wc_op_{ops}_p{perm}_m{mass}_Lambda{Lambda}.json` |
| 5 | Python `prepare_coset_input.py` | `processed/coset_reps/_coset_input_work.g` |
| 6 | GAP `generate_coset_reps.g` | `processed/coset_reps/coset_reps.json` |

Step 1 (SSCT) is skipped in production mode (`nmax_ssct = 0`) because it generates all permutation elements for each conjugacy class, which is expensive. It runs only in test mode (up to `n = 6`).

## Usage

You need GAP, Wolfram Engine (or Mathematica), and `uv` (Python package manager) installed and available in your PATH.

### Full Pipeline

```bash
# Production (ops=XXXX:2341, Lambda=14, mass=1/2)
./generate_data.sh

# Test mode — small scale, woven/coset output goes to processed/test-data/ - used by the test files in Python and Mathematica
./generate_data.sh --test

# Custom parameters
./generate_data.sh --lambda 10 --ops XXXX:2143,XPXP:2341 --mass 1/2
```

**Options:**

| Flag | Description | Production default | Test default |
|------|-------------|-------------------|--------------|
| `--test` | Small-scale test run; woven contractions and coset reps go to `processed/test-data/` | — | — |
| `--lambda LAMBDA` | Cutoff Λ | 14 | 6 |
| `--ops OPS1:PERM1,...` | Comma-separated operator specs (`OPS:PERM` where OPS is X/P string, PERM is 1-indexed permutation) | `XXXX:2341` | `XX:21,XXXX:2341` |
| `--mass MASS` | Mass parameter passed to `generate_wc.wls` | `1/2` | `1` |

**Derived parameters (not user-configurable):**

- `nmax_ct = Lambda`
- `nmax_dc = Lambda + K_max` (where `K_max` = max operator string length)
- `max_offset` parities: computed separately for even and odd K values
  - Even K: `max_offset = K/2`, `odd_difference = false`
  - Odd K: `max_offset = (K-1)/2`, `odd_difference = true`
- `nmax_ssct = 0` (production) or `6` (test)

### Cleaning Generated Data

```bash
./clean_data.sh
```

Removes all files from `processed/` subdirectories (including `test-data/`) while preserving the folder structure. Cleans: `character_tables`, `conjugacy_classes`, `double_cosets`, `coset_reps`, `probability_stores`, `woven_contractions`, `test-data`.

## Data Formats

### Character Tables and Conjugacy Elements — Mathematica (`ssct_n.txt`)

Mathematica association with keys:

- `"CharacterParameters"`: list of partitions of `n`
- `"SizesConjugacyClasses"`: conjugacy class sizes
- `"ElementsConjugacyClasses"`: all permutation elements per class
- `"CharacterTable"`: character table matrix

### Character Tables — Python (`ct_n.json`)

JSON dictionary with keys:

- `"n"`: group order
- `"CharacterParameters"`: list of partitions of `n`
- `"SizesConjugacyClasses"`: conjugacy class sizes
- `"CharacterTable"`: character table matrix

### Double Cosets — Mathematica (`dc_n{n}_p{p}_q{q}.txt`)

Mathematica association with keys:

- `"Representatives"`: list of double coset representatives
- `"Sizes"`: list of double coset sizes

### Woven Contractions — JSON (`wc_op_{ops}_p{perm}_m{mass}_Lambda{Lambda}.json`)

JSON produced by Mathematica's `ExportWovenContractions`. Permutations are 0-indexed; coefficients are polynomial coefficient lists in $d$ as rational `[numerator, denominator]` pairs.

### Coset Representatives — JSON (`coset_reps.json`)

JSON array, one entry per `(τ, n, m)` triple, each with `tau_0indexed`, `n`, `m`, and two blocks (`"left"`, `"right"`) containing `h_order`, `num_reps`, and `reps_0indexed`.

## Raw Data

The `raw/` directory contains reference data used for validation and bootstrapping, not generated by the pipeline.

### `baseline.txt`

Ground-state energy of the one-matrix model in the mean-field (large-$N$) regime, sampled at discrete coupling values. In this setting, there is an analytic formula, derived in:

> Brézin, E., Itzykson, C., Parisi, G. et al. *Planar diagrams.* Commun. Math. Phys. **59**, 35–51 (1978). <https://doi.org/10.1007/BF01614153>

Format: Mathematica-style list of `{coupling, energy}` pairs.

### `baseline_numpy.txt`

Same data as `baseline.txt` but in plain CSV format (no outer braces), suitable for direct loading with NumPy.

### `precomputed_K4_Lambda10.json`

Precomputed contraction coefficients `ComputeContractions[R, S, {pairs}]` for $K = 4$ and cutoff $\Lambda = 10$, suitable for the `PrecomputedContractions` option in Mathematica. Coefficients are polynomials in $d$ stored as `[numerator, denominator]` rational pairs, from degree 0 upward.

---

## Folder Structure

```
data/
├── README.md
├── generate_data.sh           # Master pipeline (all 6 steps)
├── clean_data.sh              # Delete all generated files
├── gap_scripts/
│   ├── main.g
│   ├── generate_ssct.g
│   ├── generate_ct.g
│   ├── generate_dc.g
│   ├── generate_coset_reps.g
│   ├── coset_reps.g
│   └── demo_*.g
├── processed/
│   ├── character_tables/      # ct_n.json (Python)
│   ├── conjugacy_classes/     # ssct_n.txt (Mathematica, test-only)
│   ├── double_cosets/         # dc_n{n}_p{p}_q{q}.txt (Mathematica)
│   ├── woven_contractions/    # wc_op_{ops}_p{perm}_m{mass}_Lambda{Lambda}.json
│   ├── coset_reps/            # coset_reps.json + _coset_input_work.g
│   ├── probability_stores/    # .npz files (Python-generated)
│   ├── hamiltonians/          # ham_*.npz files
│   ├── observables/           # Observable .npz files
│   └── test-data/             # flat test outputs (--test mode)
└── raw/
    ├── baseline.txt                   # Mean-field ground-state energy (BIPZ 1978)
    ├── baseline_numpy.txt             # Same data in plain CSV format
    └── precomputed_K4_Lambda10.json   # Precomputed contractions K=4, Λ=10
```
