# Matrix Models in the singlet state basis

[arXiv:2607.13725](https://arxiv.org/abs/2607.13725)

## Project Overview

This project implements tools for computing the Hamiltonian of $U(N)$ one-matrix models restricted to the subspace of singlet states up to a cutoff $\Lambda$. It contains:

- **Symmetric group data generation** (GAP) providing character tables, conjugacy classes, double cosets, and coset representatives
- **Symbolic computation** (Mathematica/Wolfram Language) for decomposing interactions into woven contractions and computing overlap integrals
- **Contraction assembly** (Python) implementing two computation paths — brute-force enumeration and an efficient coset-reduction + character-theory path — to assemble the full Hamiltonian
- **Observable framework** (Python) for building, saving, and evaluating Hamiltonian and trace observables in both dense and sparse representations

## Project Structure

```
matrix-models/
├── data/                          # Shared GAP-generated data and pipeline scripts
│   ├── gap_scripts/               # GAP scripts for symmetric group data
│   ├── generate_data.sh           # Master 6-step data pipeline
│   ├── clean_data.sh              # Delete all generated data
│   ├── raw/                       # Reference data
│   ├── processed/                 # Generated outputs
│   │   ├── character_tables/      # ct_n.json (Python/JSON)
│   │   ├── conjugacy_classes/     # ssct_n.txt (Mathematica, test-only)
│   │   ├── double_cosets/         # dc_n{p}_p{pp}_q{qq}.txt (Mathematica)
│   │   ├── woven_contractions/    # wc_op_*.json
│   │   ├── coset_reps/            # coset_reps.json
│   │   ├── probability_stores/    # .npz files (Python-generated)
│   │   ├── hamiltonians/          # Hamiltonian .npz files
│   │   ├── observables/           # Observable .npz files
│   │   └── test-data/             # flat test outputs (--test mode)
│   └── README.md
│
├── mathematica/                   # Symbolic calculations and testing
│   ├── generate_wc.wls            # Main script for generating woven contraction data
│   ├── packages/                  # Wolfram Language packages
│   │   ├── LadderAlgebra.wl       # Non-commutative ladder operator algebra
│   │   ├── MatrixModels.wl        # U(N) matrix model toolkit
│   │   └── OperatorTesting.wl     # Operator identity testing utilities
│   ├── notebooks/                 # Demo notebooks
│   ├── tests/                     # Test notebooks
│   └── README.md
│
└── python/                        # Computation and analysis
    ├── sym_contractions/          # Core library
    ├── scripts/                   # CLI scripts (benchmarks, data preparation)
    ├── tests/                     # Unit tests (pytest)
    ├── exploration/               # Jupyter notebooks for analysis
    ├── notebooks/                 # Demo and analysis notebooks
    ├── pyproject.toml             # Python dependencies
    └── README.md
```

## Theoretical Background

In $\mathrm{U}(N)$ one-matrix models, the degrees of freedom are arranged in a single $N\times N$ Hermitian matrix transforming in the adjoint representation. More precisely, the creation/annihilation operators $A$ and $A^\dagger$ satisfy

$$A \longrightarrow U A U^\dagger \quad \text{and} \quad A^\dagger \longrightarrow U A^\dagger U^\dagger \, .$$

The quadrature matrix operators are

$$ X = \frac{1}{\sqrt{2m}}(A + A^{\dagger}) \quad \text{and} \quad P = i \sqrt{\frac{m}{2}}(A^{\dagger} - A) \, ,$$

where $m$ is a mass parameter. The singlet states are those invariant under the adjoint action of $\mathrm{U}(N)$, and can be constructed by acting with traces of products of creation operators on the vacuum state.

This project aims to efficiently compute, up to an excitation cutoff $\Lambda$, the Hamiltonian matrix elements in the singlet state basis of one-matrix models where the interaction term is a polynomial in the quadrature operators, such as

$$ H  = \frac{1}{2} \mathrm{tr}( P^2 + m^2 X^2) + \sum_{i = 3}^{K} g(i;N, m) \mathrm{tr}_{\sigma_i}(B_{i,1}\otimes \cdots \otimes B_{i,\ell_i}) \, , $$

where $g(i;N,m)$ are coupling constants that may depend on $N$ and $m$ and where $B_{i,j}$ are either $X$ or $P$. The $`\mathrm{tr}_{\sigma_i}`$ are generalized traces defined by permutations $`\sigma_i \in S_{\ell_i}`$ that specify the contraction pattern of the indices of the $B_{i,j}$'s.

A simple non-trivial example is the quartic model in the mean-field regime, which corresponds to $K=1$, $\ell_1=4$, $B_{i,j}=X$, and $\sigma = (1\,2\,3\,4)$ (in cycle notation):
$$H = \frac{1}{2} \mathrm{tr}( P^2 + m^2 X^2) + \frac{g^2}{2N} \mathrm{tr}(X^4) \, .$$

## Quick Start

### 0. Requirements

- **GAP 4.14+** with `gap` or `gap-system` CLI — see [gap-system.org/install](https://www.gap-system.org/install/)
- **Wolfram Engine 14.0** with `wolframscript` CLI — see [wolfram.com/engine](https://www.wolfram.com/engine/)
- **Python 3.14+**
- **uv 10.6+** (Python package manager)

### 1. Setup Python Environment

```bash
cd python
uv sync
```

### 2. Generate Data (One-time setup)

All data (character tables, double cosets, woven contractions, coset representatives) is generated by a single master script.

```bash
cd data

# 1. Test mode — small scale (Lambda=6, ops=XX:21,XXXX:2341, mass=1), output to processed/test-data/
./generate_data.sh --test

# 2. Full production run (Lambda=14, ops=XXXX:2341, mass=1/2)
./generate_data.sh

# 3. Custom run (example)
./generate_data.sh --lambda 10 --ops XXXX:2143,XPXP:2341 --mass 1/2
```

See [`data/README.md`](data/README.md) for all options (`--lambda`, `--ops`, `--mass`) and details on each pipeline step.

### 3. Run Tests

```bash
cd python
uv run pytest
```

Some tests skip when GAP-generated data is missing. Use `--test` mode in the data pipeline to generate the required test data.

### 4. Build and Analyze Hamiltonians

Observables and Hamiltonians are built via the `sym_contractions` Python library. See the notebooks in `python/notebooks/` for worked examples.

## Documentation

- [`python/README.md`](python/README.md) — Python library, tests, and benchmarks
- [`mathematica/README.md`](mathematica/README.md) — Mathematica packages and notebook docs
- [`data/README.md`](data/README.md) — Data generation pipeline and formats

## Copyright License

Code and documentation copyright 2011-2026 [Jean Cazalis](https://github.com/jcazalis). Code released under the [MIT LICENSE](LICENSE).

## Citation

If you use this code in your research, please cite the following paper: [arXiv:2607.13725](https://arxiv.org/abs/2607.13725)

