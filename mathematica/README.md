# Mathematica — Woven Contractions Generator

The main purpose of this directory is to compute **woven contraction** contributions for the matrix model, using the script `generate_wc.wls`. These files are consumed by the Python pipeline.

## Requirements

- Wolfram Language 14.0.0 Engine must be installed and `wolframscript` available in your PATH. See [wolfram.com/engine](https://www.wolfram.com/engine/) for installation instructions. The script is known to be incompatible with Wolfram Language 14.3.* Engine so be sure to use 14.0.0.

## Directory Structure

```
mathematica/
├── generate_wc.wls              # Main CLI script — generates woven contractions
├── packages/                    # Wolfram Language packages
│   ├── LadderAlgebra.wl
│   ├── MatrixModels.wl
│   └── OperatorTesting.wl
├── notebooks/                   # Demo notebooks
│   ├── 01_LadderAlgebra_OperatorTesting_demo.nb
│   └── 02_MatrixModels_demo.nb
├── tests/                       # Package test notebooks
│   ├── LadderAlgebra_test.nb
│   ├── MatrixModels_test.nb
│   └── OperatorTesting_test.nb
└── README.md
```

## Packages

- **`LadderAlgebra.wl`** — Implements a non-commutative algebraic calculus for ladder operators $a_{ij}$, $a^\dagger_{ij}$ indexed by two matrix indices. Provides Fock space machinery (ket/bra states, vacuum, operator actions) and physical operators (number, quadrature, matrix $X$, $P$, $A$, $A^\dagger$).

- **`MatrixModels.wl`** — Tools for U(N) matrix models. Decomposes trace operators (for instance $\mathrm{tr}(X^K)$ or $`\mathrm{tr}(XPX) \mathrm{tr}(P)`$) into anti-normal ordered generalised traces, computes tensor contractions and overlap integrals, and implements the full woven contraction pipeline. Depends on `LadderAlgebra` and `OperatorTesting`.

- **`OperatorTesting.wl`** — Automated testing utilities for verifying operator identities. Supports both explicit index notation and implicit Einstein summation notation, using `LadderAlgebra` for simplification.

## Generating Woven Contractions

### Step 1 — Generate double coset data from GAP (prerequisite)

`generate_wc.wls` reads double coset representative files from `../data/processed/double_cosets/`. These are generated automatically by the master pipeline script `../data/generate_data.sh` (steps 1–3). If running `generate_wc.wls` standalone, ensure the double coset files exist first.

See [`../data/README.md`](../data/README.md) for details.

### Step 2 — Run the script

```bash
# Use defaults (ops=XXXX, perm=2341, Lambda=14, mass=1/2,
# output to ../data/processed/woven_contractions/)
wolframscript -script generate_wc.wls

# Override the operator string and trace permutation
wolframscript -script generate_wc.wls XX 21

# Override ops, perm, and Lambda
wolframscript -script generate_wc.wls XPXP 2143 10

# Override ops, perm, Lambda, and mass
wolframscript -script generate_wc.wls XXXX 2143 10 1

# Override all five arguments: ops, perm, Lambda, mass, outputDir
wolframscript -script generate_wc.wls XXXX 2143 10 1 /path/to/output
```

| Argument | Default | Description |
|---|---|---|
| `ops` | `"XXXX"` | Ordered operator string, for example `XX`, `XXXX`, or `XPXP` |
| `perm` | `"2341"` | Trace permutation written as a string of 1-indexed digits |
| `Lambda` | `14` | Maximum excitation level (cutoff) |
| `mass` | `1/2` | Mass parameter used in the quadrature operators |
| `outputDir` | `../data/processed/woven_contractions/` | Output directory for JSON files |

**Output filename**: `wc_op_{ops}_p{perm}_m{mass}_Lambda{Lambda}.json` (e.g., `wc_op_XXXX_p2341_m1_2_Lambda14.json`).
