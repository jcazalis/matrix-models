#!/bin/bash

# ============================================================================
# generate_data.sh — Master data generation pipeline
#
# Orchestrates all data generation steps in order:
#   1. GAP  — Symmetric character tables + conjugacy classes (Mathematica format)
#   2. GAP  — Python-format character tables
#   3. GAP  — Double cosets
#   4. WLS  — Woven contractions (wolframscript / Mathematica)
#   5. PY   — Coset input preparation (Python / uv)
#   6. GAP  — Coset representatives
#
# Usage:
#   ./generate_data.sh [--test] [--lambda LAMBDA] [--ops OPS_SPEC,...] [--mass MASS]
#
# Options:
#   --test            Small-scale data for testing.
#                     GAP files (ct, dc, ssct) still go to the standard
#                     processed/ subdirectories.
#                     Woven contractions and coset reps go to
#                     processed/test-data/ (flat, no subdirectories).
#   --lambda LAMBDA   Cutoff parameter Lambda.
#                     Default: 14 (production), 6 (test).
#   --ops OPS1:PERM1,OPS2:PERM2,...
#                     Comma-separated operator specifications.
#                     Each spec is OPS:PERM where OPS is a string of X/P
#                     operators and PERM is a 1-indexed permutation image list.
#                     Default: XXXX:2341 (production), XX:21,XXXX:2341 (test).
#                     Double cosets use K_MAX = max operator string length.
#                     Woven contractions are generated once per spec.
#   --mass MASS       Mass parameter passed to Mathematica woven-contraction step.
#                     Default: 1/2.
#
# Derived automatically (not accepted as arguments):
#   nmax_ct   = Lambda
#   nmax_dc   = Lambda + K_max  (K_max = max length of operator strings)
#   Double cosets are generated separately for even and odd operator degrees.
#   For each parity present in the ops list:
#     odd_difference = false (even K), true (odd K)
#     max_offset = K/2 (even K), (K-1)/2 (odd K)
#   nmax_ssct = 0 in production (step skipped); 6 in test.
#
# Backward compatibility: the original generate_data.sh (GAP-only, positional
# args) is still available for the GAP steps in isolation.
# ============================================================================

set -e

# ── Parse arguments ──────────────────────────────────────────────────────────
TEST_MODE=false
LAMBDA_ARG=""
OPS_ARG=""
MASS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)    TEST_MODE=true;  shift ;;
        --lambda)  LAMBDA_ARG="$2"; shift 2 ;;
        --ops)     OPS_ARG="$2";    shift 2 ;;
        --mass)    MASS_ARG="$2";   shift 2 ;;
        *)         echo "Error: Unknown argument '$1'." >&2; exit 1 ;;
    esac
done

# ── Apply mode-dependent defaults ────────────────────────────────────────────
if $TEST_MODE; then
    LAMBDA=${LAMBDA_ARG:-6}
    OPS_CSV=${OPS_ARG:-"XX:21,XXXX:2341"}
    MASS=${MASS_ARG:-"1"}
    NMAX_SSCT=6
else
    LAMBDA=${LAMBDA_ARG:-14}
    OPS_CSV=${OPS_ARG:-"XXXX:2341"}
    MASS=${MASS_ARG:-"1/2"}
    NMAX_SSCT=0
fi

# Convert comma-separated ops specs to arrays
# Each spec is "OPS:PERM" e.g. "XXXX:2341" or "XPXP:2341"
IFS=',' read -ra OPS_LIST_RAW <<< "$OPS_CSV"
OPS_LIST=()    # operator strings
PERM_LIST=()   # permutation strings
for spec in "${OPS_LIST_RAW[@]}"; do
    spec="$(echo "$spec" | tr -d ' ')"
    ops_part="${spec%%:*}"
    perm_part="${spec#*:}"
    OPS_LIST+=( "$ops_part" )
    PERM_LIST+=( "$perm_part" )
done

# K_MAX = max operator string length
K_MAX=0
for ops in "${OPS_LIST[@]}"; do
    k=${#ops}
    if [ "$k" -gt "$K_MAX" ]; then K_MAX=$k; fi
done

# Guard: permutation format encodes digits 1–9 only (single characters),
# so operator strings longer than 9 cannot be represented.  Raise early.
if [ "$K_MAX" -gt 9 ]; then
    echo "Error: operator strings longer than 9 are not supported." >&2
    echo "  The permutation encoding uses single decimal digits (1–9)," >&2
    echo "  so at most 10 operators can be specified per trace." >&2
    echo "  Longest operator string found: $K_MAX characters." >&2
    echo "  (Support for 10+ operators will be added in a future version.)" >&2
    exit 1
fi

# Double-coset requirements are parity-sensitive:
#   even K -> even (p-q), odd K -> odd (p-q)
EVEN_K_MAX=0
ODD_K_MAX=0
for ops in "${OPS_LIST[@]}"; do
    k=${#ops}
    if (( k % 2 == 0 )); then
        if [ "$k" -gt "$EVEN_K_MAX" ]; then EVEN_K_MAX=$k; fi
    else
        if [ "$k" -gt "$ODD_K_MAX" ]; then ODD_K_MAX=$k; fi
    fi
done

RUN_DC_EVEN=false
RUN_DC_ODD=false
EVEN_MAX_OFFSET=0
ODD_MAX_OFFSET=0

if [ "$EVEN_K_MAX" -gt 0 ]; then
    RUN_DC_EVEN=true
    EVEN_MAX_OFFSET=$(( EVEN_K_MAX / 2 ))
fi

if [ "$ODD_K_MAX" -gt 0 ]; then
    RUN_DC_ODD=true
    ODD_MAX_OFFSET=$(( (ODD_K_MAX - 1) / 2 ))
fi

# Derived nmax values
NMAX_CT=$LAMBDA
NMAX_DC=$(( K_MAX + LAMBDA ))

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GAP_SCRIPTS_DIR="$SCRIPT_DIR/gap_scripts"
MATHEMATICA_DIR="$( dirname "$SCRIPT_DIR" )/mathematica"
PYTHON_DIR="$( dirname "$SCRIPT_DIR" )/python"
BASE_PROCESSED="$SCRIPT_DIR/processed"

# GAP steps always write to the standard processed/ subdirectories regardless of mode.
# Only the woven contractions (Mathematica) and coset reps output differ in test mode.
if $TEST_MODE; then
    WOVEN_DIR="$BASE_PROCESSED/test-data"
    COSET_JSON="$BASE_PROCESSED/test-data/coset_reps.json"
    COSET_INPUT_FILE="$BASE_PROCESSED/test-data/_coset_input_work.g"
else
    WOVEN_DIR="$BASE_PROCESSED/woven_contractions"
    COSET_JSON="$BASE_PROCESSED/coset_reps/coset_reps.json"
    COSET_INPUT_FILE="$BASE_PROCESSED/coset_reps/_coset_input_work.g"
fi

# ── Detect GAP ───────────────────────────────────────────────────────────────
if command -v gap-system &> /dev/null; then
    GAP_CMD="gap-system"
    echo "Using gap-system command"
elif command -v gap &> /dev/null; then
    if gap --help 2>&1 | grep -q "Groups, Algorithms"; then
        GAP_CMD="gap"
    else
        echo "Error: 'gap' found but is not the GAP system (likely git-apply)." >&2
        exit 1
    fi
else
    echo "Error: GAP system not found. Please install GAP and ensure it is in PATH." >&2
    exit 1
fi

# ── Validate NMAX_SSCT ───────────────────────────────────────────────────────
if [ "$NMAX_SSCT" -gt 10 ]; then
    echo "============================================================================" >&2
    echo "ERROR: nmax_ssct=$NMAX_SSCT is too large (max allowed: 10)." >&2
    echo "Generating ALL permutations per conjugacy class is intractable for n > 10." >&2
    echo "============================================================================" >&2
    exit 1
fi

# ── Create output directories ────────────────────────────────────────────────
# Standard GAP output dirs (always needed)
mkdir -p "$BASE_PROCESSED/character_tables"
mkdir -p "$BASE_PROCESSED/conjugacy_classes"
mkdir -p "$BASE_PROCESSED/double_cosets"
mkdir -p "$BASE_PROCESSED/coset_reps"
# Woven / coset-reps output dir (test-data/ or woven_contractions/)
mkdir -p "$WOVEN_DIR"

# ── Print banner ─────────────────────────────────────────────────────────────
echo "============================================================================"
if $TEST_MODE; then
    echo "generate_all.sh — TEST MODE"
else
    echo "generate_all.sh — PRODUCTION MODE"
fi
echo "============================================================================"
echo "Parameters:"
echo "  Lambda      = $LAMBDA"
echo "  Ops list    = ${OPS_CSV}"
echo "  K max       = $K_MAX"
if $RUN_DC_EVEN && $RUN_DC_ODD; then
    echo "  DC parity   = even and odd"
    echo "  DC offsets  = even:$EVEN_MAX_OFFSET odd:$ODD_MAX_OFFSET"
elif $RUN_DC_EVEN; then
    echo "  DC parity   = even"
    echo "  DC offset   = $EVEN_MAX_OFFSET"
elif $RUN_DC_ODD; then
    echo "  DC parity   = odd"
    echo "  DC offset   = $ODD_MAX_OFFSET"
else
    echo "  DC parity   = none"
fi
echo "  mass        = $MASS"
echo "  nmax_ssct   = $NMAX_SSCT"
echo "  nmax_ct     = $NMAX_CT"
echo "  nmax_dc     = $NMAX_DC"
echo "  GAP output  = $BASE_PROCESSED/{character_tables,conjugacy_classes,double_cosets,coset_reps}"
echo "  Woven dir   = $WOVEN_DIR"
echo "  Coset JSON  = $COSET_JSON"
echo ""

# Change to gap_scripts for all GAP invocations (relative Read() paths)
cd "$GAP_SCRIPTS_DIR"

# ── Step 1: SSCT (GAP) ───────────────────────────────────────────────────────
echo "----------------------------------------------------------------------------"
echo "Step 1/6: Symmetric character tables + conjugacy classes (Mathematica format)"
echo "----------------------------------------------------------------------------"
if [ "$NMAX_SSCT" -eq 0 ]; then
    echo "Skipping (nmax_ssct=0 in production mode)."
else
    $GAP_CMD -c "nmax:=$NMAX_SSCT; Read(\"generate_ssct.g\"); quit;" -b
fi
echo ""

# ── Step 2: CT (GAP) ────────────────────────────────────────────────────────
echo "----------------------------------------------------------------------------"
echo "Step 2/6: Python-format character tables  (nmax_ct=$NMAX_CT)"
echo "----------------------------------------------------------------------------"
$GAP_CMD -c "nmax:=$NMAX_CT; Read(\"generate_ct.g\"); quit;" -b
echo ""

# ── Step 3: DC (GAP) ────────────────────────────────────────────────────────
echo "----------------------------------------------------------------------------"
echo "Step 3/6: Double cosets  (nmax_dc=$NMAX_DC)"
echo "----------------------------------------------------------------------------"
if $RUN_DC_EVEN; then
    echo "  → even p-q differences (max_offset=$EVEN_MAX_OFFSET, max K=$EVEN_K_MAX)"
    $GAP_CMD -c "nmax:=$NMAX_DC; max_offset:=$EVEN_MAX_OFFSET; odd_difference:=false; Read(\"generate_dc.g\"); quit;" -b
fi

if $RUN_DC_ODD; then
    echo "  → odd p-q differences (max_offset=$ODD_MAX_OFFSET, max K=$ODD_K_MAX)"
    $GAP_CMD -c "nmax:=$NMAX_DC; max_offset:=$ODD_MAX_OFFSET; odd_difference:=true; Read(\"generate_dc.g\"); quit;" -b
fi
echo ""

# ── Step 4: Woven contractions (Mathematica) ─────────────────────────────────
echo "----------------------------------------------------------------------------"
echo "Step 4/6: Woven contractions (Mathematica, ops=${OPS_CSV}, Lambda=$LAMBDA, mass=$MASS)"
echo "----------------------------------------------------------------------------"
cd "$MATHEMATICA_DIR"
for i in "${!OPS_LIST[@]}"; do
    ops="${OPS_LIST[$i]}"
    perm="${PERM_LIST[$i]}"
    echo "  → ops=$ops perm=$perm, Lambda=$LAMBDA, mass=$MASS  →  wc_op_${ops}_p${perm}_m${MASS}_Lambda${LAMBDA}.json"
    wolframscript -script generate_wc.wls "$ops" "$perm" "$LAMBDA" "$MASS" "$WOVEN_DIR" 
done
cd "$GAP_SCRIPTS_DIR"
echo ""

# ── Step 5: Coset input list (Python) ────────────────────────────────────────
echo "----------------------------------------------------------------------------"
echo "Step 5/6: Coset input preparation (Python)"
echo "----------------------------------------------------------------------------"
cd "$PYTHON_DIR"
uv run python scripts/prepare_coset_input.py \
    --woven-dir "$WOVEN_DIR" \
    --output "$COSET_INPUT_FILE"
cd "$GAP_SCRIPTS_DIR"
echo ""

# ── Step 6: Coset representatives (GAP) ──────────────────────────────────────
echo "----------------------------------------------------------------------------"
echo "Step 6/6: Coset representatives"
echo "----------------------------------------------------------------------------"
if [ -f "$COSET_INPUT_FILE" ]; then
    $GAP_CMD -c "Read(\"$COSET_INPUT_FILE\"); output_file:=\"$COSET_JSON\"; Read(\"generate_coset_reps.g\"); quit;" -b
    echo "Written: $COSET_JSON"
else
    echo "Warning: $COSET_INPUT_FILE not found. Skipping coset reps generation." >&2
fi
echo ""

echo "============================================================================"
echo "Pipeline complete."
echo "  GAP data : $BASE_PROCESSED/{character_tables,conjugacy_classes,double_cosets,coset_reps}"
echo "  Woven    : $WOVEN_DIR"
echo "  Cosets   : $COSET_JSON"
echo "============================================================================"
