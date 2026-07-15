#!/bin/bash

# ============================================================================
# clean_data.sh
# 
# Script to delete all generated data files while preserving the folder
# structure. Useful for development and testing.
#
# Usage:
#   ./clean_data.sh
# ============================================================================

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "============================================================================"
echo "Cleaning generated data"
echo "============================================================================"

# Clean character tables directory
if [ -d "$SCRIPT_DIR/processed/character_tables" ]; then
    echo "Removing character tables..."
    rm -f "$SCRIPT_DIR/processed/character_tables/"*.json
    FILE_COUNT=$(find "$SCRIPT_DIR/processed/character_tables" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  - Removed files from processed/character_tables/ (remaining: $FILE_COUNT)"
fi

# Clean conjugacy classes directory
if [ -d "$SCRIPT_DIR/processed/conjugacy_classes" ]; then
    echo "Removing conjugacy classes..."
    rm -f "$SCRIPT_DIR/processed/conjugacy_classes/"*.txt
    FILE_COUNT=$(find "$SCRIPT_DIR/processed/conjugacy_classes" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  - Removed files from processed/conjugacy_classes/ (remaining: $FILE_COUNT)"
fi

# Clean double cosets directory
if [ -d "$SCRIPT_DIR/processed/double_cosets" ]; then
    echo "Removing double cosets..."
    rm -f "$SCRIPT_DIR/processed/double_cosets/"*.txt
    FILE_COUNT=$(find "$SCRIPT_DIR/processed/double_cosets" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  - Removed files from processed/double_cosets/ (remaining: $FILE_COUNT)"
fi

# Clean coset reps directory
if [ -d "$SCRIPT_DIR/processed/coset_reps" ]; then
    echo "Removing coset reps..."
    find "$SCRIPT_DIR/processed/coset_reps" -type f -delete
    FILE_COUNT=$(find "$SCRIPT_DIR/processed/coset_reps" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  - Removed files from processed/coset_reps/ (remaining: $FILE_COUNT)"
fi

# Clean probability stores directory
if [ -d "$SCRIPT_DIR/processed/probability_stores" ]; then
    echo "Removing probability stores..."
    find "$SCRIPT_DIR/processed/probability_stores" -type f -delete
    FILE_COUNT=$(find "$SCRIPT_DIR/processed/probability_stores" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  - Removed files from processed/probability_stores/ (remaining: $FILE_COUNT)"
fi

# Clean woven contractions directory
if [ -d "$SCRIPT_DIR/processed/woven_contractions" ]; then
    echo "Removing woven contractions..."
    find "$SCRIPT_DIR/processed/woven_contractions" -type f -delete
    FILE_COUNT=$(find "$SCRIPT_DIR/processed/woven_contractions" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  - Removed files from processed/woven_contractions/ (remaining: $FILE_COUNT)"
fi

# Clean test-data directory
if [ -d "$SCRIPT_DIR/processed/test-data" ]; then
    echo "Removing test data..."
    find "$SCRIPT_DIR/processed/test-data" -type f -delete
    FILE_COUNT=$(find "$SCRIPT_DIR/processed/test-data" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  - Removed files from processed/test-data/ (remaining: $FILE_COUNT)"
fi

echo ""
echo "============================================================================"
echo "Data cleanup complete!"
echo "Folder structure preserved."
echo "============================================================================"
