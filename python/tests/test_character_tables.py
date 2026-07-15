"""Tests for character table loading and utilities."""

import json

import numpy as np
import pytest

from sym_contractions.character_tables import (
    find_cycle_type_index,
    get_class_weights,
    load_character_table,
    load_character_tables_range,
    partition_to_cycle_type_key,
)


@pytest.fixture
def sample_character_table_s5():
    """Sample character table data for S_5 in GAP format (lexicographic order)."""
    return {
        "n": 5,
        "CharacterParameters": [
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1],
            [2, 2, 1],
            [3, 1, 1],
            [3, 2],
            [4, 1],
            [5],
        ],
        "SizesConjugacyClasses": [1, 10, 15, 20, 20, 30, 24],
        "CharacterTable": [
            [1, -1, 1, 1, -1, -1, 1],
            [4, -2, 0, 1, 1, 0, -1],
            [5, -1, 1, -1, -1, 1, 0],
            [6, 0, -2, 0, 0, 0, 1],
            [5, 1, 1, -1, 1, -1, 0],
            [4, 2, 0, 1, -1, 0, -1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
    }


@pytest.fixture
def sample_json_file(sample_character_table_s5, tmp_path):
    """Create a temporary JSON file with sample data."""
    filepath = tmp_path / "ssct_5.json"
    with open(filepath, "w") as f:
        json.dump(sample_character_table_s5, f)
    return filepath


def test_load_character_table(sample_json_file):
    """Test loading a character table from JSON with reordering."""
    table = load_character_table(sample_json_file)

    assert table["n"] == 5
    assert len(table["cycle_types"]) == 7
    assert table["class_sizes"].shape == (7,)
    assert table["characters"].shape == (7, 7)
    # After reordering, largest partition comes first (reverse lexicographic)
    assert table["cycle_types"][0] == [5]
    assert table["cycle_types"][-1] == [1, 1, 1, 1, 1]


def test_loaded_data_types(sample_json_file):
    """Test that loaded arrays have correct dtypes."""
    table = load_character_table(sample_json_file)

    assert table["class_sizes"].dtype == np.int64
    assert table["characters"].dtype == np.int64


def test_class_sizes_sum_to_factorial(sample_json_file):
    """Test that conjugacy class sizes sum to n!."""
    import math

    table = load_character_table(sample_json_file)
    n = table["n"]

    total = np.sum(table["class_sizes"])
    expected = math.factorial(n)

    assert int(total) == expected


def test_get_class_weights(sample_json_file):
    """Test class weight computation."""
    table = load_character_table(sample_json_file)
    weights = get_class_weights(table["cycle_types"], table["n"])

    # Weights should sum to 1
    assert np.allclose(np.sum(weights), 1.0)

    # All weights should be positive
    assert np.all(weights > 0)


def test_class_sizes_reordered(sample_json_file):
    """Test that class sizes match the reordered cycle types."""
    table = load_character_table(sample_json_file)

    # After reordering: cycle types are [5], [4,1], [3,2], [3,1,1], [2,2,1], [2,1,1,1], [1,1,1,1,1]
    # Original GAP order sizes: [1, 10, 15, 20, 20, 30, 24]
    # Reversed sizes: [24, 30, 20, 20, 15, 10, 1]
    expected = np.array([24, 30, 20, 20, 15, 10, 1], dtype=np.int32)
    assert np.array_equal(table["class_sizes"], expected)


def test_partition_to_cycle_type_key():
    """Test partition to tuple conversion."""
    partition = [3, 2, 2]
    key = partition_to_cycle_type_key(partition)

    assert key == (3, 2, 2)
    assert isinstance(key, tuple)
    # Should be hashable
    assert hash(key) is not None


def test_find_cycle_type_index(sample_json_file):
    """Test finding cycle type index."""
    table = load_character_table(sample_json_file)

    # After reordering: [5], [4,1], [3,2], [3,1,1], [2,2,1], [2,1,1,1], [1,1,1,1,1]
    idx = find_cycle_type_index(table["cycle_types"], [3, 2])
    assert idx == 2

    idx = find_cycle_type_index(table["cycle_types"], [5])
    assert idx == 0

    idx = find_cycle_type_index(table["cycle_types"], [1, 1, 1, 1, 1])
    assert idx == 6

    # Test not found
    idx = find_cycle_type_index(table["cycle_types"], [2, 2, 2])
    assert idx is None


def test_load_character_tables_range(tmp_path):
    """Test loading multiple character tables."""
    # Create test files for n=3, 4, 5
    for n in [3, 4, 5]:
        data = {
            "n": n,
            "CharacterParameters": [[n]],  # Simplified
            "SizesConjugacyClasses": [1],
            "CharacterTable": [[1]],
        }
        filepath = tmp_path / f"ssct_{n}.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

    # Load range
    tables = load_character_tables_range(tmp_path, 3, 5)

    assert len(tables) == 3
    assert 3 in tables
    assert 4 in tables
    assert 5 in tables
    assert tables[3]["n"] == 3
    assert tables[5]["n"] == 5


def test_load_missing_files_warns(tmp_path, capsys):
    """Test that missing files produce warnings."""
    tables = load_character_tables_range(tmp_path, 1, 3)

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "not found" in captured.out
    assert len(tables) == 0


def test_orthogonality_relation(sample_json_file):
    """Test character orthogonality relations (basic check)."""
    import math

    table = load_character_table(sample_json_file)
    characters = table["characters"]
    class_sizes = table["class_sizes"]
    n = table["n"]

    # First orthogonality relation: Σ_R' |C_R'| χ_i(R') χ_j(R')* = n! δ_ij
    # For symmetric groups, characters are real, so χ* = χ
    factorial_n = math.factorial(n)

    for i in range(len(characters)):
        for j in range(len(characters)):
            inner_product = np.sum(class_sizes * characters[i] * characters[j])

            if i == j:
                assert np.isclose(inner_product, factorial_n)
            else:
                assert np.isclose(inner_product, 0)


def test_character_table_structure(sample_json_file):
    """Test basic properties of character table."""
    table = load_character_table(sample_json_file)

    # Should be square
    n_classes = len(table["cycle_types"])
    assert table["characters"].shape == (n_classes, n_classes)

    # After reordering, trivial representation (all ones) is the first row
    trivial = table["characters"][0]
    assert np.all(trivial == 1)

    # After reordering, sign representation is the last row
    sign = table["characters"][-1]
    # The identity class [1,1,1,1,1] is now at index 6 (last)
    # sign rep evaluated on identity is 1
    assert sign[-1] == 1
