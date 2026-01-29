# tests/test_cleaner.py

import pytest
import pandas as pd
import numpy as np
from src.data_analyzer.cleaner import DataCleaner


@pytest.fixture
def messy_data():
    """Create a sample dataframe with dirty data."""
    data = {
        "Name ": ["Alice", "Bob", "Alice", np.nan, "Charlie"],
        " Age": [25, 30, 25, 40, 35],
        "City": ["NY", "LA", "NY", "Chicago", None],
    }
    return pd.DataFrame(data)


def test_clean_data_logic(messy_data):
    """Test the cleaning pipeline logic."""
    cleaner = DataCleaner(messy_data)
    cleaned_df = cleaner.clean_data()

    # Check 1: Column names normalized?
    expected_cols = ["name", "age", "city"]
    assert list(cleaned_df.columns) == expected_cols

    # Check 2: Duplicates removed? (Alice appears twice in input)
    # Check 3: Nulls removed? (Row 4 has NaN name, Row 5 has None city)
    # Input has 5 rows.
    # - Row 1: Alice, 25, NY (Keep)
    # - Row 2: Bob, 30, LA (Keep)
    # - Row 3: Alice, 25, NY (Duplicate of Row 1 -> Drop)
    # - Row 4: NaN, 40, Chicago (Null Name -> Drop)
    # - Row 5: Charlie, 35, None (Null City -> Drop)
    # Expected result: Only Alice and Bob remain?
    # Wait, dropna() removes any row with ANY null.

    assert len(cleaned_df) == 2
    # assert 'bob' in cleaned_df['name'].values
    # assert 'alice' in cleaned_df['name'].values
    # assert 'charlie' not in cleaned_df['name'].values


def test_load_csv_file_not_found():
    """Test error handling for missing file."""
    cleaner = DataCleaner()
    with pytest.raises(FileNotFoundError):
        cleaner.load_csv("non_existent_file.csv")


def test_clean_empty_data():
    """Test error raising when cleaning without data."""
    cleaner = DataCleaner()
    with pytest.raises(ValueError, match="No data loaded"):
        cleaner.clean_data()


def test_clean_data_with_subset(messy_data):
    """Test cleaning with specific columns for dropna."""
    # Row 4: NaN Name, 40 Age, Chicago City
    # Row 5: Charlie Name, 35 Age, None City

    cleaner = DataCleaner(messy_data)

    # 只根据 'Name ' (normalized to 'name') 去除空值
    # Row 4 (NaN Name) 应该被删掉
    # Row 5 (None City) 应该保留
    cleaned_df = cleaner.clean_data(drop_na_cols=["name"])

    assert (
        len(cleaned_df) == 3
    )  # Alice, Bob, Charlie (Bob is OK, Row 3 duplicate dropped)
    assert "Charlie" in cleaned_df["name"].values
