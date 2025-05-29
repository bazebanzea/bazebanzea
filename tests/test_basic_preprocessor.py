import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
import sys
import os

# Adjust the Python path to include the 'src' directory
# This allows importing modules from src.preprocessing
# Assuming the script is in 'tests/' and 'src/' is a sibling directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.basic_preprocessor import (
    impute_numerical,
    encode_categorical,
    scale_numerical_features,
    report_missing_values
)

# --- Fixtures for test data ---

@pytest.fixture
def sample_df_with_nans():
    """DataFrame with NaNs in numerical and categorical columns."""
    data = {
        'num_col1': [1.0, 2.0, np.nan, 4.0, 5.0], # Mean: 3.0, Median: 3.0 (of 1,2,4,5)
        'num_col2': [10.0, np.nan, 12.0, 13.0, 14.0], # Mean: 12.25, Median: 12.5 (of 10,12,13,14)
        'cat_col1': ['A', 'B', 'A', 'C', np.nan],
        'num_col_no_nans': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_df_fixture():
    """An empty DataFrame."""
    return pd.DataFrame()

@pytest.fixture
def df_all_nans_col_fixture():
    """DataFrame with a column that is all NaNs."""
    data = {'all_nan_col': [np.nan, np.nan, np.nan], 'other_col': [1,2,3]}
    return pd.DataFrame(data)

# --- Tests for impute_numerical ---

def test_impute_numerical_mean(sample_df_with_nans):
    df_orig = sample_df_with_nans.copy()
    col = 'num_col1'
    nan_mask = df_orig[col].isnull()
    original_mean = df_orig[col].mean() # Mean of non-NaN values
    
    df_imputed = impute_numerical(df_orig.copy(), col, strategy='mean')
    
    assert df_imputed[col].isnull().sum() == 0, "NaNs were not filled"
    # Check that only original NaN positions are filled with the mean
    # Create a series with the mean at NaN positions for comparison
    expected_filled_values = pd.Series([original_mean] * nan_mask.sum(), index=df_orig.index[nan_mask])
    pd.testing.assert_series_equal(
        df_imputed.loc[nan_mask, col],
        expected_filled_values,
        check_dtype=False 
    )
    # Check that non-NaN values remained unchanged
    pd.testing.assert_series_equal(
        df_imputed.loc[~nan_mask, col],
        df_orig.loc[~nan_mask, col], # Compare with original non-NaN values
        check_dtype=False
    )

def test_impute_numerical_median(sample_df_with_nans):
    df_orig = sample_df_with_nans.copy()
    col = 'num_col2'
    nan_mask = df_orig[col].isnull()
    original_median = df_orig[col].median() 

    df_imputed = impute_numerical(df_orig.copy(), col, strategy='median')
    
    assert df_imputed[col].isnull().sum() == 0, "NaNs were not filled"
    expected_filled_values = pd.Series([original_median] * nan_mask.sum(), index=df_orig.index[nan_mask])
    pd.testing.assert_series_equal(
        df_imputed.loc[nan_mask, col],
        expected_filled_values,
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        df_imputed.loc[~nan_mask, col],
        df_orig.loc[~nan_mask, col],
        check_dtype=False
    )

def test_impute_numerical_no_nans(sample_df_with_nans):
    df = sample_df_with_nans.copy()
    col = 'num_col_no_nans'
    original_series = df[col].copy()
    
    df_imputed = impute_numerical(df.copy(), col, strategy='mean')
    
    assert_series_equal(df_imputed[col], original_series, check_dtype=False)

def test_impute_numerical_non_numeric_column(sample_df_with_nans):
    df = sample_df_with_nans.copy()
    col = 'cat_col1'
    with pytest.raises(TypeError, match=f"Column '{col}' is not numerical."):
        impute_numerical(df.copy(), col, strategy='mean')

def test_impute_numerical_empty_df(empty_df_fixture):
    df = empty_df_fixture.copy()
    # Add a column to avoid issues with empty df structure if impute_numerical expects columns
    df['empty_num_col'] = pd.Series(dtype=float) 
    
    df_imputed = impute_numerical(df.copy(), 'empty_num_col', strategy='mean')
    assert df_imputed['empty_num_col'].isnull().all() 

def test_impute_numerical_all_nans_column(df_all_nans_col_fixture):
    df = df_all_nans_col_fixture.copy()
    col = 'all_nan_col'
    
    df_imputed_mean = impute_numerical(df.copy(), col, strategy='mean')
    assert df_imputed_mean[col].isnull().all()

    df_imputed_median = impute_numerical(df.copy(), col, strategy='median')
    assert df_imputed_median[col].isnull().all()

def test_impute_numerical_invalid_strategy(sample_df_with_nans):
    df = sample_df_with_nans.copy()
    col = 'num_col1'
    with pytest.raises(ValueError, match="Invalid imputation strategy 'mode'. Choose 'mean' or 'median'."):
        impute_numerical(df.copy(), col, strategy='mode')

def test_impute_numerical_column_not_found(sample_df_with_nans):
    df = sample_df_with_nans.copy()
    with pytest.raises(ValueError, match="Column 'non_existent_col' not found in DataFrame."):
        impute_numerical(df.copy(), 'non_existent_col', strategy='mean')


# --- Tests for encode_categorical ---

def test_encode_categorical_basic(sample_df_with_nans):
    df_orig = sample_df_with_nans.copy()
    col = 'cat_col1'
    
    # Fill NaN for consistent testing of dummy_na=False behavior
    df_filled_na = df_orig.copy()
    df_filled_na[col] = df_filled_na[col].fillna('Unknown') 
    
    num_original_cols = len(df_filled_na.columns)
    num_unique_values = df_filled_na[col].nunique() # A, B, C, Unknown -> 4
    
    df_encoded = encode_categorical(df_filled_na.copy(), col)
    
    assert col not in df_encoded.columns, "Original column was not dropped"
    assert len(df_encoded.columns) == (num_original_cols - 1 + num_unique_values)
    
    expected_new_cols_parts = ['A', 'B', 'C', 'Unknown']
    for part in expected_new_cols_parts:
        assert f"{col}_{part}" in df_encoded.columns
    
    # Check one-hot encoding for specific rows
    # Original: row 0, cat_col1 = 'A'
    assert df_encoded.loc[0, f"{col}_A"] == 1
    assert df_encoded.loc[0, f"{col}_B"] == 0
    assert df_encoded.loc[0, f"{col}_Unknown"] == 0
    
    # Original: row 4, cat_col1 = np.nan (filled to 'Unknown')
    nan_original_index = df_orig[df_orig['cat_col1'].isnull()].index[0] # This is index 4
    assert df_encoded.loc[nan_original_index, f"{col}_Unknown"] == 1
    assert df_encoded.loc[nan_original_index, f"{col}_A"] == 0


def test_encode_categorical_with_prefix(sample_df_with_nans):
    df_orig = sample_df_with_nans.copy()
    col = 'cat_col1'
    df_filled_na = df_orig.copy()
    df_filled_na[col] = df_filled_na[col].fillna('Unknown')
    prefix = 'category_custom'
    
    df_encoded = encode_categorical(df_filled_na.copy(), col, prefix=prefix)
    
    assert col not in df_encoded.columns
    expected_new_cols_parts = ['A', 'B', 'C', 'Unknown']
    for part in expected_new_cols_parts:
         assert f"{prefix}_{part}" in df_encoded.columns

def test_encode_categorical_numeric_input(sample_df_with_nans, capsys):
    df = sample_df_with_nans.copy()
    col = 'num_col_no_nans' # An integer column [1, 2, 3, 4, 5]
    
    df_encoded = encode_categorical(df.copy(), col)
    captured = capsys.readouterr() 
    assert f"Warning: Column '{col}' is numerical." in captured.out
    
    assert col not in df_encoded.columns
    for i in df[col].unique(): # Values are 1 through 5
        assert f"{col}_{i}" in df_encoded.columns 
    assert df_encoded.loc[0, f"{col}_1"] == 1 # Original value was 1

def test_encode_categorical_missing_values_default_behavior(sample_df_with_nans):
    df = sample_df_with_nans.copy() # 'cat_col1' has a NaN at index 4
    col = 'cat_col1'
    
    df_encoded = encode_categorical(df.copy(), col) # Uses dummy_na=False
    
    assert f"{col}_nan" not in df_encoded.columns 
    
    nan_row_index = df[df[col].isnull()].index[0] 
    encoded_cols_for_cat = [c for c in df_encoded.columns if c.startswith(col + "_")]
    
    assert all(df_encoded.loc[nan_row_index, encoded_cols_for_cat] == 0)


def test_encode_categorical_column_not_found(sample_df_with_nans):
    df = sample_df_with_nans.copy()
    with pytest.raises(ValueError, match="Column 'non_existent_cat_col' not found in DataFrame."):
        encode_categorical(df.copy(), 'non_existent_cat_col')


# --- Tests for scale_numerical_features ---

def test_scale_numerical_single_column():
    df = pd.DataFrame({'A': [10., 20., 30., 40., 50.]}) 
    col_to_scale = 'A'
    
    df_scaled = scale_numerical_features(df.copy(), [col_to_scale])
    
    assert df_scaled[col_to_scale].mean() == pytest.approx(0.0, abs=1e-6)
    assert df_scaled[col_to_scale].std() == pytest.approx(1.0, abs=1e-6)

def test_scale_numerical_multiple_columns(sample_df_with_nans):
    df = sample_df_with_nans[['num_col1', 'num_col_no_nans']].copy()
    df['num_col1'] = df['num_col1'].fillna(df['num_col1'].mean()) # Impute NaNs
    cols_to_scale = ['num_col1', 'num_col_no_nans']
    
    df_scaled = scale_numerical_features(df.copy(), cols_to_scale)
    
    for col in cols_to_scale:
        assert df_scaled[col].mean() == pytest.approx(0.0, abs=1e-6)
        assert df_scaled[col].std() == pytest.approx(1.0, abs=1e-6)

def test_scale_numerical_empty_df(empty_df_fixture):
    df = empty_df_fixture.copy()
    df['A'] = pd.Series(dtype=float)
    df['B'] = pd.Series(dtype=float)

    df_scaled = scale_numerical_features(df.copy(), ['A', 'B'])
    assert df_scaled['A'].empty or df_scaled['A'].isnull().all()
    assert df_scaled['B'].empty or df_scaled['B'].isnull().all()


def test_scale_numerical_non_numeric_in_list(sample_df_with_nans):
    df = sample_df_with_nans.copy()
    cols_to_scale = ['num_col1', 'cat_col1'] 
    
    with pytest.raises(TypeError, match="Column 'cat_col1' is not numerical and cannot be scaled."):
        scale_numerical_features(df.copy(), cols_to_scale)

def test_scale_numerical_column_not_found(sample_df_with_nans):
    df = sample_df_with_nans.copy()
    with pytest.raises(ValueError, match="Column 'non_existent_num_col' not found in DataFrame."):
        scale_numerical_features(df.copy(), ['num_col1', 'non_existent_num_col'])

def test_scale_numerical_with_nans_in_input(sample_df_with_nans):
    df = sample_df_with_nans.copy() # num_col1 has nans
    with pytest.raises(ValueError, match="Input contains NaN"): 
        scale_numerical_features(df.copy(), ['num_col1'])

# --- Test for report_missing_values ---
def test_report_missing_values_runs_with_nans(sample_df_with_nans, capsys):
    report_missing_values(sample_df_with_nans.copy())
    captured = capsys.readouterr()
    output = captured.out
    assert "Missing values report:" in output
    assert "num_col1" in output 
    assert "num_col2" in output
    assert "cat_col1" in output
    # Check that num_col_no_nans is NOT in the detailed missing report part
    # This means checking that its line (e.g. "num_col_no_nans 0 0.0") is not present
    # A simpler way is to check if the "No missing values found" message is NOT there
    # and the columns with missing values ARE there.
    assert "No missing values found." not in output


def test_report_missing_values_no_missing(sample_df_with_nans, capsys):
    df_no_missing = sample_df_with_nans[['num_col_no_nans']].copy() # Only take column with no NaNs
    report_missing_values(df_no_missing)
    captured = capsys.readouterr()
    assert "No missing values found." in captured.out
    assert "num_col_no_nans" not in captured.out # Should not be listed in the table if no missing values
