# test_outlier_handler.py
import numpy as np
import pandas as pd
import pytest
from pipelines.common import OutlierHandler


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'num1': [1, 2, 3, 100, -50],
        'num2': [10, 15, 20, 25, 30],
        'cat': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def handler():
    return OutlierHandler()


def test_fit_computes_correct_bounds(sample_df, handler):
    handler.fit(sample_df)
    expected_q1 = sample_df['num1'].quantile(0.1)
    expected_q3 = sample_df['num1'].quantile(0.9)
    expected_iqr = expected_q3 - expected_q1

    assert np.isclose(handler.q1_['num1'], expected_q1)
    assert np.isclose(handler.q3_['num1'], expected_q3)
    assert np.isclose(handler.lower_bound_['num1'], expected_q1 - handler.iqr_multiplier * expected_iqr)
    assert np.isclose(handler.upper_bound_['num1'], expected_q3 + handler.iqr_multiplier * expected_iqr)
    assert handler.numeric_cols_ == ['num1', 'num2']
    assert handler.all_cols_ == ['num1', 'num2', 'cat']


def test_transform_applies_expected_mask(sample_df, handler):
    handler.fit(sample_df)
    result = handler.transform(sample_df)
    df_t = pd.DataFrame(result, columns=handler.all_cols_)

    for col in handler.numeric_cols_:
        col_vals = sample_df[col]
        lb = handler.lower_bound_[col]
        ub = handler.upper_bound_[col]
        expected_mask = (col_vals >= lb) & (col_vals <= ub)

        if not np.issubdtype(col_vals.dtype, np.number):
            continue

        actual_not_na = ~df_t[col].isna()
        assert actual_not_na.equals(expected_mask), \
            f"Column {col}: mask mismatch"

    pd.testing.assert_series_equal(df_t['cat'], sample_df['cat'], check_dtype=False)


def test_transform_with_array_input_consistency(sample_df, handler):
    handler.fit(sample_df)
    arr = sample_df.values
    result = handler.transform(arr)
    df_t = pd.DataFrame(result, columns=handler.all_cols_)

    df_direct = pd.DataFrame(handler.transform(sample_df), columns=handler.all_cols_)
    pd.testing.assert_frame_equal(df_t, df_direct)


def test_custom_multiplier_increases_outliers(sample_df):
    default_handler = OutlierHandler(iqr_multiplier=1.5)
    default_handler.fit(sample_df)
    df_default = pd.DataFrame(default_handler.transform(sample_df), columns=default_handler.all_cols_)
    nans_default = df_default['num1'].isna().sum()

    small_handler = OutlierHandler(iqr_multiplier=0.5)
    small_handler.fit(sample_df)
    df_small = pd.DataFrame(small_handler.transform(sample_df), columns=small_handler.all_cols_)
    nans_small = df_small['num1'].isna().sum()

    assert nans_small >= nans_default, \
        "The smaller IQR multiplier should generate at least as many NaN as the default or more"

def test_specific_outliers_set_to_nan_with_low_multiplier(sample_df):
    handler = OutlierHandler(iqr_multiplier=0.1)
    handler.fit(sample_df)
    df_t = pd.DataFrame(handler.transform(sample_df), columns=handler.all_cols_)

    assert pd.isna(df_t.loc[3, 'num1']), "Value 100 at index 3 should be NaN with low multiplier"
    assert pd.isna(df_t.loc[4, 'num1']), "Value -50 at index 4 should be NaN with low multiplier"

    default_handler = OutlierHandler(iqr_multiplier=1.5)
    default_handler.fit(sample_df)
    df_default = pd.DataFrame(default_handler.transform(sample_df), columns=default_handler.all_cols_)
    nans_default = df_default['num1'].isna().sum()

    small_handler = OutlierHandler(iqr_multiplier=0.5)
    small_handler.fit(sample_df)
    df_small = pd.DataFrame(small_handler.transform(sample_df), columns=small_handler.all_cols_)
    nans_small = df_small['num1'].isna().sum()

    assert nans_small >= nans_default, (
        "The smaller IQR multiplier should generate at least as many NaN as the default or more"
    )