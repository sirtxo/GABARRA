import unittest
import pandas as pd
import numpy as np

# Assuming the functions are available at these imports
from src.gabarra.data_processing import data_processing as dp


class TestFunctions(unittest.TestCase):

    def test_create_dummies(self):
        """
        Tests the `create_dummies` function (assuming it's defined in a module named `dp`).

        This test verifies that the function correctly creates dummy variables for categorical columns in a pandas DataFrame.

        Raises:
            AssertionError: If any of the assertions about the resulting DataFrame fail.
        """

        data = {'A': ['a', 'b', 'c', 'a', 'b'],
                'B': [1, 2, 3, 4, 5],
                'C': [6, 7, 8, 9, 10]}
        df = pd.DataFrame(data)

        result_df = dp.create_dummies(df)

        self.assertEqual(result_df.shape, (5, 7))
        self.assertIn('A_a', result_df.columns)
        self.assertIn('A_b', result_df.columns)
        self.assertIn('A_c', result_df.columns)
        self.assertIn('B', result_df.columns)
        self.assertIn('C', result_df.columns)
        self.assertEqual(result_df['A_a'].sum(), 2)
        self.assertEqual(result_df['A_b'].sum(), 2)
        self.assertEqual(result_df['A_c'].sum(), 1)

    def test_create_dummies_empty_df(self):
        """
        Tests the behavior of `create_dummies` when called with an empty DataFrame.

        This test verifies that the function returns an empty DataFrame if the input DataFrame is empty.

        Raises:
            AssertionError: If the resulting DataFrame is not empty.
        """
        df = pd.DataFrame()
        result_df = dp.create_dummies(df)
        self.assertTrue(result_df.empty)

    def test_create_dummies_no_object_cols(self):
        """
        Tests the behavior of `create_dummies` when called with a DataFrame containing only numeric columns.

        This test verifies that the function returns the original DataFrame unchanged if there are no object (categorical) columns to convert.

        Raises:
            AssertionError: If the resulting DataFrame is not identical to the original DataFrame.
        """
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        result_df = dp.create_dummies(df)
        self.assertTrue(result_df.equals(df))

    def test_fill_zeros_with_mean(self):
        """
        Tests the `fill_zeros_with_mean` function (assuming it's defined in a module named `dp`).

        This test verifies that the function correctly replaces zero values in a DataFrame column
        with the mean of the column, excluding zeros.

        Raises:
            AssertionError: If any of the assertions about the filled values fail.
        """
        data = {'A': [0, 1, 2, 0, 3, 0]}
        df = pd.DataFrame(data)

        result_df = dp.fill_zeros_with_mean(df, 'A')

        self.assertEqual(result_df['A'].loc[0], 2.0)
        self.assertEqual(result_df['A'].loc[3], 2.0)
        self.assertEqual(result_df['A'].loc[5], 2.0)

    def test_fill_zeros_with_mean_no_zeros(self):
        """
        Tests the behavior of `fill_zeros_with_mean` when there are no zero values in the column.

        This test verifies that the function leaves the DataFrame unchanged if the specified
        column contains no zeros.

        Raises:
            AssertionError: If the resulting DataFrame is not identical to the original DataFrame.
        """
        data = {'A': [1, 2, 3, 4, 5]}
        df = pd.DataFrame(data)

        result_df = dp.fill_zeros_with_mean(df, 'A')

        self.assertTrue(result_df.equals(df))

    def test_fill_zeros_with_mean_column_not_found(self):
        """
        Tests that `fill_zeros_with_mean` raises a ValueError if the specified column does not exist in the DataFrame.

        This test verifies that the function raises an appropriate exception when the provided column name is not found in the DataFrame.

        Raises:
            AssertionError: If a ValueError is not raised with the expected message.
        """
        data = {'A': [0, 1, 2, 0, 3, 0]}
        df = pd.DataFrame(data)

        with self.assertRaises(ValueError):
            dp.fill_zeros_with_mean(df, 'B')

    def test_fill_zeros_with_mean_no_nonzero_values(self):
        """
        Tests the behavior of `fill_zeros_with_mean` when there are no non-zero values in the column.

        This test verifies that the function issues a UserWarning and leaves the column unchanged
        if there are no values other than zero in the specified column.

        Raises:
            AssertionError: If any of the assertions about the warning or DataFrame content fail.
        """
        data = {'A': [0, 0, 0, 0, 0]}
        df = pd.DataFrame(data)

        with self.assertWarns(UserWarning):
            dp.fill_zeros_with_mean(df, 'A')

        self.assertTrue(df['A'].equals(pd.Series([0, 0, 0, 0, 0])))

    def test_fill_nans_with_mean(self):
        """
        Tests the `fill_nans_with_mean` function (assuming it's defined in a module named `dp`).

        This test verifies that the function correctly replaces NaN (Not a Number) values in a DataFrame column with the mean of the column, excluding NaNs.

        Raises:
            AssertionError: If any of the assertions about the filled values fail.
        """
        data = {'A': [1, 2, np.nan, 4, 5, np.nan]}
        df = pd.DataFrame(data)

        result_df = dp.fill_nans_with_mean(df, 'A')

        self.assertEqual(result_df['A'].loc[2], 3.0)
        self.assertEqual(result_df['A'].loc[5], 3.0)

    def test_fill_nans_with_mean_no_nans(self):
        """
        Tests the behavior of `fill_nans_with_mean` when there are no NaN values in the column.

        This test verifies that the function leaves the DataFrame unchanged if the specified
        column contains no NaN values.

        Raises:
            AssertionError: If the resulting DataFrame is not identical to the original DataFrame.
        """
        data = {'A': [1, 2, 3, 4, 5]}
        df = pd.DataFrame(data)

        result_df = dp.fill_nans_with_mean(df, 'A')

        self.assertTrue(result_df.equals(df))

    def test_fill_nans_with_mean_column_not_found(self):
        """
        Tests that `fill_nans_with_mean` raises a ValueError if the specified column does not exist in the DataFrame.

        This test verifies that the function raises an appropriate exception when the provided column name is not found in the DataFrame.

        Raises:
            AssertionError: If a ValueError is not raised with the expected message.
        """
        data = {'A': [1, 2, 3, 4, 5]}
        df = pd.DataFrame(data)

        with self.assertRaises(ValueError):
            dp.fill_nans_with_mean(df, 'B')

    def test_fill_nans_with_mean_all_nans(self):
        """
        Tests the behavior of `fill_nans_with_mean` when all values in the column are NaN.

        This test verifies that the function either:

        * Leaves the column unchanged (if it doesn't raise an exception).
        * Raises a UserWarning (if it attempts to calculate a mean from all NaNs).

        The specific behavior might depend on the implementation of the function.

        Raises:
            AssertionError: If the resulting DataFrame unexpectedly modifies the original or raises an unexpected exception.
        """
        data = {'A': [np.nan, np.nan, np.nan, np.nan, np.nan]}
        df = pd.DataFrame(data)

        with self.assertWarns(UserWarning):
            result_df = dp.fill_nans_with_mean(df, 'A')

        self.assertTrue(result_df.equals(df))

    def test_convert_to_numeric_label_encoding(self):
        """
        Tests the `convert_to_numeric` function (assuming it's defined in a module named `dp`) with LabelEncoding.

        This test verifies that the function correctly converts a categorical column in a DataFrame
        to numeric using label encoding.

        Raises:
            AssertionError: If any of the assertions about the resulting DataFrame fail.
        """
        data = {'A': ['a', 'b', 'c', 'a', 'b', 'c']}
        df = pd.DataFrame(data)

        result_df = dp.convert_to_numeric(df, motive='LabelEncoding', columns=['A'])

        self.assertEqual(result_df['A'].dtype, 'int64')
        self.assertEqual(sorted(result_df['A'].unique()), [0, 1, 2])

    def test_convert_to_numeric_one_hot_encoding(self):
        """
        Tests the `convert_to_numeric` function (assuming it's defined in a module named `dp`) with OneHotEncoding.

        This test verifies that the function correctly converts a categorical column in a DataFrame
        to numeric using one-hot encoding.

        Raises:
            AssertionError: If any of the assertions about the resulting DataFrame fail.
        """
        data = {'A': ['a', 'b', 'c', 'a', 'b', 'c']}
        df = pd.DataFrame(data)

        result_df = dp.convert_to_numeric(df, motive='OneHotEncoding', columns=['A'])

        expected_columns = ['A_a', 'A_b', 'A_c']
        self.assertEqual(len(result_df.columns), len(expected_columns))
        self.assertTrue(all(col in result_df.columns for col in expected_columns))

    def test_convert_to_numeric_frequency_encoding(self):
        """
        Tests the `convert_to_numeric` function (assuming it's defined in a module named `dp`) with FrequencyEncoding.

        This test verifies that the function correctly converts a categorical column in a DataFrame
        to numeric using frequency encoding (assigning values based on category occurrence).

        Raises:
            AssertionError: If any of the assertions about the resulting DataFrame fail.
        """
        data = {'A': ['a', 'b', 'c', 'a', 'b', 'c']}
        df = pd.DataFrame(data)

        result_df = dp.convert_to_numeric(df, motive='FrequencyEncoding', columns=['A'])

        self.assertEqual(result_df['A'].dtype, 'int64')
        self.assertEqual(sorted(result_df['A'].unique()), [2, 2, 2])

    def test_convert_to_numeric_invalid_motive(self):
        """
        Tests that `convert_to_numeric` raises a ValueError for an invalid motive.

        This test verifies that the function raises an appropriate exception when provided with
        a motive that is not supported (e.g., 'InvalidMotive').

        Raises:
            AssertionError: If a ValueError is not raised with the expected message.
        """
        data = {'A': ['a', 'b', 'c', 'a', 'b', 'c']}
        df = pd.DataFrame(data)

        with self.assertRaises(ValueError):
            dp.convert_to_numeric(df, motive='InvalidMotive', columns=['A'])

    def test_convert_to_numeric_no_columns(self):
        """
        Tests the behavior of `convert_to_numeric` when no columns are specified.

        This test verifies that the function returns the original DataFrame unchanged if no columns are provided for conversion.

        Raises:
            AssertionError: If the resulting DataFrame is not identical to the original DataFrame.
        """
        data = {'A': [1, 2, 3, 4, 5]}
        df = pd.DataFrame(data)

        result_df = dp.convert_to_numeric(df)

        self.assertTrue(result_df.equals(df))


if __name__ == '__main__':
    unittest.main()
