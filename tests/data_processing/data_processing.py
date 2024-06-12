import unittest
import pandas as pd
import numpy as np

# Assuming the functions are available at these imports
from src.gabarra.data_processing import data_processing as dp

class TestFunctions(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()

