import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch

# Assuming the functions are available at these imports
from src.gabarra.data_visualization import data_visualization as dv


class TestDataVisualization(unittest.TestCase):

    def setUp(self):
        # Set up a sample DataFrame for testing
        self.df = pd.DataFrame({
            'numeric': [1, 2, 3, np.nan, 5],
            'categorical': ['a', 'b', 'a', 'b', 'c'],
            'num_orders': [10, 15, 10, 5, 25]
        })

    @patch('matplotlib.pyplot.show')
    def test_missing_values_summary(self, mock_show):
        """
        Tests the `missing_values_summary` function to ensure it runs without errors and returns the correct DataFrame.
        """
        result = dv.missing_values_summary(self.df)
        expected = pd.DataFrame({
            'Missing Values': [1, 0, 0],
            'Percentage': [20.0, 0.0, 0.0]
        }, index=['numeric', 'categorical', 'num_orders'])
        pd.testing.assert_frame_equal(result, expected)

    @patch('matplotlib.pyplot.show')
    def test_plot_numeric_distributions(self, mock_show):
        """
        Tests the `plot_numeric_distributions` function to ensure it runs without errors.
        """
        try:
            dv.plot_numeric_distributions(self.df)
            dv.plot_numeric_distributions(self.df, hue='categorical')
        except Exception as e:
            self.fail(f"plot_numeric_distributions raised an exception: {e}")

    def test_get_viridis_colors(self):
        """
        Tests the `get_viridis_colors` function to ensure it returns the correct number of colors.
        """
        result = dv.get_viridis_colors(3)
        self.assertEqual(len(result), 3)

    @patch('matplotlib.pyplot.show')
    def test_plot_pie_charts(self, mock_show):
        """
        Tests the `plot_pie_charts` function to ensure it runs without errors.
        """
        try:
            dv.plot_pie_charts(self.df, ['categorical'])
        except Exception as e:
            self.fail(f"plot_pie_charts raised an exception: {e}")

    def test_plot_interactive_line_chart(self):
        """
        Tests the `plot_interactive_line_chart` function to ensure it runs without errors.
        """
        try:
            dv.plot_interactive_line_chart(self.df, 'numeric', 'num_orders')
            dv.plot_interactive_line_chart(self.df, 'numeric', 'num_orders', color_column='categorical')
        except Exception as e:
            self.fail(f"plot_interactive_line_chart raised an exception: {e}")

    def test_plot_interactive_pie_chart(self):
        """
        Tests the `plot_interactive_pie_chart` function to ensure it runs without errors.
        """
        try:
            dv.plot_interactive_pie_chart(self.df, 'categorical')
        except Exception as e:
            self.fail(f"plot_interactive_pie_chart raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
