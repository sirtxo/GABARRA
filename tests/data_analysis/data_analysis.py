import unittest
import pandas as pd
import numpy as np

from src.gabarra.data_analysis.data_analysis import filter_rows, outlier_meanSd, data_report, remove_outliers
from src.gabarra.data_visualization.data_visualization import missing_values_summary

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Configuración que se ejecuta antes de cada prueba
        data = {
            'columna_numerica': [1, 2, 3, 4, 5, 100],
            'columna_x': [1, 2, 3, 4, 5, 6],
            'columna_y': [2, 4, 6, 8, 10, 12]
        }
        self.df = pd.DataFrame(data)

    def test_filter_rows(self):
        condition = 'columna_numerica > 3'
        filtered_df = filter_rows(self.df, condition)
        expected_data = {
            'columna_numerica': [4, 5, 100],
            'columna_x': [4, 5, 6],
            'columna_y': [8, 10, 12]
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(filtered_df, expected_df)

    def test_remove_outliers(self):
        df_no_outliers = remove_outliers(self.df, 'columna_numerica')
        expected_data = {
            'columna_numerica': [1, 2, 3, 4, 5],
            'columna_x': [1, 2, 3, 4, 5],
            'columna_y': [2, 4, 6, 8, 10]
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(df_no_outliers, expected_df)

    def test_outlier_meanSd(self):
        df_no_outliers = outlier_meanSd(self.df, 'columna_numerica')
        expected_data = {
            'columna_numerica': [1, 2, 3, 4, 5],
            'columna_x': [1, 2, 3, 4, 5],
            'columna_y': [2, 4, 6, 8, 10]
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(df_no_outliers, expected_df)

    def test_data_report(self):
        report = data_report(self.df)
        self.assertIn('columna_numerica', report.columns)
        self.assertIn('columna_x', report.columns)
        self.assertIn('columna_y', report.columns)

    def test_missing_values_summary(self):
        df_with_missing = self.df.copy()
        df_with_missing.loc[0, 'columna_numerica'] = np.nan
        summary = missing_values_summary(df_with_missing)
        self.assertEqual(summary.loc['columna_numerica', 'Missing Values'], 1)
        self.assertAlmostEqual(summary.loc['columna_numerica', 'Percentage'], 16.67, places=2)

if __name__ == '__main__':
    unittest.main()
