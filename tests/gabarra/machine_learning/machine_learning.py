import unittest
from sklearn.datasets import make_regression
import sys
import io
from xgboost import XGBRegressor

# Assuming the functions are available at these imports
from src.gabarra.machine_learning.machine_learning import *

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        # Create a test DataFrame
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        self.df = pd.DataFrame(X, columns=['Feature'])
        self.df['Target'] = y

    def test_linear_regression(self):

        result = linear_regression(self.df, 'Target')

        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)

        # Check that the model is an instance of LinearRegression
        self.assertIsInstance(result['model'], LinearRegression)

        # Check that the predictions are a numpy array
        self.assertIsInstance(result['predictions'], np.ndarray)

        # Check that the mean squared error and R^2 are numbers
        self.assertIsInstance(result['mean_squared_error'], (int, float))
        self.assertIsInstance(result['r2_score'], (int, float))

        # Check that the mean squared error and R^2 are in the expected range
        self.assertGreaterEqual(result['mean_squared_error'], 0)
        self.assertGreaterEqual(result['r2_score'], 0)
        self.assertLessEqual(result['r2_score'], 1)

if __name__ == '__main__':
    unittest.main()

class TestCalculateMetrics(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.y_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.predictions = np.array([1.1, 2.2, 2.9, 3.8, 5.2])
        self.model_name = "Test Model"

    def test_calculate_metrics(self):

        result = calculate_metrics(self.y_test, self.predictions, self.model_name)

        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the DataFrame has the expected columns
        expected_columns = ["Modelo", "MAE", "MAPE", "MSE", "RMSE"]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check that the metrics are calculated correctly
        mae = metrics.mean_absolute_error(self.y_test, self.predictions)
        mape = metrics.mean_absolute_percentage_error(self.y_test, self.predictions)
        mse = metrics.mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        expected_data = {"Modelo": [self.model_name], 'MAE': [mae], 'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse]}
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(result, expected_df)

if __name__ == '__main__':
    unittest.main()

class TestUnSupervisedCluster(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 3, 4, 5, 6]
        })
        self.motive = 'clustering'
        self.range = 5
        self.k = 2

    def test_unSupervisedCluster(self):

        result = unSupervisedCluster(self.df, self.motive, self.range, self.k)

        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the DataFrame has the expected columns
        expected_columns = ["Cluster"]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check that the number of clusters is correct
        unique_clusters = result['Cluster'].nunique()
        self.assertEqual(unique_clusters, self.k)

if __name__ == '__main__':
    unittest.main()
    
class TestGradientBoostingRegression(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 3, 4, 5, 6]
        })
        self.target_column = 'y'
        self.test_size = 0.2
        self.random_state = 42
        self.n_estimators = 100
        self.learning_rate = 0.1

    def test_gradient_boosting_regression(self):

        result = gradient_boosting_regression(self.df, self.target_column, self.test_size, self.random_state, self.n_estimators, self.learning_rate)

        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)

        # Check that the dictionary has the expected keys
        expected_keys = ["model", "predictions", "MAE", "MAPE", "MSE", "RMSE", "R2_score"]
        self.assertListEqual(list(result.keys()), expected_keys)

        # Check that the model is a GradientBoostingRegressor
        self.assertIsInstance(result["model"], GradientBoostingRegressor)

        # Check that the predictions are a numpy array
        self.assertIsInstance(result["predictions"], np.ndarray)

        # Check that the performance metrics are floats
        self.assertIsInstance(result["MAE"], float)
        self.assertIsInstance(result["MAPE"], float)
        self.assertIsInstance(result["MSE"], float)
        self.assertIsInstance(result["RMSE"], float)
        self.assertIsInstance(result["R2_score"], float)

if __name__ == '__main__':
    unittest.main()

class TestXGBoostRegression(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'target': [3, 4, 5, 6, 7]
        })
        self.target_column = 'target'

    def test_xgboost_regression(self):
        result = xgboost_regression(self.df, self.target_column)

        # Check if the output is a dictionary
        self.assertIsInstance(result, dict)

        # Check if the dictionary has all the expected keys
        expected_keys = ["model", "predictions", "MAE", "MAPE", "MSE", "RMSE", "R2_score"]
        self.assertTrue(all(key in result for key in expected_keys))

        # Check if the model is an instance of XGBRegressor
        self.assertIsInstance(result["model"], XGBRegressor)

        # Check if predictions are a numpy array
        self.assertIsInstance(result["predictions"], np.ndarray)

        # Check if metrics are floats
        self.assertIsInstance(result["MAE"], float)
        self.assertIsInstance(result["MAPE"], float)
        self.assertIsInstance(result["MSE"], float)
        self.assertIsInstance(result["RMSE"], float)
        self.assertIsInstance(result["R2_score"], float)

if __name__ == '__main__':
    unittest.main()

class TestRandomForestRegression(unittest.TestCase):
    def setUp(self):
        # For random_forest_regression function
        self.df, self.target = make_regression(n_samples=100, n_features=20, n_informative=10, random_state=42)
        self.df = pd.DataFrame(self.df)
        self.df['target'] = self.target
        self.target_column = 'target'

    def test_random_forest_regression(self):
        # Call the function
        result = random_forest_regression(self.df, self.target_column)

        # Check if the output is a dictionary
        self.assertIsInstance(result, dict)

        # Check if the dictionary keys are correct
        expected_keys = ["model", "predictions", "MAE", "MAPE", "MSE", "RMSE", "R2_score"]
        self.assertCountEqual(result.keys(), expected_keys)

        # Check if the model is a RandomForestRegressor instance
        self.assertIsInstance(result["model"], RandomForestRegressor)

        # Check if the predictions are a numpy array
        self.assertIsInstance(result["predictions"], np.ndarray)

        # Check if the metrics are floats
        self.assertIsInstance(result["MAE"], float)
        self.assertIsInstance(result["MAPE"], float)
        self.assertIsInstance(result["MSE"], float)
        self.assertIsInstance(result["RMSE"], float)
        self.assertIsInstance(result["R2_score"], float)

if __name__ == '__main__':
    unittest.main()