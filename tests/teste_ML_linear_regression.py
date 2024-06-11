import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the function to test
from  src import linear_regression

def test_linear_regression():
    # Create a dummy dataset for testing
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    df = pd.DataFrame(data=X, columns=["Feature"])
    df["Target"] = y

    # Run the linear regression function
    result = linear_regression(df, "Target")

    # Check the type of the returned results
    assert isinstance(result, dict), "Result should be a dictionary."
    assert isinstance(result["model"], LinearRegression), "Model should be an instance of LinearRegression."
    assert isinstance(result["predictions"], np.ndarray), "Predictions should be a numpy array."
    assert isinstance(result["mean_squared_error"], float), "Mean squared error should be a float."
    assert isinstance(result["r2_score"], float), "R^2 score should be a float."

    # Check the correctness of the returned results
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Target"]), df["Target"], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    assert np.allclose(result["predictions"], y_pred), "Predictions do not match."
    assert np.isclose(result["mean_squared_error"], mse), "Mean squared error does not match."
    assert np.isclose(result["r2_score"], r2), "R^2 score does not match."