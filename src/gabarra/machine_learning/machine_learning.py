from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(df, target_column):
    """
    Performs linear regression on the given dataset without standardizing the data.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    target_column (str): The name of the target column.

    Returns:
    dict: A dictionary containing the model, predictions, mean squared error, and R^2 score.
    """
    # Splitting the data into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating and training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Calculating performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Returning the results
    return {
        "model": model,
        "predictions": y_pred,
        "mean_squared_error": mse,
        "r2_score": r2
    }

# Example usage
# result = linear_regression(df, 'target_column_name')
# print(result["mean_squared_error"], result["r2_score"])