from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score , mean_absolute_error, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn import metrics
import numpy as np
import pandas as pd

def linear_regression(df:pd, target_column:str):
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



def calculate_metrics(y_test:np, predictions:np, model_name:str, decimal_places=2):
    """
    Calculates and returns a DataFrame with regression metrics: MAE, MAPE, MSE, and RMSE.

    Parameters:
    y_test (array-like): True values.
    predictions (array-like): Predicted values.
    model_name (str): Name of the model.
    decimal_places (int, optional): Number of decimal places to display. Default is 2.

    Returns:
    pd.DataFrame: DataFrame containing the calculated metrics.
    """
    # Calculate metrics
    mae = metrics.mean_absolute_error(y_test, predictions)
    mape = metrics.mean_absolute_percentage_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Create a DataFrame with the metrics
    data = {"Modelo": [model_name], 'MAE': [mae], 'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse]}
    df_metrics = pd.DataFrame(data)

    # Format the DataFrame to show specified decimal places
    pd.options.display.float_format = f'{{:.{decimal_places}f}}'.format

    return df_metrics

def unSupervisedCluster(df:pd,motive='analisys', range=20 ,k=3 ):
    
    '''
    Function:
    -----------

    This function works with the unsupervised model of Kmeans, and its objective is to show you how depending the number of
    clusters that you want the inertia and the silhouette score are going to go up or down to facilitate your choose oof k, and also
    have the model of Kmeans to see thoose clusters.


    Parameters:
    -----------
    df: Pandas DataFrame
        Data that the function is going to analyze
    motive: str
        Depend in wich word you use the function is going to ralize different things, for example 'Analysis' show you 2 graphs
        and 'clustering' give you in wich cluster is every target
    Range: int
        Range of k's that are in the graph showing the inertia and the silhouette score for each one of them
    K: int
        number that indicates how much clusters do you want in the modeling of Kmeans
    Returns:
    -----------
    Pandas DataFrame
        The function returns a dataframe with an aditional column wich have in wich cluster each target is in

    '''
    if motive=='analisys':
        km_list = [KMeans(n_clusters=a, random_state=42).fit(df) for a in range(2,range)]
        inertias = [model.inertia_ for model in km_list]
        silhouette_score_list = [silhouette_score(df, model.labels_) for model in km_list]

        plt.figure(figsize=(20,5))

        plt.subplot(121)
        sns.set(rc={'figure.figsize':(10,10)})
        plt.plot(range(2,range), inertias)
        plt.xlabel('k')
        plt.ylabel("inertias")
        sns.despine()

        plt.subplot(122)
        sns.set(rc={'figure.figsize':(10,10)})
        plt.plot(range(2,range), silhouette_score_list)
        plt.xlabel('k')
        plt.ylabel("silhouette_score")
        sns.despine()

    if motive =='clustering':
        kmeans = KMeans(n_clusters=k,n_init=10, random_state=42).fit(df)
        df_clusters = pd.DataFrame(kmeans.labels_, columns=['Cluster'])
        return df_clusters



def gradient_boosting_regression(df:pd, target_column:str,test_size=0.2,random_state=42,n_estimators=100,learning_rate=0.1):
    """
    Performs Gradient Boosting regression on the given dataset.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    target_column (str): The name of the target column.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split. Default is 42.
    n_estimators (int, optional): The number of boosting stages to be run. Default is 100.
    learning_rate (float, optional): Learning rate shrinks the contribution of each tree by learning_rate. Default is 0.1.

    Returns:
    dict: A dictionary containing the model, predictions, MAE, MAPE, MSE, RMSE, and R^2 score.
    """
    # Splitting the data into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Creation and training the Gradient Boosting model
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Results
    return {
        "model": model,
        "predictions": y_pred,
        "MAE": mae,
        "MAPE": mape,
        "MSE": mse,
        "RMSE": rmse,
        "R2_score": r2
    }



def xgboost_regression(df:pd, target_column:str, test_size=0.2, random_state=42, n_estimators=100, learning_rate=0.1):
    """
    Performs XGBoost regression on the given dataset.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    target_column (str): The name of the target column.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split. Default is 42.
    n_estimators (int, optional): The number of boosting stages to be run. Default is 100.
    learning_rate (float, optional): Learning rate shrinks the contribution of each tree by learning_rate. Default is 0.1.

    Returns:
    dict: A dictionary containing the model, predictions, MAE, MAPE, MSE, RMSE, and R^2 score.
    """
    # Splitting the data into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Creating and training the XGBoost model
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Results
    return {
        "model": model,
        "predictions": y_pred,
        "MAE": mae,
        "MAPE": mape,
        "MSE": mse,
        "RMSE": rmse,
        "R2_score": r2
    }
    
def most_common_words(texts:list, nwords:int, language:str):

    '''
    From a list of texts returns n most common words in a given language excluding stop words.

    Parameters:
    texts (list): List of texts
    nwords (int): number of most common words
    language (str): language of the text

    Return: print n most common worlds with their quantity

    Example:

    >>> most_common_words(['I hate cats', 'I love dogs', 'My dog love cats'], 3, 'english')
    Most common words:
    cats: 2
    love: 2
    dog: 1
    '''
    vectorizer_count = CountVectorizer(max_features=nwords, stop_words=language)
    texts = vectorizer_count.fit_transform(texts)
    vocabulary = vectorizer_count.vocabulary_
    most_common_words = {word: texts[:, index].sum() for word, index in vocabulary.items()}
    most_common_words = sorted(most_common_words.items(), key=lambda x: x[1], reverse=True)
    print("Most common words:")
    for word, frequency in most_common_words:
        print(f'{word}: {frequency}')

import os

def y_generator(path, labels, separator):

    ''' 
    Categorize a set of images for the training of multi-class learning models.
    Based on the image name.
    path[str]: Path to the folder.
    labels[list]: Possible output labels.
    separator[str]: Separator in the image name.
     
    '''

    y = []

    for i in os.listdir(path):
        s = i
        if separator:
            s = i.split(separator)
        j = labels
        c =""
        for w in s:
            for r in j:
                if r.lower() in w.lower():
                    c = w
                for x,y in enumerate(labels):
                    if y == c:
                        arr = np.zeros(len(labels))
                        arr[x] = 1
                        y.append(arr)
    return y