import pandas as pd

def filter_rows(df, condition):
    '''
    Filters rows in a DataFrame based on a condition.

    Parameters:
    df(pd.DataFrame): The DataFrame to filter.
    condition (str): The condition to filter the rows. Must be a valid Pandas expression.

    Returns:
    pd.DataFrame: A filtered DataFrame that meets the specified condition.
    '''
    try:
        filtered_df = df.query(condition)
        return filtered_df
    except Exception as e:
        print(f"Error al filtrar filas: {e}")
        return None



import numpy as np

def remove_outliers(df, column_name):
     
        '''
        Define the remove_outliers function that takes a DataFrame df and a column_name as arguments.
        Calculate the first quartile (Q1), which is the value that separates the lowest 25% of the data. Use nanquantile to ignore any NaN values in the column.
        Calculate the third quartile (Q3), which is the value that separates the highest 25% of the data.
        Calculate the lower (lower_fence) and upper (upper_fence) limits to determine what values are considered outliers. Values below lower_fence or above upper_fence are considered outliers.
        Return a filtered DataFrame containing only the values within the calculated limits, thereby excluding outliers
        '''
    Q1 = np.nanquantile(df[column_name], 0.25)
    Q3  =np.nanquantile(df[column_name], 0.75)
	IQ = Q3 - Q1
	lower_fence = Q1 - 1.5 * IQ
	upper_fence = Q3 + 1.5 * IQ
    return df[(df[column_name] <= upper_fence) & (df[column_name] >= lower_fence)]

def basic_data_analysis(df):

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    """
    Performs a basic data analysis of a Pandas Dataframe
    Args:df (pd.DataFrame)
    Returns:None
    """
    # Explore the first records
    print("Primeras filas:")
    print(df.head())

    # Data cleaning
    df.dropna(inplace=True)  # Remove rows with null values
    df = df.astype({"columna_numerica": float})  # Fix data types

    # Visualization
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(df["columna_numerica"], bins=20, kde=True)
    plt.title("Histograma de la columna_numerica")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.show()

    # Statistic analysis
    print("\nEstadísticas descriptivas:")
    print(df.describe())

    # Correlation
    print("\nMatriz de correlación:")
    print(df.corr())

    # Linear regression (Example)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["columna_x"], df["columna_y"])
    print(f"\nRegresión lineal: Pendiente={slope:.2f}, Intercepto={intercept:.2f}, R^2={r_value**2:.2f}")


def outlier_meanSd(df, feature, param=3):  

    """"
    This function removes outliers and null values from a pandas DataFrame by:
        1.Calculate the mean (media) and standard deviation (desEst) of the specified feature (feature) in the DataFrame df.
        2.Define two thresholds (th1 and th2) using the mean and standard deviation multiplied by a parameter (param). These thresholds are used to identify outliers.
        3.Filter the original DataFrame (df) based on the following conditions:
            a.Values must fall within the range [th1, th2].
            b.Include any null (NaN) values.
        4.Finally, return a new DataFrame with the filtered values.
    """
    media = df[feature].mean()
    desEst = df[feature].std()

    th1 = media - desEst*param
    th2 = media + desEst*param

    return df[((df[feature] >= th1) & (df[feature] <= th2))  | (df[feature].isnull())].reset_index(drop=True)


def data_report(df):

    """
    This function generates a comprehensive report for a DataFrame using the pandas library.  It provides a detailed overview of statistics and features for the input DataFrame. This is useful for efficient data exploration and analysis. 
        1.Column Names (`COL_N`): Creates a DataFrame called `cols` with the column names from the input DataFrame `df`.
        2.Data Types (`DATA_TYPE`): Creates another DataFrame called `types` with the data types of the columns in `df`.
        3.Missing Values (`MISSINGS (%)`): Calculates the percentage of missing values (NaN) in each column and creates a DataFrame called `percent_missing_df`.
        4.Unique Values (`UNIQUE_VALUES`): Calculates the number of unique values in each column and creates a DataFrame called `unicos`.
        5.Cardinality (`CARDIN (%)`): Computes the percentage of cardinality (number of unique values relative to the DataFrame size) and creates a DataFrame called `percent_cardin_df`.
        6.Concatenation and Transposition: Combines all the above DataFrames into one called `concatenado`. Then, it transposes this DataFrame so that columns become indices and vice versa.
    """

    import pandas as pd
    # Get the NAMES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Get the TYPES
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Get the MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Get the UNIQUE VALUES
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])

    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)

    return concatenado.T


def missing_values_summary(df):
    """
    Generates a summary of missing values in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to analyze.

    Returns:
    DataFrame: A DataFrame with the count and percentage of missing values per column.
    """
    missing_summary = df.isnull().sum().to_frame(name='Missing Values')
    missing_summary['Percentage'] = (missing_summary['Missing Values'] / len(df)) * 100
    return missing_summary

    