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