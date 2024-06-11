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