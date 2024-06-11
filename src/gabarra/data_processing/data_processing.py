
import pandas as pd

def create_dummies(df):
    """
    This function takes a DataFrame and creates dummy variables for all object columns.
    The resulting DataFrame includes the dummy variables along with the original numeric variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame

    Returns:
    pd.DataFrame: The DataFrame with dummy variables and numeric columns
    """
    # Separate the object columns and numeric columns
    object_cols = df.select_dtypes(include=['object']).columns
    numeric_df = df.select_dtypes(exclude=['object'])

    # Create dummy variables for object columns
    dummies_df = pd.get_dummies(df[object_cols], drop_first=True)

    # Concatenate the numeric columns and dummy variables
    final_df = pd.concat([numeric_df, dummies_df], axis=1)

    return final_df


# Replace zeros with the mean of non-zero values in a specified column of a DataFrame
def fill_zeros_with_mean(df, column):
    # Calculate the mean of the non-zero values in the specified column
    mean_value = df[df[column] != 0][column].mean()
    
    # Replace zeros in the specified column with the calculated mean value
    df[column] = df[column].replace(0, mean_value)
    
    return df



# Replace NaN values with the mean of the specified column in a DataFrame
def fill_nans_with_mean(df, column):
    # Calculate the mean of the specified column, ignoring NaN values
    mean_value = df[column].mean(skipna=True)
    
    # Replace NaN values in the specified column with the calculated mean value
    df[column] = df[column].fillna(mean_value)
    
    return df