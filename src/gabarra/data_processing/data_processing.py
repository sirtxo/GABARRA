
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