
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def create_dummies(df):
    """
    This function takes a DataFrame and creates dummy variables for all object columns.
    The resulting DataFrame includes the dummy variables along with the original numeric variables.

    Parameters:pip
    df (pd.DataFrame): The input DataFrame

    Returns:
    pd.DataFrame: The DataFrame with dummy variables and numeric columns
    """
    
    object_cols = df.select_dtypes(include=['object']).columns
    numeric_df = df.select_dtypes(exclude=['object'])

    
    dummies_df = pd.get_dummies(df[object_cols], drop_first=False) 

    
    final_df = pd.concat([numeric_df, dummies_df], axis=1)

    return final_df



def fill_zeros_with_mean(df, column):
    """
    Fills zero values in a specified column of a DataFrame with the column's mean, excluding zeros.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be imputed.
        column (str): The name of the column containing zero values.

    Returns:
        pandas.DataFrame: The modified DataFrame with zeros replaced by the mean.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.

    Warns:
        UserWarning: If there are no non-zero values in the column, a warning is issued
            indicating that the mean cannot be calculated and the column remains unchanged.


    """
    
    mean_value = df[df[column] != 0][column].mean()
    df[column] = df[column].replace(0, mean_value)

    return df




def fill_nans_with_mean(df, column):
    """
     Fills NaN (Not a Number) values in a specified column of a DataFrame with the column's mean, excluding NaNs.

     Args:
        df (pandas.DataFrame): The DataFrame containing the column to be imputed.
        column (str): The name of the column containing NaN values.

     Returns:
        pandas.DataFrame: The modified DataFrame with NaN values replaced by the mean.

     Raises:
        ValueError: If the specified column does not exist in the DataFrame.

    """
    mean_value = df[column].mean(skipna=True)

    df[column] = df[column].fillna(mean_value)
        
    return df


def convert_to_numeric(df, motive='LabelEncoding', columns=list):
    """
    Converts categorical variables in a pandas DataFrame to numerical features.

    Args:
        df (pd.DataFrame): The DataFrame containing the categorical variables.
        motive (str, optional): The type of encoding to apply. Defaults to 'LabelEncoding'.
            - 'LabelEncoding': Assigns a unique integer to each category.
            - 'OneHotEncoding': Creates binary features for each category.
            - 'FrequencyEncoding': Assigns a value based on category frequency.
        columns (list): The columns to be converted. Defaults to all categorical columns.

    Returns:
        pd.DataFrame: The DataFrame with converted categorical columns.

    Raises:
        ValueError: If an invalid 'motive' is provided.
    """

    if motive not in ['LabelEncoding', 'OneHotEncoding', 'FrequencyEncoding']:
        raise ValueError(f"Invalid motive: '{motive}'. Valid options are 'LabelEncoding', 'OneHotEncoding', and 'FrequencyEncoding'.")

    if not columns:
        columns = df.select_dtypes(include=['category', 'object']).columns

    for col in columns:
        if motive == 'LabelEncoding':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
        elif motive == 'OneHotEncoding':
            encoder = OneHotEncoder(sparse=False)  
            encoded_df = pd.DataFrame(encoder.fit_transform(df[[col]]), columns=[f'{col}_{c}' for c in encoder.categories_[0]])
            df = pd.concat([df, encoded_df], axis=1).drop(col, axis=1)
        elif motive == 'FrequencyEncoding':
            category_counts = df[col].value_counts().to_dict()
            df[col] = df[col].replace(category_counts)

    return df

