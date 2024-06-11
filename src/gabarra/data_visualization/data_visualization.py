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