import matplotlib.pyplot as plt
import seaborn as sns

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
    
def plot_numeric_distributions(df, hue=None):
    """
    Plots histograms and boxplots for all numeric columns in the DataFrame.
    If a categorical column is specified as hue, the plots will be differentiated by this column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    hue (str, optional): The name of the categorical column used for differentiation. Default is None.

    Returns:
    None
    """
    # Filter numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Iterate over each numeric column
    for col in numeric_columns:
        if col != hue:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

            # Histogram
            sns.histplot(data=df, x=col, hue=hue, kde=True, element='step', stat='density', ax=axes[0])
            axes[0].set_title(f'Histogram of {col}' + (f' differentiated by {hue}' if hue else ''))

            # Boxplot
            sns.boxplot(data=df, x=hue, y=col, ax=axes[1])
            axes[1].set_title(f'Boxplot of {col}' + (f' differentiated by {hue}' if hue else ''))

            plt.tight_layout()
            plt.show()



import matplotlib.cm as cm

def get_viridis_colors(n_bins):
    """
    Obtiene una lista de colores del colormap viridis.

    :param n_bins: Número de bins (categorías) en el gráfico.
    :return: Lista de colores en formato hexadecimal.
    """
    cmap = cm.get_cmap('viridis', n_bins)
    return [cmap(i) for i in range(n_bins)]

def plot_pie_charts(df, columns):
    """
    Crea gráficos de pastel para las columnas especificadas en el DataFrame.

    :param df: DataFrame que contiene los datos.
    :param columns: Lista de columnas para las que se crearán los gráficos de pastel.
    """
    for column in columns:
        # Verificar si la columna existe en el DataFrame
        if column not in df.columns:
            print(f"Columna {column} no encontrada en el DataFrame.")
            continue

        # Número de categorías en la columna
        n_bins = df[column].nunique()

        # Obtener colores viridis
        colors = get_viridis_colors(n_bins)

        # Agrupar por la columna y sumar las órdenes
        data = df.groupby([column]).num_orders.sum()

        # Crear el gráfico de pastel
        plt.figure(figsize=(6, 6))
        plt.pie(data,
                labels=data.index,
                shadow=False,
                colors=colors,
                explode=[0.05] * n_bins,
                startangle=90,
                autopct='%1.1f%%', pctdistance=0.9,
                textprops={'fontsize': 8})
        plt.title(f"% de pedidos por {column}")
        plt.show()

# Interactive line chart

import plotly.express as px

def plot_interactive_line_chart(df, x_column, y_column, color_column=None):
    """
    Creates an interactive line chart for the specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x_column (str): The name of the column to be used for the x-axis.
    y_column (str): The name of the column to be used for the y-axis.
    color_column (str, optional): The name of the column used for color differentiation. Default is None.

    Returns:
    None
    """
    # Check if the columns exist in the DataFrame
    if x_column not in df.columns:
        print(f"Column {x_column} not found in the DataFrame.")
        return
    if y_column not in df.columns:
        print(f"Column {y_column} not found in the DataFrame.")
        return
    if color_column and color_column not in df.columns:
        print(f"Column {color_column} not found in the DataFrame.")
        return

    # Create the line chart
    fig = px.line(df, x=x_column, y=y_column, color=color_column, title=f'{y_column} over {x_column}')

    # Update the layout for better appearance
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else 'Legend',
        hovermode='x unified'
    )

    fig.show()

# Interactive pie chart

import plotly.express as px
import plotly.graph_objects as go

def plot_interactive_pie_chart(df, column):
    """
    Creates an interactive pie chart for the specified column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column for which the pie chart will be created.

    Returns:
    None
    """
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        print(f"Column {column} not found in the DataFrame.")
        return

    # Aggregate the data
    data = df[column].value_counts().reset_index()
    data.columns = [column, 'counts']

    # Create the pie chart
    fig = px.pie(data, values='counts', names=column, title=f'Percentage of Orders by {column}')

    # Update the layout for better appearance
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

    fig.show()