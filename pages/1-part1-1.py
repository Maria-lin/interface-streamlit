import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter
from pprint import pprint
from matplotlib import pyplot as plt
import seaborn as sns
from enum import Enum
from pathlib import Path
import altair as alt
import math
st.set_page_config(page_title="Data Exploration Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("Data Exploration Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
alt.themes.enable("dark")




fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
   
    df = pd.read_csv("data/Dataset2.csv", encoding = "ISO-8859-1")
    st.subheader("Loaded Dataset")

col1, col2 = st.columns((2))
#col = st.columns((8.5,2), gap='medium')
_, view4, dwn4 = st.columns([0.5,0.45,0.45])

with col1:
   
   st.dataframe(df.head())

with col2:
    with st.expander('About', expanded=False):
        st.write('''
            - Data: [U.S. Census Bureau](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html).
            - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
            - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
            ''')
    result2 = df.head()
    st.download_button("Get Data", data=result2.to_csv().encode("utf-8"),
                       file_name="Dataset1.csv", mime="text/csv")


df_numeric = df.apply(pd.to_numeric, errors='coerce')

df_filled = df_numeric.fillna(0)

correlation_matrix = df_filled.corr()

col1, col2 = st.columns((2))

with col1: 
    st.subheader("Correlation Matrix Heatmap")
    plt.figure(figsize=(10, 4))  
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot(plt)

with col2: 
   fig, ax = plt.subplots(figsize=(10, 4))
   st.subheader('Distribution of pH')
   sns.histplot(df['pH'], kde=True, ax=ax)
   ax.set_xlabel('pH Value')
   ax.set_ylabel('Frequency')
   st.pyplot(fig)

    
def check_weird_values(data):
    for col in data.columns:
        try:
            data[col] = data[col].astype(float)
        except ValueError as err:
            print(f'could not convert data on column "{col}" with error {err}')
check_weird_values(df)
def check_duplicated(data: pd.DataFrame):
    if num_duplicated := sum(data.duplicated()):
        print(f'df has {num_duplicated} duplicated rows')
    else:
        print('data frame has no duplicated rows')
check_duplicated(df)
st.divider()
st.subheader('Number of :blue[ empty values ] in each column:')
col1, col2 = st.columns((2))

with col1:

    missing_values_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum()
    })

    with st.expander("Missing Values Summary", expanded=False):

        formatted_missing_values_df = missing_values_df.copy()
        formatted_missing_values_df['Missing Values'] = formatted_missing_values_df['Missing Values'].apply(lambda x: f'{x:.0f}')

        st.dataframe(
            formatted_missing_values_df,
            width=None,
            height=None,
            hide_index=True
        )
with col2:
    
    with st.expander("Visualize Missing Values"):
        
        missing_values_df = pd.DataFrame({
            'Attribute': df.columns,
            'Missing Values': df.isnull().sum()
        })

      
        fig = px.bar(missing_values_df, x='Attribute', y='Missing Values',
                     labels={'Missing Values': 'Number of Missing Values'},
                     title='Missing Values in Each Attribute',
                     width=400, 
                     height=400)
        st.plotly_chart(fig)


st.divider()




def remove_rows_with_errors(input_df: pd.DataFrame) -> pd.DataFrame:
    error_rows = []

    for col in input_df.columns:
        try:
            input_df[col] = input_df[col].astype(float)
        except ValueError as e:
            print(f'could not convert data on column "{col}" with error {e}')
            error_rows.extend(input_df[col][pd.to_numeric(input_df[col], errors='coerce').isna()].index.tolist())

    error_rows = np.unique(error_rows)
    df_cleaned = df.drop(index=error_rows)
    print(f'removed rows are : {error_rows}')

    return df_cleaned


def remove_duplicates_from_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    seen_rows = set()
    output_rows = []

    for index, row in input_df.iterrows():
        row_tuple = tuple(row)
        if row_tuple not in seen_rows:
            seen_rows.add(row_tuple)
            output_rows.append(row)

    output_df = pd.DataFrame(output_rows, columns=input_df.columns)
    return output_df

def remove_rows_with_missing_values(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = input_df[~input_df.isna().any(axis=1)]
    return output_df

def clean_df(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df = remove_rows_with_errors(input_df)
    input_df = remove_duplicates_from_dataframe(input_df)
    input_df = remove_rows_with_missing_values(input_df)
    return input_df
df = clean_df(df)
check_duplicated(df)
check_weird_values(df)

def global_describe(input_data):
    """
    Provide a simple global description for a pandas DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary containing the number of rows, number of columns, and data type of each column.
    """
    num_rows = len(input_data)
    num_columns = len(input_data.columns)
    column_types = input_data.dtypes.to_dict()

    global_desc = {
        'num_rows': num_rows,
        'num_columns': num_columns,
        'column_types': column_types,
    }

    return global_desc
all_columns = global_describe(df)



def custom_describe(input_df: pd.DataFrame):
    result = {}

    for column in input_df.columns:
        sorted_values = sorted(input_df[column].tolist())
        # Maximum
        max_val = sorted_values[-1]

        # Minimum
        min_val = sorted_values[0]

        # Mean
        mean = sum(sorted_values) / len(sorted_values)

        # Mode
        counter = Counter(input_df[column])
        mode = counter.most_common(1)[0][0]

        # Median
        n = len(sorted_values)
        if n % 2 == 0:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            median = sorted_values[n // 2]

        # Standard Deviation
        std_val = (sum((x - mean) ** 2 for x in sorted_values) / len(sorted_values)) ** 0.5

        # Quantiles
        quantiles = {
            f'{column}_Q0': min_val,
            f'{column}_Q1': sorted_values[int(0.25 * n)],
            f'{column}_Q2': median,
            f'{column}_Q3': sorted_values[int(0.75 * n)],
            f'{column}_Q4': max_val
        }

        result[column] = {
            'max': max_val,
            'min': min_val,
            'mean': mean,
            'mode': mode,
            'median': median,
            'std': std_val,
            'quantiles': quantiles
        }

    return result

result_custom = custom_describe(df)
result_default = df.describe().transpose()
custom_stats = {}
for column, stats in result_custom.items():
    if column not in result_default.index:
        custom_stats[column] = stats
with col1:
     with st.expander("Data description", expanded=False):
        st.write(all_columns)   

with col2:
     with st.expander("Combined Describe Result", expanded=False):
      st.subheader("Default Statistics (df.describe()):")
      st.table(result_default)

      st.subheader("Custom Statistics:")
      for column, stats in custom_stats.items():
          st.write(f"Column: {column}")
          st.table(pd.DataFrame(stats).transpose())





st.subheader("Data :blue[Visualization]") 

class PlotType(Enum):
    BOX = 'Box Plot'
    HIST = 'Histogram'
    SCATTER = 'Scatter Plots'
    
def plottest(input_df, plot_types) -> None:
    for plot_type in plot_types:
        num_cols = len(input_df.columns)
        num_rows = (num_cols + 1) // 7
        fig, axes = plt.subplots(num_rows, 7, figsize=(15, num_rows * 4))
        fig.suptitle(f'{plot_type.value} of Data', y=1.02)

        axes = axes.flatten()

        for i, column in enumerate(input_df.columns):
            match plot_type:
                case PlotType.BOX:
                    sns.boxplot(y=input_df[column], ax=axes[i])
                case PlotType.HIST:
                    sns.histplot(input_df[column], ax=axes[i], kde=True)
                case PlotType.SCATTER:
                    sns.scatterplot(input_df[column], ax=axes[i])
            axes[i].set_title(column)

        plt.tight_layout()
        st.pyplot(fig)

selected_plots = st.multiselect('Select Plot Types', [plot_type.value for plot_type in PlotType])
selected_plots = [PlotType(plot_type) for plot_type in selected_plots]
plottest(df, selected_plots)

def plot(input_df, *, plot_type: PlotType) -> None:
    num_cols = len(input_df.columns)
    num_rows = (num_cols + 1) // 7
    fig, axes = plt.subplots(num_rows, 7, figsize=(15, num_rows * 4))
    fig.suptitle(f'{plot_type.value} of Data', y=1.02)

    axes = axes.flatten()

    for i, column in enumerate(input_df.columns):
        match plot_type:
            case PlotType.BOX:
                sns.boxplot(y=input_df[column], ax=axes[i])
            case PlotType.HIST:
                sns.histplot(input_df[column], ax=axes[i], kde=True)
            case PlotType.SCATTER:
                sns.scatterplot(input_df[column], ax=axes[i])
        axes[i].set_title(column)

    plt.tight_layout()

    st.pyplot(fig)
form = st.form(key="my_form_2")

with form:
    all_columns_option = 'all'
    columns_with_all = df.columns.insert(0, all_columns_option)
    selected_columns = st.multiselect('Select Columns', columns_with_all, default=[all_columns_option])


    if all_columns_option in selected_columns:
         selected_columns = df.columns
     
    selected_plot_type = st.selectbox('Select Plot Type', [plot_type.value for plot_type in PlotType])
button_pressed = form.form_submit_button("Generate Plot")

 
df_selected = df[selected_columns]

if button_pressed:
    st.subheader(f'Plot of {", ".join(selected_columns)}')

    if selected_plot_type == PlotType.BOX.value:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_selected, ax=ax)
        st.pyplot(fig)
    elif selected_plot_type == PlotType.HIST.value:
        fig, ax = plt.subplots()
        sns.histplot(data=df_selected, kde=True, ax=ax)
        st.pyplot(fig)
    elif selected_plot_type == PlotType.SCATTER.value:
        fig = sns.pairplot(df_selected)
        st.pyplot(fig)

def custom_describe(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.describe()


def has_outliers(input_df: pd.DataFrame, *, threshold: float = 1.5) -> bool:
    data_description = custom_describe(input_df)
    for column in input_df.columns:
        values = input_df[column]
        q1 = data_description.loc['25%', column]
        q3 = data_description.loc['75%', column]
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        column_outliers = values[(values < lower_bound) | (values > upper_bound)].index.to_list()

        if column_outliers:
            return True

    return False


def detect_and_treat_outliers(dataframe: pd.DataFrame, *, threshold: float = 1.5, show: bool = False) -> pd.DataFrame:
    data_description = dataframe.describe()
    for column in dataframe.columns:
        values = dataframe[column]
        q1 = data_description.loc['25%', column]
        q3 = data_description.loc['75%', column]
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        column_outliers = values[(values < lower_bound) | (values > upper_bound)].index.to_list()

        if show and column_outliers:
            print(f"Outliers in column '{column}': {column_outliers}")

        values[column_outliers] = values.mean()
        dataframe[column] = values
    return dataframe

def treat_outliers(input_df : pd.DataFrame, show : bool = False) -> pd.DataFrame:
    while True:
        df_without_outliers = detect_and_treat_outliers(input_df, show=show)
        if not has_outliers(df_without_outliers):
            return df_without_outliers
        input_df = df_without_outliers

df = treat_outliers(df)

st.subheader("after pretretement ")
form = st.form(key="my_form_after")

with form:
    all_columns_option = 'all'
    columns_with_all = df.columns.insert(0, all_columns_option)
    selected_columns = st.multiselect('Select Columns', columns_with_all, default=[all_columns_option])


    if all_columns_option in selected_columns:
         selected_columns = df.columns
     
    selected_plot_type = st.selectbox('Select Plot Type', [plot_type.value for plot_type in PlotType])
button_pressed = form.form_submit_button("Generate Plot")

 
df_selected = df[selected_columns]

if button_pressed:
    st.subheader(f'Plot of {", ".join(selected_columns)}')

    if selected_plot_type == PlotType.BOX.value:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_selected, ax=ax)
        st.pyplot(fig)
    elif selected_plot_type == PlotType.HIST.value:
        fig, ax = plt.subplots()
        sns.histplot(data=df_selected, kde=True, ax=ax)
        st.pyplot(fig)
    elif selected_plot_type == PlotType.SCATTER.value:
        fig = sns.pairplot(df_selected)
        st.pyplot(fig)

      
def min_max_scaling(input_df):
    copy_df = input_df.copy()
    min_values = [input_df[column].min() for column in input_df.columns]
    max_values = [input_df[column].max() for column in input_df.columns]
    diff = [max_values[i] - min_values[i] for i in range(len(max_values))]
    copy_df = (copy_df - min_values) / diff
    return copy_df

# Fonction de normalisation Z-score
def z_score_normalization(input_df):
    copy_df = input_df.copy()
    means = [input_df[column].mean() for column in input_df.columns]
    stds = [input_df[column].std() for column in input_df.columns]
    copy_df = (copy_df - means) / stds
    return copy_df

# Fonction pour le tracé des graphiques
def plot_chart(df, plot_type, selected_columns):
    num_cols = len(selected_columns)
    fig, axes = plt.subplots(num_cols, 1, figsize=(8, 4 * num_cols))

    axes = np.ravel(axes)
    for i, column in enumerate(selected_columns):
        ax = axes[i]
        if plot_type == 'Histogram':
            sns.histplot(df[column], kde=True, ax=ax)
            ax.set_title(f'Histogram of {column}')
        elif plot_type == 'Scatter Plot':
            sns.scatterplot(x=df.index, y=df[column], ax=ax)
            ax.set_title(f'Scatter Plot of {column}')
        elif plot_type == 'Box Plot':
            sns.boxplot(x=df[column], ax=ax)
            ax.set_title(f'Box Plot of {column}')

    plt.tight_layout()
    st.pyplot(fig)
    
st.subheader("Data Normalization and Visualization")


with st.form("my_form"):
    normalization_method = st.radio("Choose normalization method:", ('Min-Max Scaling', 'Z-score Normalization'))
    plot_type = st.selectbox("Choose plot type:", ('Histogram', 'Scatter Plot', 'Box Plot'))

  
    selected_columns = st.multiselect("Select columns:", df.columns)

    submit_button = st.form_submit_button(label='Generate')

if submit_button:
    if normalization_method == 'Min-Max Scaling':
        df_normalized = min_max_scaling(df)
        pprint(df)
    else:
        df_normalized = z_score_normalization(df)

    if selected_columns:
        plot_chart(df_normalized[selected_columns], plot_type, selected_columns)
    else:
        st.warning("Please select at least one column.")
        
        
        

