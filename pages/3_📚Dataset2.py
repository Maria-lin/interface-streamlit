import base64
import altair as alt
import pandas as pd
import seaborn as sns
from datetime import datetime
from enum import Enum
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px


st.set_page_config(page_title="Superstore!!!",
                   page_icon=":bar_chart:", layout="wide")

st.title(" :bar_chart:  EDA")
st.markdown(
    '<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_local('data\huh.png')


df = pd.read_csv("data/Dataset2.csv", encoding="ISO-8859-1")
st.subheader("Loaded Dataset")

col1, col2 = st.columns((2))
# col = st.columns((8.5,2), gap='medium')
_, view4, dwn4 = st.columns([0.5, 0.45, 0.45])

with col1:
    st.dataframe(df)

with col2:

    with st.expander("Data overview", expanded=True):
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

    with st.expander('Remarque', expanded=True):
        st.write('''
            - :orange[**Missing/Values**]: we have some missing values in the columns 'test count', 'case count' and 'positive tests'
            - :orange[**type columns**]: we have two columns with type object that represents dates, we should transform them into time series
            ''')

    st.download_button("Get Data", data=df.to_csv().encode("utf-8"),
                       file_name="Dataset2.csv", mime="text/csv")


df.dropna(inplace=True)


def float_to_int(input_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        input_df[column] = input_df[column].astype(int)


float_to_int(df, ['test count', 'case count', 'positive tests'])


st.divider()
st.subheader('Start and end :blue[ dates ]:')
col1, col2 = st.columns((2))

with col1:

    missing_values_df = df['Start date'].unique()

    with st.expander("start date ", expanded=False):
        # Format the DataFrame for better display
        formatted_missing_values_df = missing_values_df.copy()
        st.dataframe(
            formatted_missing_values_df,
            width=None,
            height=None,
            hide_index=True
        )


with col2:

    missing_values_df = df['end date'].unique()

    with st.expander("End date ", expanded=False):

        formatted_missing_values_df = missing_values_df.copy()
        st.dataframe(
            formatted_missing_values_df,
            width=None,
            height=None,
            hide_index=True
        )

st.write(
    "we notice that :orange[**we have different format of dates**], we should fix that")


st.divider()
st.subheader('Scatter Plot of Case Count and Start Date Year')

df_data = df.copy()
df_data['end date'] = pd.to_datetime(df_data['end date'], errors='coerce')
df_data['year'] = df_data['end date'].dt.year
df_data = df_data.dropna(subset=['year'])
df_data['year'] = df_data['year'].astype(int)
df_data = df_data.sort_values('year')

col1, col2 = st.columns((2))  # Ici, nous créons deux colonnes de largeur égale

# Utilisation de la première colonne pour le graphique de dispersion
with col1:
    st.vega_lite_chart(df_data, {
        'mark': {
            'type': 'circle',
            'tooltip': True,
            'size': 10  # Ajuster la taille des points ici, la valeur est arbitraire
        },
        'encoding': {
            'x': {'field': 'time_period', 'type': 'quantitative', 'title': 'Case Count'},
            'y': {
                'field': 'year',
                'type': 'ordinal',
                'title': 'Year',
                'sort': 'ascending'  # Assurez-vous que les années sont triées en ordre croissant
            }
        },
        'width': 300,  # Ajuster la largeur du graphique
        'height': 300,  # Ajuster la hauteur du graphique
        'title': 'Scatter Plot of Case Count and End Date Year',
    })


def get_max_min_period_by_year(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.groupby(pd.to_datetime(input_df['end date'], errors='coerce').dt.year).agg({'time_period': ['min', 'max']}).reset_index()


result_df = get_max_min_period_by_year(df_data)

with col2:

    with st.expander('What we noticed', expanded=True):
        st.write('''
                 
            We can notice that :blue[the year of the start date is related to the time period], so we can fix the dates by adding the current year to the date corresponding to the time period
            - 0-20 days => :orange[**2019**]
            - 20-35 days => :orange[**2020**]
            - 35-53 days => :orange[**2021**]
            - 53-155 days => :orange[**2022**]
            ''')


def transform_date(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix dates that has different formats (e.g. 1-Jan-20, 1-Jan-2020, 1/1/2020) or has no year (e.g. 1-Jan)
    :param input_df:
    :return:
    """
    def time_period_to_year(time_period: int) -> int:
        if time_period <= 20:
            return 2019
        elif 20 < time_period <= 35:
            return 2020
        elif 35 < time_period <= 53:
            return 2021
        elif 53 < time_period:
            return 2022

    for index, row in input_df.iterrows():
        try:
            start_date = datetime.strptime(row['Start date'], '%m/%d/%Y')
            end_date = datetime.strptime(row['end date'], '%m/%d/%Y')
        except ValueError:
            start_date = datetime.strptime(row['Start date'], '%d-%b')
            end_date = datetime.strptime(row['end date'], '%d-%b')
            new_year = time_period_to_year(row['time_period'])
            start_date = start_date.replace(year=new_year)
            end_date = end_date.replace(
                year=new_year) if end_date.month >= start_date.month else end_date.replace(year=new_year + 1)

        input_df.at[index, 'Start date'] = start_date
        input_df.at[index, 'end date'] = end_date

    input_df['Start date'] = pd.to_datetime(input_df['Start date'])
    input_df['end date'] = pd.to_datetime(input_df['end date'])
    return input_df


st.divider()
st.subheader("DataFrame transformé")
col1, col2 = st.columns((2))

with col1:

    transformed_df = transform_date(df)

    with st.expander("View DataFrame transformé", expanded=True):
        st.dataframe(transformed_df)
with col2:
    with st.expander('Remarque', expanded=True):
        st.write('''
            - :orange[**Missing/Values**]: we have some missing values in the columns 'test count', 'case count' and 'positive tests'
            - :orange[**type columns**]: we have two columns with type object that represents dates, we should transform them into time series
            ''')

st.divider()


def treat_data(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df.drop_duplicates(inplace=True)
    input_df = input_df[input_df['test count'] >= input_df['positive tests']]
    input_df = input_df[input_df['Start date'] <= input_df['end date']]
    return input_df


def regroup_data_by_date(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.groupby(['Start date', 'end date']).agg({'test count': 'sum', 'case count': 'sum', 'positive tests': 'sum'}).reset_index()


df = treat_data(df)
df_by_date = regroup_data_by_date(df)

st.subheader('Data Grouped by Date')

st.dataframe(df_by_date)


class PlotType(Enum):
    LINE = 'line plot'
    BOX = 'box plot'


def plot(input_df, *, plot_type: PlotType) -> None:
    if plot_type == PlotType.LINE:
        fig, ax = plt.subplots()
        sns.lineplot(x=input_df['Start date'],
                     y=input_df['test count'], label='Tests', ax=ax)
        sns.lineplot(
            x=input_df['Start date'], y=input_df['case count'], label='Confirmed Cases', ax=ax)
        sns.lineplot(
            x=input_df['Start date'], y=input_df['positive tests'], label='Positive Tests', ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel('Time')
        plt.ylabel('Count')

    elif plot_type == PlotType.BOX:
        fig, ax = plt.subplots()
        sns.boxplot(
            data=input_df[['test count', 'case count', 'positive tests']], ax=ax)
        plt.ylabel('Count')

    else:
        raise ValueError('Invalid plot type')

    plt.title('Evolution of COVID-19 Tests and Cases Over Time')
    plt.legend()

    # Afficher le plot dans Streamlit
    st.pyplot(fig)


with st.form(key='my_form'):
    # Afficher le sous-titre
    st.subheader("Evolution of COVID-19 Tests and Cases Over Time")

    selected_plot_type = st.radio("Select Plot Type:", [
                                  PlotType.LINE, PlotType.BOX], format_func=lambda x: x.value)

    submitted = st.form_submit_button("Generate Plot")
    if submitted:
        with st.expander("Evolution of COVID-19 Tests and Cases Over Time"):
            plot(df_by_date, plot_type=selected_plot_type)


def regroup_data_by_zone(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.groupby(['zcta']).agg({'case count': 'sum', 'positive tests': 'sum'}).reset_index()


df_by_zone = regroup_data_by_zone(df)


melted_df = pd.melt(df_by_zone, id_vars='zcta',
                    var_name='variable', value_name='value')

chart = px.bar(melted_df, x='zcta', y='value', color='variable',
               barmode='group', title='Distribution of COVID-19 Tests and Cases by Zone')
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader('Distribution of COVID-19 Tests and Cases by Zone')
    st.plotly_chart(chart, use_container_width=True)
with col2:

    st.markdown("""
        ### Explication
    
        Voici un catplot qui montre la distribution des tests et des cas de COVID-19 par zone.
    
        - **Axe X:** Zones
        - **Axe Y:** Nombre (Count)
        - **Légende:** Variable (Tests, Cases, etc.)
    
        Le graphique présente une vue d'ensemble des données par zone.
        
    """)
    with st.expander("View DataFrame (df_by_zone)"):
        st.write(df_by_zone)


st.subheader("Positive Tests Over Time")
chart_data = pd.DataFrame({
    "positive tests": df["positive tests"],
    "population": df["population"],
    "test count": df["test count"]

})

st.line_chart(chart_data[["test count", "positive tests", "population"]])


def get_random_zone_df(input_df: pd.DataFrame, zone: int = None) -> pd.DataFrame:
    if zone is None:
        zone = input_df['zcta'].sample(1).values[0]
    df_random_zone = input_df[input_df['zcta'] == zone]
    return df_random_zone


def regroup_data_by_year(input_df: pd.DataFrame, zone=None) -> pd.DataFrame:
    df_random_zone = get_random_zone_df(input_df, zone)
    return df_random_zone.groupby([df['Start date'].dt.year, 'zcta']).agg({'test count': 'sum', 'case count': 'sum', 'positive tests': 'sum'}).reset_index()


def regroup_data_by_month(input_df: pd.DataFrame, zone=None) -> pd.DataFrame:
    df_random_zone = get_random_zone_df(input_df, zone)
    input_df = df_random_zone.resample('M', on='Start date').agg(
        {'zcta': 'first', 'test count': 'sum', 'case count': 'sum', 'positive tests': 'sum'}).reset_index()
    return input_df[input_df['test count'] != 0]


def regroup_data_by_week(input_df: pd.DataFrame, zone=None) -> pd.DataFrame:
    df_random_zone = get_random_zone_df(input_df, zone)
    input_df = df_random_zone.resample('W-Mon', on='Start date').agg(
        {'zcta': 'first', 'test count': 'sum', 'case count': 'sum', 'positive tests': 'sum'}).reset_index()
    return input_df[input_df['test count'] != 0]


df_year = regroup_data_by_year(df, 95129)
df_week = regroup_data_by_week(df, 95129)
df_month = regroup_data_by_month(df, 95129)


def plot_zone_data(input_df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figsize as needed
    sns.lineplot(x=input_df['Start date'],
                 y=input_df['test count'], label='Tests', ax=ax)
    sns.lineplot(x=input_df['Start date'],
                 y=input_df['case count'], label='Confirmed Cases', ax=ax)
    sns.lineplot(x=input_df['Start date'],
                 y=input_df['positive tests'], label='Positive Tests', ax=ax)

    plt.xlabel('Period')
    plt.ylabel('Rate')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.legend()

    return fig


tabs = st.tabs(["Yearly Data", "Monthly Data", "Weekly Data"])

with tabs[0]:
    st.header("Yearly Data")
    expander_year = st.expander("Plot Details (Yearly)")
    with expander_year:
        st.write("Explanation for Yearly Data")
        fig_year = plot_zone_data(
            df_year, f'Case and Positive Rates for {df_year["zcta"].values[0]}')
        st.pyplot(fig_year)


with tabs[1]:
    st.header("Monthly Data")
    expander_month = st.expander("Plot Details (Monthly)")
    with expander_month:
        st.write("Explanation for Monthly Data")
        fig_month = plot_zone_data(
            df_month, f'Case and Positive Rates for {df_month["zcta"].values[0]}')
        st.pyplot(fig_month)


with tabs[2]:
    st.header("Weekly Data")
    expander_week = st.expander("Plot Details (Weekly)")
    with expander_week:
        st.write("Explanation for Weekly Data")
        fig_week = plot_zone_data(
            df_week, f'Case and Positive Rates for {df_week["zcta"].values[0]}')
        st.pyplot(fig_week)

st.divider()
st.subheader("Positive Tests by Zone and Year")


def group_by_year_zone(input_df):
    return input_df.groupby([input_df['Start date'].dt.year, 'zcta']).agg({'positive tests': 'sum'}).reset_index()


df_year_zone = group_by_year_zone(df)

pivot = df_year_zone.pivot(
    index='zcta', columns='Start date', values='positive tests')

pivot_long = pivot.reset_index().melt(
    id_vars='zcta', var_name='Start date', value_name='positive tests')

chart = alt.Chart(pivot_long).mark_bar().encode(
    x='zcta:N',
    y='positive tests:Q',
    color='Start date:N',
    tooltip=['zcta', 'Start date', 'positive tests']
).properties(
    width=alt.Step(20)
).configure_axis(
    labelAngle=45
)

st.altair_chart(chart, use_container_width=True)


def population_by_zone(input_df: pd.DataFrame):
    return input_df.groupby(['zcta']).agg({'population': 'first', 'test count': 'sum'}).reset_index()


df_population = population_by_zone(df)
st.divider()
st.subheader('Population Test Count by Zone')

chart = alt.Chart(df_population).mark_bar().encode(
    x='zcta:N',
    y='test count:Q',
    tooltip=['zcta:N', 'test count:Q']
).properties(
    width=600,
    height=400,
    title='Population Test Count by Zone'
)

st.altair_chart(chart, use_container_width=True)

st.subheader('Scatter Plot of Population and Tests')

chart = alt.Chart(df_population).mark_circle().encode(
    x='population:Q',
    y='test count:Q',
    tooltip=['zcta:N', 'population:Q', 'test count:Q']
).properties(
    width=600,
    height=400,
    title='Scatter Plot of Population and Tests'
)

st.altair_chart(chart, use_container_width=True)


def get_most_affected_zones(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.groupby(['zcta']).agg({'case count': 'sum'}).reset_index().sort_values(by='case count', ascending=False)


def get_most_affected_zones_by_case_ratio(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.groupby(['zcta']).agg({'case count': 'sum', 'test count': 'sum'}).reset_index().assign(ratio=lambda x: x['case count'] / x['test count']).sort_values(by='ratio', ascending=False)


def get_most_affected_zones_by_positive_rate(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.groupby(['zcta']).agg({'positive tests': 'sum', 'test count': 'sum'}).reset_index().assign(ratio=lambda x: x['positive tests'] / x['test count']).sort_values(by='ratio', ascending=False)


st.subheader('Analysis of Most Affected Zones')

analysis_type = st.selectbox("Select Analysis Type:", [
                             "Most Affected Zones", "Case Ratio", "Positive Rate"])

# Perform analysis based on the selected type
if analysis_type == "Most Affected Zones":
    result_df = get_most_affected_zones(df)
elif analysis_type == "Case Ratio":
    result_df = get_most_affected_zones_by_case_ratio(df)
else:
    result_df = get_most_affected_zones_by_positive_rate(df)

# Display the result using Altair chart
chart = alt.Chart(result_df).mark_bar().encode(
    x='zcta:N',
    y='case count:Q' if analysis_type == "Most Affected Zones" else 'ratio:Q',
    tooltip=['zcta:N', 'case count:Q'] if analysis_type == "Most Affected Zones" else [
        'zcta:N', 'ratio:Q']
).properties(
    width=600,
    height=400,
    title=f'{analysis_type} Analysis of Most Affected Zones'
)

# Display the Altair chart using Streamlit
st.altair_chart(chart, use_container_width=True)


def plot_most_affected_zone_by_case_count(input_df: pd.DataFrame, number_of_zones=None) -> None:
    data = input_df.head(
        number_of_zones) if number_of_zones is not None else input_df

    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('zcta:N', sort='-y'),
        y=alt.Y('case count:Q'),
        tooltip=['zcta:N', 'case count:Q']
    ).properties(
        width=600,
        height=400,
        title='Most Affected Zone by Case Count'
    )

    st.altair_chart(chart, use_container_width=True)


st.subheader('Analysis of Most Affected Zones')
number_of_zones = st.slider(
    'Select Number of Zones:', min_value=1, max_value=len(df['zcta']), value=5)

most_affected_zones = get_most_affected_zones(df)

# Plot most affected zones by case count
plot_most_affected_zone_by_case_count(most_affected_zones, number_of_zones)
