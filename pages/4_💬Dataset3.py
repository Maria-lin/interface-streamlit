import math
import base64
from itertools import combinations
from collections import Counter
from typing import Literal
import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np


st.set_page_config(page_title="Dataset3!!!",
                   page_icon=":bar_chart:", layout="wide")


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


st.title(" :bar_chart: Apriori")
st.markdown(
    '<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

df = pd.read_csv("data/Dataset3.csv", encoding="ISO-8859-1")
st.subheader("Loaded Dataset")

col1, col2 = st.columns((2))
_, view4, dwn4 = st.columns([0.5, 0.45, 0.45])

with col1:

    st.dataframe(df)

with col2:
    st.write("### Data overview")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.write(df.describe().T)

    st.download_button("Get Data", data=df.to_csv().encode("utf-8"),
                       file_name="Dataset3.csv", mime="text/csv")


st.divider()


discretize_column = 'Temperature'


def discretize_equal_width(input_df, column, *, n_bins=0):
    input_df = input_df.copy()
    sorted_data = input_df[column].sort_values()
    if n_bins == 0:
        n_bins = int(1 + (10 / 3) * math.log10(len(sorted_data)))
    ranges = np.linspace(sorted_data.min(), sorted_data.max(), n_bins + 1)
    labels = [str(i) for i in range(1, n_bins + 1)]
    current_bin = 0
    class_column = []
    for i, row_temp in enumerate(sorted_data):
        if row_temp > ranges[current_bin + 1]:
            current_bin += 1
        class_column.append(labels[current_bin])
    input_df[column] = class_column
    return input_df


def plot_classes(input_df, column=discretize_column):
    chart = alt.Chart(input_df).mark_bar().encode(
        x=alt.X(column, title='Classes'),
        y=alt.Y('count()', title='Count'),
    ).properties(
        title='Class Distribution'
    )
    return chart


def discretize_equal_freq(input_df, column, *, n_bins=0):
    input_df = input_df.copy()

    if n_bins == 0:
        n_bins = int(1 + (10 / 3) * math.log10(len(input_df[column])))

    sorted_values = sorted(input_df[column])

    bin_edges = [sorted_values[i * len(sorted_values) // n_bins]
                 for i in range(n_bins)]
    bin_edges.append(sorted_values[-1])  # Include the maximum value
    labels = [str(i) for i in range(1, n_bins + 1)]

    discrete_column = []
    current_bin = 0

    for i, value in enumerate(sorted_values):
        if value > bin_edges[current_bin + 1]:
            current_bin += 1
        discrete_column.append(labels[current_bin])

    input_df[column] = discrete_column

    return input_df


st.subheader('Equal Width/Frequency Discretization and Class Distribution')


df_width = discretize_equal_width(df, discretize_column)


df_freq = discretize_equal_freq(df, discretize_column)


col1, col2 = st.columns(2)


with st.expander("Equal Width Discretization"):

    st.dataframe(df_width.head())
    st.altair_chart(plot_classes(df_width), use_container_width=True)


with st.expander("Equal Frequency Discretization"):

    st.dataframe(df_freq.head())
    st.altair_chart(plot_classes(
        df_freq, discretize_column), use_container_width=True)

st.divider()

# discretisize all numeric columns

numerical_columns = [column for column in df.columns if column not in [
    'Fertilizer', 'Soil', 'Crop']]

df_width = df.copy()

for column in numerical_columns:
    df_width = discretize_equal_width(df_width, column)

predictable_classes = list(df.columns)
grouping_classes = predictable_classes.copy()

predict_class = st.selectbox("Select column to predict", predictable_classes)
grouping_classes.remove(predict_class)

groupe_by_classes = st.multiselect(
    "Select columns to group-by", grouping_classes, default=grouping_classes)


def get_grouped_df(input_df, columns=None):
    if columns is None:
        columns = grouping_classes
    return input_df.groupby(columns).agg({
        predict_class: set,
    }).rename(columns={predict_class: 'Items'})


grouped_df = get_grouped_df(df_width, columns=groupe_by_classes)

st.subheader('Apriori and association rules')
with st.expander("Grouped DataFrame"):
    st.table(grouped_df)


class Apriori:
    Metric = Literal['confidence', 'cosine', 'lift',
                     'all_confidence', 'max_confidence', 'jaccard', 'kulczynski']

    def __init__(self, min_support, min_confidence):
        self.df = None
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []

    def set_params(self, *, min_support, min_confidence):
        self.min_support = min_support if min_support else self.min_support
        self.min_confidence = min_confidence if min_confidence else self.min_confidence

    def fit(self, input_df):
        self.frequent_itemsets = []
        self.df = input_df
        self._get_frequent_itemsets()
        self._get_rules()

    def _get_frequent_itemsets(self):
        self._get_frequent_1_itemsets()
        k = 2
        while True:
            frequent_itemsets = self._get_frequent_k_itemsets(k)
            if not frequent_itemsets:
                break
            self.frequent_itemsets.append(frequent_itemsets)
            k += 1

    def _get_frequent_1_itemsets(self):
        self.item_counts = Counter()
        for itemset in self.df['Items']:
            self.item_counts.update(itemset)

        n = len(self.df)
        self.frequent_itemsets.append([
            (frozenset([item]), support / n)
            for item, support in self.item_counts.items()
            if support / n >= self.min_support
        ])

    def _get_frequent_k_itemsets(self, k):
        itemsets = self.frequent_itemsets[k - 2]
        frequent_itemsets = []
        for itemset1, support1 in itemsets:
            for itemset2, support2 in itemsets:
                if itemset1 == itemset2:
                    continue
                union = itemset1 | itemset2
                if len(union) != k:
                    continue
                if union in (itemset for itemset, _ in frequent_itemsets):
                    continue
                support = self._get_support(union)
                if support >= self.min_support:
                    frequent_itemsets.append((union, support))
        return frequent_itemsets

    def _get_rules(self):
        self.rules = []
        for itemsets in self.frequent_itemsets:
            for itemset, support in itemsets:
                if len(itemset) < 2:
                    continue
                for antecedent in self._get_antecedents(itemset):
                    antecedent_support = self._get_support(antecedent)
                    if support / antecedent_support >= self.min_confidence:
                        self.rules.append(
                            (antecedent, itemset - antecedent, support / antecedent_support))
        return self.rules

    @staticmethod
    def _get_antecedents(itemset: set) -> list[tuple]:
        antecedents = []
        for i in range(1, len(itemset)):
            current_antecedents = combinations(itemset, i)
            current_antecedents = {frozenset(antecedent)
                                   for antecedent in current_antecedents}
            antecedents.extend(current_antecedents)
        return antecedents

    def _get_support(self, itemset):
        return sum(1 for itemset2 in self.df['Items'] if itemset.issubset(itemset2)) / len(self.df)

    def _get_cosines(self):
        self.cosines = []
        for itemset1, consequent, confidence in self.rules:
            cosine = self._get_support(itemset1 | consequent) / math.sqrt(
                self._get_support(itemset1) * self._get_support(consequent))
            self.cosines.append((itemset1, consequent, cosine))
        return self.cosines

    def _get_lifts(self):
        self.lifts = []
        for itemset1, consequent, confidence in self.rules:
            lift = confidence / self._get_support(consequent)
            self.lifts.append((itemset1, consequent, lift))
        return self.lifts

    def _get_jaccard(self):
        self.jaccard = []
        for itemset1, consequent, confidence in self.rules:
            jaccard = self._get_support(itemset1 | consequent) / (self._get_support(
                itemset1) + self._get_support(consequent) - self._get_support(itemset1 | consequent))
            self.jaccard.append((itemset1, consequent, jaccard))
        return self.jaccard

    def _get_kulczynski(self):
        self.kulczynski = []
        for itemset1, consequent, confidence in self.rules:
            join_support = self._get_support(itemset1 | consequent)
            kulczynski = ((join_support / self._get_support(itemset1)) +
                          (join_support / self._get_support(consequent))) / 2
            self.kulczynski.append((itemset1, consequent, kulczynski))
        return self.kulczynski

    def _get_max_confidence(self):
        self.max_confidence = []
        for itemset1, consequent, confidence in self.rules:
            join_support = self._get_support(itemset1 | consequent)
            max_confidence = max(self._get_support(
                itemset1) / join_support, self._get_support(consequent) / join_support)
            self.max_confidence.append((itemset1, consequent, max_confidence))
        return self.max_confidence

    def _get_all_confidence(self):
        self.all_confidence = []
        for itemset1, consequent, confidence in self.rules:
            all_confidence = self._get_support(
                itemset1 | consequent) / max(self._get_support(itemset1), self._get_support(consequent))
            self.all_confidence.append((itemset1, consequent, all_confidence))
        return self.all_confidence

    def get_strong_rules(self, *, metric: Metric = 'confidence', n_rules: int = 10):
        sorting_functions = {
            'confidence': lambda: sorted(self.rules, key=lambda x: x[2], reverse=True),
            'cosine': lambda: sorted(self._get_cosines(), key=lambda x: x[2], reverse=True),
            'lift': lambda: sorted(self._get_lifts(), key=lambda x: x[2], reverse=True),
            'all_confidence': lambda: sorted(self._get_all_confidence(), key=lambda x: x[2], reverse=True),
            'jaccard': lambda: sorted(self._get_jaccard(), key=lambda x: x[2], reverse=True),
            'kulczynski': lambda: sorted(self._get_kulczynski(), key=lambda x: x[2], reverse=True),
            'max_confidence': lambda: sorted(self._get_max_confidence(), key=lambda x: x[2], reverse=True),
        }

        if metric in sorting_functions:
            return sorting_functions[metric]()[:n_rules]
        else:
            raise ValueError(
                f'metric should be one of {", ".join(sorting_functions.keys())}')

    def predict(self, items: list[str], *, metric: Metric = 'confidence'):
        prediction_functions = {
            'confidence': self._predict_confidence,
            'cosine': self._predict_cosine,
            'lift': self._predict_lift,
            'all_confidence': self._predict_all_confidence,
            'jaccard': self._predict_jaccard,
            'kulczynski': self._predict_kulczynski,
            'max_confidence': self._predict_max_confidence,
        }

        if metric in prediction_functions:
            return prediction_functions[metric](items)
        else:
            raise ValueError(
                f'metric should be one of {", ".join(prediction_functions.keys())}')

    def _predict_confidence(self, items: list[str]):
        items = set(items)
        predictions = []
        for itemset1, consequent, confidence in self.rules:
            if itemset1 == items:
                predictions.append((consequent, confidence))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_cosine(self, items: list[str]):
        items = set(items)
        predictions = []
        for itemset1, consequent, cosine in self._get_cosines():
            if itemset1 == items:
                predictions.append((consequent, cosine))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_lift(self, items: list[str]):
        items = set(items)
        predictions = []
        for itemset1, consequent, lift in self._get_lifts():
            if itemset1 == items:
                predictions.append((consequent, lift))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_all_confidence(self, items: list[str]):
        items = set(items)
        predictions = []
        for itemset1, consequent, all_confidence in self._get_all_confidence():
            if itemset1 == items:
                predictions.append((consequent, all_confidence))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_jaccard(self, items: list[str]):
        items = set(items)
        predictions = []
        for itemset1, consequent, jaccard in self._get_jaccard():
            if itemset1 == items:
                predictions.append((consequent, jaccard))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_kulczynski(self, items: list[str]):
        items = set(items)
        predictions = []
        for itemset1, consequent, kulczynski in self._get_kulczynski():
            if itemset1 == items:
                predictions.append((consequent, kulczynski))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def _predict_max_confidence(self, items: list[str]):
        items = set(items)
        predictions = []
        for itemset1, consequent, max_confidence in self._get_max_confidence():
            if itemset1 == items:
                predictions.append((consequent, max_confidence))
        return sorted(predictions, key=lambda x: x[1], reverse=True)


st.subheader("Apriori Algorithm Interface")


with st.form(key='algorithm_parameters'):
    st.subheader("Algorithm Parameters")
    min_support = st.slider("Minimum Support", 0.1, 1.0)
    min_confidence = st.slider(
        "Minimum Confidence", 0.1, 1.0)

    submit_button = st.form_submit_button(label='Run Apriori Algorithm')

# Display frequent itemsets and rules if the algorithm parameters form is submitted
if submit_button:
    st.session_state['apriori'] = Apriori(
        min_support=min_support, min_confidence=min_confidence)
    # Fit the algorithm with the data
    st.session_state['apriori'].fit(grouped_df)


if 'apriori' in st.session_state:
    st.subheader("Frequent Itemsets")
    st.write(st.session_state['apriori'].frequent_itemsets)

    st.subheader("Rules")
    st.write(st.session_state['apriori'].rules)

    # Metric selection form
    st.header("Metric Selection")

    # Form for choosing the metric
    metric = st.selectbox("Select Metric", [
        'confidence', 'cosine', 'lift', 'all_confidence', 'jaccard', 'kulczynski', 'max_confidence'])
    n_rules = st.slider("Number of Rules to Display", 1, 20, 2)

    st.header(f"Strong Rules - Metric: {metric}")
    st.write(st.session_state['apriori'].get_strong_rules(
        metric=metric, n_rules=n_rules))
