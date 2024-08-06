from time import sleep

import pandas as pd
import streamlit as st
from pathlib import Path

from atgen.visualize.plot_line import plot_line
from atgen.utils.get_last_workdir import get_last_workdir


METRICS_MAP = {
    'BARTScore s->h': 'BARTScore-sh',
    'BARTScore h->r': 'BARTScore-hr',
    'AlignScore': 'alignscore',
    'ROUGE-1': 'rouge1',
    'ROUGE-2': 'rouge2',
    'ROUGE-L': 'rougeL',

}

# Initialize an empty DataFrame
df = None

# Iterate through the JSON files and load the data
workdir = get_last_workdir()
path = Path(workdir) / "metrics.csv"

if path.exists():
    # Give it a few seconds to sleep: maybe it is being saved now
    sleep(1)
    df = pd.read_csv(path)
    df = df.loc[:, [x for x in df.columns if not x.startswith('time_')]]

# If data is available, create a DataFrame
if df is not None:
    # Set 'iteration' as the index
    df.index.name = "Iteration"
    # Display the DataFrame in Streamlit
    st.dataframe(df, width=100000)
    metrics = list(df.columns.tolist())
    user_metric_input = st.radio("Choose the metric to visualize:", list(METRICS_MAP))
    chart = plot_line(workdir, METRICS_MAP[user_metric_input])
    st.plotly_chart(chart)
else:
    st.text("No results found at the moment.")
