import json
from pathlib import Path
from time import sleep

import numpy as np
import streamlit as st

from atgen.utils.get_last_workdir import get_last_workdir

NUM_RANDOM_GENS_TO_SHOW = 10

# Initialize an empty list to store the data for the DataFrame
data = []
column_names = []

# Initialize iteration number
iteration_number = 0
folder_exists = True

# Iterate through the JSON files and load the data
while folder_exists:
    workdir = get_last_workdir()
    path = Path(workdir) / f"iter_{iteration_number}" / "generations.json"
    if path.exists():
        sleep(1)
        with open(path, "r") as file:
            json_data = json.load(file)
            data.append(json_data)
        iteration_number += 1
    else:
        folder_exists = False

# If data is available, create a DataFrame
if data:
    data_to_show = np.random.choice(data[-1], min(len(data), NUM_RANDOM_GENS_TO_SHOW), False)
    for i, text in enumerate(data_to_show):
        st.text(f'**Generated text {i}**:\n{text}\n\n')
else:
    st.text("No results found at the moment.")
