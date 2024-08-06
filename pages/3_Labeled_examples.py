import os
from pathlib import Path
from time import sleep

import numpy as np
import streamlit as st
import yaml
from datasets import load_from_disk

from atgen.utils.get_last_workdir import get_last_workdir

with open("configs/data/user_data.yaml", "r") as file:
    config = yaml.safe_load(file)

output_column_name = config.get("output_column_name")


num_random_examples_to_display = 5
workdir = get_last_workdir()
dataset_path = str(Path(workdir) / f"query")

if not os.path.exists(dataset_path):
    st.text("No results found at the moment.")
else:
    sleep(3)

    dataset = load_from_disk(dataset_path)
    random_idx = np.random.choice(range(len(dataset)), min(len(dataset), num_random_examples_to_display), False)
    subset = dataset[random_idx]
    inputs = subset["input"]
    outputs = subset[output_column_name]

    for i in range(len(inputs)):
        st.write(f"**Input #{i+1}**\n")
        st.write(f"{inputs[i]}\n")

        st.write(f"**Annotation #{i+1}**\n")
        st.write(f"{outputs[i]}\n")
