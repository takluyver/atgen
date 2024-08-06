import os
import json
from json.decoder import JSONDecodeError
from datetime import datetime
from omegaconf import OmegaConf

import streamlit as st
import yaml

from hydra import compose, initialize

from pathlib import Path
from scripts.run_active_learning import run_active_learning


LABELLER_MAP = {
    "Custom / Open-source LLM": "custom_llm",
    "API LLM": "api_llm",
    "Human": "human",
    "Golden (only for benchmarking)": "golden"
}

# st.set_page_config(page_title="Input Form", layout="wide")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**General and active learning parameters:**")
    config_name = st.radio("Config name (from 'configs' folder)", options=["Base", "Custom"]).lower()
    strategy = st.radio("AL strategy", options=["Random", "Hadas", "Huds", "te_delfy"]).lower()
    query_size = st.number_input(
        "AL query size", min_value=1, step=1, value=1, placeholder="Type a number..."
    )

    st.write("**Fill in at least one of the parameters below:**")

    num_iterations = st.number_input(
        "Number of AL iterations",
        min_value=0,
        max_value=100,
        step=1,
        value=None,
        placeholder="Type a number...",
    )
    required_performance = st.text_area(
        "Required performance (JSON format):", placeholder='Example: {"rouge1": 0.5}'
    )
    budget = st.number_input(
        "Budget (in $)", step=1, value=None, placeholder="Type a number..."
    )


with col2:
    st.write("**Labeller parameters:**")
    labeller = st.radio(
        "Labeller", ["Custom / Open-source LLM", "API LLM", "Human", "Golden (only for benchmarking)"]
    )
    labeller = LABELLER_MAP[labeller]
    if labeller == "human":
        price_input_per_example = st.number_input(
            "Price_input_per_example (in $)",
            step=0.01,
            value=0.,
            min_value=0.,
            max_value=1000.,
            placeholder="Type a number...",
        )
    elif labeller == "custom_llm":
        model_checkpoint = st.text_input(
            "Model checkpoint from HuggingFace",
            value="meta-llama/Meta-Llama-3.1-8B-Instruct",
        )
    elif labeller == "api_llm":
        api_key = st.text_input("api_key", placeholder="Your api key", type="password")
        provider = st.radio("Provider", ["openai", "antropic"])
        model = st.text_input("Model", value="gpt-4o")
        input_per_1m = st.number_input("Price per 1M input tokens (in $)")
        output_per_1m = st.number_input("Price per 1M output tokens (in $)")

with col3:
    st.write("**Data parameters:**")
    dataset = st.text_input("Dataset or path to data", value="SpeedOfMagic/gigaword_tiny")
    input_column_name = st.text_input("Input column name", value="document")
    output_column_name = st.text_input(
        "Output column name (only for benchmarking)", value="output"
    )
    prompt = st.text_area(
        "Prompt", height=220, value="Write a summary for the article in lowercase:",
        placeholder="Example:Rewrite the text: {text}"
    )

with col4:
    st.write("**Model and inference parameters:**")
    model = st.text_input("Model checkpoint from HuggingFace or path to the saved model", value="microsoft/Phi-3-mini-4k-instruct")
    inference_framework = st.radio(
        "Inference framework", options=["vLLM", "TensorRT-LLM", "Transformers"]
    )
    input_max_length = st.number_input(
        "Model maximum length for training", step=1, value=256
    )
    output_max_length = st.number_input("Output maximum length", step=1, value=20)

destination_path = Path("configs/data/user_data.yaml")
with destination_path.open() as file:
    config_data = yaml.safe_load(file)

if st.button("Run the script"):
    is_valid_required_performance = True
    if required_performance:
        required_performance = required_performance.strip()
        try:
            required_performance = json.loads(required_performance)
        except JSONDecodeError:
            is_valid_required_performance = False
            st.error("Provide a valid json input for 'required_performace' field")

    if (
        strategy
        and (
            num_iterations is not None
            or budget is not None
            or (required_performance and required_performance != "{}")
        )
        and is_valid_required_performance
    ):
        config_data["dataset"] = dataset
        config_data["input_column_name"] = input_column_name
        config_data["output_column_name"] = output_column_name
        config_data["input_max_length"] = input_max_length
        config_data["output_max_length"] = output_max_length
        config_data["prompt"] = prompt

        with destination_path.open("w") as file:
            yaml.safe_dump(config_data, file)

        st.text(
            "Script is running...It might take a while. Meanwhile, you can check the intermediate results, generations of the model, and the labeled examples."
        )
        if st.button("Cancel the script."):
            st.rerun()
        with initialize(config_path="configs", version_base=None):
            config = compose(
                config_name=config_name, overrides=["data=user_data", f"labeller={labeller}"]
            )

            config["al"]["strategy"] = strategy
            config["al"]["num_iterations"] = num_iterations
            if query_size:
                query_size = int(query_size)
                config["al"]["query_size"] = query_size
                config["al"]["init_query_size"] = query_size
            if budget:
                config["al"]["budget"] = budget
            if required_performance:
                config["al"]["required_performance"] = required_performance

            if labeller == "human":
                config["labeller"]["price_input_per_example"] = price_input_per_example
            elif labeller == "custom_llm":
                config["labeller"]["model"]["checkpoint"] = model_checkpoint
            elif labeller == "api_llm":
                config["labeller"]["api_key"] = api_key
                config["labeller"]["provider"] = provider
                config["labeller"]["parameters"]["model"] = model
                config["labeller"]["price"]["input_per_1m"] = input_per_1m
                config["labeller"]["price"]["output_per_1m"] = output_per_1m

            config["model"]["checkpoint"] = model
            config["inference"]["framework"] = inference_framework.lower()

            # Set `output_dir`
            current_time = datetime.now()
            output_dir = os.path.join(
                "outputs",
                current_time.strftime("%Y-%m-%d"),
                current_time.strftime("%H-%M-%S"),
            )
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(config))
            config["output_dir"] = output_dir

            if labeller == "human":
                st.subheader("Human labelling has been selected. For annotator: kindly switch to the tab `Annotation` on the left.")

            run_active_learning(config)
            st.success(
                "Script has finished running. Check the results on the pages to the left."
            )
    elif is_valid_required_performance:
        st.error("You didn't fill one of the required arguments.")
