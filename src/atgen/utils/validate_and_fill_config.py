from omegaconf import DictConfig, open_dict


def validate_field(config, field):
    subfields = field.split(".")
    try:
        for subfield in subfields:
            config = config[subfield]
    except:
        raise Exception(f"{field} is not defined in config")


def validate_and_fill_config(config: DictConfig) -> DictConfig:
    with open_dict(config):
        # Common fields
        config.setdefault("name", "I have no name!")
        config.setdefault("seed", 42)
        config.setdefault("cache_dir", "cache")
        config.setdefault("save_model", True)

        # Experiment
        validate_field(config, "al")
        validate_field(config, "al.init_query_size")
        validate_field(config, "al.budget")
        validate_field(config, "al.num_iterations")
        validate_field(config, "al.query_size")
        # TODO choose strategy automatically
        validate_field(config, "al.strategy")
        config.al.setdefault("required_performance", {})
        config.al.setdefault("additional_metrics", [])

        # Data
        validate_field(config, "data")
        validate_field(config, "data.dataset")
        validate_field(config, "data.input_column_name")
        validate_field(config, "data.output_column_name")
        validate_field(config, "data.prompt")
        config.data.setdefault("train_subset_name", "train")
        config.data.setdefault("test_subset_name", "test")
        config.data.setdefault("train_subset_size", None)
        config.data.setdefault("test_subset_size", None)
        config.data.setdefault("num_proc", 16)
        config.data.setdefault("fetch_kwargs", {})
        config.data.setdefault("few_shot", {})
        config.data.few_shot.setdefault("count", 0)
        config.data.few_shot.setdefault("separator", "\n\n")

        # Model
        validate_field(config, "model")
        validate_field(config, "model.checkpoint")
        config.model.setdefault("quantize", False)
        config.model.setdefault("model_max_length", None)
        config.model.setdefault(
            "separator",
            (
                "<|assistant|>\n"
                if "stable" in config.model.checkpoint
                else "<|assistant|>"
            ),
        )

        # Labeler
        validate_field(config, "labeller")
        validate_field(config, "labeller.type")
        config.labeller.setdefault("budget", None)

        # Training
        config.setdefault("training", {})
        config.training.setdefault("num_epochs", 1)
        config.training.setdefault("train_batch_size", 64)
        config.training.setdefault("eval_batch_size", 64)
        config.training.setdefault("gradient_accumulation_steps", 2)
        config.training.setdefault("lr", 0.00003)
        config.training.setdefault("warmup_ratio", 0.03)
        config.training.setdefault("weight_decay", 0.01)
        config.training.setdefault("max_grad_norm", 1.0)
        config.training.setdefault("early_stopping_patience", 5)

        # Inference
        config.setdefault("inference", {})
        config.inference.setdefault("batch_size", 64)
        config.inference.setdefault("framework", "vllm")
        config.inference.setdefault("max_new_tokens", None)

    return config
