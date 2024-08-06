from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_tokenizer(
    model_config: DictConfig,
    cache_dir: str,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.checkpoint,
        model_max_length=model_config.model_max_length,
        cache_dir=cache_dir,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    checkpoint: str | Path,
    model_config: DictConfig,
    cache_dir: str,
) -> PreTrainedModel:
    kwargs = {
        "cache_dir": cache_dir,
        "torch_dtype": model_config.dtype,
        "trust_remote_code": True,
    }
    if model_config.quantize:
        bnb_config = _get_bnb_config()
        kwargs["quantization_config"] = bnb_config

    # TODO: add support of seq2seq models
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
    return model


def _get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
