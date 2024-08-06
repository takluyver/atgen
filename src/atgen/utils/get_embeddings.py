import gc
import logging
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from torch import cuda


log = logging.getLogger()


def get_embeddings(
    texts,
    model: str | PreTrainedModel,
    tokenizer: str | PreTrainedTokenizer | None = None,
    batch_size: int = 64,
    model_is_causallm: bool = False,
    model_max_length: int = 1024,
    device: str = "cuda",
    **model_kwargs,
) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(model, str):
        if model_is_causallm:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                model_max_length=model_max_length,
                cache_dir=model_kwargs.get("cache_dir", None),
                padding_side="left",
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs).to(
                device
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                model_max_length=model_max_length,
                cache_dir=model_kwargs.get("cache_dir", None),
            )
            model = AutoModel.from_pretrained(model, **model_kwargs).to(device)
    else:
        model = model.to(device)
    embeddings = _get_embeddings(texts, model, tokenizer, batch_size, model_is_causallm)
    del model, tokenizer
    gc.collect()
    cuda.empty_cache()

    return embeddings


def _get_embeddings(
    texts, model, tokenizer, batch_size: int = 64, model_is_causallm: bool = False
) -> np.ndarray:
    n_texts = len(texts)
    embeddings = np.empty((n_texts, model.config.hidden_size), dtype=np.float32)
    start = 0
    end = 0

    with tqdm(
        total=int(n_texts / batch_size) + 1, desc="Generating embeddings..."
    ) as pbar:
        while end < n_texts:
            end += batch_size
            batch_idx = slice(start, end)
            # Tokenize sentences
            encoded = tokenizer(
                texts[batch_idx], padding=True, truncation=True, return_tensors="pt"
            )
            output = model(
                input_ids=encoded["input_ids"].cuda(),
                attention_mask=encoded["attention_mask"].cuda(),
                output_hidden_states=True,
                return_dict=True,
            )
            # Perform pooling
            if model_is_causallm:
                batch_embeddings = output.hidden_states[-1][:, -1].cpu()
            else:
                batch_embeddings = output.last_hidden_state[:, 0].cpu()
            # Normalize embeddings
            embeddings[batch_idx] = batch_embeddings.detach().numpy()
            start = end
            pbar.update()

    return embeddings
