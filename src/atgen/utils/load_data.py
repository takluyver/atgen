import logging
import os
from typing import Union

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer


def fetch_dataset(
    dataset_name_or_path: Union[str, list[str]],
    subset_name: str,
    fetch_kwargs: dict | DictConfig,
) -> Dataset:
    # Load local dataset
    if os.path.exists(dataset_name_or_path):
        # Load a saved on disk dataset
        if os.path.isdir(dataset_name_or_path):
            dataset = load_from_disk(dataset_name_or_path, **fetch_kwargs)
        # Load csv dataset
        elif dataset_name_or_path.endswith("csv"):
            dataset = Dataset.from_csv(dataset_name_or_path, **fetch_kwargs)
        # Load json dataset
        elif dataset_name_or_path.endswith("json"):
            dataset = Dataset.from_json(dataset_name_or_path, **fetch_kwargs)
        else:
            raise NotImplementedError(
                f"Unexpected format {dataset_name_or_path.split('.')[-1]} of the dataset. Supported formats: csv, json."
            )
    # Load dataset from HuggingFace
    elif isinstance(dataset_name_or_path, list):
        dataset = load_dataset(*dataset_name_or_path, **fetch_kwargs)
    else:
        dataset = load_dataset(dataset_name_or_path, **fetch_kwargs)

    if isinstance(dataset, DatasetDict):
        return dataset[subset_name]
    else:
        return dataset


def _add_id_column(dataset: Dataset) -> Dataset:
    if "id" in dataset.column_names:
        dataset = dataset.remove_columns(["id"])
    dataset = dataset.add_column("id", list(range(len(dataset))))
    return dataset


def _take_subset(dataset_subset: Dataset, size: int, seed: int) -> Dataset:
    dataset_subset = dataset_subset.shuffle(seed=seed)
    dataset_subset = dataset_subset.select(range(size))
    return dataset_subset


def load_data(
    config: DictConfig,
    split: str,
    cache_dir: str,
    seed: int,
) -> Dataset:
    if split == "train":
        subset_name = config.get("train_subset_name", split)
        subset_size = config.get("train_subset_size")
    elif split == "test":
        subset_name = config.get("test_subset_name", split)
        subset_size = config.get("test_subset_size")
    else:
        raise NotImplementedError(
            f"Unexpected split {split}; Please specify either `train` or `test`."
        )
    dataset = fetch_dataset(
        config.dataset, subset_name, dict(config.fetch_kwargs, cache_dir=cache_dir)
    )

    # Add `id` column to the dataset (practical use) or to train subset (benchmarking)
    dataset = _add_id_column(dataset)
    if subset_size is not None:
        dataset = _take_subset(dataset, subset_size, seed)

    return dataset


def load_data_with_prompt(
    config: DictConfig,
    prompt: str,
    split: str,
    cache_dir: str,
    seed: int,
):
    data = load_data(config, split, cache_dir, seed)

    inputs = []
    for inst in data:
        inputs.append(prompt.format(text=inst[config.input_column_name]))

    return data.add_column("input", inputs)


# TODO: move to another file
def tokenize_dataset(
    data_config: DictConfig,
    model_config: DictConfig,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    split: str,
):
    separator_token = model_config.separator
    separator_token_ids = tokenizer.encode(separator_token, add_special_tokens=False)

    def find_idx_beginning_answer(ids: list[int]) -> int:
        idx_beginning_answer = None
        for i in range(len(ids) - len(separator_token_ids)):
            if _are_ids_equal(
                ids[i : i + len(separator_token_ids)],
                separator_token_ids,
                tokenizer,
            ):
                idx_beginning_answer = i + len(separator_token_ids)
                break
        return idx_beginning_answer

    if split == "train":

        def tokenize_fn(instance):
            ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": instance["input"]},
                    {
                        "role": "assistant",
                        "content": instance[data_config.output_column_name],
                    },
                ],
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=model_config.model_max_length,
            )
            if "stablelm" in model_config.checkpoint:  # stablelm returns
                ids = ids[:-1]

            instance["input_ids"] = ids
            instance["attention_mask"] = [1 for _ in range(len(ids))]

            idx_beginning_answer = find_idx_beginning_answer(ids=ids)
            if idx_beginning_answer is None:
                instance["labels"] = None
            else:
                instance["labels"] = [-100 for _ in range(idx_beginning_answer)] + ids[
                    idx_beginning_answer:
                ]
            return instance

    else:

        def tokenize_fn(instance):
            ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": instance["input"]},
                    {"role": "assistant", "content": ""},
                ],
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=model_config.model_max_length,
            )
            idx_beginning_answer = find_idx_beginning_answer(ids=ids)
            # In this case, we exceeded max length. Let's throw away info from the middle.
            if idx_beginning_answer is None:
                ids = _retokenize_long_text_for_test(
                    instance, tokenizer, model_config.model_max_length
                )
            ids = ids[:idx_beginning_answer]
            instance["input_ids"] = ids
            instance["attention_mask"] = [1 for _ in range(len(ids))]
            return instance

    dataset = dataset.map(
        tokenize_fn,
        batched=False,
        num_proc=data_config.num_proc,
    )
    if split == "train":
        dataset = dataset.filter(lambda x: x["labels"] is not None, batched=False)
    return dataset


def _are_ids_equal(ids1, ids2, tokenizer):
    if len(ids1) != len(ids2):
        return False
    for a, b in zip(ids1, ids2):
        if a != b and tokenizer.decode([a]) != tokenizer.decode(b):
            return False
    return True


def _retokenize_long_text_for_test(
    instance: dict[str, str], tokenizer: PreTrainedTokenizer, model_max_length: int
) -> list[int]:
    # Remove 20 tokens (in total) because we also want to add some text indicating that
    # we removed some text from the middle
    half_model_max_length = model_max_length // 2 - 10

    ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": instance["input"]},
            {"role": "assistant", "content": ""},
        ],
        tokenize=True,
        add_generation_prompt=False,
        truncation=False,
    )
    middle_ids = tokenizer("\n\n...some text was skipped...\n\n")["input_ids"]
    ids = ids[:half_model_max_length] + middle_ids + ids[-half_model_max_length:]
    return ids


def get_initial_labeled_data(
    config: DictConfig, unlabeled_data: Dataset, labeller
) -> tuple[Dataset, list[int]]:
    random_data_to_label_size = max(
        config.al.init_query_size, config.data.few_shot.count
    )
    random_data_to_label = unlabeled_data.train_test_split(
        train_size=random_data_to_label_size, shuffle=True, seed=config.seed
    )["train"]

    labeled_data = labeller(random_data_to_label)
    if labeller.is_out_of_budget:
        logging.info("Labeler ran out of budget when labeling the initial query.")

    return labeled_data, labeled_data["id"]


def add_prefix_to_input(row, prefix):
    row["input"] = prefix + row["input"]
    return row
