import os
import torch
from shutil import rmtree
import gc
import json
import hydra
from pathlib import Path
from typing import Union
import logging

from atgen.utils.main_decorator import main_decorator


log = logging.getLogger()


@main_decorator
def run_experiment_design(config, workdir: Union[str, Path]):
    from transformers import set_seed

    from atgen.metrics.compute_metrics import compute_metrics
    from atgen.utils.load_data import load_data_with_prompt, tokenize_dataset
    from atgen.utils.load_model_tokenizer import load_model, load_tokenizer
    from atgen.utils.prepare_model_for_peft import prepare_model_for_training
    from atgen.utils.training_utils import get_trainer
    from atgen.labellers import get_labeller, OutOfBudgetException
    from atgen.utils.generate import generate
    from atgen.utils.save_labeled_data import save_labeled_data

    seed = config.seed
    cache_dir = config.cache_dir
    input_column_name = config.data.input_column_name
    output_column_name = config.data.output_column_name
    model_name = config.model.checkpoint
    num_al_iterations = config.al.num_iterations
    al_query_size = config.al.query_size

    log.info(
        f"""Running Active Learning...
{num_al_iterations} iterations
Query size: {al_query_size}
Dataset: {config.data.dataset}
Seed: {seed}
Model: {model_name}
Config: {config.name}
"""
    )

    if isinstance(workdir, str):
        workdir = Path(workdir)

    unlabeled_data = load_data_with_prompt(config.data, "train", config.cache_dir, seed)
    test_data = load_data_with_prompt(config.data, "test", config.cache_dir, seed)

    tokenizer = load_tokenizer(config.model, cache_dir)
    test_data = tokenize_dataset(
        config.data, config.model, test_data, tokenizer, "test"
    )
    log.info("Model loading started...")
    model = load_model(model_name, config.model, cache_dir)
    model = prepare_model_for_training(model, config.model.peft)
    log.info(f"Model loading done.")

    labeller = get_labeller(
        config.labeller,
        output_column_name,
        cache_dir=cache_dir,
        data_config=config.data,  # if labeller is a custom LLM on transformers
        model_config=config.model,  # if labeller is a custom LLM on transformers
    )

    random_data_to_label = unlabeled_data.train_test_split(
        train_size=config.al.init_query_size, shuffle=True, seed=seed
    )["train"]
    try:
        labeled_data = labeller(random_data_to_label)
    except OutOfBudgetException:
        print("Labeler ran out of budget when labeling initial dataset")
        return
    labeled_ids = labeled_data["id"]
    set_labeled_ids = set(labeled_ids)
    unlabeled_data = unlabeled_data.filter(lambda x: x["id"] not in set_labeled_ids)
    save_labeled_data(
        labeled_data=labeled_data,
        labeled_query=labeled_data,
        workdir=workdir,
        iter_dir=workdir,
        labeled_ids=labeled_ids,
        query_ids=labeled_ids,
    )

    train_data = tokenize_dataset(
        data_config=config.data,
        model_config=config.model,
        dataset=labeled_data,
        tokenizer=tokenizer,
        split="train",
    )
    train_dev_data = train_data.train_test_split(
        test_size=config.training.dev_split_size, shuffle=True, seed=seed
    )

    train_output_dir = workdir / "tmp"
    save_dir = workdir / "tmp_best"

    # Set seed for reproducibility
    set_seed(seed)
    trainer = get_trainer(
        config.training.hyperparameters,
        model,
        tokenizer,
        train_dev_data,
        seed,
        train_output_dir,
        config.model.quantize,
    )

    # Launch training
    train_result = trainer.train()

    model = model.to(torch.float32).eval().merge_and_unload()
    generations = generate(
        config.inference,
        data=test_data,
        model=model,
        tokenizer=tokenizer,
        save_dir=save_dir,
    )
    if os.path.exists(save_dir):
        rmtree(save_dir)

    metrics = compute_metrics(
        generations, test_data[output_column_name], test_data[input_column_name]
    )
    log.info(metrics)
    with open(workdir / "train_result.json", "w") as f:
        json.dump(train_result, f)
    with open(workdir / "results.json", "w") as f:
        json.dump(metrics, f)
    with open(workdir / "generations.json", "w") as f:
        json.dump(generations, f)

    model.save_pretrained(workdir / "model.bin")


@hydra.main(
    config_path=os.environ.get("HYDRA_CONFIG_PATH", "../configs/"),
    config_name=os.environ.get("HYDRA_CONFIG_NAME"),
    version_base="1.1",
)
def main(config):
    if getattr(config, "debug", True):
        try:
            run_experiment_design(config)
        except Exception as e:
            print(e)
            import sys, pdb

            exc_type, exc_value, exc_traceback = sys.exc_info()
            pdb.post_mortem(exc_traceback)
    else:
        run_experiment_design(config)


if __name__ == "__main__":
    main()
