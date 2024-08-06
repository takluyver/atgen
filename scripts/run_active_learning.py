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
def run_active_learning(config, workdir: Union[str, Path]):
    from transformers import set_seed
    from datasets import concatenate_datasets

    from atgen.metrics.compute_metrics import compute_metrics
    from atgen.utils.load_data import (
        load_data_with_prompt,
        tokenize_dataset,
        get_initial_labeled_data,
        add_prefix_to_input,
    )
    from atgen.utils.load_model_tokenizer import load_model, load_tokenizer
    from atgen.utils.prepare_model_for_peft import prepare_model_for_training
    from atgen.utils.training_utils import get_trainer
    from atgen.strategies.get_strategy import get_strategy
    from atgen.labellers import get_labeller
    from atgen.utils.generate import generate
    from atgen.utils.check_required_performance import check_required_performance
    from atgen.utils.save_labeled_data import save_labeled_data
    from atgen.utils.combine_results import combine_results

    seed = config.seed
    cache_dir = config.cache_dir
    input_column_name = config.data.input_column_name
    output_column_name = config.data.output_column_name
    prompt = config.data.prompt.strip()
    # Verify the prompt
    if "{text}" not in prompt:
        prompt += "\n{text}"
    # TODO: fix
    has_test = config.data.has_test
    dev_split_size = config.training.dev_split_size

    model_name = config.model.checkpoint

    num_al_iterations = config.al.num_iterations
    al_query_size = config.al.query_size
    required_performance_dict = check_required_performance(
        config.al.required_performance
    )
    budget = config.al.budget
    if budget is None:
        budget = 1e10

    # Stopping criteria due to reaching required performance
    is_performance_reached = False

    log.info(
        f"""Running Active Learning...
Strategy: {config.al.strategy}
{num_al_iterations} iterations
Query size: {al_query_size}
Dataset: {config.data.dataset}
Seed: {seed}
Model: {model_name}
Config: {config.name}
Prompt:\n{prompt})
"""
    )

    if isinstance(workdir, str):
        workdir = Path(workdir)
    tokenizer = load_tokenizer(config.model, cache_dir)

    log.info("Loading data.")
    unlabeled_data = load_data_with_prompt(config.data, prompt, "train", config.cache_dir, seed)
    if has_test:
        test_data = load_data_with_prompt(config.data, prompt, "test", config.cache_dir, seed)
    else:
        test_data = None

    log.info("Initial iteration: loading model.")
    model = load_model(model_name, config.model, cache_dir)

    log.info("Loading AL strategy.")
    al_strategy = get_strategy(
        config.al.strategy,
        subsample_size=config.al.subsample_size,
        unlabeled_pool=unlabeled_data[input_column_name],
        model=model,
        tokenizer=tokenizer,
        inference_config=config.inference,  # for hadas
        cache_dir=cache_dir,  # for hadas, huds, graph_cut
        seed=seed,
        **config.al.strategy_kwargs,
    )

    # TODO: unsure whether need to log here since may be confusing for a human labeller
    # log.info("Loading labeller.")
    labeller = get_labeller(
        config.labeller,
        output_column_name,
        cache_dir=cache_dir,
        budget=budget,
        workdir=workdir,  # if labeller is a human
        data_config=config.data,  # if labeller is a custom LLM on transformers
        model_config=config.model,  # if labeller is a custom LLM on transformers
    )

    # if al_strategy.random_init or config.data.few_shot.count > 0:
    if al_strategy.random_init:
        labeled_data, labeled_ids = get_initial_labeled_data(
            config, unlabeled_data, labeller
        )
        unlabeled_data = unlabeled_data.filter(
            lambda x: x["id"] not in set(labeled_ids)
        )

        save_labeled_data(
            labeled_data=labeled_data,
            labeled_query=labeled_data,
            workdir=workdir,
            iter_dir=workdir / "init_iteration",
            labeled_ids=labeled_ids,
            query_ids=labeled_ids,
        )
    else:
        labeled_data = unlabeled_data.select(range(0, 0))
        labeled_ids = []

    # if config.data.few_shot.count > 0:
    #     few_shot_examples = labeled_data.train_test_split(
    #         train_size=config.data.few_shot.count, shuffle=True, seed=config.seed
    #     )["train"]
    #     with open(workdir / "few_shot_ids.json", "w") as f:
    #         json.dump(few_shot_examples["id"], f)
    #     few_shot_prefix = config.data.few_shot.separator.join(few_shot_examples["input"]) + config.data.few_shot.separator
    #     unlabeled_data = unlabeled_data.map(lambda row: add_prefix_to_input(row, few_shot_prefix))
    #     labeled_data = labeled_data.map(lambda row: add_prefix_to_input(row, few_shot_prefix))
    #     test_data = test_data.map(lambda row: add_prefix_to_input(row, few_shot_prefix))

    if has_test:
        test_data = tokenize_dataset(
            config.data, config.model, test_data, tokenizer, "test"
        )
    # Start AL cycle. Use `num_al_iterations + 1` because we do not label data on the last iteration.
    for al_iter in range(num_al_iterations + 1):
        log.info(f"Starting AL iteration #{al_iter}.")

        iter_dir = workdir / ("iter_" + str(al_iter))
        iter_dir.mkdir(exist_ok=True)

        log.info(f"Iteration {al_iter}: model loading started...")
        model = load_model(model_name, config.model, cache_dir)
        model = prepare_model_for_training(model, config.model.peft)
        log.info(f"Iteration {al_iter}: model loading done.")

        train_eval_data = tokenize_dataset(
            data_config=config.data,
            model_config=config.model,
            dataset=labeled_data,
            tokenizer=tokenizer,
            split="train",
        )
        if dev_split_size > 0 and len(train_eval_data) > 1:
            train_eval_data = train_eval_data.train_test_split(
                test_size=dev_split_size, shuffle=True, seed=seed
            )
            train_data = train_eval_data["train"]
            eval_data = train_eval_data["test"]
        else:
            train_data = train_eval_data
            eval_data = None

        train_output_dir = workdir / "tmp"
        save_dir = workdir / "tmp_best"

        # Set seed for reproducibility
        set_seed(seed)
        trainer = get_trainer(
            config=config.training.hyperparameters,
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            eval_data=eval_data,
            seed=seed,
            output_dir=train_output_dir,
            do_quantize=config.model.quantize,
        )

        # Launch training
        if len(train_data) > 0:
            train_result = trainer.train()
        else:
            train_result = {}
        del trainer
        rmtree(train_output_dir)

        model = model.to(torch.float32).eval().merge_and_unload()

        if not has_test:
            if dev_split_size > 0:
                test_data = eval_data
        if test_data is not None and len(test_data) > 0:
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

            if required_performance_dict is not None:
                # TODO: let the user decide any / all
                is_performance_reached = all(
                    [
                        metrics[metric] >= required_performance
                        for metric, required_performance in required_performance_dict.items()
                    ]
                )
        else:
            metrics = {}
            generations = []
        log.info(metrics)
        with open(iter_dir / "train_result.json", "w") as f:
            json.dump(train_result, f)
        with open(iter_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)
        with open(iter_dir / "generations.json", "w") as f:
            json.dump(generations, f)
        combine_results(workdir, al_iter)

        log.info(f"Iteration {al_iter}: saving the trained model...")
        model.save_pretrained(workdir / "model.bin")

        # Make AL query for the next round if we have not run out of iterations
        if al_iter != num_al_iterations:
            log.info(f"Making AL query at iteration {al_iter}.")
            query_ids = al_strategy(
                model=model,
                tokenizer=tokenizer,
                unlabeled_pool=unlabeled_data.remove_columns(output_column_name),
                labeled_pool=labeled_data,
                input_column_name=input_column_name,
                output_column_name=output_column_name,
                num_to_label=al_query_size,
                batch_size=config.inference.batch_size,
                max_new_tokens=config.inference.max_new_tokens,
            )

            query = unlabeled_data.filter(lambda x: x["id"] in query_ids)
            unlabeled_data = unlabeled_data.filter(lambda x: x["id"] not in query_ids)
            labeled_query = labeller(query)
            if labeller.is_out_of_budget:
                log.info(f"Labeler ran out of budget at iteration {al_iter}.")
            labeled_data = concatenate_datasets([labeled_data, labeled_query])
            labeled_ids += query_ids

            log.info(f"Saving labeled data at iteration #{al_iter}.")
            save_labeled_data(
                labeled_data=labeled_data,
                labeled_query=labeled_query,
                workdir=workdir,
                iter_dir=iter_dir,
                labeled_ids=labeled_ids,
                query_ids=query_ids,
            )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if labeller.is_out_of_budget:
            log.info("Labeler ran out of budget. Finishing active learning.")
            return
        if is_performance_reached:
            logging.info("Stopping AL since the required performance is reached.")

    log.info("Active learning is done.")


@hydra.main(
    config_path=os.environ.get("HYDRA_CONFIG_PATH", "../configs/"),
    config_name=os.environ.get("HYDRA_CONFIG_NAME"),
    version_base="1.1",
)
def main(config):
    if getattr(config, "debug", True):
        try:
            run_active_learning(config)
        except Exception as e:
            print(e)
            import sys, pdb

            exc_type, exc_value, exc_traceback = sys.exc_info()
            pdb.post_mortem(exc_traceback)
    else:
        run_active_learning(config)


if __name__ == "__main__":
    main()
