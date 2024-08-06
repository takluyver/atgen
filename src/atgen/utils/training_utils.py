from pathlib import Path

import torch
from datasets import Dataset
from omegaconf import DictConfig
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def get_training_args(
    config: DictConfig,
    train_data: Dataset,
    eval_data: Dataset,
    seed: int,
    output_dir: str,
    do_quantize: bool,
) -> TrainingArguments:
    if eval_data is None:
        evaluation = "none"
    else:
        evaluation = "epoch"
    return TrainingArguments(
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=min(config.train_batch_size, len(train_data)),
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="linear",
        fp16=False if do_quantize or not torch.cuda.is_available() else True,
        fp16_full_eval=False,
        bf16=True if do_quantize and torch.cuda.is_available() else False,
        bf16_full_eval=False,
        load_best_model_at_end=True,
        evaluation_strategy=evaluation,
        logging_strategy=evaluation,
        save_strategy=evaluation,
        save_total_limit=1,
        seed=seed,
        output_dir=output_dir,
        optim="paged_adamw_8bit" if do_quantize else config.optimizer,
        report_to="none",
        include_num_input_tokens_seen=True,
        remove_unused_columns=True,
    )


def get_trainer(
    config: DictConfig,
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_data: Dataset,
    eval_data: Dataset,
    seed: int,
    output_dir: str | Path,
    do_quantize: bool,
) -> Trainer:
    train_args = get_training_args(config, train_data, eval_data, seed, output_dir, do_quantize)
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
    ] if eval_data is not None else []
    return Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=train_args,
        data_collator=data_collator,
        callbacks=callbacks,
    )
