import gc
from math import ceil
from pathlib import Path
import subprocess

from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch import bfloat16, float16, cuda, no_grad
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, DataCollatorForSeq2Seq

from .load_data import tokenize_dataset


def generate_vllm(
    config: DictConfig,
    data: Dataset,
    model: PreTrainedModel = None,
    tokenizer: PreTrainedTokenizer = None,
    save_dir: str | Path = "tmp",
    model_tokenizer_dir: str | Path = None,
    llm_runner: "LLM" = None,
    **useless_kwargs,
) -> list[str]:
    """
    TODO: improve the description.
    Function for generating with the vLLM framework.
    Requires either model + tokenizer + save_dir or the path to the saved model and tokenizer.
    In the last case, they need to be stored inside "PATH/model" and "PATH/tokenizer".
    """
    from vllm import LLM, SamplingParams

    delete_vllm_after_inference = False
    if llm_runner is None:
        if model_tokenizer_dir is None:
            model.save_pretrained(f"{save_dir}/model")
            tokenizer.save_pretrained(f"{save_dir}/tokenizer")
            model_tokenizer_dir = save_dir
        gpu_memory_utilization = getattr(config, "gpu_memory_utilization", 0.5)
        llm_runner = LLM(
            f"{model_tokenizer_dir}/model",
            f"{model_tokenizer_dir}/tokenizer",
            gpu_memory_utilization=gpu_memory_utilization,  # TODO: make arbitrary
            dtype=bfloat16,
            trust_remote_code=True,
        )
        delete_vllm_after_inference = True

    del model
    gc.collect()
    cuda.empty_cache()

    params = SamplingParams(
        temperature=config.temperature,
        seed=42,  # TODO: make arbitrary
        max_tokens=config.max_new_tokens,
        top_p=config.top_p,
    )

    generations = []
    num_batches = ceil(len(data) / config.batch_size)
    for i in tqdm(range(num_batches)):
        batch = data[i * config.batch_size : (i + 1) * config.batch_size]["input"]
        out = llm_runner.generate(batch, params, use_tqdm=False)
        outputs = [x.outputs[0].text for x in out]
        generations += outputs

    if delete_vllm_after_inference:
        del llm_runner
        gc.collect()
        cuda.empty_cache()
    return generations


def generate_transformers(
    config: DictConfig,
    data: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data_config: DictConfig = None,
    model_config: DictConfig = None,
    **useless_kwargs,
) -> list:
    # Tokenize dataset if necessary
    if "input_ids" not in data.column_names:
        data = tokenize_dataset(
            data_config=data_config,
            model_config=model_config,
            dataset=data,
            tokenizer=tokenizer,
            split="test",
        )
    dataloader = DataLoader(
        data.remove_columns(
            [x for x in data.column_names if x not in ("input_ids", "attention_mask")]
        ),
        batch_size=config.batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer),
        shuffle=False,
    )

    if model.dtype != float16:
        model = model.to(float16)
    if cuda.is_available() and model.device.type != "cuda":
        model = model.cuda()
    generations = []
    with no_grad():
        for batch in tqdm(dataloader):
            out = model.generate(
                batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                return_dict_in_generate=True,
                output_scores=True,
            )
            # outputs = tokenizer.batch_decode(out.sequences, True)
            outputs_only = [
                tokenizer.decode(x[len(y) :], True).strip()
                for (x, y) in zip(out.sequences, batch["input_ids"])
            ]
            generations += outputs_only
    return generations


def generate_tllm(
    config: DictConfig,
    data: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    **useless_kwargs,
) -> list:
    from tensorrt_llm.runtime.model_runner_cpp_new import ModelRunnerCpp

    model.save_pretrained(f"tmp/{exp_name}/model")
    tokenizer.save_pretrained(f"tmp/{exp_name}/model")

    os.chdir("TensorRT-LLM/examples/llama/")
    subprocess.run(
        [
            "python",
            "./convert_checkpoint.py",
            "--model_dir",
            f"../../../tmp/{exp_name}_best/model",
            "--output_dir",
            f"../../../tmp/{exp_name}_best/tllm_checkpoint",
            "--dtype",
            "float16",
        ]
    )
    os.chdir("../../..")
    subprocess.run(
        [
            "trtllm-build",
            "--checkpoint_dir",
            f"./tmp/{exp_name}_best/tllm_checkpoint",
            "--output_dir",
            f"./tmp/{exp_name}_best/tllm_model",
            "--gemm_plugin",
            "float16",
            "--max_batch_size",
            str(config.inference.batch_size),
            "--max_input_len",
            "1250",
            "--max_output_len",
            str(max_new_tokens),
            "--gather_all_token_logits",
        ]
    )

    runner = ModelRunnerCpp.from_dir(
        engine_dir=f"./tmp/{exp_name}_best/tllm_model",
        lora_dir=None,
        rank=0,
        lora_ckpt_source="hf",
        max_batch_size=config.inference.batch_size,
        max_input_len=1250,
        max_output_len=max_new_tokens,
        max_beam_width=1,
        max_attention_window_size=4096,
        sink_token_length=None,
        free_gpu_memory_fraction=0.2,
    )

    generations_with_inputs = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs = runner.generate(
                batch["input_ids"],
                max_new_tokens=max_new_tokens,
                end_id=tokenizer.eos_token_id,
                pad_id=tokenizer.eos_token_id,
                do_sample=False,
                streaming=False,
                output_sequence_lengths=True,
                return_dict=True,
            )
            generations_with_inputs += tokenizer.batch_decode(
                outputs["output_ids"][:, 0], True
            )

    generations = []
    for i, gen_with_input in enumerate(generations_with_inputs):
        gen_splitted = gen_with_input.split("\nSummary:\n")
        if len(gen_splitted) > 1:
            gen = gen_splitted[1].strip().split("\n")[0].strip()
        else:
            gen = tokenizer.decode(
                tokenizer(gen_with_input, max_length=2048)["input_ids"][
                    -max_new_tokens:
                ]
            )
        generations.append(gen)


def generate(
    config: DictConfig,
    data: Dataset,
    **kwargs,
) -> list[str]:
    framework = config.framework
    if framework == "vllm":
        return generate_vllm(
            config=config,
            data=data,
            **kwargs,
        )
    elif framework == "transformers":
        return generate_transformers(
            config=config,
            data=data,
            **kwargs,
        )
    elif framework == "tllm":
        return generate_tllm(
            config=config,
            data=data,
            **kwargs,
        )
    else:
        raise NotImplementedError
