from os import listdir

from datasets import load_dataset
from transformers import (
    GenerationMixin,
    PreTrainedTokenizerBase,
)

"""
https://arxiv.org/abs/2210.09261v1

Prompt options (answer only or chain-of-thought) can be found in
../../prompts/big-bench-hard
"""


class BigBenchHardMetric:
    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: PreTrainedTokenizerBase,
        prompts_path: str = "../../prompts/big-bench-hard/answer-only",
        cache_dir: str = "./cache",
        **generation_kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        self.cache_dir = cache_dir

        self.prompts = {}
        for filename in listdir(prompts_path):
            if not filename.endswith(".txt"):
                continue

            with open(f"{prompts_path}/{filename}") as file:
                content = file.read()
                SEP = "-----\n"
                if SEP in content:
                    content = content.split(SEP)[-1]

                self.prompts[filename.split(".")[0]] = content

    def task_score(self, task_name: str) -> float:
        tasks = load_dataset(
            "maveriq/bigbenchhard",
            task_name,
            cache_dir=self.cache_dir,
        )["train"][:]

        prompt_template = self.prompts[task_name]
        prompts = [
            prompt_template.format(text=task_input) for task_input in tasks["input"]
        ]
        targets = [task_target.lower().strip() for task_target in tasks["target"]]

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model.generate(
            **inputs,
            **self.generation_kwargs,
        )
        generated = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        n_correct = 0
        for text, target in zip(generated, targets):
            if text.endswith("."):
                text = text[:-1]
            if task_name in ["dyck_languages", "word_sorting"]:  # multiple word answers
                text = (
                    text.split("A:")[-1].lower().split("so the answer is")[-1].strip()
                )
            else:
                text = text.split()[-1].strip().lower()

            n_correct += int(text == target)

        return n_correct / len(targets)

    def score(self) -> float:
        scores = [self.task_score(task_name) for task_name in self.prompts]
        return sum(scores) / len(scores)
