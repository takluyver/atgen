from typing import List, Dict
from tqdm.notebook import tqdm
import re
import openai
import fuckit
import pandas as pd
import numpy as np
from time import sleep, time
from IPython.display import display, clear_output
from itertools import chain


openai.api_key = os.environ['OPENAI_API_KEY']


POST_PROCESS_FUNCS = {
    "single": lambda x: int(x[0]),
    "multiple": lambda x: list(map(int, x)),
}


PROMPT_CONFIGS = {
    "best": dict(
        system="Determine the best summary for the source text.",
        user=lambda x: "Text:\n{text}\n\n"
        + "".join((f"Summary {i}:\n{{sum_{i}}}\n\n" for i in range(1, x + 1)))
        + "Output the id of the summary, which suits the source text the best:",
        output="single",
    ),
    "consistency": dict(
        system="Determine all summaries that are inconsistent with the source text or contradict it. If there are no inconsistent summaries, output 0.",
        user=lambda x: (
            "Source text:\n{text}\n\n"
            + "".join((f"Summary {i}:\n{{sum_{i}}}\n\n" for i in range(1, x + 1)))
            + "Output the ids of the summaries that contradict or are inconsistent with the source text. "
            + "If there are no such summaries, output 0:"
        ),
        output="multiple",
    ),
    "informativeness": dict(
        system="Determine all summaries that do not capture the key information from the source text.",
        user=lambda x: (
            "Source text:\n{text}\n\n"
            + "".join((f"Summary {i}:\n{{sum_{i}}}\n\n" for i in range(1, x + 1)))
            + "Output the ids of the summaries that struggle to maintain the most important information or do not capture the key information from the source text. If there are no such summaries, output 0:"
        ),
        output="multiple",
    ),
    "fluency": dict(
        system="Your goal is to determine the texts that are do not sound fluent or do not look like written by a human. This includes texts that look like AI-written, contain errors, do not look fluent or lack coherence among sentences.",
        user=lambda x: (
            "".join((f"Text {i}:\n{{sum_{i}}}\n\n" for i in range(1, x + 1)))
            + "Output the ids of the texts that look like AI-written, are erroneous, are not fluent, or are poorly written. "
            + "If there are no such texts, output 0:"
        ),
        output="multiple",
    ),
    "hallucination": dict(
        system="Determine all summaries that hallucinate facts from the source text. If there are no hallucinating summaries, output 0.",
        user=lambda x: (
            "Source text:\n{text}\n\n"
            + "".join((f"Summary {i}:\n{{sum_{i}}}\n\n" for i in range(1, x + 1)))
            + "Output the ids of the summaries that hallucinate information from the source text. "
            + "If there are no such summaries, output 0:"
        ),
        output="multiple",
    ),
#     "coherence": dict(
#         system="Determine the texts that look incoherent.",
#         user=lambda x: (
#             "".join((f"Text {i}:\n{{sum_{i}}}\n\n" for i in range(1, x + 1)))
#             + "Output the ids of the texts that lack coherence among sentences. "
#             + "If there are no such texts, output 0:"
#         ),
#         output="multiple",
#     ),
}


@fuckit
def openai_gpt4_request(prompt_config, text_kwargs):
    # Number of summaries == `len(kwargs) - 1` (excluding one source text)
    user_message = prompt_config["user"](len([x for x in text_kwargs if x.startswith("sum_")]))
    kwargs = {
        k: v for k, v in text_kwargs.items() if (f"{{{k}}}") in user_message
    }
    sleep_time = iter((0, 5, 30, 120, 600, 1000, None))
    output = None
    while output is None:
        iter_sleep_time = next(sleep_time)
        if iter_sleep_time is None:
            return
        elif iter_sleep_time >= 30:
            print(f"Sleeping {iter_sleep_time:.0f} s")
        sleep(iter_sleep_time)
        output = openai.ChatCompletion.create(
            model="gpt-4o",
            # model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=15,
            messages=[
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": user_message.format(**kwargs)},
            ],
        )
        content = output["choices"][0]["message"]["content"].strip()
        content = re.findall("\d+", content)
        output = POST_PROCESS_FUNCS[prompt_config["output"]](content)
    return output


def calculate_metrics(text_kwargs: Dict[str, str]):
    results = {}
    for metric_name, prompt_config in PROMPT_CONFIGS.items():
        openai_result = openai_gpt4_request(prompt_config, text_kwargs)
        if openai_result is None:
            raise ValueError
        results[metric_name] = openai_result
    return results


def compare_results(
    source_texts: List[str], golden_output: List[str] = None, *summaries: List[str], columns: List[str] = None
) -> Dict[str, List[str]]:
    if columns is None:
        columns = list(range(len(summaries)))
    if golden_output is None:
        all_data = (source_texts,) + summaries
    else:
        all_data = (source_texts, golden_output) + summaries
    # TODO: add comparison with golden_output
    results = {key: [] for key in PROMPT_CONFIGS.keys()}
    pbar = tqdm(total=len(source_texts), desc="Num instances processed")

    for text_data in zip(*all_data):
        text_kwargs = {"text": text_data[0]}
        text_kwargs.update(
            {f"sum_{i}": summary for i, summary in enumerate(text_data[1:], 1)}
        )
        instance_results = calculate_metrics(text_kwargs)
        for key, value in instance_results.items():
            results[key].append(value)
        clear_output(wait=True)
        pbar.update()
        df = pd.DataFrame(columns=["None"] + columns)
        for key, value in results.items():
            if isinstance(value[0], int):
                df.loc[key] = np.bincount(value, minlength=df.shape[1])
            else:
                df.loc[key] = np.bincount(list(chain.from_iterable(value)), minlength=df.shape[1])
        display(df)

    return results

