from time import time
from typing import List, Dict, Tuple, Union
import logging


import numpy as np
from evaluate import load
from tqdm import tqdm

from .metrics import (
    pair_bleu,
    calculate_bart_score,
    calculate_alignscore,
    calculate_summac_score,
    calculate_gpt4o_score,
)


log = logging.getLogger()


def compute_metrics(
    generated_texts,
    reference_texts,
    original_texts,
    add_metrics_to_use: Union[Tuple[str], List[str]] = (
        "bartscore",
        "alignscore",
    ),
    cache_dir: str = "cache",
) -> Dict[str, float]:
    sacrebleu = load("sacrebleu", cache_dir=cache_dir)
    rouge = load("rouge", cache_dir=cache_dir)

    result = {}
    result["word_length_gen"] = np.array(
        [len(text.split()) for text in generated_texts]
    )

    time_dict = {}

    ### Metrics that use both the generated texts and the original texts and
    ### those than are not ''obliged'' to use reference texts
    # Lengths
    src_word_lengths = [len(text.split()) for text in original_texts]
    result["word_length_src_rel"] = result["word_length_gen"] / src_word_lengths
    if "summac" in add_metrics_to_use:
        start_time = time()
        result.update(
            calculate_summac_score(generated_texts, original_texts, reference_texts)
        )
        time_dict["time_summac"] = time() - start_time
    if "bartscore" in add_metrics_to_use:
        log.info("Calculating BARTScore scores...")
        start_time = time()
        result.update(
            calculate_bart_score(
                preds=generated_texts,
                texts=original_texts,
                refs=reference_texts,
                batch_size=4,
                cache_dir=cache_dir,
            )
        )
        time_dict["time_bartscore"] = time() - start_time
    ### Metrics that use both the generated texts and the reference texts
    if reference_texts is not None:
        # BLEU
        start_time = time()
        result["bleu"] = np.array(
            [
                pair_bleu(pred, ref)
                for pred, ref in tqdm(zip(generated_texts, reference_texts))
            ]
        )
        time_dict["time_bleu"] = time() - start_time
        # ROUGE
        start_time = time()
        result.update(
            rouge.compute(
                predictions=generated_texts,
                references=reference_texts,
                use_stemmer=True,
            )
        )
        time_dict["time_rouge"] = time() - start_time
        # Sacrebleu
        start_time = time()
        sacrebleu_result = sacrebleu.compute(
            predictions=generated_texts, references=[[ref] for ref in reference_texts]
        )
        result["sacrebleu"] = sacrebleu_result.pop("score")
        time_dict["time_sacrebleu"] = time() - start_time
        # Lengths
        ref_word_lengths = [len(text.split()) for text in reference_texts]
        result["word_length_rel"] = result["word_length_gen"] / ref_word_lengths

        if "alignscore" in add_metrics_to_use:
            log.info("Calculating AlignScore scores...")
            start_time = time()
            result.update(
                calculate_alignscore(generated_texts, reference_texts, original_texts)
            )
            time_dict["time_alignscore"] = time() - start_time

        if "gp4o" in add_metrics_to_use:
            log.info("Calculating gpt4o scores...")
            start_time = time()
            result.update(
                calculate_alignscore(generated_texts, reference_texts, original_texts)
            )
            time_dict["time_gpt4o"] = time() - start_time

    for key, value in result.items():
        result[key] = float(np.mean(value))
    result = {key: result[key] for key in sorted(result.keys())}
    result.update(time_dict)

    return result
