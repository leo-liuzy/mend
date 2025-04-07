import copy
import importlib
import logging
import random

import os
import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from trainer import EditTrainer
from knowledge_propagation.utils import io, vars, extractor
from knowledge_propagation.modules.inferencers import QAInferencer
from experiments.musique.inference_only import eval_inferencer, macro_averaging
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

from knowledge_propagation.modules.evaluators import (
    ExactMatchEvaluator,
    RougeEvaluator,
    OpenAIEvaluator,
    NumDiffEvaluator,
)

import models
import utils
from utils import EditLoss, EditInput
from typing import Iterable, List

OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

logging.basicConfig(format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


em_evaluator = ExactMatchEvaluator()
rouge_evaluator = RougeEvaluator()
llm_evaluator = OpenAIEvaluator()
year_diff_evaluator = NumDiffEvaluator()

icl_prompt = "\n".join(
    [
        "Q: When did the simpsons first air on television?",
        "A: 1989",
        "Q: When was Jesus born?",
        "A: 6 to 4 BC",
        "Q: What year did the United State declare independence?",
        "A: 1776",
    ]
)


def score_df(df):
    # em_per_example = em_evaluator.compute_metric(
    #     predictions=df["predicted_answer"],
    #     references=df["answer"],
    #     use_aggregator=False,
    # )
    # rouge_per_example = rouge_evaluator.compute_metric(
    #     predictions=df["predicted_answer"],
    #     references=df["answer"],
    #     use_aggregator=False,
    # )

    diff_per_example = year_diff_evaluator.compute_metric(
        predictions=df["predicted_answer"],
        references=df["answer"],
        use_aggregator=False,
    )

    model_response_w_score = df.join(pd.DataFrame({**diff_per_example}))
    return model_response_w_score


def add_padding(tokenizer, model):
    import transformers

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM):
        #     model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight.mean(0)
        # else:
        model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


def add_eos(tokenizer_output, eos_token_id, ignore=False):
    if ignore:
        return tokenizer_output
    return {
        k: torch.concat(
            [
                v,
                torch.full(
                    (v.shape[0], 1),  # shape of the constant tensor
                    (
                        1
                        if k == "attention_mask"
                        else eos_token_id  # this is to teach the model to end after outputing the answer.
                    ),
                ),
            ],
            dim=-1,
        )
        for k, v in tokenizer_output.items()
    }


def generate(
    context: str,
    answer: str,
    config,
    model,
    tokenizer,
    generation_config,
):
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.gen_w_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]

    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    print(
        "Input for generation:",
        "[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"])) + "]",
    )
    print("Label for generation:", "[" + answer + "]")
    print("--------------------")

    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    generated_texts = [t.replace(ctx_decoded, "") for t in generated_texts]
    predicted_answer = generated_texts[0]
    if hasattr(config, "add_icl") and config.add_icl:
        # if using ICL, extract by the first new line
        if "\n" in predicted_answer:
            predicted_answer = predicted_answer[: predicted_answer.find("\n")]

    model_response = pd.DataFrame(
        [
            {
                "question": context,
                "answer": answer.strip(),
                "predicted_answer_idx": 0,
                "predicted_answer": predicted_answer.strip(),
            }
        ]
    )
    return score_df(model_response)


def generate_multi_answers(
    context: str,
    answers: List[str],
    config,
    model,
    tokenizer,
    generation_config,
):
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.gen_w_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]

    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    print(
        "Input for generation:",
        "[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"])) + "]",
    )
    print("Label for generation:", "[" + str(answers) + "]")
    print("--------------------")

    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    generated_texts = [t.replace(ctx_decoded, "") for t in generated_texts]
    predicted_answer = generated_texts[0]
    if hasattr(config, "add_icl") and config.add_icl:
        # if using ICL, extract by the first new line
        if "\n" in predicted_answer:
            predicted_answer = predicted_answer[: predicted_answer.find("\n")]

    model_response = pd.DataFrame(
        [
            {
                "question": context,
                "answer": answers,
                "predicted_answer_idx": 0,
                "predicted_answer": predicted_answer.strip(),
            }
        ]
    )
    return score_df(model_response)


@hydra.main(config_path="config", config_name="config")
def run(config):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)
    add_padding(tokenizer, model)

    from data_classes.zsre import ZsreDataset

    train_set = ZsreDataset(
        tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-train-new_annotated_final.jsonl", config
    )
    val_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config)

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))

    generation_config = GenerationConfig(
        do_sample=False,  # Greedy
        top_k=None,
        top_p=None,
        temperature=None,
        max_new_tokens=20,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    trainer = EditTrainer(alg, config, train_set, val_set)
    assert hasattr(config, "date_data")
    if config.date_data == "common":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_date_data/valid.jsonl")
    elif config.date_data == "bio_syn_v2":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/test.jsonl")
    elif config.date_data == "bio_syn_v2_ood":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/test_ood.jsonl")
    else:
        raise ValueError(f"Unknown date_data: {config.date_data}")

    all_results = []
    edit_model_infos = []
    # trainer.validate(log=True)
    assert config.val_steps <= len(val_data)
    assert config.eval_only

    assert hasattr(config, "ice")
    
    if hasattr(config, "add_icl") and config.add_icl:
        eos_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    else:
        eos_token_id = tokenizer.eos_token_id

    # question_key = config.quesiton_key if hasattr(config, "quesiton_key") else "question"
    
    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        # for i in tqdm([717, 718, 719], desc=f"Running eval on {config.task}"):
        # for i in tqdm(range(1), desc=f"Running eval on {config.task}"):
        datum = val_data[i]
        # import pdb

        # pdb.set_trace()
        if config.date_data == "common":
            test_queries = [
                {"question": datum["question"], "answer": datum["answer"]}
                # {"question": datum["year_after_question"], "answer": datum["year_after_answer"]}
            ]
        elif "_ood" in config.date_data:
            test_queries = datum["ood_questions"]
        else:
            test_queries = datum["questions"]

        # prepare [Q][A] accuracy and generation inputs

        # import pdb

        # pdb.set_trace()
        # assert len(test_queries) == 1, "# TODO: make this support multiple input"
        for question_key in ["question", "unaliased_question"]:
            for test_query in test_queries:
                if config.ice:
                    test_queries_q_str = datum["text"] + "\n\n" + test_query[question_key].strip()
                else:
                    test_queries_q_str = test_query[question_key].strip()

                if config.do_generation:
                    pre_result_df = generate_multi_answers(
                        test_queries_q_str, test_query["answer"], config, trainer.model.model, tokenizer, generation_config
                    )
                else:
                    pre_result_df = pd.DataFrame([{"predicted_answer_idx": 0}])
                assert len(pre_result_df) == 1

                pre_result_df.insert(0, "question_key", question_key)
                pre_result_df.insert(0, "input", "\n\n".join(f"[[{s}]]" for s in [test_queries_q_str]))
                pre_result_df.insert(1, "stage", "pre-edit")
                pre_result_df.insert(0, "question_tag", test_query["question_type"])
                pre_result_df.insert(0, "id", str(i))

                all_results.append(pre_result_df)

    all_results = pd.concat(all_results)

    if config.generation.save_dir:
        save_dir = config.generation.save_dir
        if os.path.abspath(config.generation.save_dir) != config.generation.save_dir:
            # using relative path
            save_dir = f"{base_dir}/{config.generation.save_dir}"
        save_dir = os.path.join(save_dir, config.date_data)
        LOG.info(f"Saving to dir: {save_dir}")

        os.makedirs(save_dir, exist_ok=True)
        all_results.to_excel(
            f"{save_dir}/base_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if config.do_generation else 'wo'}-gen_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl_ice={config.ice}.xlsx",
            index=False,
        )


if __name__ == "__main__":
    run()
