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

# from experiments.musique.inference_only import eval_inferencer, macro_averaging
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from copy import deepcopy

from knowledge_propagation.modules.evaluators import (
    ExactMatchEvaluator,
    RougeEvaluator,
    OpenAIEvaluator,
    NumDiffEvaluator,
)

import models
import utils
from utils import EditLoss, EditInput
from typing import Iterable

OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

logging.basicConfig(format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


em_evaluator = ExactMatchEvaluator()
rouge_evaluator = RougeEvaluator()
llm_evaluator = OpenAIEvaluator()
year_diff_evaluator = NumDiffEvaluator()


def score_df(df):
    em_per_example = em_evaluator.compute_metric(
        predictions=df["predicted_answer"],
        references=df["answer"],
        use_aggregator=False,
    )

    model_response_w_score = df.join(
        pd.DataFrame(
            {
                **em_per_example,
            }
        )
    )
    return model_response_w_score


def add_padding(tokenizer, model):
    import transformers

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM) and not isinstance(model, transformers.Qwen2ForCausalLM):
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
    # import pdb; pdb.set_trace()
    # if "32B" not in os.path.basename(config.model.name):
    model.to(config.device)
    
    generation_config = GenerationConfig(
        do_sample=False,  # Greedy
        top_k=None,
        top_p=None,
        temperature=None,
        max_new_tokens=20, # 32768
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # trainer = EditTrainer(alg, config, train_set, val_set)
    assert hasattr(config, "date_data")
    if config.date_data == "4K_test_id":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_id_entity152_rel31.jsonl")
        config.val_steps = 500
        assert len(val_data) == config.val_steps
    elif config.date_data == "4K_test_ood":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_ood_entity37_rel7.jsonl")
        config.val_steps = 350
        assert len(val_data) == config.val_steps
    elif config.date_data == "4K_test_ood-relation":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_ood-relation_entity152_rel7.jsonl")
        config.val_steps = 350
        assert len(val_data) == config.val_steps
    elif config.date_data == "4K_test_ood-entity":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_ood-entity_entity37_rel31.jsonl")
        config.val_steps = 350
        assert len(val_data) == config.val_steps
    elif config.date_data == "30K_test_id":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/30Ktrain_data_100percent_frozen/test_text_data_id_entity152_rel31.jsonl")
        config.val_steps = 500
    elif config.date_data == "profiling":
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_id_entity152_rel31.jsonl")
        config.val_steps = 1 # 50
        val_data = val_data[: config.val_steps]
        assert len(val_data) == config.val_steps
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
    assert config.do_generation, "Generation is required for this script"
    # import pdb; pdb.set_trace()
    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        datum = val_data[i]
        
        if config.ice:
            assert datum["text"].startswith(datum["subject"])
            if datum["subject_type"] == "person":
                prepend_txt = "Imagine that someone named " + datum["text"]
            else:
                assert datum["subject_type"] == "company"
                prepend_txt = "Imagine that a company named " + datum["text"]

        question_types = [
            ("efficacy", datum["questions"]),
        ]
        
        for question_type, test_queries in question_types:
            for question_key in ["alias_question", "unalias_question"][:]:
                for q_i, test_query in enumerate(test_queries):
                    q_str_only = test_query[question_key].strip()
                    test_queries_q_str = deepcopy(q_str_only)
                    if config.ice:
                        # import pdb; pdb.set_trace()
                        test_queries_q_str = prepend_txt + " " + test_query[question_key].strip()
                    
                    test_queries_a_str = str(test_query["answer"]).strip()
                    
                    
                    messages = [
                        # {
                        #     "role": "system",
                        #     "content": ""
                        # },
                        {
                            "role": "user",
                            "content": test_queries_q_str,
                        }
                    ]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # import pdb; pdb.set_trace()
                    pre_result_df = generate(
                        text, test_queries_a_str, config, model, tokenizer, generation_config
                    )
                    # import pdb; pdb.set_trace()
                    pre_result_df.insert(0, "full_input", text)
                    pre_result_df.insert(0, "input", test_queries_q_str)
                    pre_result_df.insert(0, "stage", "pre-edit")
                    pre_result_df["question"] = q_str_only
                    pre_result_df.insert(0, "question_type", question_type)
                    pre_result_df.insert(0, "question_key", question_key)
                    pre_result_df.insert(0, "id", str(i))
                    # import pdb; pdb.set_trace()
                    all_results.append(pre_result_df)

    all_results = pd.concat(all_results)

    if config.generation.save_dir:
        save_dir = config.generation.save_dir
        if os.path.abspath(config.generation.save_dir) != config.generation.save_dir:
            # using relative path
            save_dir = f"{base_dir}/{config.generation.save_dir}"
        save_dir = os.path.join(save_dir, config.date_data)
        # LOG.info(f"Saving to dir: {save_dir}")

        os.makedirs(save_dir, exist_ok=True)
        fpath = f"{save_dir}/base_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if config.do_generation else 'wo'}-gen_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl_ice={config.ice}.xlsx"
        LOG.info(f"Saving to dir: {fpath}")
        all_results.to_excel(
            fpath,
            index=False,
        )


if __name__ == "__main__":
    run()
