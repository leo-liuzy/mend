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

import gc
from trainer import EditTrainer
from knowledge_propagation.utils import io, vars, extractor
from knowledge_propagation.modules.inferencers import QAInferencer

# from experiments.musique.inference_only import eval_inferencer, macro_averaging
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

from knowledge_propagation.modules.evaluators import (
    ExactMatchEvaluator,
    RougeEvaluator,
    OpenAIEvaluator,
)
from typing import List, Dict

from copy import deepcopy
import models
import utils
from utils import EditLoss, StrEnum

from utils_leo_date import get_eval_result, add_eos


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

logging.basicConfig(format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


icl_prompt = "\n".join(
    [
        "Q: When did the simpsons first air on television?",
        "A: December 17, 1989",
        "Q: Who has more super bowl wins afc or nfc?",
        "A: NFC",
        "Q: Is the federal court the same as the supreme court?",
        "A: No",
    ]
)


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
    return model_response  # score_df(model_response)


def add_padding(tokenizer, model):
    import transformers

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM) and not isinstance(model, transformers.Qwen2ForCausalLM):
        #     model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight.mean(0)
        # else:
        model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


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

    
    print("Task: ", config.task)

    assert hasattr(config, "spec_question")
    assert hasattr(config, "date_data")
    # assert hasattr(config, "n_edit")
    # import pdb

    # pdb.set_trace()
    if config.date_data == "4K_test_id":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_id_entity152_rel31.jsonl"
        # )
        config.val_steps = 500
        # assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "4K_test_ood":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_ood_entity37_rel7.jsonl"
        # )
        config.val_steps = 350
        # assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "4K_test_ood-relation":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_ood-relation_entity152_rel7.jsonl"
        # )
        config.val_steps = 350
        # assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "4K_test_ood-entity":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_ood-entity_entity37_rel31.jsonl"
        # )
        config.val_steps = 350
        # assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "30K_test_id":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/30Ktrain_data_100percent_frozen/test_text_data_id_entity152_rel31.jsonl"
        # )
        config.val_steps = 500
        # assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "30K_test_ood":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/30Ktrain_data_100percent_frozen/test_text_data_ood_entity37_rel7.jsonl"
        # )
        config.val_steps = 100
        # assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "30K_test_ood-relation":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/30Ktrain_data_100percent_frozen/test_text_data_ood-relation_entity152_rel7.jsonl"
        # )
        config.val_steps = 350
        # assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "30K_test_ood-entity":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/30Ktrain_data_100percent_frozen/test_text_data_ood-entity_entity37_rel31.jsonl"
        # )
        config.val_steps = 350
        # assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "profile":
        # edit_dev_dataset = io.load_jsonlines(
        fpath = f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_text_data_id_entity152_rel31.jsonl"
        # )
        config.val_steps = 50
        # assert len(edit_dev_dataset) == config.val_steps
    else:
        raise NotImplementedError(f"date_data `{config.date_data}` is not supported")
    # else:
    # else:
    #     assert config.date_data == "n"
    #     edit_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/test_n_question.jsonl")
    from data_classes.syn_story import SynStoryDataset
    train_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/train_text_data_id_entity152_rel31.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
    edit_dev_dataset = SynStoryDataset(tokenizer, fpath, config, size=config.val_steps, is_eval=True, max_length=tokenizer.model_max_length, ) # 
    # import pdb; pdb.set_trace()
    trainer = EditTrainer(alg, config, train_set, edit_dev_dataset)
    all_results = []
    edit_model_infos = []
    # trainer.validate(log=True)
    assert config.val_steps <= len(edit_dev_dataset)
    assert config.eval_only
    if hasattr(config, "add_icl") and config.add_icl:
        eos_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    else:
        eos_token_id = tokenizer.eos_token_id
    trainer.model.train(False)
    print(f"Running eval on {config.task}")
    print(trainer.validate(steps=100, log=True))


if __name__ == "__main__":
    run()
