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
from experiments.musique.inference_only import eval_inferencer, macro_averaging
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

from knowledge_propagation.modules.evaluators import (
    ExactMatchEvaluator,
    RougeEvaluator,
    OpenAIEvaluator,
)

from copy import deepcopy
import models
import utils
from utils import EditLoss, StrEnum

from utils_leo_date import get_analysis_result, add_eos


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


def add_padding(tokenizer, model):
    import transformers

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM):
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

    from data_classes.zsre import ZsreDataset

    train_set = ZsreDataset(
        tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-train-new_annotated_final.jsonl", config
    )
    val_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config)
    tokenizer = val_set.tok

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
    print("Task: ", config.task)

    assert hasattr(config, "date_data")

    if config.date_data == "n+1":
        edit_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data/test.jsonl")
    else:
        assert config.date_data == "n"
        edit_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data/test_n_question.jsonl")

    # spec_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_date_data/valid.jsonl")

    all_results = []
    edit_model_infos = []
    # trainer.validate(log=True)
    assert config.val_steps <= len(edit_dev_dataset)
    assert config.eval_only
    if hasattr(config, "add_icl") and config.add_icl:
        eos_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    else:
        eos_token_id = tokenizer.eos_token_id

    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        # for i in tqdm([717, 718, 719], desc=f"Running eval on {config.task}"):
        # for i in tqdm(range(1), desc=f"Running eval on {config.task}"):
        datum = edit_dev_dataset[i]

        sentences = [datum["text"]]

        assert config.edit_loss == EditLoss.clm, f"edit_loss `{config.edit_loss}` is not supported"
        sentences_toks = targets_toks = add_eos(
            tokenizer(sentences, padding=True, return_tensors="pt", add_special_tokens=True),
            eos_token_id,
            ignore=not config.add_eos,
        )

        edit_inner = {
            "input_ids": sentences_toks["input_ids"],
            "attention_mask": sentences_toks["attention_mask"],
            "labels": val_set.get_edit_labels(targets_toks["input_ids"]),
        }

        # import pdb; pdb.set_trace()
        print("Input for EDIT: ")
        print("[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(edit_inner["input_ids"])) + "]")
        print("Label for EDIT: ")
        print("[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(targets_toks["input_ids"])) + "]")
        print()

        # import pdb; pdb.set_trace()
        edit_inner = utils.dict_to(edit_inner, config.device)

        # edit the model with MEND
        edited_model, model_info = trainer.model.edit(edit_inner)
        model_info["input"] = sentences[0]
        model_info["decoded_input_ids"] = tokenizer.decode(targets_toks["input_ids"][0])

        question_type = "efficacy"

        analysis_result_dict = get_analysis_result(
            question=datum["question"],
            answer=datum["answer"],
            model=edited_model.model,
            tokenizer=tokenizer,
            config=config,
            generation_config=generation_config,
        )
        analysis_result_dict["question_type"] = question_type
        analysis_result_dict["question"] = datum["question"]
        analysis_result_dict["answer"] = datum["answer"]
        analysis_result_dict["id"] = i
        assert all(k not in model_info for k in analysis_result_dict.keys())
        model_info.update(analysis_result_dict)
        edit_model_infos.append(model_info)

        del edited_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if config.generation.save_dir:
        save_dir = config.generation.save_dir
        if os.path.abspath(config.generation.save_dir) != config.generation.save_dir:
            # using relative path
            save_dir = f"{base_dir}/{config.generation.save_dir}"

        LOG.info(f"Saving to dir: {save_dir}")

        os.makedirs(save_dir, exist_ok=True)

        io.dump_jsonlines(
            edit_model_infos,
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl"
            + f"_{config.date_data}-question"
            + "_edit-model-infos.jsonl",
        )


if __name__ == "__main__":
    run()
