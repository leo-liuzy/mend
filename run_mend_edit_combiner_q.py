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
from utils import StrEnum

from utils_leo import get_eval_result, add_eos


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())
    
logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)

class EditInput(StrEnum):
    two_single_hop = "two-1hop"
    first_single_hop = "first-1hop"
    second_single_hop = "second-1hop"
    

icl_prompt = "\n".join([
    "Q: When did the simpsons first air on television?",
    "A: December 17, 1989",
    "Q: Who has more super bowl wins afc or nfc?",
    "A: NFC",
    "Q: Is the federal court the same as the supreme court?",
    "A: No"
])


def add_padding(tokenizer, model):
    import transformers
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM):
    #     model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight.mean(0)
    # else:
        model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)
        
question_types = [
    "single_hop_efficacy",
    "multi_hop_efficacy",
]

@hydra.main(config_path='config', config_name='config')
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

    train_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-train-new_annotated_final.jsonl", config)
    val_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config)
    tokenizer = val_set.tok
    
    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))
    
    generation_config = GenerationConfig(
        do_sample=False, # Greedy
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
    if config.task == "zsre":
        val_data = val_set
    else:
        assert config.task == "musique"
        # if config.edit_input == EditInput.two_doc:
        #     # val_data = io.load_jsonlines(f"{vars.DATA_DIR}/musique_c_small/examples-paragraph.jsonl")
            
        # else:
        #     # val_data = io.load_jsonlines(f"{vars.DATA_DIR}/musique_c_small/examples-paragraph-seen.jsonl")
        #     val_data = io.load_jsonlines(f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev-seen.jsonl")
        val_data = io.load_jsonlines(f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev.jsonl")
    
    all_results = []
    edit_model_infos = []
    # trainer.validate(log=True)
    assert config.val_steps <= len(val_data)
    assert config.eval_only
    if hasattr(config, "add_icl") and config.add_icl:
        eos_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    else:
        eos_token_id = tokenizer.eos_token_id
            
    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
    # for i in tqdm([717, 718, 719], desc=f"Running eval on {config.task}"):
    # for i in tqdm(range(1), desc=f"Running eval on {config.task}"):
        datum = val_data[i]
        
        assert config.task == "musique"

        if config.edit_input == EditInput.first_single_hop:
            test_queries = [datum["single_hop_efficacy"][0]]
        elif config.edit_input == EditInput.second_single_hop:
            test_queries = [datum["single_hop_efficacy"][1]]
        else:
            assert config.edit_input == EditInput.two_single_hop
            test_queries = datum["single_hop_efficacy"]
            
        if hasattr(config, "add_icl") and config.add_icl:
            new_test_queries = []
            for q in test_queries:
                new_q = deepcopy(q)
                new_q["question"] = icl_prompt + "\nQ: " + test_queries[0]["question"] + "\nA:"
                new_test_queries.append(new_q)
            test_queries = new_test_queries
        
        targets =  [(" " if q["answer"][0] != " " else "") + q["answer"] for q in test_queries]
        sentences = [q["question"] + targets[q_i] for q_i, q in enumerate(test_queries)]

        
        sentences_toks = add_eos(tokenizer(sentences, padding=True, return_tensors="pt"), eos_token_id, ignore=not config.add_eos)
        targets_toks = add_eos(tokenizer(targets, padding=True, return_tensors="pt", add_special_tokens=False), eos_token_id, ignore=not config.add_eos)
    
        
        edit_inner = {
            # "input_ids": torch.concat([sentences_toks["input_ids"], sentences_toks["input_ids"].flip(-1)], dim=-1),
            "input_ids": sentences_toks["input_ids"],
            "attention_mask": sentences_toks["attention_mask"],
            # "labels": val_set.get_edit_labels(torch.concat([targets_toks["input_ids"], sentences_toks["input_ids"].flip(-1)], dim=-1))d,
            "labels": val_set.get_edit_labels(targets_toks["input_ids"]),
        }
        
        # import pdb; pdb.set_trace()
        print("Input for EDIT: ")
        print("["+"\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(edit_inner["input_ids"]))+"]")
        print("Label for EDIT: ")
        print("["+"\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(targets_toks["input_ids"]))+"]")
        print()
        
        
        edit_inner = utils.dict_to(edit_inner, config.device)
        
        all_datum_result_df = []
        for question_type in question_types:
            questions = datum[question_type]
            logging.info(f"Question type: {question_type}")
            
            for question in questions:
        
                pre_result_df = get_eval_result(
                    question=question["question"], 
                    answer=question["answer"],
                    model=trainer.model.model,
                    tokenizer=tokenizer, 
                    config=config,
                    generation_config=generation_config
                )
                pre_result_df.insert(0, "stage", "pre-edit")
                pre_result_df.insert(0, "sft_input", "\n\n".join(
                        f"[[{tokenizer.decode(s)}]]"
                        for s in sentences_toks["input_ids"]
                    )
                )
                pre_result_df.insert(0, "question_type", question_type)
                pre_result_df.insert(0, "id", datum["id"])
                all_datum_result_df.append(pre_result_df)
        
        # edit the model with MEND
        edited_model, model_info = trainer.model.edit(edit_inner)
        model_info["input"] = sentences[0]
        model_info["target"] = tokenizer.decode(targets_toks["input_ids"][0])
        edit_model_infos.append(model_info)
        
        for question_type in question_types:
            questions = datum[question_type]
            logging.info(f"Question type: {question_type}")
            
            for question in questions:
        
                post_result_df = get_eval_result(
                    question=question["question"], 
                    answer=question["answer"],
                    model=edited_model.model,
                    tokenizer=tokenizer, 
                    config=config,
                    generation_config=generation_config
                )
                post_result_df.insert(0, "stage", "post-edit")
                post_result_df.insert(0, "sft_input", "\n\n".join(
                        f"[[{tokenizer.decode(s)}]]"
                        for s in sentences_toks["input_ids"]
                    )
                )
                post_result_df.insert(0, "question_type", question_type)
                post_result_df.insert(0, "id", datum["id"])
                all_datum_result_df.append(post_result_df)
        
        del edited_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        all_datum_result_df = pd.concat(all_datum_result_df)
        all_results.append(all_datum_result_df)
        
    all_results = pd.concat(all_results)
    
    if config.generation.save_dir:
        save_dir = config.generation.save_dir
        if os.path.abspath(config.generation.save_dir) != config.generation.save_dir:
            # using relative path
            save_dir = f"{base_dir}/{config.generation.save_dir}"
        
        LOG.info(f"Saving to dir: {save_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
        all_results.to_excel(
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if config.do_generation else 'wo'}-gen_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl.xlsx",
            index=False,
        )
        io.dump_jsonlines(
            edit_model_infos,
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl_edit-model-infos.jsonl"
        )
    
if __name__ == "__main__":
    run()
