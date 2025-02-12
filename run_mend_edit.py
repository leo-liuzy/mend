import copy
import importlib
import logging
import random
from enum import Enum

import os
import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from trainer import EditTrainer
from knowledge_propagation.utils import io, vars
from knowledge_propagation.modules.inferencers import QAInferencer
from experiments.musique.inference_only import eval_inferencer, macro_averaging
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

import models
import utils


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class EditLoss(StrEnum):
    sft = "sft"

    clm = "clm"
    
class EditInput(StrEnum):
    question = "question"

    two_doc = "2doc"
    
    single_doc = "1doc"
    

OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())
    
logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


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
    
    from data_classes.zsre import ZsreDataset

    train_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-train-new_annotated_final.jsonl", config)
    val_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config)
    tokenizer = val_set.tok
    
    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))
    
    
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(config_path="../KE-by-CP/configs/evaluator", version_base=None,):
        if config.generation.prompt == "urial":
            inferencer_cfg = hydra.compose(config_name="base.yaml",)["inferencers"][0]
        else:
            assert config.generation.prompt == "no"
            inferencer_cfg = hydra.compose(config_name="base_null.yaml",)["inferencers"][0]
        inferencer_cfg.max_new_tokens = config.generation.max_new_tokens
        
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(config_path="../KE-by-CP/configs/generation", version_base=None,):
        generation_cfg = hydra.compose(config_name="greedy.yaml",)
        generation_cfg.max_new_tokens = config.generation.max_new_tokens
    
    generation_config = GenerationConfig(
        do_sample=generation_cfg.do_sample,
        top_k=generation_cfg.top_k,
        top_p=generation_cfg.top_p,
        temperature=generation_cfg.temperature,
        max_new_tokens=generation_cfg.max_new_tokens,
        num_return_sequences=generation_cfg.n_decoding_example,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    trainer = EditTrainer(alg, config, train_set, val_set)
    
    if config.task == "zsre":
        val_data = val_set
    else:
        assert config.task == "musique"
        if config.edit_input == EditInput.two_doc:
            # val_data = io.load_jsonlines(f"{vars.DATA_DIR}/musique_c_small/examples-paragraph.jsonl")
            val_data = io.load_jsonlines(f"{vars.DATA_DIR}/musique_mend/2hop_musique_ans_v1.0_dev.jsonl")
        else:
            # val_data = io.load_jsonlines(f"{vars.DATA_DIR}/musique_c_small/examples-paragraph-seen.jsonl")
            val_data = io.load_jsonlines(f"{vars.DATA_DIR}/musique_mend/2hop_musique_ans_v1.0_dev-seen.jsonl")
    
    all_results = []
    # trainer.validate(log=True)
    assert config.val_steps <= len(val_data)
    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
    # for i in tqdm(range(1), desc=f"Running eval on {config.task}"):
        datum = val_data[i]
        if config.task == "zsre":
            assert config.edit_input == EditInput.question
            targets =  [(" " if datum["alt"][0] != " " else "") + datum["alt"]]
            sentences = [datum["src"] + targets[0]]
            test_queries = [{"question": datum["src"], "answer": datum["alt"]}]
        else:
            assert config.task == "musique"
            test_queries = [
                {
                    "question": datum["multi_hop_efficacy"][0]["question"],
                    "answer": datum["multi_hop_efficacy"][0]["answer"]
                 }
            ]
            if config.edit_input == EditInput.question:
                question = datum["multi_hop_efficacy"][0]
                targets =  [(" " if question["answer"][0] != " " else "") + question["answer"]]
                sentences = [question["question"] + targets[0]]
            else:
                assert config.edit_loss == EditLoss.clm, f"Input `{config.edit_input}` is only supported for CLM loss"
                sentences = targets = datum["texts"]
                
                
        # generate 
        inferencer = QAInferencer(
            inferencer_cfg,
            config.seed,
            rag_model=None,
            queries=test_queries,
        )
        
        if config.edit_loss == EditLoss.sft:
            sentences_toks = tokenizer(sentences, padding=True, return_tensors="pt")
            targets_toks = tokenizer(targets, padding=True, return_tensors="pt", add_special_tokens=False)
        else:
            assert config.edit_loss == EditLoss.clm, f"edit_loss `{config.edit_loss}` is not supported"
            sentences_toks = targets_toks = tokenizer(sentences, padding=True, return_tensors="pt", add_special_tokens=False)
        
        edit_inner = {
            "input_ids": sentences_toks["input_ids"],
            "attention_mask": sentences_toks["attention_mask"],
            "labels": val_set.get_edit_labels(targets_toks["input_ids"]),
        }
        
        edit_inner = utils.dict_to(edit_inner, config.device)
        with torch.no_grad():
            pre_edit_logits = trainer.model(
                input_ids=edit_inner["input_ids"], 
                attention_mask=edit_inner["attention_mask"]
            )
            pre_edit_dict = trainer.model.edit_loss_fn(pre_edit_logits, edit_inner["labels"], exact_match=config.edit_loss == EditLoss.sft)
            
        pre_result_df = eval_inferencer(
            inferencer,
            trainer.model.model,
            tokenizer=tokenizer,
            generation_cfg=generation_config,
        )
        assert len(pre_result_df) == 1
        
        pre_result_df.insert(0, "input", sentences[0])
        pre_result_df.insert(1, "stage", "pre-edit")
        pre_result_df.insert(pre_result_df.shape[-1], "[Q][A] acc", pre_edit_dict["acc"].item())
        all_results.append(pre_result_df)
            
        # edit the model with MEND
        edited_model, model_info = trainer.model.edit(edit_inner)
        
        with torch.set_grad_enabled(not config.eval_only):
            post_edit_logits = edited_model(input_ids=edit_inner["input_ids"], attention_mask=edit_inner["attention_mask"])
            post_edit_dict = trainer.model.edit_loss_fn(post_edit_logits, edit_inner["labels"], exact_match=config.edit_loss == EditLoss.sft)

        
        post_result_df = eval_inferencer(
            inferencer,
            edited_model.model,
            tokenizer=tokenizer,
            generation_cfg=generation_config,
        )
        assert len(post_result_df) == 1
        del edited_model
        
        post_result_df.insert(0, "input", sentences[0])
        post_result_df.insert(1, "stage", "post-edit")
        post_result_df.insert(post_result_df.shape[-1], "[Q][A] acc", post_edit_dict["acc"].item())
        all_results.append(post_result_df)
    
    all_results = pd.concat(all_results)
    
    if config.generation.save_dir:
        save_dir = config.generation.save_dir
        if os.path.abspath(config.generation.save_dir) != config.generation.save_dir:
            # using relative path
            save_dir = f"{base_dir}/{config.generation.save_dir}"
        
        LOG.info(f"Saving to dir: {save_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
        all_results.to_excel(
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}.xlsx",
            index=False,
        )
        
    
if __name__ == "__main__":
    run()
