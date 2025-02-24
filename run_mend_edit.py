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
)

import models
import utils
from utils import EditLoss, EditInput


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())
    
logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


em_evaluator = ExactMatchEvaluator()
rouge_evaluator = RougeEvaluator()
llm_evaluator = OpenAIEvaluator()

icl_prompt = "\n".join([
    "Q: When did the simpsons first air on television?",
    "A: December 17, 1989",
    "Q: Who has more super bowl wins afc or nfc?",
    "A: NFC",
    "Q: Is the federal court the same as the supreme court?",
    "A: No"
])

def score_df(df):
    em_per_example = em_evaluator.compute_metric(
        predictions=df["predicted_answer"],
        references=df["answer"],
        use_aggregator=False,
    )
    rouge_per_example = rouge_evaluator.compute_metric(
        predictions=df["predicted_answer"],
        references=df["answer"],
        use_aggregator=False,
    )
    llm_acc_per_example = llm_evaluator.compute_metric(
        questions=df["question"],
        predictions=df["predicted_answer"],
        references=df["answer"],
        use_aggregator=False,
        rescale_to_one=True,
    )
        
    model_response_w_score = df.join(pd.DataFrame({**em_per_example, **rouge_per_example, **llm_acc_per_example}))
    return model_response_w_score

def add_padding(tokenizer, model):
    import transformers
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
                    (v.shape[0], 1), # shape of the constant tensor
                    (
                        1 
                        if k == "attention_mask" else
                        eos_token_id # this is to teach the model to end after outputing the answer.
                    )
                )
            ], 
            dim=-1
        )
        for k, v in tokenizer_output.items()
    }


def generate(context: str, answer: str, config, model, tokenizer, generation_config, ):
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.gen_w_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"])[0]
    
    if hasattr(config, "add_icl") and config.add_icl:
        eos_token = "\n"
    else:
        eos_token = tokenizer.eos_token
        
    if hasattr(config, "add_eos") and config.add_eos:
        answer += eos_token
    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    print("Input for generation:", "["+ "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"])) +"]")
    print("Label for generation:", "["+ answer +"]")

    
    generation_output = model.generate(
        **inputs,
        # input_ids = inputs["input_ids"],
        # attention_mask = inputs["attention_mask"],
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)
    generated_texts = [t.replace(ctx_decoded.strip(), "") for t in generated_texts]
    
    model_response = pd.DataFrame([
        {
            "question": context, "answer": answer, "predicted_answer_idx": 0,
            "predicted_answer": generated_texts[0], 
        }
    ])
    return score_df(model_response)

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
    print("Task: ", config.task)
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
        
        if config.task == "zsre":
            assert config.edit_input == EditInput.question
            targets =  [(" " if datum["answers"][0][0] != " " else "") + datum["answers"][0]]
            sentences = [datum["src"] + targets[0]]
            test_queries = [{"question": datum["src"], "answer": datum["answers"][0]}]

            if hasattr(config, "add_icl") and config.add_icl:
                test_queries[0]["question"] = icl_prompt + "\nQ: " + test_queries[0]["question"] + "\nA:",
        else:
            assert config.task == "musique"
            test_queries = [
                {
                    "question": datum["multi_hop_efficacy"][0]["question"],
                    "answer": datum["multi_hop_efficacy"][0]["answer"]
                 }
            ]
            if hasattr(config, "add_icl") and config.add_icl:
                test_queries[0]["question"] = icl_prompt + "\nQ: " + test_queries[0]["question"] + "\nA:"
            
            if config.edit_input == EditInput.question:
                question = test_queries[0]
                targets =  [(" " if question["answer"][0] != " " else "") + question["answer"]]
                sentences = [question["question"] + targets[0]]
            else:
                assert config.edit_loss == EditLoss.clm, f"Input `{config.edit_input}` is only supported for CLM loss"
                sentences = targets = datum["texts"]
        
        # prepare [Q][A] accuracy and generation inputs
        
        assert len(test_queries) == 1, "# TODO: make this support multiple input"
        test_queries_q_str = [test_queries[0]["question"]]
        test_queries_a_str = [test_queries[0]["answer"]]
        test_queries_str = [test_queries_q_str[0] + (" " if test_queries_a_str[0][0] != " " else "") + test_queries_a_str[0]]
        
        acc_toks = add_eos(tokenizer(test_queries_str, padding=True, return_tensors="pt", add_special_tokens=True), eos_token_id, ignore=not config.add_eos)
        acc_toks = utils.dict_to(acc_toks, config.device)
        sft_labels = val_set.get_edit_labels(
            add_eos(
                tokenizer(
                    [
                        (" " if test_queries_a_str[0][0] != " " else "") + test_queries_a_str[0]
                    ], padding=True, return_tensors="pt", add_special_tokens=False), 
                eos_token_id, ignore=not config.add_eos
            )["input_ids"]
        ).to(config.device)
        
        clm_labels = val_set.get_edit_labels(acc_toks["input_ids"]).to(config.device)
        
        if config.edit_loss == EditLoss.sft:
            sentences_toks = add_eos(tokenizer(sentences, padding=True, return_tensors="pt"), eos_token_id, ignore=not config.add_eos)
            targets_toks = add_eos(tokenizer(targets, padding=True, return_tensors="pt", add_special_tokens=False), eos_token_id, ignore=not config.add_eos)
        else:
            assert config.edit_loss == EditLoss.clm, f"edit_loss `{config.edit_loss}` is not supported"
            sentences_toks = targets_toks = add_eos(tokenizer(sentences, padding=True, return_tensors="pt", add_special_tokens=True), eos_token_id, ignore=not config.add_eos)
        
        
        
        edit_inner = {
            # "input_ids": torch.concat([sentences_toks["input_ids"], sentences_toks["input_ids"].flip(-1)], dim=-1),
            "input_ids": sentences_toks["input_ids"],
            "attention_mask": sentences_toks["attention_mask"],
            # "labels": val_set.get_edit_labels(torch.concat([targets_toks["input_ids"], sentences_toks["input_ids"].flip(-1)], dim=-1))d,
            "labels": val_set.get_edit_labels(targets_toks["input_ids"]),
        }
        
        print("Input for EDIT: ")
        print("["+"\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(sentences_toks["input_ids"]))+"]")
        print("Label for EDIT: ")
        print("["+"\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(sentences_toks["input_ids"]))+"]")
        print()
        
        
        edit_inner = utils.dict_to(edit_inner, config.device)
        
        print("Input for [Q][A] Accuracy: ")
        print("["+tokenizer.decode(acc_toks["input_ids"][0])+"]")
        print("SFT label:", "["+tokenizer.decode(sft_labels[0])+"]")
        print("CLM label(before ShiftLeft):", "["+tokenizer.decode(clm_labels[0])+"]")
        print()
        with torch.no_grad():
            
            pre_edit_logits = trainer.model(
                input_ids=acc_toks["input_ids"],
                attention_mask=acc_toks["attention_mask"]
            )
            
            pre_edit_sft_pm_dict = trainer.model.edit_loss_fn(pre_edit_logits, sft_labels, exact_match=False)
            pre_edit_sft_em_dict = trainer.model.edit_loss_fn(pre_edit_logits, sft_labels, exact_match=True)
            pre_edit_clm_pm_dict = trainer.model.edit_loss_fn(pre_edit_logits, clm_labels, exact_match=False)
            pre_edit_clm_em_dict = trainer.model.edit_loss_fn(pre_edit_logits, clm_labels, exact_match=True)
            
        if config.do_generation:
            
            pre_result_df = generate(test_queries_q_str[0], test_queries_a_str[0], config, trainer.model.model, tokenizer, generation_config)
        else:
            pre_result_df = pd.DataFrame([{"predicted_answer_idx": 0}])
        assert len(pre_result_df) == 1
        
        pre_result_df.insert(0, "input", "\n\n".join(f"[[{s}]]" for s in sentences))
        pre_result_df.insert(1, "stage", "pre-edit")
        pre_result_df.insert(pre_result_df.shape[-1], "[Q][A] Acc EM", pre_edit_clm_em_dict["acc"].item())
        pre_result_df.insert(pre_result_df.shape[-1], "[Q][A] Acc PM", pre_edit_clm_pm_dict["acc"].item())
        pre_result_df.insert(pre_result_df.shape[-1], "[A]|[Q] Acc EM", pre_edit_sft_em_dict["acc"].item())
        pre_result_df.insert(pre_result_df.shape[-1], "[A]|[Q] Acc PM", pre_edit_sft_pm_dict["acc"].item())
        all_results.append(pre_result_df)
            
        # edit the model with MEND
        edited_model, model_info = trainer.model.edit(edit_inner)
        model_info["input"] = sentences[0]
        model_info["target"] = tokenizer.decode(targets_toks["input_ids"][0])
        edit_model_infos.append(model_info)
        
        with torch.set_grad_enabled(not config.eval_only):
            # post_edit_logits = edited_model(input_ids=edit_inner["input_ids"], attention_mask=edit_inner["attention_mask"])
            # post_edit_dict = trainer.model.edit_loss_fn(post_edit_logits, edit_inner["labels"], exact_match=config.edit_loss == EditLoss.sft)
            post_edit_logits = edited_model(
                input_ids=acc_toks["input_ids"],
                attention_mask=acc_toks["attention_mask"]
            )
            post_edit_sft_pm_dict = trainer.model.edit_loss_fn(post_edit_logits, sft_labels, exact_match=False)
            post_edit_sft_em_dict = trainer.model.edit_loss_fn(post_edit_logits, sft_labels, exact_match=True)
            post_edit_clm_pm_dict = trainer.model.edit_loss_fn(post_edit_logits, clm_labels, exact_match=False)
            post_edit_clm_em_dict = trainer.model.edit_loss_fn(post_edit_logits, clm_labels, exact_match=True)

        if config.do_generation:
            post_result_df = generate(test_queries_q_str[0], test_queries_a_str[0], config, edited_model.model, tokenizer, generation_config)
        else:
            post_result_df = pd.DataFrame([{"predicted_answer_idx": 0}])
        assert len(post_result_df) == 1
        del edited_model
        
        post_result_df.insert(0, "input", "\n\n".join(f"[[{s}]]" for s in sentences))
        post_result_df.insert(1, "stage", "post-edit")
        # post_result_df.insert(post_result_df.shape[-1], "[Q][A] acc", post_edit_dict["acc"].item())
        post_result_df.insert(post_result_df.shape[-1], "[Q][A] Acc EM", post_edit_clm_em_dict["acc"].item())
        post_result_df.insert(post_result_df.shape[-1], "[Q][A] Acc PM", post_edit_clm_pm_dict["acc"].item())
        post_result_df.insert(post_result_df.shape[-1], "[A]|[Q] Acc EM", post_edit_sft_em_dict["acc"].item())
        post_result_df.insert(post_result_df.shape[-1], "[A]|[Q] Acc PM", post_edit_sft_pm_dict["acc"].item())
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
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if config.do_generation else 'wo'}-gen_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl.xlsx",
            index=False,
        )
        io.dump_jsonlines(
            edit_model_infos,
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl_edit-model-infos.jsonl"
        )
    
if __name__ == "__main__":
    run()
