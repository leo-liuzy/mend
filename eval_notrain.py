import os
import json
from datasets import Dataset
from typing import Optional
import pickle as pkl
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, GenerationConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from knowledge_propagation.utils import vars, io
import torch
import gc
import utils
from utils import StrEnum
from knowledge_propagation.modules.evaluators import (
    ExactMatchEvaluator,
    RougeEvaluator,
    OpenAIEvaluator,
)
import pandas as pd
from losses import multiclass_log_probs

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

em_evaluator = ExactMatchEvaluator()
rouge_evaluator = RougeEvaluator()
llm_evaluator = OpenAIEvaluator()



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
    # llm_acc_per_example = llm_evaluator.compute_metric(
    #     questions=df["question"],
    #     predictions=df["predicted_answer"],
    #     references=df["answer"],
    #     use_aggregator=False,
    #     rescale_to_one=True,
    # )
        
    model_response_w_score = df.join(pd.DataFrame({**em_per_example, **rouge_per_example,}))
    return model_response_w_score


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
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.add_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]
    
    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    logging.info("Input for generation: " + "["+ "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"])) +"]")
    logging.info("Label for generation: " + "["+ answer +"]")

    
    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    generated_texts = [t.replace(ctx_decoded, "") for t in generated_texts]
    # predicted_answer = generated_texts[0]
    model_response_content = []
    for g_i, generated_text in enumerate(generated_texts):
        predicted_answer = generated_text.strip()
        model_response_content.append(
            {
                "question": context, "answer": answer.strip(), 
                "predicted_answer_idx": g_i,
                "predicted_answer": predicted_answer, 
            }
        )
    model_response = pd.DataFrame(model_response_content)
    
    # if hasattr(config, "add_icl") and config.add_icl:
    #     # if using ICL, extract by the first new line
    #     if "\n" in predicted_answer:
    #         predicted_answer = predicted_answer[:predicted_answer.find("\n")]
    
    
    return score_df(model_response)    


def get_edit_labels(labels, tokenizer):
    return labels.masked_fill(labels == tokenizer.pad_token_id, -100)



    
@dataclass
class CustomConfig:
    device: Optional[str] = "cuda:0"
    add_eos_accuracy: Optional[bool] = True
    add_bos: Optional[bool] = True
    base_model_name: Optional[str] = "Llama-3.2-1B-eos-sft"
    save_dir_suffix: Optional[str] = None
    spec_question: bool = False
    add_icl: bool = False

parser = HfArgumentParser((SFTConfig, CustomConfig))
(args, custom_cfg) = parser.parse_args_into_dataclasses()
model_name_or_path = f"{os.getcwd()}/models/{custom_cfg.base_model_name}"

logging.info(f"CustomConfig: {custom_cfg}")

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False,)
model = model.to(custom_cfg.device)
tokenizer = AutoTokenizer.from_pretrained(f"{os.environ['SHARE_RES_DIR']}/models/llama3/hf/Llama-3.2-1B", add_eos_token=True, use_fast=False)
tokenizer.padding_side = 'right'
original_vocab_size = len(tokenizer)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<sft_token_1>']}, replace_additional_special_tokens=False)
model.resize_token_embeddings(len(tokenizer))

tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

assert tokenizer.eos_token != tokenizer.pad_token
assert tokenizer.eos_token_id != tokenizer.pad_token_id


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

all_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev_w-spec.jsonl")


for instance in all_dev_dataset:
    model.eval()
    logging.info(f"Example ID: {instance['id']}")

    eos_token_id = tokenizer.eos_token_id
    if custom_cfg.spec_question:
        question_types = [
            "single_hop_specificity",
            "multi_hop_specificity",
        ]
    else:
        question_types = [
            "single_hop_efficacy",
            "multi_hop_efficacy",
        ]
    logging.info("Start evaluating model: Generation, Accuracy")

    all_result_df = []
    
    ctx = "\n\n".join([
        # f"<doc{i}>\n{t}\n</doc{i}>"
        f"{t}"
        for i, t in enumerate(instance["texts"])
    ])
    for question_type in question_types:
        questions = instance[question_type]
        logging.info(f"Question type: {question_type}")
        
        for question in questions:
            if custom_cfg.add_icl:
                test_queries_q_str = ctx + "\n\n" + question["question"]
            else:
                test_queries_q_str = question["question"]
            
            test_queries_a_str = question["answer"]
            test_queries_str = [test_queries_q_str + (" " if test_queries_a_str[0] != " " else "") + test_queries_a_str]

            acc_toks = add_eos(tokenizer(test_queries_str, padding=True, return_tensors="pt", add_special_tokens=custom_cfg.add_bos), eos_token_id, ignore=not custom_cfg.add_eos_accuracy)
            acc_toks = utils.dict_to(acc_toks, custom_cfg.device)
            sft_labels = get_edit_labels(
                add_eos(
                    tokenizer(
                        [
                            (" " if test_queries_a_str[0] != " " else "") + test_queries_a_str
                        ], padding=True, return_tensors="pt", add_special_tokens=False), 
                    eos_token_id, ignore=not custom_cfg.add_eos_accuracy
                )["input_ids"], tokenizer
            ).to(custom_cfg.device)

            clm_labels = get_edit_labels(acc_toks["input_ids"], tokenizer).to(custom_cfg.device)
            
            logging.info("Input for [Q][A] Accuracy: ")
            logging.info("["+tokenizer.decode(acc_toks["input_ids"][0])+"]")
            logging.info("SFT label: " + "["+tokenizer.decode(sft_labels[0])+"]")
            logging.info("CLM label(before ShiftLeft): " + "["+tokenizer.decode(clm_labels[0])+"]")
            logging.info("")
            
                    
            with torch.no_grad():
                
                post_edit_output = model(
                    input_ids=acc_toks["input_ids"],
                    attention_mask=acc_toks["attention_mask"]
                )
                post_edit_logits = post_edit_output.logits
                post_edit_sft_em_dict = multiclass_log_probs(post_edit_logits, sft_labels, exact_match=True)
                post_edit_sft_pm_dict = multiclass_log_probs(post_edit_logits, sft_labels, exact_match=False)
                post_edit_clm_em_dict = multiclass_log_probs(post_edit_logits, clm_labels, exact_match=True)
                post_edit_clm_pm_dict = multiclass_log_probs(post_edit_logits, clm_labels, exact_match=False)
                
            post_result_df = generate(test_queries_q_str, test_queries_a_str, custom_cfg, model, tokenizer, generation_config)
            
            post_result_df.insert(1, "stage", "pre-edit")
            post_result_df.insert(0, "question_type", question_type)
            post_result_df.insert(0, "id", instance["id"])
            post_result_df.insert(post_result_df.shape[-1], "[A]|[Q] Acc EM", post_edit_sft_em_dict["acc"].item())
            post_result_df.insert(post_result_df.shape[-1], "[A]|[Q] Acc PM", post_edit_sft_pm_dict["acc"].item())
            post_result_df.insert(post_result_df.shape[-1], "[Q][A] Acc EM", post_edit_clm_em_dict["acc"].item())
            post_result_df.insert(post_result_df.shape[-1], "[Q][A] Acc PM", post_edit_clm_pm_dict["acc"].item())
            
            all_result_df.append(post_result_df)
    all_result_df = pd.concat(all_result_df)

    exp_save_dir = f"{os.getcwd()}/exp_output/{custom_cfg.base_model_name}_sh+mh"
    os.makedirs(exp_save_dir, exist_ok=True)

    individual_result_save_dir = f"{exp_save_dir}/individual_results"
    if custom_cfg.spec_question:
        individual_result_save_dir += "_spec"
    os.makedirs(individual_result_save_dir, exist_ok=True)

    logging.info(f"Saving individual results to {individual_result_save_dir}")

    all_result_df.to_excel(
        f"{individual_result_save_dir}/{instance['id']}_eval_results.xlsx",
        index=False,
    )
