import datetime
import typing
import numpy as np
import struct
import os
import getpass
import hydra
import logging
import torch
from collections import defaultdict
import math
from knowledge_propagation.modules.evaluators import (
    ExactMatchEvaluator,
    RougeEvaluator,
    OpenAIEvaluator,
    NumDiffEvaluator,
)
import pandas as pd
from losses import multiclass_log_probs
import utils

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
    diff_per_example = year_diff_evaluator.compute_metric(
        predictions=df["predicted_answer"],
        references=df["answer"],
        use_aggregator=False,
    )
    
    model_response_w_score = df.join(pd.DataFrame({**em_per_example, **diff_per_example, }))
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


def get_eval_result(question, answer, model, tokenizer, config, generation_config):
    test_queries_str = [question + (" " if answer[0] != " " else "") + answer]

    eos_token_id = tokenizer.eos_token_id
    
    acc_toks = add_eos(tokenizer(test_queries_str, padding=True, return_tensors="pt", add_special_tokens=config.add_bos), eos_token_id, ignore=not config.add_eos_accuracy)
    acc_toks = utils.dict_to(acc_toks, config.device)
    sft_labels = get_edit_labels(
        add_eos(
            tokenizer(
                [
                    (" " if answer[0] != " " else "") + answer
                ], padding=True, return_tensors="pt", add_special_tokens=False), 
            eos_token_id, ignore=not config.add_eos_accuracy
        )["input_ids"], tokenizer
    ).to(config.device)

    clm_labels = get_edit_labels(acc_toks["input_ids"], tokenizer).to(config.device)
    
    logging.info("Input for [Q][A] Accuracy: ")
    logging.info("["+tokenizer.decode(acc_toks["input_ids"][0])+"]")
    logging.info("SFT label: " + "["+tokenizer.decode(sft_labels[0])+"]")
    logging.info("CLM label(before ShiftLeft): " + "["+tokenizer.decode(clm_labels[0])+"]")
    logging.info("")
    
    model.eval()
            
    with torch.no_grad():
        
        model_output = model(
            input_ids=acc_toks["input_ids"],
            attention_mask=acc_toks["attention_mask"]
        )
        if isinstance(model_output, torch.Tensor):
            model_logits = model_output
        else:
            model_logits = model_output.logits
        model_sft_em_dict = multiclass_log_probs(model_logits, sft_labels, exact_match=True)
        model_sft_pm_dict = multiclass_log_probs(model_logits, sft_labels, exact_match=False)
        model_clm_em_dict = multiclass_log_probs(model_logits, clm_labels, exact_match=True)
        model_clm_pm_dict = multiclass_log_probs(model_logits, clm_labels, exact_match=False)
        
        model_result_df = generate(question, answer, config, model, tokenizer, generation_config)
        
    model_result_df.insert(model_result_df.shape[-1], "[A]|[Q] Acc EM", model_sft_em_dict["acc"].item())
    model_result_df.insert(model_result_df.shape[-1], "[A]|[Q] Acc PM", model_sft_pm_dict["acc"].item())
    model_result_df.insert(model_result_df.shape[-1], "[Q][A] Acc EM", model_clm_em_dict["acc"].item())
    model_result_df.insert(model_result_df.shape[-1], "[Q][A] Acc PM", model_clm_pm_dict["acc"].item())
    
    return model_result_df
