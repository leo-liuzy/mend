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
from utils_leo import get_eval_result
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


def prepare_sft_text(args, custom_cfg, instance, tokenizer):
    # assert len(tokenizer.additional_special_tokens) == 1
    if custom_cfg.input_format == InputFormat.two_single_hop:
        dataset = instance["single_hop_efficacy"]
    elif custom_cfg.input_format == InputFormat.first_single_hop:
        dataset = [instance["single_hop_efficacy"][0]]
        
    elif custom_cfg.input_format == InputFormat.second_single_hop:
        dataset = [instance["single_hop_efficacy"][1]]
    else:
        assert custom_cfg.input_format == InputFormat.two_hop
        dataset = [instance["multi_hop_efficacy"][0]]
        
    new_dataset = []
    for datum in dataset:
        q = datum["question"]
        a = datum["answer"]
        t = f"{q}{a}" if a[0] == " " else f"{q} {a}"
        t += tokenizer.eos_token
        datum[args.dataset_text_field] = t
        new_dataset.append(datum)
    return new_dataset



class InputFormat(StrEnum):
    two_single_hop = "two-1hop"
    first_single_hop = "first-1hop"
    second_single_hop = "second-1hop"
    two_hop = "2hop"

    
@dataclass
class CustomConfig:
    example_idx: int
    input_format: InputFormat 
    device: Optional[str] = "cuda:0"
    add_eos_accuracy: Optional[bool] = True
    add_bos: Optional[bool] = True
    base_model_name: Optional[str] = "Llama-3.2-1B-eos-sft"
    save_dir_suffix: Optional[str] = None
    spec_question: bool = False

parser = HfArgumentParser((SFTConfig, CustomConfig))
(args, custom_cfg) = parser.parse_args_into_dataclasses()
model_name_or_path = f"{os.getcwd()}/models/{custom_cfg.base_model_name}"

logging.info(f"CustomConfig: {custom_cfg}")

exp_save_dir = f"{os.getcwd()}/exp_output/{custom_cfg.base_model_name}_sft-baseline_input={custom_cfg.input_format}_lr={args.learning_rate}_epoch={args.num_train_epochs}{'_' + custom_cfg.save_dir_suffix if custom_cfg.save_dir_suffix is not None else ''}"

os.makedirs(exp_save_dir, exist_ok=True)

individual_result_save_dir = f"{exp_save_dir}/individual_results"
if custom_cfg.spec_question:
        individual_result_save_dir += "_spec"
os.makedirs(individual_result_save_dir, exist_ok=True)

all_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev_w-spec.jsonl")
instance = all_dev_dataset[custom_cfg.example_idx]


logging.info(f"Example ID: {instance['id']}")

fpath = f"{individual_result_save_dir}/{instance['id']}_eval_results.xlsx"
if os.path.exists(fpath):
    logging.info("=" * 20 + "Already evaluated" + "=" * 20)
    exit(0)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False, device_map=custom_cfg.device)
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

train_dataset = prepare_sft_text(args, custom_cfg, instance, tokenizer)

logging.info(f"Setting per_device_train_batch_size == {len(train_dataset)}")
args.per_device_train_batch_size = len(train_dataset)
# valid_dataset = prepare_sft_text(args, io.load_jsonlines(f"{vars.DATA_DIR}/trivia_qa_wiki_sft/valid.jsonl"), tokenizer)

train_dataset = Dataset.from_list(train_dataset)
# valid_dataset = Dataset.from_list(valid_dataset)

response_template = "?" # tokenizer.additional_special_tokens[0] # "?" alternative "Ġ?"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset, # type: ignore
    # eval_dataset=valid_dataset, # type: ignore
    args=args,
    data_collator=collator,
)

trainer.train()
trainer.accelerator.wait_for_everyone()


model = trainer.model
# clear internal pointer in trainer/accelerator
trainer.accelerator.free_memory(trainer.model, trainer.optimizer, trainer.lr_scheduler)
del trainer.model, trainer.optimizer, trainer.lr_scheduler
del trainer
# clear cache to make spaces in GPU and CPU
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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
for question_type in question_types:
    questions = instance[question_type]
    logging.info(f"Question type: {question_type}")
    
    for question in questions:
        
        sft_eval_result = get_eval_result(
            question=question["question"], 
            answer=question["answer"],
            model=model,
            tokenizer=tokenizer, 
            config=custom_cfg,
            generation_config=generation_config
        )
        sft_eval_result.insert(0, "stage", "post-edit")
        sft_eval_result.insert(0, "sft_input", "\n\n".join(
                f"[[{tokenizer.decode(s)}]]"
                for s in tokenizer(
                    train_dataset["text"], 
                    add_special_tokens=True
                )["input_ids"]
            )
        )
        sft_eval_result.insert(0, "question_type", question_type)
        sft_eval_result.insert(0, "id", instance["id"])
        
        all_result_df.append(sft_eval_result)
all_result_df = pd.concat(all_result_df)

# exp_save_dir = f"{os.getcwd()}/exp_output/{custom_cfg.base_model_name}_sft-baseline_input={custom_cfg.input_format}_lr={args.learning_rate}_epoch={args.num_train_epochs}{'_' + custom_cfg.save_dir_suffix if custom_cfg.save_dir_suffix is not None else ''}"

logging.info(f"Saving individual results to {individual_result_save_dir}")

all_result_df.to_excel(
    fpath,
    index=False,
)
