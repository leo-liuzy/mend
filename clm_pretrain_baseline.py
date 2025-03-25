import os
import json
from datasets import Dataset
from typing import Optional
import pickle as pkl
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from knowledge_propagation.utils import vars, io


response_template = "@@@\n"  # tokenizer.additional_special_tokens[0] # "?" alternative "Ġ?"

def prepare_sft_text(args, custom_cfg, dataset: list, tokenizer):
    # assert len(tokenizer.additional_special_tokens) == 1

    new_dataset = []
    has_show_example = False
    for datum in dataset[:custom_cfg.train_size]:
        
        ctx = datum["text"]
        
        for qa in datum["questions"]:
            q = qa["question"]
            a = str(qa["answer"])
            
            t = f"{ctx}\n{response_template}" + (f"{q}{a}" if a[0] == " " else f"{q} {a}")
            
            t += tokenizer.eos_token
            if not has_show_example:
                print(f"Example: -> {t}")
                has_show_example = True
            new_dataset.append({
                args.dataset_text_field: t,
            })
    return new_dataset

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@dataclass
class CustomConfig:
    syn_data: str
    tunable_params: str
    train_size: Optional[int] = 3_000

parser = HfArgumentParser((SFTConfig, CustomConfig))
(args, custom_cfg) = parser.parse_args_into_dataclasses()
model_name_or_path = f"/u/zliu/datastor1/mend/models/Llama-3.2-1B-eos-sft"


model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_eos_token=True, use_fast=False)
tokenizer.padding_side = "right"
original_vocab_size = len(tokenizer)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<sft_token_1>']}, replace_additional_special_tokens=False)
model.resize_token_embeddings(len(tokenizer))

tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

assert tokenizer.eos_token != tokenizer.pad_token
assert tokenizer.eos_token_id != tokenizer.pad_token_id

if custom_cfg.tunable_params != "all":
    if custom_cfg.tunable_params == "top3-mlp":
        params = [
            "model.layers.13.mlp.gate_proj.weight",
            "model.layers.13.mlp.up_proj.weight",
            "model.layers.13.mlp.down_proj.weight",
            "model.layers.14.mlp.gate_proj.weight",
            "model.layers.14.mlp.up_proj.weight",
            "model.layers.14.mlp.down_proj.weight",
            "model.layers.15.mlp.gate_proj.weight",
            "model.layers.15.mlp.up_proj.weight",
            "model.layers.15.mlp.down_proj.weight",
        ]
    elif custom_cfg.tunable_params == "midupper3-mlp":
        params = [
            "model.layers.10.mlp.gate_proj.weight",
            "model.layers.10.mlp.up_proj.weight",
            "model.layers.10.mlp.down_proj.weight",
            "model.layers.11.mlp.gate_proj.weight",
            "model.layers.11.mlp.up_proj.weight",
            "model.layers.11.mlp.down_proj.weight",
            "model.layers.12.mlp.gate_proj.weight",
            "model.layers.12.mlp.up_proj.weight",
            "model.layers.12.mlp.down_proj.weight",
        ]
    else:
        raise ValueError(f"Unknown tunable_params: {custom_cfg.tunable_params}")
    
    for n, param in model.named_parameters():
        
        if any(p in n for p in params):
            param.requires_grad = True
        else:
            param.requires_grad = False

print_trainable_parameters(model)
# train_dataset = prepare_sft_text(args, io.load_jsonlines(f"{vars.DATA_DIR}/trivia_qa_wiki_sft/train.jsonl"), tokenizer)
# valid_dataset = prepare_sft_text(args, io.load_jsonlines(f"{vars.DATA_DIR}/trivia_qa_wiki_sft/valid.jsonl"), tokenizer)

# train_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_date_data/train.jsonl"), tokenizer
# )
# valid_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_date_data/valid.jsonl"), tokenizer
# )
if custom_cfg.syn_data == "bio_syn_v2":
    train_dataset = prepare_sft_text(
        args, custom_cfg, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/train.jsonl"), tokenizer
    )
    valid_dataset = prepare_sft_text(
        args, custom_cfg, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/valid.jsonl"), tokenizer
    )
else:
    assert custom_cfg.syn_data == "country_syn"
    train_dataset = prepare_sft_text(
        args, custom_cfg, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/country_syn_data/train.jsonl"), tokenizer
    )
    valid_dataset = prepare_sft_text(
        args, custom_cfg, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/country_syn_data/valid.jsonl"), tokenizer
    )
    
train_dataset = Dataset.from_list(train_dataset)
valid_dataset = Dataset.from_list(valid_dataset)



collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
# import pdb; pdb.set_trace()
trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,  # type: ignore
    eval_dataset=valid_dataset,  # type: ignore
    args=args,
    data_collator=collator,
)


trainer.train()

trainer.model.config.pad_token_id = None
print_trainable_parameters(trainer.model)
trainer.model.resize_token_embeddings(original_vocab_size)
trainer.model.save_pretrained(save_directory=args.output_dir)

trainer.accelerator.wait_for_everyone()
