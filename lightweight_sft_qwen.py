import os
import json
from datasets import Dataset
from typing import Optional
import pdb
import pickle as pkl
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from peft import LoraConfig, TaskType, get_peft_model, LoraModel, PeftModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from knowledge_propagation.utils import vars, io


@dataclass
class CustomConfig:
    is_peft: bool
    eos_sft: bool

def prepare_sft_text(args, dataset: list, tokenizer):
    # assert len(tokenizer.additional_special_tokens) == 1

    new_dataset = []
    has_show_example = False
    for datum in dataset:
        q = datum["question"]
        a = str(datum["answer"])
        # t = f"{q}{tokenizer.additional_special_tokens[0]}{a}" if a[0] == " " else f"{q}{tokenizer.additional_special_tokens[0]} {a}"
        t = f"{q}{a}" if a[0] == " " else f"{q} {a}"
        t += tokenizer.eos_token
        if not has_show_example:
            print(f"Example: -> {t}")
            has_show_example = True
        datum[args.dataset_text_field] = t
        new_dataset.append(datum)
    # pdb.set_trace()
    return new_dataset


parser = HfArgumentParser((SFTConfig, CustomConfig))
(args, custom_args) = parser.parse_args_into_dataclasses()
print(custom_args)
print(args.output_dir)
if custom_args.eos_sft:
    model_name_or_path = f"{os.environ['SHARE_RES_DIR']}/models/qwen/Qwen2.5-1.5B"
else:
    model_name_or_path = "/u/zliu/datastor1/mend/models/Qwen2.5-1.5B-eos-sft"


model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_eos_token=True, use_fast=False)
tokenizer.padding_side = "right"
original_vocab_size = len(tokenizer)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<sft_token_1>']}, replace_additional_special_tokens=False)
model.resize_token_embeddings(len(tokenizer))

tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

# import pdb; pdb.set_trace()
assert tokenizer.eos_token != tokenizer.pad_token
assert tokenizer.eos_token_id != tokenizer.pad_token_id

# train_dataset = prepare_sft_text(args, io.load_jsonlines(f"{vars.DATA_DIR}/trivia_qa_wiki_sft/train.jsonl"), tokenizer)
# valid_dataset = prepare_sft_text(args, io.load_jsonlines(f"{vars.DATA_DIR}/trivia_qa_wiki_sft/valid.jsonl"), tokenizer)

# train_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_date_data/train.jsonl"), tokenizer
# )
# valid_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_date_data/valid.jsonl"), tokenizer
# )

if custom_args.eos_sft:
    train_dataset = prepare_sft_text(args, io.load_jsonlines(f"{vars.DATA_DIR}/trivia_qa_wiki_sft/train.jsonl"), tokenizer)
else:
    train_dataset = prepare_sft_text(
        args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/model_prep/light_weight_sft_content_curated_v1_sample=10.jsonl"), tokenizer
    )

# train_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_country_data/train.jsonl"), tokenizer
# )
# valid_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_country_data/valid.jsonl"), tokenizer
# )

if custom_args.is_peft:

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r = 128,
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    args.output_dir += f"-r{peft_config.r}"

response_template = "?"  # tokenizer.additional_special_tokens[0] # "?" alternative "Ġ?"

# filtered based on whether "?" could be detected
filtered_train_dataset = []
for datum in train_dataset:
    text = datum[args.dataset_text_field]
    if all([x in tokenizer(text)["input_ids"] for x in tokenizer(response_template)["input_ids"]]):
        filtered_train_dataset.append(datum)
# pdb.set_trace()
train_dataset = Dataset.from_list(filtered_train_dataset)
# valid_dataset = Dataset.from_list(valid_dataset) 

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,  # type: ignore
    # eval_dataset=valid_dataset,  # type: ignore
    args=args,
    data_collator=collator,
)

trainer.train()

trainer.model.config.pad_token_id = None
trainer.model.resize_token_embeddings(original_vocab_size)
trainer.model.save_pretrained(save_directory=args.output_dir)

trainer.accelerator.wait_for_everyone()

