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


def add_padding(tokenizer, model):
    import transformers

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM):
        #     model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight.mean(0)
        # else:
        model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


def interpolate_params(delta_params_A, delta_params_B, alpha):
    """
    alpha: weights for delta_params_A
    """
    assert set(delta_params_A.keys()) == set(delta_params_B.keys())

    delta_params = {}
    for k in delta_params_A.keys():
        delta_params[k] = (1 - alpha) * delta_params_B[k] + alpha * delta_params_A[k]

    return delta_params


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

    assert hasattr(config, "alpha")
    assert hasattr(config, "tunable_params")

    if config.tunable_params == "midupper3-mlp":
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
        raise ValueError(f"Unknown tunable_params: {config.tunable_params}")
    original_params = {}
    for n, param in model.named_parameters():
        if any(p in n for p in params):
            original_params[n] = param.clone().cpu()
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainer = EditTrainer(alg, config, train_set, val_set)
    print("Task: ", config.task)

    assert hasattr(config, "spec_question")
    assert hasattr(config, "date_data")
    # import pdb

    # pdb.set_trace()
    if config.date_data == "recent+popular":
        edit_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/test.jsonl")
    elif config.date_data == "recent":
        edit_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/test.jsonl")
    else:
        raise NotImplementedError("Only all_propagation is supported for date_data")

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
        if config.tunable_params == "midupper3-mlp":
            # Load CPT update
            delta_cpt_params = torch.load(
                f"/data/users/zliu/mend/ripple_exp_output/Llama-3.2-1B-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tuned-params=midupper3-mlp/delta_params_recent+popular/{i}_midupper3-mlp.pt",
                weights_only=True,
            )
        sentences = [datum["edit"]["prompt"]]

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

        all_datum_result_df = []
        outerloop_queries = []
        for k in ["Logical_Generalization", "Compositionality_I", "Compositionality_II", "Subject_Aliasing"]:
            for instance in datum[k]:
                for q in instance["test_queries"]:
                    if (
                        len(q["answers"]) > 0
                        and len([a["value"] for a in q["answers"] if len(a["value"].strip()) > 0]) > 0
                    ):
                        q["question_type"] = k
                        outerloop_queries.append(q)

        assert len(outerloop_queries) > 0

        locality_queries = []
        for k in ["Relation_Specificity", "Forgetfulness"]:
            for instance in datum[k]:
                for q in instance["test_queries"]:
                    if (
                        len(q["answers"]) > 0
                        and len([a["value"] for a in q["answers"] if len(a["value"].strip()) > 0]) > 0
                    ):
                        q["question_type"] = k
                        locality_queries.append(q)
        assert len(locality_queries) > 0

        question_types = [
            ("efficacy", outerloop_queries),
            ("specificity", locality_queries),
        ]

        # for question_type, questions in question_types:
        #     logging.info(f"Question type: {question_type}")

        #     for q_i, question in enumerate(questions):
        #         answer_candidates = [a["value"] for a in question["answers"]]
        #         answer = answer_candidates[0]

        #         pre_result_df = get_eval_result(
        #             question=question["prompt"],
        #             answer=answer,
        #             model=trainer.model.model,
        #             tokenizer=tokenizer,
        #             config=config,
        #             generation_config=generation_config,
        #         )
        #         pre_result_df.insert(0, "stage", "pre-edit")
        #         pre_result_df.insert(
        #             0, "edit_input", "\n\n".join(f"[[{tokenizer.decode(s)}]]" for s in sentences_toks["input_ids"])
        #         )
        #         pre_result_df.insert(0, "relation", f"{question['relation']}")
        #         pre_result_df.insert(0, "question_tag", f"{question_type}_{question['question_type']}")
        #         pre_result_df.insert(0, "question_type", question_type)
        #         pre_result_df.insert(0, "id", str(i))
        #         # import pdb
        #         # pdb.set_trace()
        #         all_datum_result_df.append(pre_result_df)

        # edit the model with MEND
        edited_model, model_info = trainer.model.edit(edit_inner)
        model_info["input"] = sentences[0]
        model_info["target"] = tokenizer.decode(targets_toks["input_ids"][0])
        edit_model_infos.append(model_info)

        # Get update from MEND
        # TODO: make this more elegant
        delta_mend_params = {}
        for n, param in edited_model.model.named_parameters():
            if any(p in n for p in params):
                # updated param - original param
                delta_mend_params[n] = param.clone().cpu() - original_params[n]

        # Interpolate update from MEND and CPT
        # alpha * mend_update + (1-alpha) * cpt_update
        delta_params = interpolate_params(delta_mend_params, delta_cpt_params, config.alpha)
        delta_params = utils.dict_to(delta_params, config.device)

        # Apply the interpolated update to the model
        for n, param in edited_model.model.named_parameters():
            if any(p in n for p in params):
                # original param + delta param
                param.data = original_params[n].to(config.device).data + delta_params[n].data

        for question_type, questions in question_types:
            logging.info(f"Question type: {question_type}")

            for q_i, question in enumerate(questions):
                answer_candidates = [a["value"] for a in question["answers"]]
                answer = answer_candidates[0]

                post_result_df = get_eval_result(
                    question=question["prompt"],
                    answer=answer,
                    model=edited_model.model,
                    tokenizer=tokenizer,
                    config=config,
                    generation_config=generation_config,
                )
                post_result_df.insert(0, "stage", "post-edit")
                post_result_df.insert(
                    0, "edit_input", "\n\n".join(f"[[{tokenizer.decode(s)}]]" for s in sentences_toks["input_ids"])
                )
                post_result_df.insert(0, "relation", f"{question['relation']}")
                post_result_df.insert(0, "question_tag", f"{question_type}_{question['question_type']}")
                post_result_df.insert(0, "question_type", question_type)
                post_result_df.insert(0, "id", str(i))
                # import pdb
                # pdb.set_trace()
                all_datum_result_df.append(post_result_df)

        del edited_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_datum_result_df = pd.concat(all_datum_result_df)
        all_results.append(all_datum_result_df)

    all_results = pd.concat(all_results)

    if config.generation.save_dir:
        save_dir = config.generation.save_dir + "_alpha=" + str(config.alpha)
        if os.path.abspath(save_dir) != config.generation.save_dir:
            # using relative path
            save_dir = f"{base_dir}/{save_dir}"

        LOG.info(f"Saving to dir: {save_dir}")

        os.makedirs(save_dir, exist_ok=True)
        fpath = (
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if config.do_generation else 'wo'}-gen_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl"
            + ("_e+s" if config.spec_question else "_e")
            + f"_{config.date_data}-question"
            + ".xlsx"
        )

        all_results.to_excel(fpath, index=False)
        io.dump_jsonlines(
            edit_model_infos,
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl_edit-model-infos.jsonl",
        )


if __name__ == "__main__":
    run()
