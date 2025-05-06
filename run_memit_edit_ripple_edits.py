import copy
import pdb
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
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

from easyeditor import MEMITHyperParams
from easyeditor import BaseEditor

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


@hydra.main(config_path="config", config_name="config")
def run(config):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)
    # add_padding(tokenizer, model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    from data_classes.zsre import ZsreDataset

    val_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config)

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

    hparams = MEMITHyperParams.from_hparams(f"/data/users/zliu/EasyEdit/hparams/MEMIT/{config.config_name}")
    hparams.mom2_dataset = config.mom2_dataset
    # hparams.mom2_dataset = "ripple_recent+popular"
    # hparams.mom2_dataset = "wikipedia"

    print("Task: ", config.task)

    assert hasattr(config, "spec_question")
    assert hasattr(config, "date_data")

    if config.date_data == "recent":
        edit_dev_dataset = io.load_jsonlines(
            f"{vars.DATA_DIR}/ripple_edits/meta_train_old/meta_train_recent/test_mend.jsonl"
        )
    elif config.date_data == "recent+popular":
        edit_dev_dataset = io.load_jsonlines(
            f"{vars.DATA_DIR}/ripple_edits/meta_train_old/meta_train_recent+popular/test_aug.jsonl"
        )
    elif config.date_data == "all":
        edit_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/ripple_edits/meta_train/all/test_aug.jsonl")
        config.val_steps = 500
    else:
        raise NotImplementedError("Only all_propagation is supported for date_data")
    #     assert config.date_data == "n"
    #     edit_dev_dataset = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/test_n_question.jsonl")

    all_results = []
    edit_model_infos = []
    # trainer.validate(log=True)
    assert config.val_steps <= len(edit_dev_dataset)
    assert config.eval_only
    if hasattr(config, "add_icl") and config.add_icl:
        eos_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    else:
        eos_token_id = tokenizer.eos_token_id
    editor = BaseEditor.from_hparams(hparams)

    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        # for i in tqdm([717, 718, 719], desc=f"Running eval on {config.task}"):
        # for i in tqdm(range(1), desc=f"Running eval on {config.task}"):
        datum = edit_dev_dataset[i]

        if datum["edit"]["target"] + "." in datum["edit"]["prompt"]:
            prefix = datum["edit"]["prompt"].replace(datum["edit"]["target"] + ".", "").strip()
        prompts = [tokenizer.bos_token + prefix]
        objects = [datum["edit"]["target"]]
        assert datum["edit"]["subject"] is not None, "subject is None"
        subjects = [datum["edit"]["subject"]]

        assert config.edit_loss == EditLoss.clm, f"edit_loss `{config.edit_loss}` is not supported"

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
        # import pdb

        # pdb.set_trace()

        if datum["edit"]["subject"] == "":
            edited_model = editor.model
            weights_copy = {}
            metrics = {}
        else:
            # edit the model with MEND
            metrics, edited_model, weights_copy = editor.edit(
                prompts=prompts,
                ground_truth=None,
                target_new=objects,
                subject=subjects,
                keep_original_weight=True,
            )
        edit_model_infos.append(metrics)
        # pdb.set_trace()
        for question_type, questions in question_types:
            logging.info(f"Question type: {question_type}")

            for q_i, question in enumerate(questions):
                answer_candidates = [a["value"] for a in question["answers"]]
                answer = answer_candidates[0]
                # import pdb

                # pdb.set_trace()
                post_result_df = get_eval_result(
                    question=question["prompt"],
                    answer=answer,
                    model=edited_model,
                    tokenizer=tokenizer,
                    config=config,
                    generation_config=generation_config,
                )
                post_result_df.insert(0, "stage", "post-edit")
                post_result_df.insert(0, "edit_input", f"{datum['edit']['prompt']}")
                post_result_df.insert(0, "relation", f"{question['relation']}")
                post_result_df.insert(0, "question_tag", f"{question_type}_{question['question_type']}")
                post_result_df.insert(0, "question_type", question_type)
                post_result_df.insert(0, "id", str(i))
                # import pdb

                # pdb.set_trace()
                all_datum_result_df.append(post_result_df)

        # rollback the model
        # pdb.set_trace()
        for name, param in edited_model.named_parameters():
            if name in weights_copy:
                # pdb.set_trace()
                param.data = weights_copy[name].data.clone()
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
        fpath = (
            f"{save_dir}/memit({hparams.mom2_dataset})_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if config.do_generation else 'wo'}-gen_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl"
            + ("_e+s" if config.spec_question else "_e")
            + f"_{config.date_data}-question"
            + ".xlsx"
        )

        all_results.to_excel(fpath, index=False)
        io.dump_jsonlines(
            edit_model_infos,
            f"{save_dir}/memit_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl_edit-model-infos.jsonl",
        )


if __name__ == "__main__":
    run()
