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

    hparams = MEMITHyperParams.from_hparams(f"{os.getenv('PROJ_PLAYGROUND')}/EasyEdit/hparams/MEMIT/{config.config_name}")
    hparams.mom2_dataset = config.mom2_dataset
    # hparams.mom2_dataset = "ripple_recent+popular"
    # hparams.mom2_dataset = "wikipedia"

    print("Task: ", config.task)

    assert hasattr(config, "spec_question")
    assert hasattr(config, "date_data")

    if config.date_data == "4K_test_id":
        edit_dev_dataset = io.load_jsonlines(
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_structure_data_id_entity152_rel31.jsonl"
        )
        config.val_steps = 500

    elif config.date_data == "4K_test_ood":
        edit_dev_dataset = io.load_jsonlines(
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_structure_data_ood_entity37_rel7.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "4K_test_ood-relation":
        edit_dev_dataset = io.load_jsonlines(
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_structure_data_ood-relation_entity152_rel7.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "4K_test_ood-entity":
        edit_dev_dataset = io.load_jsonlines(
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_structure_data_ood-entity_entity37_rel31.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.date_data == "profile":
        edit_dev_dataset = io.load_jsonlines(
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/4Ktrain_data_100percent_frozen/test_structure_data_id_entity152_rel31.jsonl"
        )
        config.val_steps = 50
        edit_dev_dataset = edit_dev_dataset[:config.val_steps]
        assert len(edit_dev_dataset) == config.val_steps
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
    target_modules_names = [hparams.rewrite_module_tmp.format(x) + ".weight" for x in hparams.layers]
    weights_copy = {n: p.clone() for n, p in editor.model.named_parameters() if n in target_modules_names}
    # pdb.set_trace()
    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        # for i in tqdm([717, 718, 719], desc=f"Running eval on {config.task}"):
        # for i in tqdm(range(1), desc=f"Running eval on {config.task}"):
        datum = edit_dev_dataset[i]

        edits = []
        prompts = []
        subjects = []
        objects = []
        for fact in datum["facts"]:
            edits.append(fact["fact"])
            prompts.append(tokenizer.bos_token + fact["prefix"])
            subjects.append(datum["subject"])
            objects.append(str(fact["target"]))

        assert config.edit_loss == EditLoss.clm, f"edit_loss `{config.edit_loss}` is not supported"

        all_datum_result_df = []
        efficacy_questions = [
            {"question": q["alias_question"], "answer": q["answer"], "relation": q["question_template"]}
            for q in datum["questions"]
        ]
        specificity_questions = [
            {"question": q["unalias_question"], "answer": q["answer"], "relation": q["question_template"]}
            for q in datum["questions"]
        ]

        question_types = [
            ("efficacy", efficacy_questions),
            ("specificity", specificity_questions),
        ]
        # import pdb

        # pdb.set_trace()
        assert all([len(x) > 0 for x in edits])
        assert all([len(x) > 0 for x in prompts])
        assert all([len(x) > 0 for x in subjects])
        assert all([len(x) > 0 for x in objects])

        # edit the model with MEND
        metrics, edited_model, _ = editor.edit(
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
                # import pdb

                # pdb.set_trace()
                post_result_df = get_eval_result(
                    question=question["question"],
                    answer=str(question["answer"]),
                    model=edited_model,
                    tokenizer=tokenizer,
                    config=config,
                    generation_config=generation_config,
                )
                post_result_df.insert(0, "stage", "post-edit")
                post_result_df.insert(0, "edit_input", "\n\n".join(edits))
                post_result_df.insert(0, "relation", f"{question['relation']}")
                if question_type == "efficacy":
                    post_result_df.insert(0, "question_key", "alias_question")
                else:
                    assert question_type == "specificity"
                    post_result_df.insert(0, "question_key", "unalias_question")
                post_result_df.insert(0, "question_type", question_type)
                post_result_df.insert(0, "id", str(i))
                # import pdb

                # pdb.set_trace()
                all_datum_result_df.append(post_result_df)

        # ! rollback the model
        # pdb.set_trace()
        assert len([name for name, param in edited_model.named_parameters() if name in weights_copy]) == len(
            weights_copy
        )
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
        # io.dump_jsonlines(
        #     edit_model_infos,
        #     f"{save_dir}/memit_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl_edit-model-infos.jsonl",
        # )


if __name__ == "__main__":
    run()
