import os
import sys
import json
from sqlalchemy import all_
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from loguru import logger
import pandas as pd
from tqdm import tqdm

from knowledge_propagation.utils import vars, io, misc
from knowledge_propagation.modules.inferencers import QAInferencer
from knowledge_propagation.modules.rag_model import (
    RAGModel,
    OracleRAGModel,
    RAGModelType,
)
from transformers import PreTrainedTokenizer, GenerationConfig
import gc

from typing import List
from knowledge_propagation.model import ContinuedPretrainedModel


def eval_inferencer(
    inferencer: QAInferencer,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationConfig,
    save_dir: str = None,
    rerun: bool = True,
    return_df: bool = True,
):
    logger.info(f"Evaluating model type: {type(model)}")
    # logger.info(f"Generation config: {generation_cfg}")
    if not return_df:
        assert save_dir is not None, "save_dir must be provided when return_df is False"
        os.makedirs(save_dir, exist_ok=True)

    if generation_cfg.do_sample:
        response_filepath = f"{save_dir}/template={inferencer.template_name}_seed={inferencer.seed}_p={generation_cfg.top_p}_tok{inferencer.max_new_tokens}_{inferencer.label}.xlsx"
        logger.info("Using Top-p decoding")
    else:
        assert generation_cfg.num_beams == 1  # doing greedy
        response_filepath = f"{save_dir}/template={inferencer.template_name}_greedy_tok{inferencer.max_new_tokens}_{inferencer.label}.xlsx"
        logger.info("Using Greedy decoding")

    if save_dir is not None:
        logger.info(f"Save path: {response_filepath}")

    if os.path.exists(response_filepath) and not rerun:
        model_responses = pd.read_excel(response_filepath)
    else:
        model_responses = inferencer.infer_w_model(tokenizer, model, generation_cfg)

    if not inferencer.has_answer:
        return

    metric_scores = inferencer.aggregate_metric_from_df(model_responses)
    if return_df:
        return model_responses

    model_responses.to_excel(response_filepath, index=False)
    for s_i, metric_score in enumerate(metric_scores):
        agg_metric_filepath = io.remove_last_extension(response_filepath) + f"_metric-{s_i}.json"
        logger.info(f"Metric scores: {metric_score}")
        io.dump_json(metric_score, agg_metric_filepath)


def macro_averaging(df: pd.DataFrame, metrics: List[str], multi_level_averaging: List[str]):
    """
    Do macro-averaging over the given metrics and multi-level averaging categories.
    """
    extracted_multi_level_cols = [[m, "mean"] for m in metrics]
    while len(multi_level_averaging) > 0:
        # first take the mean over each generation,
        # and, only take `mean` of `rouge1` and  `llm_accuracy` column groups
        df_over_cols = df.groupby(multi_level_averaging, observed=True).describe()[extracted_multi_level_cols]
        # remove the multi-level column indices, since there's only one sub-level -- "mean"
        df_over_cols.columns = df_over_cols.columns.get_level_values(0)

        # reset index to flatten the multi-level column indices for the next macro-averaging class
        df = df_over_cols.reset_index(inplace=False)
        multi_level_averaging.pop(-1)
    return df


@hydra.main(
    version_base=None,
    config_path=f"{vars.PROJ_DIR}/configs",
    config_name="generate_only.yaml",
)
def main(cfg: OmegaConf):
    model_name = os.path.basename(cfg.model.model_name_or_path)

    tag = "page" if "page" in cfg.data.train_file else "paragraph"
    data_name = HydraConfig.get().runtime.choices["data"]

    out_dir = f"{vars.EXP_OUTPUT_ROOT}/{data_name}_{model_name}/rag={cfg.evaluator.rag.label}_tag={tag}"
    # import pdb; pdb.set_trace()
    if cfg.tune_suffix:
        out_dir = f"{out_dir}_tune/{cfg.tune_suffix}"
    os.makedirs(out_dir, exist_ok=True)

    out_f_stem = f"{out_dir}/all_results"

    logger.add(f"{out_f_stem}.log")
    logger.add(
        f"{out_f_stem}.debug",
        level="DEBUG",
    )
    logger.info(f"Command runned: python {' '.join(sys.argv)}")
    OmegaConf.save(config=cfg, f=f"{out_dir}/config.yaml")

    raw_dataset = io.load_jsonlines(f"{vars.PROJ_DIR}/{cfg.data.data_dir}/{cfg.data.train_file}")
    logger.info(f"{vars.PROJ_DIR}/{cfg.data.data_dir}/{cfg.data.train_file}")

    all_results = []

    cpt_model = ContinuedPretrainedModel(cfg, do_train=cfg.do_train)
    question_types = [
        # "single_hop_efficacy",
        "multi_hop_efficacy",
        # "single_hop_specificity",
        # "multi_hop_specificity",
    ]
    # for raw_instance in tqdm(raw_dataset[11:13], desc="Running Inference"): # debug
    # debug_example_ids = ["2hop__132710_120035"]
    for raw_instance in tqdm(raw_dataset, desc="Running Inference"):
        # if raw_instance["id"] not in debug_example_ids:
        #     continue
        logger.info(f"Example ID: {raw_instance['id']}")
        ref_dataset = raw_instance["texts"]

        for question_type in question_types:
            logger.info(f"Evaluating on [{question_type}]")
            # make a rag model
            if cfg.evaluator.rag.label == "none" or "specificity" in question_type:
                rag_model = None
            elif "oracle" in cfg.evaluator.rag.label:
                # implement by a fixed mapping from question to document

                query2doc_map = {
                    q["question"]: "\n\n".join([ref_dataset[i] for i in q["supporting_text_ids"]])
                    if "supporting_text_ids" in q
                    else ""
                    for q in raw_instance[question_type]
                }
                rag_model = OracleRAGModel(query2doc_map, cfg.evaluator.rag, cfg.seed)
            else:
                assert misc.is_valid_strType(
                    cfg.evaluator.rag.label, RAGModelType
                ), f"`{cfg.evaluator.rag.label}` is not supported."
                # TODO: take a set over all the supporting text ids
                rag_model = RAGModel(
                    [
                        "\n\n".join([ref_dataset[i] for i in q["supporting_text_ids"]])
                        if "supporting_text_ids" in q
                        else ""
                        for q in raw_instance[question_type]
                    ],
                    cfg.evaluator.rag,
                    cfg.seed,
                )

            questions = raw_instance[question_type]
            inferencer = QAInferencer(
                cfg.evaluator.inferencers[0],
                cfg.seed,
                rag_model=rag_model,
                queries=questions,
            )
            result_df = eval_inferencer(
                inferencer,
                cpt_model.model,
                tokenizer=cpt_model.tokenizer,
                generation_cfg=cpt_model.generation_config,
            )
            result_df.insert(0, "question_type", question_type)
            result_df.insert(0, "id", raw_instance["id"])
            all_results.append(result_df)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_results = pd.concat(all_results)

    all_results.to_excel(
        f"{out_f_stem}.xlsx",
        index=False,
    )
    logger.info(f"Result save to: {out_f_stem}")
    # calculate and printthe macro-averaged results
    metrics = ["rouge1", "llm_accuracy"]
    multi_level_averaging = ["question_type", "id", "question"]

    # Doing macro-averaging over: generations -> questions -> id
    result_df = macro_averaging(all_results, metrics, multi_level_averaging).round(2)
    q_cat_dtype = pd.CategoricalDtype(
        categories=question_types,
        ordered=True,
    )

    result_df["question_type"] = result_df["question_type"].astype(q_cat_dtype)
    logger.info(result_df.sort_values(by=["question_type"], inplace=False))


if __name__ == "__main__":
    main()
