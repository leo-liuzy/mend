from datasets import load_dataset
from tqdm import tqdm
from knowledge_propagation.utils import vars, io, extractor
from scipy.stats import describe
from typing import List, Dict
import re
from copy import deepcopy
import pandas as pd

from bespokelabs import curator
from datasets import Dataset

score_tag_extractor = extractor.tag_content_extractor("score")

from knowledge_propagation.modules.evaluators import (
    NumDiffEvaluator,
    ExactMatchEvaluator,
)
year_diff_evaluator = NumDiffEvaluator()
em_evaluator = ExactMatchEvaluator()

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

    model_response_w_score = df.join(pd.DataFrame({
        **em_per_example, 
        # **diff_per_example,
    }))
    
    return model_response_w_score


class LlmAsJudge(curator.LLM):
    MAX_VAL: float = 10.0
    PROMPT: str = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
accuracy: 
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numerical score.

[Question]
{question}

[The Start of Ground truth]
{reference}
[The End of Ground truth]

[The Start of Assistant's Answer]
{prediction}
[The End of Assistant's Answer]

Return the numerical score wrapped in <score>..</score> tag
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            question=input["question"], prediction=input["predicted_answer"], reference=input["answer"]
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        score_ = score_tag_extractor(response)
        assert len(score_) == 1
        score = score_[0].strip()
        assert score.isdigit()
        assert 1 <= float(score) <= 10
        score = float(score)
        score /= self.MAX_VAL
        if "llm_accuracy" in input:
            existing_llm_acc = input["llm_accuracy"]
            del input["llm_accuracy"]
        input["llm_accuracy"] = score

        return {**input}



llm_judge = LlmAsJudge(
    model_name="gpt-4o-mini", backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
)
fpath = "/u/zliu/datastor1/mend/debug_exp_output/Llama-3.2-1B-common-date-year-after-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tunable-params=all/all_results_id+ood_v2_text_w-icl=True.xlsx"
# fpath = "/u/zliu/datastor1/mend/exp_output/eos-sft_musique_propagator_text_hidden_w-atomq/musique/mend_eval_loss=clm_input=hidden_n=1000_prompt=no_w-gen_wo-icl_spec.xlsx"
scored_df = pd.read_excel(fpath)
scored_df["predicted_answer"] = scored_df["predicted_answer"].astype(str)
scored_df["answer"] = scored_df["answer"].astype(str)

scored_df = scored_df.drop(columns=["__index_level_0__"], inplace=False, errors="ignore")
scored_df = score_df(scored_df)
if "is_num" in scored_df.columns:
    non_numeric_df = scored_df[~scored_df["is_num"]]
else:
    non_numeric_df = scored_df
scored_dataset = Dataset.from_pandas(non_numeric_df)
scored_dataset = llm_judge(
    scored_dataset,
)

non_numeric_scored_df = scored_dataset.to_pandas()

if "is_num" in scored_df.columns:
    pd.concat([scored_df[scored_df["is_num"]], non_numeric_scored_df]).sort_values(by=["id", "question_tag"]).to_excel(fpath, index=False)
else:
    non_numeric_scored_df.sort_values(by=["id", "question_tag"]).to_excel(fpath, index=False)
print(fpath)
