from datasets import load_dataset
from tqdm import tqdm
from knowledge_propagation.utils import vars, io
from scipy.stats import describe
from typing import List, Dict
import re
from copy import deepcopy
import pandas as pd

from bespokelabs import curator
from datasets import Dataset
    
    
class LlmAsJudge(curator.LLM):
    PROMPT : str = """
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
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(question=input["question"], prediction=input["predicted_answer"], reference=input["answer"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        
        return {**input, "nl_question": response}

llm_judge = LlmAsJudge(model_name="gpt-4o-mini")
fpath = "/u/zliu/datastor1/mend/exp_output/eos-sft_musique_propagator_p0_w-atomq/musique/mend_eval_loss=clm_input=2doc_n=1000_prompt=no_w-gen_wo-icl.xlsx"

nl_zsre_question_df = pd.read_excel(fpath)

if "llm_accuracy" in nl_zsre_question_df.columns:
    exit(0)

nl_zsre_question_dataset = Dataset.from_pandas(nl_zsre_question_df)

nl_zsre_question_dataset = llm_judge(nl_zsre_question_dataset)
MAX_VAL = 10
llm_acc = []
for score in nl_zsre_question_dataset['nl_question']:
    try:
        score: str
        if score.isdigit():
            score = float(score)
        else:
            assert "Score " in score, "Failing heuristic fix"
            score = float(score[len("Score ") :])

        assert 1 <= score <= 10, "Score needs to be in scale of [1, 10]"
    except Exception as e:
        print(e)
        score = 0.0
    score /= MAX_VAL
    llm_acc.append(score)
nl_zsre_question_df["llm_accuracy"] = llm_acc
# nl_zsre_question_dataset.to_json(f"/data/users/zliu/KE-by-CP/data/musique_mend/2hop_musique_ans_v1.0_{split}_zsre-questions.jsonl")
nl_zsre_question_df.to_excel(fpath, index=False)
print()
