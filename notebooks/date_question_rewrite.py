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

rewrite_tag_extractor = extractor.tag_content_extractor("rewrite")
    
class DateQuestionRewrite(curator.LLM):
    PROMPT : str = """
[Instruction]: Change the question to be "When was the year after the year that [X] happened?"

[Question]: {question}


Return the re-written question wrapped in <rewrite>..</rewrite> tag.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(question=input["question"], )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        question_ = rewrite_tag_extractor(response)
        assert len(question_) == 1
        question = question_[0].strip()
        
        input["rewrite_question"] = question
        return {**input}

llm_judge = DateQuestionRewrite(model_name="gpt-4o")
fpath = "/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/common_date_data/common_fact_date.xlsx"
# fpath = "/u/zliu/datastor1/mend/exp_output/eos-sft_musique_propagator_text_hidden_w-atomq/musique/mend_eval_loss=clm_input=hidden_n=1000_prompt=no_w-gen_wo-icl_spec.xlsx"
df = pd.read_excel(fpath)
df["answer"] = df["answer"].astype(str)


dataset = Dataset.from_pandas(df)
dataset = llm_judge(dataset)

dataset.to_pandas().to_excel("/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/common_date_data/common_fact_date_w-rewrite.xlsx", index=False)
print()
