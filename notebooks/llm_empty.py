from datasets import load_dataset
from tqdm import tqdm
from knowledge_propagation.utils import vars, io, extractor
from scipy.stats import describe
from typing import List, Dict
import re
import os
from copy import deepcopy
import pandas as pd
from glob import glob
from bespokelabs import curator
from datasets import Dataset

score_tag_extractor = extractor.tag_content_extractor("score")


class LlmPrompt(curator.LLM):
    MAX_VAL: float = 10.0
    PROMPT: str = """
{input}
""".strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            input=input["input"], 
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        input["predicted_answer"] = response
        return {**input}

model_name = "gpt-4o"
llm_prompt = LlmPrompt(
    model_name=model_name, 
    generation_params={
        "max_tokens": 20,
    },
    backend_params={
        "max_requests_per_minute": 30_000, 
        "max_tokens_per_minute": 150_000_000,
    }
)
# all_files = glob(
#     "/data/users/zliu/mend/country_exp_out/3K_heavy_noshare_mid-upper3_template-5_seen-350/**/*.xlsx", recursive=True
# )

fpath = "/u/zliu/datastor1/mend/synstory_exp_output/curator/4K_test_id/base_n=500_prompt=no_w-gen_wo-icl_ice=True.xlsx"
fdir = os.path.dirname(fpath)
fname = os.path.basename(fpath)

model_fpath = f"{fdir}/{model_name}_{fname}"

df = pd.read_excel(fpath)

dataset = Dataset.from_pandas(df[:])
answered_dataset = llm_prompt(
    dataset,
)

answered_dataset.to_pandas().to_excel(model_fpath, index=False)

print(fpath)


# from pdb import set_trace
# set_trace()
# fpath = "/datastor1/zliu/mend/debug_exp_output/llama3.2-1B-eos-sft/ood_v3_prefilter/base_n=1185_prompt=no_w-gen_wo-icl_ice=False.xlsx"
# fpath = "/u/zliu/datastor1/mend/exp_output/eos-sft_musique_propagator_text_hidden_w-atomq/musique/mend_eval_loss=clm_input=hidden_n=1000_prompt=no_w-gen_wo-icl_spec.xlsx"
# import pdb; pdb.set_trace()
