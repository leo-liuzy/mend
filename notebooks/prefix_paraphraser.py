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


paraphrase_extractor = extractor.tag_content_extractor("paraphrase")


class PrefixParaphraser(curator.LLM):
    PROMPT: str = """
Given a context and a completion of the context, return the a paraphrase of the context so that context concatenated with the completion is semantically equivalent to the paraphrase concatednated with the completion.

<context>{context}</context>
<completion>{completion}</completion>

Return the paraphrased context in <paraphrase>..</paraphrase> tag.
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            context=input["context"],
            completion=input["completion"],
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        paraphrase_ = paraphrase_extractor(response)
        assert len(paraphrase_) == 1
        paraphrase = paraphrase_[0].strip()

        input["paraphrase"] = paraphrase

        return {**input}


# fact_generator = CommonCitiesGenerator(model_name="gpt-4o")
# df = pd.DataFrame(countries, columns=["country"])

# dataset = Dataset.from_pandas(df)
# dataset = fact_generator(dataset)

# dataset.save_to_disk("/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/country_data/common_cities_generation.hf",)
split = "test"
test_instances = list(io.load_jsonlines(f"{vars.DATA_DIR}/ripple_edits/meta_train/all/{split}_mend_nophrase.jsonl"))

triplet_extractor = PrefixParaphraser(
    model_name="gpt-4o",
    backend_params={
        "max_requests_per_minute": 30_000,
        "max_tokens_per_minute": 150_000_000,
        "require_all_responses": True,
    },
)

df = pd.DataFrame(test_instances)

dataset = Dataset.from_pandas(df)
new_ds = triplet_extractor(
    dataset,
)
new_ds.save_to_disk(
    f"{vars.DATA_DIR}/ripple_edits/meta_train/all/{split}_w_paraphrase.hf",
)


print()
