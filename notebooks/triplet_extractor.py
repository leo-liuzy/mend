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


subject_extractor = extractor.tag_content_extractor("subject")
object_extractor = extractor.tag_content_extractor("object")


class TripletExtractor(curator.LLM):
    PROMPT: str = """
Return the subject and the object of the following sentence:
{sentence}

Example 1:
The name of the ethnic group which bell hooks is associated with is Arbëreshë.
<subject>bell hooks</subject>
<object>Arbëreshë</object>

Example 2:
The name of the country which 2008 United States presidential election is associated with is Mordovia.
<subject>2008 United States presidential election</subject>
<object>Mordovia</object>

Example 3:
The name of the author of Relative contribution of groundwater to plant transpiration estimated with stable isotopes is Rick G Pleijhuis.
<subject>Relative contribution of groundwater to plant transpiration estimated with stable isotopes</subject>
<object>Rick G Pleijhuis</object>

Only return the subject and the object wrapped in <subject>..</subject> and <object>..</object> tag.
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            sentence=input["prompt"],
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        subject_ = subject_extractor(response)
        assert len(subject_) == 1
        subject = subject_[0]

        object_ = object_extractor(response)
        assert len(object_) == 1
        obj = object_[0]
        assert "subject" not in input
        assert "object" not in input

        input["subject"] = subject
        input["object"] = obj

        # assert obj in input["prompt"]
        # assert subject in input["prompt"]

        return {**input}


context_extractor = extractor.tag_content_extractor("context")
paraphrase_extractor = extractor.tag_content_extractor("paraphrase")


class ContextParaphraser(curator.LLM):
    PROMPT: str = """
Write a paraphrase for the context of the following sentence.
[Sentence]
{sentence}

[Context]
{context}

Only return the paraphrased context in <paraphrase>..</paraphrase> tag.
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            sentence=input["prompt"],
            context=input["context"],
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""

        paraphrase_ = paraphrase_extractor(response)
        assert len(paraphrase_) == 1
        paraphrase = paraphrase_[0].strip()

        input["paraphrase"] = paraphrase
        # import pdb

        # pdb.set_trace()
        # assert obj in input["prompt"]
        # assert sub in input["prompt"]

        return {**input}


# fact_generator = CommonCitiesGenerator(model_name="gpt-4o")
# df = pd.DataFrame(countries, columns=["country"])

# dataset = Dataset.from_pandas(df)
# dataset = fact_generator(dataset)

# dataset.save_to_disk("/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/country_data/common_cities_generation.hf",)
split = "valid"
# manual_process_ids = io.load_jsonlines(f"/data/users/zliu/mend/notebooks/{split}_manual_process_ids.jsonl")
test_instances = list(io.load_jsonlines(f"{vars.DATA_DIR}/ripple_edits/meta_train/all/{split}_missing_parahrase.jsonl"))
# test_instances = [test_instances[i] for i in manual_process_ids]


triplet_extractor = ContextParaphraser(
    model_name="gpt-4o",
    backend_params={
        "max_requests_per_minute": 30_000,
        "max_tokens_per_minute": 150_000_000,
        "require_all_responses": True,
    },
)

# df = pd.DataFrame([deepcopy(x["edit"]) for x in test_instances])
df = pd.DataFrame([deepcopy(x) for x in test_instances])

dataset = Dataset.from_pandas(df)
new_ds = triplet_extractor(
    dataset,
)
new_ds.save_to_disk(
    f"{vars.DATA_DIR}/ripple_edits/meta_train/all/{split}_missing_parahrase_w_paraphrase.hf",
)


print()
