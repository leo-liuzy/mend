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


class PrefixSplit(curator.LLM):
    PROMPT: str = """
Split the following sentence into a context that preceeds its object and the object:
{sentence}

Then, paraphrase the context so that the context concatenated with the object is semantically equivalent to the original sentence.
Also, return the subject of the sentence.

Only return the context and the object wrapped in <context>..</context> and <object>..</object> tag. And return the paraphrased context in <paraphrase>..</paraphrase> tag. And return the subject in <subject>..</subject> tag.
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            sentence=input["prompt"],
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        context_ = context_extractor(response)
        assert len(context_) == 1
        context = context_[0].strip()

        paraphrase_ = paraphrase_extractor(response)
        assert len(paraphrase_) == 1
        paraphrase = paraphrase_[0].strip()

        object_ = object_extractor(response)
        assert len(object_) == 1
        obj = object_[0].strip()
        subject_ = subject_extractor(response)
        assert len(subject_) == 1
        sub = subject_[0].strip()
        assert "subject" not in input
        assert "object" not in input

        input["context"] = context
        input["paraphrase"] = paraphrase
        input["object"] = obj
        input["subject"] = sub
        import pdb

        pdb.set_trace()
        # assert obj in input["prompt"]
        # assert subject in input["prompt"]

        return {**input}


# fact_generator = CommonCitiesGenerator(model_name="gpt-4o")
# df = pd.DataFrame(countries, columns=["country"])

# dataset = Dataset.from_pandas(df)
# dataset = fact_generator(dataset)

# dataset.save_to_disk("/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/country_data/common_cities_generation.hf",)
split = "train"
test_instances = list(io.load_jsonlines(f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/{split}.jsonl"))

triplet_extractor = PrefixSplit(
    model_name="gpt-4o",
    backend_params={
        "max_requests_per_minute": 30_000,
        "max_tokens_per_minute": 150_000_000,
        "require_all_responses": True,
    },
)

df = pd.DataFrame([deepcopy(x["edit"]) for x in test_instances])

dataset = Dataset.from_pandas(df)
new_ds = triplet_extractor(
    dataset,
)
new_ds.save_to_disk(
    f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/{split}_w_prefix.hf",
)


print()
