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


entity_extractor = extractor.tag_content_extractor("entity")
answer_extractor = extractor.tag_content_extractor("answer")

class EntityGenerator(curator.LLM):
    PROMPT : str = """
Generate a long list of well-known entities for {entity_type}.

Only return each entity wrapped in <entity>..</entity> tag.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(entity_type=input["entity_type"],)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        entities_ = entity_extractor(response)
        assert len(entities_) > 1
        input["entities"] = entities_
        # import pdb; pdb.set_trace()
        return {**input}

class QuestionAnswerer(curator.LLM):
    PROMPT : str = """
Answer the following question. Make the answer concise and direct. The answer should be informative and within a few words.
[Question]
{question}

Only return answer wrapped in <answer>..</answer> tag.
If you don't know the answer or are not sure about the answer, return "I don't know" wrapped in <answer>..</answer> tag.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(question=input["question"],)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        answer_ = answer_extractor(response)
        assert len(answer_) == 1
        answer = answer_[0].strip()
        input["answer"] = answer
        return {**input}
    

class AnswerShortener(curator.LLM):
    PROMPT : str = """
Answer to the following question is too long (e.g., sentence). Make the answer concise and direct. The answer should be a single sentence or a few words. The answer should be as short as possible while still being equally informative.

[Question]
{question}
[Answer]
{answer}

Only return shortened answer wrapped in <answer>..</answer> tag.
If the answer is already concise, return the original answer wrapped in <answer>..</answer> tag.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(question=input["question"], answer=input["answer"],)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        answer_ = answer_extractor(response)
        assert len(answer_) == 1
        answer = answer_[0].strip()
        # import pdb; pdb.set_trace()
        assert 0 < len(answer) <= len(input["answer"])
        input["shortened_answer"] = answer
        return {**input}

# entity_generator = EntityGenerator(model_name="gpt-4o")
question_answerer = QuestionAnswerer(model_name="gpt-4.1", backend_params={
        "max_requests_per_minute": 30_000,
        "max_tokens_per_minute": 150_000_000,
        "require_all_responses": True,
},)
answer_shortener = AnswerShortener(model_name="gpt-4.1", backend_params={
        "max_requests_per_minute": 30_000,
        "max_tokens_per_minute": 150_000_000,
        "require_all_responses": True,
})
n_trial_per_entity_type = 10
version=2

# entity2templates = io.load_json(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/data_gen/entity2templates_v{version}.json")
# entity_types = list(entity2templates.keys())
# # import pdb; pdb.set_trace()
# df = pd.DataFrame([x for et in entity_types for x in [{"entity_type": et}] * n_trial_per_entity_type])
# dataset = Dataset.from_pandas(df)
# dataset_generated = entity_generator(dataset)
# dataset_generated.save_to_disk(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/data_gen/entities_v{version}.hf",)

# question_df = pd.read_csv(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/data_gen/entity_type_name_template_curated_v1.csv")
# dataset = Dataset.from_pandas(question_df)
# # import pdb; pdb.set_trace()
# dataset_answered = question_answerer(dataset)
# dataset_answered.save_to_disk(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/data_gen/entity_type_name_template_curated_v1_answered.hf",)

long_answer_df = pd.read_csv(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/data_gen/entity_type_name_template_curated_v1_answered_long.csv")
dataset = Dataset.from_pandas(long_answer_df)
dataset_shortened = answer_shortener(dataset)
dataset_shortened.save_to_disk(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/data_gen/entity_type_name_template_curated_v1_answered_long_shortened.hf",)
