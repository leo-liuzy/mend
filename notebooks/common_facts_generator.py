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

event_categories = [
    "Historic Figures' Birthday",
    "Political Events",
    "Wars and Conflicts",
    "Revolutions and Rebellions",
    "Technological Advancements",
    "Scientific Discoveries",
    "Social Movements",
    "Economic Events",
    "Financial Crises and Depressions",
    "Natural Disasters",
    "Cultural Milestones",
    "Exploration and Discovery",
    "Colonization and Empire Building",
    "Human Rights Developments",
    "Religious Events",
    "Medical and Health Advancements",
    "Pandemics and Epidemics",
    "Environmental Events",
    "Military Developments",
    "Government and Law Changes",
    "Diplomatic Agreements and Treaties",
    "Industrial and Agricultural Revolutions",
    "Communication and Media Innovations",
    "Space Exploration",
    "Sports and Entertainment Events",
    "Transportation and Infrastructure Developments",
    "Education and Knowledge Advancements",
    "Crime and Legal History",
    "Trade and Commerce Expansions",
    "Cultural and Artistic Movements",
    "Migration and Population Shifts"
]

qa_pair_extractor = extractor.tag_content_extractor("qa_pair")
question_extractor = extractor.tag_content_extractor("question")
answer_extractor = extractor.tag_content_extractor("answer")
    
class CommonFactsGenerator(curator.LLM):
    N_QA: int = 50
    PROMPT : str = """
Give me {n_qa} question-answer pairs. The question asks for important historical date (in year) regarding {topic}. The answer is the date (in year). Make the questions diverse.

Return the question wrapped in <question>..</question> tag, and answer wrapped in <answer>..</answer> tag. Each pair should be wrapped in <qa_pair>..</qa_pair> tag.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(topic=input["topic"], n_qa=self.N_QA)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        qa_pairs = qa_pair_extractor(response)
        assert len(qa_pairs) > 0
        qa_pairs = [x.strip() for x in qa_pairs]
        processed_qa_pairs = []
        for qa_pair in qa_pairs:
            quetsion = question_extractor(qa_pair)
            assert len(quetsion) == 1
            question = quetsion[0].strip()
            
            answer = answer_extractor(qa_pair)
            assert len(answer) == 1
            answer = answer[0].strip()
            processed_qa_pairs.append({"question": question, "answer": answer})
            
        input["qa_pairs"] = processed_qa_pairs
        return {**input}
    
class CommonFactsAnswerer(curator.LLM):
    PROMPT : str = """
Give me answer(s) to the question below. 

[Question]
{question}

Return answer wrapped in <answer>..</answer> tag.
If the question is not answerable, please give "<answer>I don't know</answer>" as the answer.
If the question has multiple answers, wrap each answer in <answer>..</answer> tag.
Each answer should be short and concise.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(question=input["question"],)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        answers_ = answer_extractor(response)
        # import pdb; pdb.set_trace()
        assert len(answers_) > 0
        answers = [a.strip() for a in answers_]
            
        input["answers"] = answers
        return {**input}
    


answer_generator = CommonFactsAnswerer(model_name="gpt-4o")

questions = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/popular_ood_questions.jsonl")
df = pd.DataFrame(questions,)
dataset = Dataset.from_pandas(df)
dataset = answer_generator(dataset)

dataset.save_to_disk(f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/popular_ood_questions_answers.hf",)
print()
