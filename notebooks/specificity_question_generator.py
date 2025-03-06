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

output_tag_extractor = extractor.tag_content_extractor("output")
onehop_tag_extractor = extractor.tag_content_extractor("1hop")
twohop_tag_extractor = extractor.tag_content_extractor("2hop")
q_tag_extractor = extractor.tag_content_extractor("q")
a_tag_extractor = extractor.tag_content_extractor("a")

N_SPEC_QUESTIONS = 2

class SpecificityQuestionWriter(curator.LLM):
    PROMPT : str = """
You will be given single-hop questions (each wrapped in <1hop>...</1hop> tags) and the corresponding multi-hop questions (wrapped in <2hop>...</2hop> tags) about context (in <context>...</context> tags). 

Your task is to generate another {n_spec_questions} set of questions that mimic the style of the given questions.

Rules:
1. The generated pairs are thematically and semantically similar to the given questions.
2. However, the generated questions' answers are NOT affected by the given context.
3. The generated questions should be about a **well-known** facts/event/etc.


<context>
{context}
</context>

<1hop>
<q>{single_hop_q_1}</q>
<a>{single_hop_a_1}</a>
</1hop>

<1hop>
<q>{single_hop_q_2}</q>
<a>{single_hop_a_2}</a>
</1hop>

<2hop>
<q>{multi_hop_q}</q>
<a>{multi_hop_a}</a>
</2hop>

Wrap each set of single-hop question and multi-hop question in <output>...</output>. You should have {n_spec_questions} <output>...</output> tags in total. In each <output> tag, include 2 single-hop questions (two <1hop>...</1hop> tags) and 1 multi-hop questions (<2hop>...</2hop>).
    """.strip()
    n_spec_questions: int = N_SPEC_QUESTIONS
    
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(**input, n_spec_questions=self.n_spec_questions)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        question_sets = [x.strip() for x in output_tag_extractor(response)]
        assert len(question_sets) == self.n_spec_questions
        
        processed_questions = {}
        for q_i, q_set in enumerate(question_sets):
            onehops = [x.strip() for x in onehop_tag_extractor(q_set)]
            assert len(onehops) == 2
            for onehop_i, onehop in enumerate(onehops):
                q = [x.strip() for x in q_tag_extractor(onehop)]
                a = [x.strip() for x in a_tag_extractor(onehop)]
                assert len(q) == 1 and len(a) == 1
                q, a = q[0], a[0]
                processed_questions[f"spec_single_hop_q_{onehop_i}-{q_i}"] = q
                processed_questions[f"spec_single_hop_a_{onehop_i}-{q_i}"] = a
                
            # process multi-hop specific questions
            twohop = [x.strip() for x in twohop_tag_extractor(q_set)]
            assert len(twohop) == 1
            twohop = twohop[0]
            q = [x.strip() for x in q_tag_extractor(twohop)]
            a = [x.strip() for x in a_tag_extractor(twohop)]
            assert len(q) == 1 and len(a) == 1
            q, a = q[0], a[0]
            processed_questions[f"spec_multi_hop_q-{q_i}"] = q
            processed_questions[f"spec_multi_hop_a-{q_i}"] = a
        assert all(k not in input for k in processed_questions.keys())
        return {**input, **processed_questions}

question_writer = SpecificityQuestionWriter(model_name="gpt-4o")
fpath = f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev.jsonl"
raw_data = io.load_jsonlines(fpath)

# scored_df = pd.read_excel(fpath)
# scored_df["predicted_answer"] = scored_df["predicted_answer"].astype(str)

df_content = []
for datum in tqdm(raw_data):
    context = "\n\n".join(datum["texts"])
    assert len(datum["single_hop_efficacy"]) == 2
    single_hop_q_1 = datum["single_hop_efficacy"][0]["question"].strip()
    single_hop_a_1 = datum["single_hop_efficacy"][0]["answer"].strip()
    single_hop_q_2 = datum["single_hop_efficacy"][1]["question"].strip()
    single_hop_a_2 = datum["single_hop_efficacy"][1]["answer"].strip()
    
    assert len(datum["multi_hop_efficacy"]) == 1
    multi_hop_q = datum["multi_hop_efficacy"][0]["question"].strip()
    multi_hop_a = datum["multi_hop_efficacy"][0]["answer"].strip()

    df_content.append({
        "id": datum["id"],
        "context": context,
        "single_hop_q_1": single_hop_q_1,
        "single_hop_a_1": single_hop_a_1,
        "single_hop_q_2": single_hop_q_2,
        "single_hop_a_2": single_hop_a_2,
        "multi_hop_q": multi_hop_q,
        "multi_hop_a": multi_hop_a
    })
df = pd.DataFrame(df_content[:])

dataset = Dataset.from_pandas(df)
spec_dataset = question_writer(dataset)
spec_df = spec_dataset.to_pandas()

for i, spec_row in spec_df.iterrows():
    assert spec_row["id"] == raw_data[i]["id"]
    multi_hop_specificity = []
    single_hop_specificity = []
    
    for q_i in range(N_SPEC_QUESTIONS):
        multi_hop_specificity.append({
            "question": spec_row[f"spec_multi_hop_q-{q_i}"],
            "answer": spec_row[f"spec_multi_hop_a-{q_i}"]
        })
        for onehop_i in range(2):
            single_hop_specificity.append({
                "question": spec_row[f"spec_single_hop_q_{onehop_i}-{q_i}"],
                "answer": spec_row[f"spec_single_hop_a_{onehop_i}-{q_i}"]
            })
    raw_data[i]["multi_hop_specificity"] = multi_hop_specificity
    raw_data[i]["single_hop_specificity"] = single_hop_specificity
    
# scored_dataset.to_pandas().to_excel(fpath, index=False)
io.dump_jsonlines(raw_data, io.remove_last_extension(fpath) + "_w-spec.jsonl")
print()
