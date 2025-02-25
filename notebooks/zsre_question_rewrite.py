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


def resolved_answer_references(singleton_questions: List[Dict]):
    """
    The single-hop questions per MuSiQue instance contains reference to answers in other questions. This function replace the reference with actual value.
    """
    pattern = r"#(\d+)"

    resolved_singleton_questions = [None] * len(singleton_questions)
    for q_i, q in enumerate(singleton_questions):
        resolved_q = deepcopy(q)
        match = re.findall(pattern, q["question"])
        # replace every answer reference with the actual value
        resolved_question = q["question"]
        for ans_i in match:
            try:
                assert int(ans_i) - 1 >= 0
                resolved_question = resolved_question.replace(
                    f"#{ans_i}", singleton_questions[int(ans_i) - 1]["answer"].strip()
                )
            except Exception:
                continue

        resolved_q["question"] = resolved_question
        resolved_singleton_questions[q_i] = resolved_q
    assert not any(q is None for q in resolved_singleton_questions)
    return resolved_singleton_questions

split = "train"
dataset_unresolved = io.load_jsonlines(f"/data/users/zliu/KE-by-CP/data/musique_mend/2hop_musique_ans_v1.0_{split}.jsonl")

dataset_reference_resolved = []
for datum in dataset_unresolved:
    new_datum = deepcopy(datum)
    new_datum["question_decomposition"] = resolved_answer_references(datum["question_decomposition"])
    dataset_reference_resolved.append(new_datum)
    
zsre_question_list = []

for datum in dataset_reference_resolved:
    for decomp_q in datum["question_decomposition"]:
        if ">>" in decomp_q["question"]:
            assert " >> " in decomp_q["question"]
            new_decomp_q = deepcopy(decomp_q)
            new_decomp_q["id"] = datum["id"] + "::" + str(decomp_q["id"])
            new_decomp_q["text"] = datum["texts"][new_decomp_q['supporting_text_id']]
            zsre_question_list.append(new_decomp_q)
    
    
class zsREQuestioner(curator.LLM):
    PROMPT : str = """
<text>
Green is the fourth studio album by British progressive rock musician Steve Hillage. Written in spring 1977 at the same time as his previous album, the funk-inflected "Motivation Radio" (1977), "Green" was originally going to be released as "The Green Album" as a companion to "The Red Album" (the originally intended name for "Motivation Radio"). However, this plan was dropped and after a US tour in late 1977, "Green" was recorded alone, primarily in Dorking, Surrey, and in London.
</text>

<triplet>
Green >> performer >> Steve Hillage
</triplet>

<question>
Who is the performer of Green?
</question>

<text>
Starting from this edition, the UEFA Europa League winners automatically qualify for the subsequent UEFA Champions League season even if they do not qualify for the Champions League through their domestic performance. Therefore, the winners of this tournament qualify for the 2015–16 UEFA Champions League. They are guaranteed to enter at least the play-off round, and since the group stage berth reserved for the Champions League title holders will not be used (the winners of the 2014–15 UEFA Champions League are guaranteed to qualify for the group stage through domestic performance), they will be elevated to enter the group stage via this berth.
</text>

<triplet>
Gibraltar Football Association >> member of >> UEFA
</triplet>

<question>
What organization is the Gibraltar Football Association a member of?
</question>

You receive an knowledge triplet of form
"[Subject] >> [Relation] >> [Object]", wrapped in <triplet>...</triplet>. And text wrapped in <text>...</text> serves as a context for understanding.

Your task is to turn the input to be a question about [Object], wrapped in <question>...</question>. 

Here are some detailed instructions:
* Avoid Yes/No question.
* Include [Relation] to make the question as clear as possible.
* Do not include [Object] in the question.
* Avoid question like "What is the relationship between [Subject] and [Object]?"

<text>
{text}
</text>

<triplet>
{question} >> {answer}
</triplet>
"""
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(text=input["text"], question=input["question"], answer=input["answer"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        
        return {**input, "nl_question": response}
    
dedup_zsre_question_list = []
id2zsre_question = {}
# atom_q_ids = set([x["id"] for x in zsre_question_list])
print("# zsre question (before dedup):", len(zsre_question_list))
for q in zsre_question_list:
    if q["id"] not in id2zsre_question:
        id2zsre_question[q["id"]] = q
        dedup_zsre_question_list.append(q)
    else:
        assert q["question"] == id2zsre_question[q["id"]]["question"]
print("# zsre question (after dedup):", len(dedup_zsre_question_list))

zsre_questioner = zsREQuestioner(model_name="gpt-4o")
zsre_question_df = pd.DataFrame(dedup_zsre_question_list)
zsre_question_dataset = Dataset.from_pandas(zsre_question_df)
nl_zsre_question_dataset = zsre_questioner(zsre_question_dataset)
nl_zsre_question_dataset.to_json(f"/data/users/zliu/KE-by-CP/data/musique_mend/2hop_musique_ans_v1.0_{split}_zsre-questions.jsonl")
print()