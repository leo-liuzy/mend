import jsonlines
from torch.utils.data import Dataset
import random
from utils import EditBatchSampler, dict_to
import torch
from transformers import BartTokenizerFast, BartTokenizer
import logging
import typing
import json
from knowledge_propagation.utils import io, vars
from utils import EditLoss, EditInput
import numpy as np


LOG = logging.getLogger(__name__)


class MusiqueDataset(Dataset):
    """
    ! Leo: adding support for running zsre with Decoder-only model

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        tokenizer,
        data_path,
        config,
        size: typing.Optional[int] = None,
        max_length=32,
    ):
        super().__init__()
        self.tok = tokenizer
        self.data = io.load_jsonlines(data_path)
        self.config = config
        
        self.n_example_per_instance = 2 # !! we are using single hop question independently
        
        # if self.config.add_icl:
        #     self.eos_token_id = self.tok("\n", add_special_tokens=False)["input_ids"][0]
        # else:
        self.eos_token_id = self.tok.eos_token_id
        
        self.icl_prompt = "\n".join([
            "Q: When did the simpsons first air on television?",
            "A: December 17, 1989",
            "Q: Who has more super bowl wins afc or nfc?",
            "A: NFC",
            "Q: Is the federal court the same as the supreme court?",
            "A: No"
        ])
        
        self.show_first_example = False
        assert self.config.data.rephrase, "propogation question must be used."
        self.max_length = max_length
        if self.config.data.zsre_nq: # ! Leo: original if-condition: `and "train" not in data_path`
            self.use_nq = True
            LOG.info("** Using natural questions for zsre base samples **")
            from data_classes.nq import NQDataset
            self.nq = NQDataset(self.config.data.nq_path + ("/train.json" if "train" in data_path else "/validation.json"),tokenizer, config)
        else:
            self.use_nq = False

    def __len__(self):
        return len(self.data) * self.n_example_per_instance

    def __getitem__(self, item, seed=None):
        
            
        data_item = item // self.n_example_per_instance
        doc_idx = item % self.n_example_per_instance
        assert len(self.data[data_item]["single_hop_efficacy"]) == self.n_example_per_instance
        assert len(self.data[data_item]["texts"]) == self.n_example_per_instance
        
        injection_question = self.data[data_item]["single_hop_efficacy"][doc_idx]
        
        if self.config.add_icl:
            output = {
                "src": [self.data[data_item]["texts"][injection_question['supporting_text_id']]],
                "question": self.icl_prompt + "\nQ: " + injection_question["question"] + "\nA:",
                "answer": injection_question["answer"],
            }
        else:
            output = {
                "src": [self.data[data_item]["texts"][injection_question['supporting_text_id']]],
                "question": injection_question["question"],
                "answer": injection_question["answer"],
            }
        
        return output

    def collate_fn(self, batch):
        src = [s for b in batch for s in b["src"]]
        
        """ 
        ! original line
        trg = (
            [b["answers"][0] for b in batch[:-ne]] +
            [b["alt"] for b in batch[-ne:]]
        )
        """
        answers = [b["answer"] for b in batch]
        questions = [b["question"] for b in batch]
        
        answers = [("" if len(ans_) != 0 and ans_[0] == " " else " ") + ans_ for ans_ in answers]
        questions = [q_ + ans_ for q_, ans_ in zip(questions, answers)]
        

        batches = {
            f"{k1}_{k2}": 
                torch.concat(
                    [
                        v2, 
                        torch.full(
                            (v2.shape[0], 1), # shape of the constant tensor
                            (
                                1 
                                if k2 == "attention_mask" else
                                self.eos_token_id # this is to teach the model to end after outputing the answer.
                            )
                        )
                    ], dim=-1)
            for k1, v1 in {
                "src": src,
                "question": questions,
                "answer": answers,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                add_special_tokens="answer" not in k1,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch
        return batches

    def _check_padding(self, ids):
        if (ids[:, 0] == self.tok.pad_token_id).any():
            raise ValueError("Left-padding not supported")

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(n, memorize_mode=self.config.single_batch, loc_disjoint=not self.use_nq, seed=self.config.seed)

        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)
            assert len(edit_idxs) == 1
            # idxs = loc_idxs + edit_idxs
            toks = self.collate_fn([self[idx] for idx in edit_idxs])

            # ne = self.config.data.n_edits
            edit_inner = {}
            
            edit_inner["input_ids"] = toks["src_input_ids"]
            edit_inner["attention_mask"] = toks["src_attention_mask"]
            edit_inner["labels"] = self.get_edit_labels(toks["src_input_ids"])
            
            assert edit_inner["labels"].size(1) <= edit_inner["input_ids"].size(1)

            if self.config.data.rephrase:
                # in this case, rephrase means using propogation questions for L_e
                edit_outer = {}
                edit_outer["input_ids"] = toks["question_input_ids"]
                edit_outer["attention_mask"] = toks["question_attention_mask"]
                edit_outer["labels"] = self.get_edit_labels(toks["answer_input_ids"])
                if self.config.data.musique_propagation_only:
                    edit_inner = edit_outer
            else:
                edit_outer = edit_inner
            if not self.show_first_example:
                print("Edit_inner:")
                print("Input:", self.tok.batch_decode(edit_inner["input_ids"]))
                print("Label:", self.tok.batch_decode(torch.where(edit_inner["labels"] == -100, self.tok.pad_token_id, edit_inner["labels"])))
                
                print("Edit_outer:")
                print("Input:", self.tok.batch_decode(edit_outer["input_ids"]))
                print("Label:", self.tok.batch_decode(torch.where(edit_outer["labels"] == -100, self.tok.pad_token_id, edit_outer["labels"])))
                self.show_first_example = True
            loc = {}
            if self.use_nq:
                batch = [self.nq[idx] for idx in loc_idxs]
                questions = [b[0] for b in batch]
                answers = [b[1] for b in batch]
                answers = [("" if answer[0] == " " else " ") + answer for answer in answers]
                questions = [q + a for (q, a) in zip(questions, answers) ]
                
                loc = dict(self.tok(questions, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True))
                trg_toks = dict(self.tok(answers, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True, add_special_tokens=False))
                loc["labels"] = self.get_edit_labels(trg_toks["input_ids"])
            else:
                loc = edit_inner

            # cond = {k[5:]: v for k, v in toks.items() if k.startswith("cond")}

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": None,
                "raw": toks["raw"]
            }

            yield dict_to(batch, self.config.device)
