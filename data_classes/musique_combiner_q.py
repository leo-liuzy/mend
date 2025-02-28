import jsonlines
from torch.utils.data import Dataset
import random
from utils import EditBatchSampler, dict_to
import torch
from transformers import BartTokenizerFast, BartTokenizer
import logging
import typing
import json
from knowledge_propagation.utils import io
from utils import StrEnum
import numpy as np


LOG = logging.getLogger(__name__)

class EditInput(StrEnum):
    two_single_hop = "two-1hop"
    first_single_hop = "first-1hop"
    second_single_hop = "second-1hop"
    

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
        if not self.config.two_doc_at_same_time:
            return len(self.data) * self.n_doc_per_instance
        else:
            return len(self.data)

    def __getitem__(self, item, seed=None):
        assert len(self.data[item]["multi_hop_efficacy"]) == 1
        propagation_question = self.data[item]["multi_hop_efficacy"][0]
        if self.config.edit_input == EditInput.first_single_hop:
            qas = [self.data[item]["single_hop_efficacy"][0]]
        elif self.config.edit_input == EditInput.second_single_hop:
            qas = [self.data[item]["single_hop_efficacy"][1]]
        else:
            assert self.config.edit_input == EditInput.two_single_hop
            qas = self.data[item]["single_hop_efficacy"]
        
        np.random.shuffle(qas) # ! this is to avoid model exploiting heuristics
        
        single_hop_answers = [("" if len(qa["answer"]) != 0 and qa["answer"][0] == " " else " ") + qa["answer"] for qa in qas]
        
        single_hop_questions = [qa["question"] for qa in qas]
        single_hop_questions = [q_ + ans_ for q_, ans_ in zip(single_hop_questions, single_hop_answers)]
        
        output = {
            # "src": texts,
            "single_hop_questions": single_hop_questions,
            "single_hop_answers": single_hop_answers,
            
            "propagation_question": propagation_question["question"],
            "propagation_answer": propagation_question["answer"],
        }
        return output

    def collate_fn(self, batch):
        single_hop_questions = [s for b in batch for s in b["single_hop_questions"]]
        single_hop_answers = [s for b in batch for s in b["single_hop_answers"]]
        
        """ 
        ! original line
        trg = (
            [b["answers"][0] for b in batch[:-ne]] +
            [b["alt"] for b in batch[-ne:]]
        )
        """
        answers = [b["propagation_answer"] for b in batch]
        questions = [b["propagation_question"] for b in batch]
        
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
                                self.tok.eos_token_id # this is to teach the model to end after outputing the answer.
                            )
                        )
                    ], dim=-1)
            for k1, v1 in {
                "single_hop_questions": single_hop_questions,
                "single_hop_answers": single_hop_answers,
                "propagation_question": questions,
                "propagation_answer": answers,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                add_special_tokens="answer" not in k1, # make the SFT label free of BOS
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
            edit_inner["input_ids"] = toks["single_hop_questions_input_ids"]
            edit_inner["attention_mask"] = toks["single_hop_questions_attention_mask"]
            edit_inner["labels"] = self.get_edit_labels(toks["single_hop_answers_input_ids"])
                
            assert edit_inner["labels"].size(1) <= edit_inner["input_ids"].size(1)

            if self.config.data.rephrase:
                # in this case, rephrase means using propogation questions for L_e
                edit_outer = {}
                edit_outer["input_ids"] = toks["propagation_question_input_ids"]
                edit_outer["attention_mask"] = toks["propagation_question_attention_mask"]
                edit_outer["labels"] = self.get_edit_labels(toks["propagation_answer_input_ids"])
                if self.config.data.musique_propagation_only:
                    edit_inner = edit_outer
            else:
                edit_outer = edit_inner
                
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
            
            if not self.show_first_example:
                LOG.info("Edit_inner:")
                LOG.info("Input: " +  "\n@@\n".join(self.tok.batch_decode(edit_inner["input_ids"])))
                LOG.info("Label:" +  "\n@@\n".join(self.tok.batch_decode(torch.where(edit_inner["labels"] == -100, self.tok.pad_token_id, edit_inner["labels"]))))
                
                LOG.info("Edit_outer:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(edit_outer["input_ids"])))
                LOG.info("Label: " +  "\n@@\n".join(self.tok.batch_decode(torch.where(edit_outer["labels"] == -100, self.tok.pad_token_id, edit_outer["labels"]))))
                
                LOG.info("loc:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(loc["input_ids"])))
                LOG.info("Label: " +  "\n@@\n".join(self.tok.batch_decode(torch.where(loc["labels"] == -100, self.tok.pad_token_id, loc["labels"]))))
                
                self.show_first_example = True

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": None,
                "raw": toks["raw"]
            }

            yield dict_to(batch, self.config.device)
