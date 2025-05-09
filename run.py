import copy
import random
import importlib
import logging

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils
from utils import StrEnum

from knowledge_propagation.utils import vars
from trainer import EditTrainer
import models
import transformers


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM) and not isinstance(model, transformers.Qwen2ForCausalLM):
        #     model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight.mean(0)
        # else:
        model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


@hydra.main(config_path="config", config_name="config")
def run(config):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)

    if config.task == "gen" or config.task == "wiki":
        add_padding(tokenizer, model)
        from data_classes.wiki import GenDataset

        train_set = GenDataset("train", tokenizer, config, config.data.path, pct=10)
        val_set = GenDataset("validation", tokenizer, config, config.data.path, pct=10)
    elif config.task == "fc" or config.task == "fever":
        from data_classes.fever import BinaryAugmentedKILT

        train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever-train-kilt.jsonl", config)
        val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever-dev-kilt.jsonl", config)
    elif config.task == "qa" or config.task == "zsre":
        from data_classes.zsre import ZsreDataset

        add_padding(tokenizer, model)

        train_set = ZsreDataset(
            tokenizer,
            f"{base_dir}/data/zsre/structured_zeroshot-train-new_annotated_final.jsonl",
            config,
            size=getattr(config, "train_size", None),
        )
        val_set = ZsreDataset(
            tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config
        )
    elif config.task == "qa" or config.task == "musique":
        from data_classes.musique import MusiqueDataset

        train_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend/2hop_musique_ans_v1.0_train.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend/2hop_musique_ans_v1.0_dev.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
    elif config.task == "qa" or config.task == "musique_dropout":
        add_padding(tokenizer, model)
        from data_classes.musique_dropout import MusiqueDataset

        train_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend/2hop_musique_ans_v1.0_train.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend/2hop_musique_ans_v1.0_dev.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
    elif config.task == "qa" or config.task == "musique_dropout_better":
        add_padding(tokenizer, model)
        from data_classes.musique_dropout_better import MusiqueDataset

        train_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend/2hop_musique_ans_v1.0_train.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend/2hop_musique_ans_v1.0_dev.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )

    elif config.task == "qa" or config.task == "musique_combiner_q":
        add_padding(tokenizer, model)
        from data_classes.musique_combiner_q import MusiqueDataset

        train_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_train.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
    elif config.task == "qa" or config.task == "musique_combiner_text":
        add_padding(tokenizer, model)
        from data_classes.musique_combiner_text import MusiqueDataset

        train_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_train.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
    elif config.task == "qa" or config.task == "musique_propagator_text":
        add_padding(tokenizer, model)
        from data_classes.musique_propagator_text import MusiqueDataset

        class EditInput(StrEnum):
            seen_doc = "seen"
            hidden_doc = "hidden"
            first_single_hop = "first-1hop"
            second_single_hop = "second-1hop"

        if config.edit_input == EditInput.seen_doc:
            suffix = "-seen"
        elif config.edit_input == EditInput.hidden_doc:
            suffix = "-hidden"
        else:
            assert config.edit_input == EditInput.all_doc
            suffix = ""

        train_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_train{suffix}.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev{suffix}.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
    elif config.task == "qa" or config.task == "musique_propagator_text_special":
        add_padding(tokenizer, model)
        from data_classes.musique_propagator_text import MusiqueDataset

        class EditInput(StrEnum):
            seen_doc = "seen"
            hidden_doc = "hidden"
            all_doc = "all"

        if config.edit_input == EditInput.seen_doc:
            suffix = "-seen"
        elif config.edit_input == EditInput.hidden_doc:
            suffix = "-hidden"
        else:
            assert config.edit_input == EditInput.all_doc
            suffix = ""

        train_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_train{suffix}.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev{suffix}.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
    elif config.task == "qa" or config.task == "musique_injector":
        add_padding(tokenizer, model)
        from data_classes.musique_injector import MusiqueDataset

        train_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_train.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = MusiqueDataset(
            tokenizer,
            f"{vars.DATA_DIR}/musique_mend_converted/2hop_musique_ans_v1.0_dev.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
    elif config.task == "qa" or config.task == "bio_syn":
        add_padding(tokenizer, model)
        from data_classes.bio_syn import BioSynDataset

        assert hasattr(config, "train_set_size"), "bio_syn config must be provided"
        train_set = BioSynDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data/train.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = BioSynDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        # import pdb; pdb.set_trace()
    elif config.task == "qa" or config.task == "bio_syn_v2":
        add_padding(tokenizer, model)
        from data_classes.bio_syn_v2 import BioSynDataset

        assert hasattr(config, "train_set_size"), "bio_syn config must be provided"
        train_set = BioSynDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/train.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = BioSynDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/bio_syn_data_v2/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        # import pdb; pdb.set_trace()
    elif config.task == "qa" or config.task == "country_syn":
        add_padding(tokenizer, model)
        from data_classes.bio_syn_v2 import BioSynDataset

        assert hasattr(config, "train_set_size"), "bio_syn config must be provided"
        train_set = BioSynDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/country_syn_data/train.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = BioSynDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/country_syn_data/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "country_syn_v2":
        add_padding(tokenizer, model)
        from data_classes.bio_syn_v2 import BioSynDataset

        assert hasattr(config, "train_set_size"), "bio_syn config must be provided"
        sub_data_dir = f"n_template_{config.n_template}_n_seen_pairs_{config.n_seen_pair}"
        config.dataset += f"-{config.train_set_size}-({config.n_template}, {config.n_seen_pair})"
        train_set = BioSynDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/country_syn_data_v2/{sub_data_dir}/train.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = BioSynDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/country_syn_data_v2/{sub_data_dir}/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits import RippleEditsDataset

        assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/train.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/train.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/valid.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits_all":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits import RippleEditsDataset

        assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/train.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train/all/train.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/valid.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train/all/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits_recent_popular":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits import RippleEditsDataset

        assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/train.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train/recent+popular/train.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/valid.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train/all/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits_w_aug":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits import RippleEditsDataset

        assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/train.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/train_w_random.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/valid.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits_w_ekp":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits_w_ekp import RippleEditsPlusEKPDataset
        from data_classes.ripple_edits import RippleEditsDataset

        assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsPlusEKPDataset(
            tokenizer,
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/train_w_ekp.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/valid.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits_w_recoe":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits_w_recoe import RippleEditsPlusReCoEDataset
        from data_classes.ripple_edits import RippleEditsDataset

        assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsPlusReCoEDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/train.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/train_w_recoe.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/valid.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits_mend":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits_mend import RippleEditsMENDDataset

        # assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsMENDDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/train_mend.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/train_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsMENDDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/valid_mend.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train_recent+popular/valid_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits_all_mend":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits_mend import RippleEditsMENDDataset

        # assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsMENDDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/train_mend.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train/all/train_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsMENDDataset(
            tokenizer,
            # f"{vars.DATA_DIR}/ripple_edits/meta_train_recent/valid_mend.jsonl",
            f"{vars.DATA_DIR}/ripple_edits/meta_train/all/valid_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "drop":
        add_padding(tokenizer, model)
        from data_classes.drop import DropDataset

        # assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = DropDataset(
            tokenizer,
            f"{vars.DATA_DIR}/drop_dataset_converted/drop_dataset_train.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = DropDataset(
            tokenizer,
            f"{vars.DATA_DIR}/drop_dataset_converted/drop_dataset_dev.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "syn_story":
        add_padding(tokenizer, model)
        from data_classes.syn_story import SynStoryDataset

        assert hasattr(config, "train_set_size"), "bio_syn config must be provided"
        config.dataset += f"-{config.train_prefix}train"
        train_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/{config.train_prefix}train_data_100percent_frozen/train_text_data_id_entity152_rel31.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/{config.train_prefix}train_data_100percent_frozen/valid_text_data_id_entity152_rel31.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        LOG.info(f"model_max_length: {tokenizer.model_max_length}")
    elif config.task == "qa" or config.task == "syn_story_mend":
        add_padding(tokenizer, model)
        from data_classes.syn_story_mend import SynStoryMENDDataset

        assert hasattr(config, "train_set_size"), "bio_syn config must be provided"
        config.dataset += f"-{config.train_prefix}train"
        train_set = SynStoryMENDDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/{config.train_prefix}train_data_100percent_frozen/train_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = SynStoryMENDDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/{config.train_prefix}train_data_100percent_frozen/valid_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        LOG.info(f"model_max_length: {tokenizer.model_max_length}")
    elif config.task == "qa" or config.task == "syn_story_ablate_paraphrase":
        add_padding(tokenizer, model)
        from data_classes.syn_story import SynStoryDataset

        assert hasattr(config, "train_set_size"), "bio_syn config must be provided"
        config.dataset += f"-{config.train_prefix}train"
        train_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/{config.train_prefix}train_data_100percent_frozen/train_text_data_id_entity152_rel31_paraphrase-only.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/{config.train_prefix}train_data_100percent_frozen/valid_text_data_id_entity152_rel31_paraphrase-only.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        LOG.info(f"model_max_length: {tokenizer.model_max_length}")
    elif config.task == "qa" or config.task == "syn_story_ablate_cpt":
        add_padding(tokenizer, model)
        from data_classes.syn_story_sft import SynStorySFTDataset

        assert hasattr(config, "train_set_size"), "bio_syn config must be provided"
        config.dataset += f"-{config.train_prefix}train"
        train_set = SynStorySFTDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/{config.train_prefix}train_data_100percent_frozen/train_structure_data_id_entity152_rel31.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = SynStorySFTDataset(
            tokenizer,
            f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/{config.train_prefix}train_data_100percent_frozen/valid_structure_data_id_entity152_rel31.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        LOG.info(f"model_max_length: {tokenizer.model_max_length}")

    else:
        raise ValueError(f"Unrecognized task {config.task}")
    # train_set[0]
    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))

    if config.alg == "ft" and config.ft.locality.enabled:
        if config.ft.locality.oracle:
            alg.loc_sampler = train_set.edit_generator(config.ft.locality.batch_size + 1)
        else:
            state = np.random.get_state()
            np.random.seed(0)
            loc_batch = next(train_set.edit_generator(config.ft.locality.batch_size + 1))["loc"]
            np.random.set_state(state)
            alg.loc_ids = loc_batch["input_ids"]
            alg.loc_masks = loc_batch["attention_mask"]

    trainer = EditTrainer(alg, config, train_set, val_set)
    trainer.run()


if __name__ == "__main__":
    run()
