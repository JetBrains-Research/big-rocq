import torch
import wandb
import random
import numpy as np
from omegaconf import OmegaConf
import logging
from transformers import RobertaTokenizer, BertTokenizer

from src import load_dataset, split_dataset
from src import TrainingDataset, ValidationDataset
from src import train_loop
from src.dataset import RankingDataset, visualize_dataset

logger = logging.getLogger(__name__)


def main(cfg_path: str = "config.yaml"):
    cfg = OmegaConf.load(cfg_path)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    log_level_str = cfg.get("log_level", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
        )
    else:
        wandb.init(mode="disabled")

    data = load_dataset(cfg.dataset.dataset_path, 1)
    train_data, val_data, _ = split_dataset(
        data,
        cfg.dataset.train_split,
        cfg.dataset.val_split,
        cfg.dataset.test_split,
        remove=[cfg.dataset.rankin_ds_path_statements],
    )
    tokenizer = RobertaTokenizer.from_pretrained(cfg.base_model_name)

    train_dataset = TrainingDataset(
        train_data,
        tokenizer,
        cfg.max_seq_length,
        cfg.threshold_pos,
        cfg.threshold_neg,
        cfg.threshold_hard_neg,
        cfg.dataset.samples_from_single_anchor,
        cfg.k_negatives,
    )

    # visualize_dataset(train_dataset)
    #
    # raise NotImplementedError

    val_dataset = TrainingDataset(
        val_data,
        tokenizer,
        cfg.max_seq_length,
        cfg.threshold_pos,
        cfg.threshold_neg,
        cfg.threshold_hard_neg,
        cfg.dataset.samples_from_single_anchor,
        cfg.k_negatives,
    )

    val_dataset_ranking = ValidationDataset(
        val_data,
        tokenizer,
        cfg.max_seq_length,
        cfg.threshold_pos,
        cfg.threshold_neg,
        cfg.threshold_hard_neg,
        cfg.dataset.samples_from_single_anchor,
        cfg.k_negatives,
        cfg.evaluation.query_size_in_eval,
    )

    extra_imm_validation = RankingDataset(
        cfg.max_seq_length,
        tokenizer,
        cfg.dataset.rankin_ds_path_statements,
        cfg.dataset.rankin_ds_path_references,
    )

    logger.info(
        f"Created train, validation with sizes: {len(train_dataset)}, {len(val_dataset)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_loop(
        train_dataset,
        val_dataset,
        cfg,
        device,
        val_dataset_ranking,
        extra_imm_validation,
        tokenizer,
    )

    model.to(device)

    wandb.finish()


if __name__ == "__main__":
    main()
