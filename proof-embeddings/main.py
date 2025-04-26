import torch
import wandb
import random
import numpy as np
from omegaconf import OmegaConf
import logging
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, BertTokenizer

from src import load_dataset, split_dataset
from src import TheoremDataset, PairTheoremDataset
from src import train_loop, evaluate, evaluate_classifier
from src.dataset import RankingDataset

logger = logging.getLogger(__name__)


def main(cfg_path: str = "config.yaml"):
    cfg = OmegaConf.load(cfg_path)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    log_level_str = cfg.get("log_level", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags
        )
    else:
        wandb.init(mode="disabled")

    data = load_dataset(cfg.dataset_path)
    train_data, val_data, test_data = split_dataset(data, cfg.train_split, cfg.val_split, cfg.test_split)
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)

    train_dataset = PairTheoremDataset(
        train_data, tokenizer, cfg.max_seq_length,
        cfg.threshold_pos, cfg.threshold_neg,
        cfg.samples_from_single_anchor
    )
    # For correlation and recall calc during eval
    val_dataset = PairTheoremDataset(
        val_data, tokenizer, cfg.max_seq_length,
        cfg.threshold_pos, cfg.threshold_neg,
        cfg.samples_from_single_anchor
    )
    # visualize_dataset(train_dataset)
    # For validation loss calc
    # val_dataset_triplet = TheoremDataset(
    #     val_data, tokenizer, cfg.max_seq_length,
    #     cfg.threshold_pos, cfg.threshold_neg,
    #     cfg.samples_from_single_anchor
    # )
    test_dataset = PairTheoremDataset(
        test_data, tokenizer, cfg.max_seq_length,
        cfg.threshold_pos, cfg.threshold_neg,
        cfg.samples_from_single_anchor
    )

    ranking_validation_dataset = RankingDataset(
        cfg.max_seq_length, tokenizer,
        cfg.rankin_ds_path_statements,
        cfg.rankin_ds_path_references
    )

    logger.info(f"Created train, validation, and test datasets with sizes: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_loop(train_dataset, val_dataset, cfg, device, ranking_validation_dataset)
    
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    logger.info("Evaluating on test set of size: %d", len(test_dataset))

    test_results = evaluate_classifier(model, test_loader, device, cfg.threshold_pos, cfg.evaluation.k_values, ranking_validation_dataset, cfg.evaluation.f_score_beta)
    
    wandb.log({
        "events_accuracy": test_results["events_accuracy"],
        "accuracy": test_results["accuracy"],
        "precision": test_results["precision"],
        "recall": test_results["recall"],
        "f1": test_results["f1"]
    })

    wandb.finish()

if __name__ == "__main__":
    main()
