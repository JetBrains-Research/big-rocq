import torch
import wandb
import random
import numpy as np
from omegaconf import OmegaConf
import logging
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src import load_dataset, split_dataset
from src import TheoremDataset
from src import train_loop, evaluate

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

    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)

    train_dataset = TheoremDataset(
        train_data, tokenizer, cfg.max_seq_length, 
        cfg.threshold_pos, cfg.threshold_neg
    )

    # train_dataset.preview_dataset(10)

    val_dataset = TheoremDataset(
        val_data, tokenizer, cfg.max_seq_length, 
        cfg.threshold_pos, cfg.threshold_neg
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_loop(train_dataset, val_dataset, cfg, device)
    
    model.to(device)
    test_dataset = TheoremDataset(
        test_data, tokenizer, cfg.max_seq_length, 
        cfg.threshold_pos, cfg.threshold_neg
    )

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_results = evaluate(model, test_loader, device, cfg.threshold_pos)
    
    wandb.log({
        "test_pearson": test_results["pearson"],
        "test_spearman": test_results["spearman"]
    })

    wandb.finish()

if __name__ == "__main__":
    main()
