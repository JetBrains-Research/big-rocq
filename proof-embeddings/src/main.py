import torch
import wandb
import random
import numpy as np
from omegaconf import OmegaConf
from data import load_json_dataset, split_dataset
from dataset import TheoremDataset
from train import evaluate, train_loop

def main(cfg_path: str = "config.yaml"):
    cfg = OmegaConf.load(cfg_path)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

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

    data = load_json_dataset(cfg.dataset_path)
    train_data, val_data, test_data = split_dataset(data, cfg.train_split, cfg.val_split, cfg.test_split)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)

    train_dataset = TheoremDataset(
        train_data, tokenizer, cfg.max_seq_length, 
        cfg.threshold_pos, cfg.threshold_neg, 
        mode="train"
    )
    val_dataset = TheoremDataset(
        val_data, tokenizer, cfg.max_seq_length, 
        cfg.threshold_pos, cfg.threshold_neg, 
        mode="train"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_loop(train_dataset, val_dataset, cfg, device)
    
    model.to(device)
    test_dataset = TheoremDataset(
        test_data, tokenizer, cfg.max_seq_length, 
        cfg.threshold_pos, cfg.threshold_neg, 
        mode="test"
    )

    test_loader = TheoremDataset(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_results = evaluate(model, test_loader, device)
    
    wandb.log({
        "test_pearson": test_results["pearson"],
        "test_spearman": test_results["spearman"]
    })

    wandb.finish()

if __name__ == "__main__":
    main()
