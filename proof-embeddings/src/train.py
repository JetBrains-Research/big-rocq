import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
import itertools
import logging
from typing import Optional

from .losses import TripletMarginLoss
from .metrics import compute_correlation_scores, recall_at_k, precision_at_k, f_scores
from .model import BERTStatementEmbedder

logger = logging.getLogger(__name__)


def train_one_epoch(
    model,
    loss_fn,
    dataloader,
    optimizer,
    device,
    cfg,
    val_loader,
    eval_freq: Optional[int] = None
):
    model.train()
    total_loss = 0.0
    batch_index = 0
    for batch in tqdm(dataloader, desc="Training Iter", leave=False):
        optimizer.zero_grad()

        input_ids_anchor = batch["input_ids_anchor"].to(device)
        attn_anchor = batch["attention_mask_anchor"].to(device)

        input_ids_pos = batch["input_ids_pos_sample"].to(device)
        attn_pos = batch["attention_mask_pos_sample"].to(device)

        input_ids_neg = batch["input_ids_neg_sample"].to(device)
        attn_neg = batch["attention_mask_neg_sample"].to(device)

        emb_anchor = model(input_ids_anchor, attn_anchor)
        emb_pos = model(input_ids_pos, attn_pos)
        emb_neg = model(input_ids_neg, attn_neg)

        loss = loss_fn(emb_anchor, emb_pos, emb_neg)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if eval_freq is not None and batch_index % eval_freq == 0:
            metrics_val = evaluate(model, val_loader, device, cfg.threshold_pos, cfg.evaluation.k_values, cfg.evaluation.f_score_beta)
            wandb.log({
                "val_pearson": metrics_val["pearson"],
                "val_spearman": metrics_val["spearman"]
            })

        batch_index += 1

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    pos_threshold,
    k_values,
    f_score_beta=1.0
):
    model.eval()

    dist_preds = []
    dist_true = []

    anchor2pred = {}
    anchor2truth = {}

    for batch in tqdm(dataloader, desc="Eval Iter", leave=False):
        input_ids_i = batch["input_ids_i"].to(device)
        attn_i = batch["attention_mask_i"].to(device)
        emb_i = model(input_ids_i, attn_i)

        input_ids_j = batch["input_ids_j"].to(device)
        attn_j = batch["attention_mask_j"].to(device)
        emb_j = model(input_ids_j, attn_j)

        dist_golden = batch["dist"].float()
        dist_pred = torch.norm(emb_i - emb_j, p=2, dim=1)

        dist_preds.extend(dist_pred.cpu().numpy().tolist())
        dist_true.extend(dist_golden.cpu().numpy().tolist())

        i_indices = batch["idx_i"].cpu().numpy()
        j_indices = batch["idx_j"].cpu().numpy()

        for index in range(len(i_indices)):
            anchor = int(i_indices[index])
            other = int(j_indices[index])
            dpred_val = float(dist_pred[index].item())

            if anchor not in anchor2pred:
                anchor2pred[anchor] = []

            anchor2pred[anchor].append((dpred_val, other))

            if anchor not in anchor2truth:
                anchor2truth[anchor] = set()

            if dist_golden[index].item() <= pos_threshold:
                anchor2truth[anchor].add(other)

    query2predictions = {}
    for anchor, dist_list in anchor2pred.items():
        # Sort by distance, we simulate the retrieval task
        # by sorting the distances and taking the top k
        dist_list.sort(key=lambda x: x[0])
        sorted_idx = [v[1] for v in dist_list]
        # Remove duplicates while preserving order
        sorted_idx = [key for key, _ in itertools.groupby(sorted_idx)]
        query2predictions[anchor] = sorted_idx

    recs = recall_at_k(anchor2truth, query2predictions, k_values)
    precisions = precision_at_k(anchor2truth, query2predictions, k_values)
    fs_at_k = f_scores(recs, precisions, k_values, f_score_beta)

    pearson_corr, p_pearson, spearman_corr, p_spearman = compute_correlation_scores(dist_preds, dist_true)

    return {
        "pearson": pearson_corr,
        "p_pearson": p_pearson,
        "spearman": spearman_corr,
        "p_spearman": p_spearman,
        "recall": recs,
        "precision": precisions,
        "f_score": fs_at_k
    }


def train_loop(
    train_dataset,
    val_dataset,
    cfg,
    device
):
    model = BERTStatementEmbedder(
        model_name=cfg.model_name,
        freeze_bert=cfg.freeze_bert,
        embedding_dim=cfg.embedding_dim
    )
    model.to(device)

    loss_fn = TripletMarginLoss(margin=cfg.margin).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    logger.info(f"Logging pre-training metrics")
    metrics_val = evaluate(model, val_loader, device, cfg.threshold_pos, cfg.evaluation.k_values, cfg.evaluation.f_score_beta)
    wandb.log({
        "val_pearson": metrics_val["pearson"],
        "val_spearman": metrics_val["spearman"]
    })

    for epoch in range(cfg.epochs):
        # eval_freq = 100 if epoch == 0 else None
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device, cfg, val_loader)
        metrics_val = evaluate(model, val_loader, device, cfg.threshold_pos, cfg.evaluation.k_values, cfg.evaluation.f_score_beta)

        wandb.log({
            "train_loss": train_loss,
            "val_pearson": metrics_val["pearson"],
            "val_spearman": metrics_val["spearman"]
        })

        for k in cfg.evaluation.k_values:
            wandb.log({f"val_recall@{k}": metrics_val["recall"][k]})
            wandb.log({f"val_precision@{k}": metrics_val["precision"][k]})
            wandb.log({f"val_f_score@{k}": metrics_val["f_score"][k]})

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_pearson={metrics_val['pearson']:.4f} val_spearman={metrics_val['spearman']:.4f}")

    return model
