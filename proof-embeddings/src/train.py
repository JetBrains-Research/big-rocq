import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
import itertools
import logging
from typing import Optional

from .metrics import compute_correlation_scores, recall_at_k, precision_at_k, f_scores
from .model import RulerModel
import torch.nn as nn

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

        input_ids_i = batch["input_ids_i"].to(device)
        attn_i = batch["attention_mask_i"].to(device)
        input_ids_j = batch["input_ids_j"].to(device)
        attn_j = batch["attention_mask_j"].to(device)

        pred_distance = model(input_ids_i, attn_i, input_ids_j, attn_j)
        true_distance = batch["dist"].float().to(device).unsqueeze(1)

        loss = loss_fn(pred_distance, true_distance)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # if eval_freq is not None and batch_index % eval_freq == 0:
        #     metrics_val = evaluate(model, val_loader, device, cfg.threshold_pos, cfg.evaluation.k_values, cfg.evaluation.f_score_beta, loss_fn)
        #     wandb.log({
        #         "val_loss": metrics_val["val_loss"],
        #         "val_pearson": metrics_val["pearson"],
        #         "val_spearman": metrics_val["spearman"]
        #     })

        batch_index += 1

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    pos_threshold,
    k_values,
    ranking_validation_dataset,
    f_score_beta=1.0,
    loss_function=None
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    dist_preds = []
    dist_true = []

    anchor2pred = {}
    anchor2truth = {}

    for batch in tqdm(dataloader, desc="Eval Iter", leave=False):
        input_ids_i = batch["input_ids_i"].to(device)
        attn_i = batch["attention_mask_i"].to(device)
        input_ids_j = batch["input_ids_j"].to(device)
        attn_j = batch["attention_mask_j"].to(device)

        pred_distance = model(input_ids_i, attn_i, input_ids_j, attn_j)
        pred_distance = pred_distance.squeeze(1)
        true_distance = batch["dist"].to(device)

        if loss_function is not None:
            loss = loss_function(pred_distance.unsqueeze(1), true_distance.unsqueeze(1))
            total_loss += loss.item()

        dist_preds.extend(pred_distance.cpu().numpy().tolist())
        dist_true.extend(true_distance.cpu().numpy().tolist())

        i_indices = batch["idx_i"].cpu().numpy()
        j_indices = batch["idx_j"].cpu().numpy()

        for index in range(len(i_indices)):
            anchor = int(i_indices[index])
            other = int(j_indices[index])
            dpred_val = float(pred_distance[index].item())

            if anchor not in anchor2pred:
                anchor2pred[anchor] = []
            anchor2pred[anchor].append((dpred_val, other))

            if anchor not in anchor2truth:
                anchor2truth[anchor] = set()
            if true_distance[index].item() <= pos_threshold:
                anchor2truth[anchor].add(other)

        num_batches += 1

    avg_val_loss = total_loss / num_batches if num_batches > 0 and loss_function is not None else None

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
    metrics = {
        "pearson": pearson_corr,
        "p_pearson": p_pearson,
        "spearman": spearman_corr,
        "p_spearman": p_spearman,
        "recall": recs,
        "precision": precisions,
        "f_score": fs_at_k,
        "val_loss": avg_val_loss,
        "events_accuracy": ranking_validation_dataset.get_accuracy_score(model, device),
    }
    return metrics


def train_loop(
    train_dataset,
    val_dataset,
    cfg,
    device,
    ranking_validation_dataset
):
    model = RulerModel(
        model_name=cfg.model_name,
        freeze_bert=cfg.freeze_bert,
        embedding_dim=cfg.embedding_dim
    )

    model.to(device)

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    logger.info(f"Logging pre-training metrics")
    metrics_val = evaluate(model, val_loader, device, cfg.threshold_pos, cfg.evaluation.k_values, ranking_validation_dataset, cfg.evaluation.f_score_beta, loss_fn)
    wandb.log({
        "val_pearson": metrics_val["pearson"],
        "val_spearman": metrics_val["spearman"]
    })

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device, cfg, val_loader)
        metrics_val = evaluate(model, val_loader, device, cfg.threshold_pos, cfg.evaluation.k_values, ranking_validation_dataset, cfg.evaluation.f_score_beta, loss_fn)
        wandb.log({
            "train_loss": train_loss,
            "val_loss": metrics_val["val_loss"],
            "val_pearson": metrics_val["pearson"],
            "val_spearman": metrics_val["spearman"],
            "events_accuracy": metrics_val["events_accuracy"]
        })
        for k in cfg.evaluation.k_values:
            wandb.log({f"val_recall@{k}": metrics_val["recall"][k]})
            wandb.log({f"val_precision@{k}": metrics_val["precision"][k]})
            wandb.log({f"val_f_score@{k}": metrics_val["f_score"][k]})

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={metrics_val['val_loss']:.4f} val_pearson={metrics_val['pearson']:.4f} val_spearman={metrics_val['spearman']:.4f}")

    return model
