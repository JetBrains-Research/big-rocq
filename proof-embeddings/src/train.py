import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from .losses import ContrastiveLoss, TripletMarginLoss
from .metrics import compute_correlation_scores, recall_at_k
from .model import BERTStatementEmbedder

def train_one_epoch(
    model,
    loss_fn,
    dataloader,
    optimizer,
    device
):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Iter", leave=False):
        optimizer.zero_grad()

        input_ids_i = batch["input_ids_i"].to(device)
        attn_i = batch["attention_mask_i"].to(device)
        input_ids_j = batch["input_ids_j"].to(device)
        attn_j = batch["attention_mask_j"].to(device)

        labels = batch["label"].to(device)
        emb_i = model(input_ids_i, attn_i)
        emb_j = model(input_ids_j, attn_j)

        if isinstance(loss_fn, ContrastiveLoss):
            loss = loss_fn(emb_i, emb_j, labels.float())
        elif isinstance(loss_fn, TripletMarginLoss):
            loss = torch.tensor(0.0, requires_grad=True).to(device)
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(device)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    pos_threshold,
    k_values=[1,5,10]
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

        dist = batch["dist"].float()
        dist_pred = torch.norm(emb_i - emb_j, p=2, dim=1)

        dist_preds.extend(dist_pred.cpu().numpy().tolist())
        dist_true.extend(dist.cpu().numpy().tolist())

        idx_i = batch["idx_i"].cpu().numpy()
        idx_j = batch["idx_j"].cpu().numpy()

        # Print statements in the batch
        # for i in range(len(idx_i)):
        #     print(f"Statement pair: {dataloader.dataset.statements[idx_i[i]]} and {dataloader.dataset.statements[idx_j[i]]}")
        #     print(f"Distance: {dist[i].item()}, Predicted Distance: {dist_pred[i].item()}")
        #
        # print(f"Statements in batch: { [dataloader.dataset.statements[i] for i in idx_i] }")

        for b in range(len(idx_i)):
            anchor = int(idx_i[b])
            other = int(idx_j[b])
            dpred_val = float(dist_pred[b].item())
        
            if anchor not in anchor2pred:
                anchor2pred[anchor] = []
            anchor2pred[anchor].append((dpred_val, other))

            if anchor not in anchor2truth:
                anchor2truth[anchor] = set()
            if dist[b].item() <= pos_threshold:
                anchor2truth[anchor].add(other)

    # Print anchor2truth
    # for anchor, truth in anchor2truth.items():
    #     print(f"Statement {dataloader.dataset.statements[anchor]}, Truth: { [dataloader.dataset.statements[t] for t in truth] }")

    query2preds = {}
    for anchor, dist_list in anchor2pred.items():
        dist_list.sort(key=lambda x: x[0])
        sorted_idx = [v[1] for v in dist_list]
        query2preds[anchor] = sorted_idx

    recs = recall_at_k(anchor2truth, query2preds, k_values)

    pearson_corr, p_pearson, spearman_corr, p_spearman = compute_correlation_scores(dist_preds, dist_true)

    return {
        "pearson": pearson_corr,
        "p_pearson": p_pearson,
        "spearman": spearman_corr,
        "p_spearman": p_spearman,
        "recall": recs
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

    if cfg.loss_type == "contrastive":
        loss_fn = ContrastiveLoss(margin=cfg.margin).to(device)
    else:
        loss_fn = TripletMarginLoss(margin=cfg.margin).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
        metrics_val = evaluate(model, val_loader, device, cfg.threshold_pos, cfg.evaluation.recall_k)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pearson": metrics_val["pearson"],
            "val_spearman": metrics_val["spearman"]
        })

        for k in cfg.evaluation.recall_k:
            wandb.log({f"val_recall@{k}": metrics_val["recall"][k]})

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_pearson={metrics_val['pearson']:.4f} val_spearman={metrics_val['spearman']:.4f}")

    return model
