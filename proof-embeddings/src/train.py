import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import logging
import os
from transformers import get_linear_schedule_with_warmup
from datetime import datetime

from .dataloader import DynamicQueryDataLoader, RankingDataLoader
from .losses import InfoNCELoss
from .metrics import precision_at_k, recall_at_k, f_scores, compute_correlation_scores
from .model import RocqStatementEmbedder

logger = logging.getLogger(__name__)


def train_one_step(
    model,
    loss_fn,
    batch,
    optimizer,
    device,
):
    model.train()

    optimizer.zero_grad()

    anchor_ids = batch["input_ids_anchor"].to(device)
    anchor_mask = batch["attention_mask_anchor"].to(device)
    pos_ids = batch["input_ids_positive"].to(device)
    pos_mask = batch["attention_mask_positive"].to(device)
    neg_ids = batch["input_ids_negatives"].to(device)
    neg_mask = batch["attention_mask_negatives"].to(device)

    emb_anchor = model(anchor_ids, anchor_mask)
    emb_positive = model(pos_ids, pos_mask)

    batch_size, k_neg, seq_len = neg_ids.shape
    emb_negatives = model(
        neg_ids.reshape(-1, seq_len), neg_mask.reshape(-1, seq_len)
    ).reshape(batch_size, k_neg, -1)

    loss = loss_fn(emb_anchor, emb_positive, emb_negatives)

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_ranking_ability(
    model,
    dataloader,
    device,
    pos_threshold,
    k_values,
    extra_imm_validation_ds,
    eval_steps=100,
):
    model.eval()
    steps = 0

    precisions = []
    recalls = []
    f1s = []

    pearson_correlations = []
    spearman_correlations = []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        anchor_ids = batch["input_ids_anchor"].to(device)
        anchor_mask = batch["attention_mask_anchor"].to(device)
        others_ids = batch["input_ids_others"].to(device)
        others_mask = batch["attention_mask_others"].to(device)

        other_idxs = batch["other_idxs"]
        true_distances = batch["distances"]

        emb_anchor = model(anchor_ids, anchor_mask)
        batch_size, others_count, seq_len = others_ids.shape

        emb_others = model(
            others_ids.reshape(-1, seq_len), others_mask.reshape(-1, seq_len)
        ).reshape(batch_size, others_count, -1)

        emb_anchor = F.normalize(emb_anchor, dim=-1)
        emb_others = F.normalize(emb_others, dim=-1)

        similarities = torch.einsum("bd,bnd->bn", emb_anchor, emb_others)
        predicted_distances = 1 - similarities

        for b in range(batch_size):
            predicted = predicted_distances[b].cpu().numpy().tolist()
            ids = [int(x) for x in other_idxs[b]]
            trues = [float(x) for x in true_distances[b]]
            relevant_items = [idx for idx, d in zip(ids, trues) if d <= pos_threshold]
            pairs = list(zip(ids, predicted))

            pairs.sort(key=lambda x: x[1])
            ranked_indices = [idx for idx, _ in pairs]

            precision = precision_at_k(ranked_indices, relevant_items, k_values)
            recall = recall_at_k(ranked_indices, relevant_items, k_values)
            f1 = f_scores(recall, precision, k_values)

            pearson_corr, _, spearman_corr, _ = compute_correlation_scores(
                predicted, trues
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            pearson_correlations.append(pearson_corr)
            spearman_correlations.append(spearman_corr)

        steps += 1
        if steps >= eval_steps:
            break

    avg_precision = {}
    avg_recall = {}
    avg_f1 = {}

    for k in k_values:
        avg_precision[k] = sum(p[k] for p in precisions) / len(precisions)
        avg_recall[k] = sum(r[k] for r in recalls) / len(recalls)
        avg_f1[k] = sum(f1[k] for f1 in f1s) / len(f1s)

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "pearson": sum(pearson_correlations) / len(pearson_correlations),
        "spearman": sum(spearman_correlations) / len(spearman_correlations),
        "imm_extra_eval": extra_imm_validation_ds.get_accuracy_score(model, device),
    }


@torch.no_grad()
def evaluate_validation_loss(
    model,
    dataloader,
    device,
    loss_fn,
    eval_steps=100,
):
    model.eval()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        anchor_ids = batch["input_ids_anchor"].to(device)
        anchor_mask = batch["attention_mask_anchor"].to(device)
        pos_ids = batch["input_ids_positive"].to(device)
        pos_mask = batch["attention_mask_positive"].to(device)
        neg_ids = batch["input_ids_negatives"].to(device)
        neg_mask = batch["attention_mask_negatives"].to(device)

        emb_anchor = model(anchor_ids, anchor_mask)
        emb_positive = model(pos_ids, pos_mask)

        batch_size, k_neg, seq_len = neg_ids.shape
        emb_negatives = model(
            neg_ids.reshape(-1, seq_len), neg_mask.reshape(-1, seq_len)
        ).reshape(batch_size, k_neg, -1)

        emb_anchor = F.normalize(emb_anchor, dim=-1)
        emb_positive = F.normalize(emb_positive, dim=-1)
        emb_negatives = F.normalize(emb_negatives, dim=-1)

        loss = loss_fn(emb_anchor, emb_positive, emb_negatives)
        total_loss += loss.item()

        steps += 1
        if steps >= eval_steps:
            break

    avg_val_loss = total_loss / steps if loss_fn else None

    metrics = {
        "val_loss": avg_val_loss,
    }

    return metrics


def train_loop(
    train_dataset,
    val_dataset,
    cfg,
    device,
    val_dataset_ranking,
    extra_imm_validation_ds,
    tokenizer,
):
    model = RocqStatementEmbedder(
        model_name=cfg.base_model_name, embedding_dim=cfg.embedding_dim
    ).to(device)

    loss_fn = InfoNCELoss(temperature=0.07).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    total_steps = cfg.steps
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_loader = DynamicQueryDataLoader(train_dataset, batch_size=cfg.batch_size)
    val_loader = DynamicQueryDataLoader(val_dataset, batch_size=cfg.batch_size)
    val_ranking_loader = RankingDataLoader(
        val_dataset_ranking, batch_size=cfg.batch_size
    )

    step = 0
    train_losses = []

    while step < cfg.steps:
        for batch in tqdm(train_loader, desc="Train", leave=False):
            model.train()
            loss = train_one_step(model, loss_fn, batch, optimizer, device)
            train_losses.append(loss)

            scheduler.step()
            step += 1

            if step % cfg.evaluation.evaluate_freq == 0 or step == cfg.steps:
                avg_train_loss = sum(
                    train_losses[-cfg.evaluation.evaluate_freq :]
                ) / len(train_losses[-cfg.evaluation.evaluate_freq :])
                val_loss_metrics = evaluate_validation_loss(
                    model,
                    val_loader,
                    device,
                    loss_fn,
                    eval_steps=cfg.evaluation.eval_steps,
                )
                val_ranking_metrics = evaluate_ranking_ability(
                    model,
                    val_ranking_loader,
                    device,
                    cfg.threshold_pos,
                    cfg.evaluation.k_values,
                    extra_imm_validation_ds,
                    eval_steps=cfg.evaluation.eval_steps,
                )

                wandb.log(
                    {
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss_metrics["val_loss"],
                        "val_pearson": val_ranking_metrics["pearson"],
                        "val_spearman": val_ranking_metrics["spearman"],
                        "val_imm_extra_eval": val_ranking_metrics["imm_extra_eval"],
                    }
                )

                for k in cfg.evaluation.k_values:
                    wandb.log(
                        {
                            f"val_precision_at_{k}": val_ranking_metrics["precision"][
                                k
                            ],
                            f"val_recall_at_{k}": val_ranking_metrics["recall"][k],
                            f"val_f1_at_{k}": val_ranking_metrics["f1"][k],
                        }
                    )

            if step % cfg.evaluation.save_freq == 0 or step == cfg.steps:
                date = datetime.now().strftime("%Y-%m-%d_%H-%M")
                ckpt_dir = os.path.join(
                    cfg.evaluation.output_dir, f"checkpoint-{step}-{date}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)

                model.bert.save_pretrained(ckpt_dir)

                tokenizer.save_pretrained(ckpt_dir)
                with open(os.path.join(ckpt_dir, "training_args.txt"), "w") as f:
                    f.write(f"step = {step}\n")
                    for k, v in vars(cfg).items():
                        f.write(f"{k} = {v}\n")
                logger.info(f"Saved checkpoint to {ckpt_dir}")

            if step >= cfg.steps:
                break

            if step >= cfg.steps:
                break

    return model
