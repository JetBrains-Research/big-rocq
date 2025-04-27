from torch.utils.data import DataLoader
import torch

def dynamic_collate_fn(batch):
    input_ids_anchor = torch.stack([item["input_ids_anchor"] for item in batch])
    attn_mask_anchor = torch.stack([item["attention_mask_anchor"] for item in batch])

    input_ids_pos = torch.stack([item["input_ids_positive"] for item in batch])
    attn_mask_pos = torch.stack([item["attention_mask_positive"] for item in batch])

    input_ids_neg = torch.stack([item["input_ids_negatives"] for item in batch])
    attn_mask_neg = torch.stack([item["attention_mask_negatives"] for item in batch])

    anchor_idxs = [item["anchor_idx"] for item in batch]
    positive_idxs = [item["positive_idx"] for item in batch]
    negative_idxs = [item["negative_idxs"] for item in batch]

    positive_distances = torch.tensor([item["positive_distance"] for item in batch])
    negative_distances = torch.tensor([item["negative_distances"] for item in batch])

    return {
        "input_ids_anchor": input_ids_anchor,
        "attention_mask_anchor": attn_mask_anchor,
        "input_ids_positive": input_ids_pos,
        "attention_mask_positive": attn_mask_pos,
        "input_ids_negatives": input_ids_neg,
        "attention_mask_negatives": attn_mask_neg,
        "anchor_idxs": anchor_idxs,
        "positive_idxs": positive_idxs,
        "negative_idxs": negative_idxs,
        "positive_distances": positive_distances,
        "negative_distances": negative_distances
    }


def ranking_collate_fn(batch):
    input_ids_anchor = torch.stack([item["input_ids_anchor"] for item in batch], dim=0)
    mask_anchor = torch.stack([item["attention_mask_anchor"] for item in batch], dim=0)

    input_ids_others = torch.stack([item["input_ids_others"] for item in batch], dim=0)
    mask_others = torch.stack([item["attention_mask_others"] for item in batch], dim=0)

    other_idxs = torch.tensor([item["other_idxs"] for item in batch], dtype=torch.long)
    distances = torch.tensor([item["distances"] for item in batch], dtype=torch.float)

    return {
        "input_ids_anchor": input_ids_anchor,
        "attention_mask_anchor": mask_anchor,
        "input_ids_others": input_ids_others,
        "attention_mask_others": mask_others,
        "other_idxs": other_idxs,
        "distances": distances,
    }


class DynamicQueryDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=0):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dynamic_collate_fn,
            drop_last=False
        )


class RankingDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=0):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=ranking_collate_fn,
            drop_last=False
        )