import json
import torch.nn as nn

from torch.utils.data import Dataset
from torch import stack
import torch.nn.functional as F
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import logging

from .data import load_single_file_json_dataset
from .similarity import proof_distance, split_tactics
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class DatasetItem:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    idx: int
    original_statement: str
    proof_str: str
    positive_distances: List[Tuple[int, float]]
    negative_distances: List[Tuple[int, float]]

    def __init__(
        self,
        original_statement: str,
        proof_str: str,
        idx: int,
        tokenizer,
        max_seq_length: int,
        threshold_positive: float,
        threshold_negative: float,
        threshold_hard_negative: float,
        vectorizer: TfidfVectorizer,
    ):
        encoded_statement = tokenizer(
            original_statement,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

        self.input_ids = encoded_statement["input_ids"].squeeze(0)
        self.attention_mask = encoded_statement["attention_mask"].squeeze(0)
        self.idx = idx
        self.original_statement = original_statement
        self.proof_str = proof_str
        self.threshold_positive = threshold_positive
        self.threshold_negative = threshold_negative
        self.threshold_hard_negative = threshold_hard_negative
        self.positive_distances = []
        self.negative_distances = []
        self.tf_idf_normalized = vectorizer.transform([original_statement])

    def add_distance(self, idx: int, distance: float):
        if distance <= self.threshold_positive:
            self.positive_distances.append((idx, distance))
        elif distance >= self.threshold_negative:
            self.negative_distances.append((idx, distance))
        elif distance >= self.threshold_hard_negative:
            thirty_percent_chance = random.uniform(0, 1)
            if thirty_percent_chance <= 0.3:
                self.negative_distances.append((idx, distance))

    def has_enough_distances(self, neg_min: int, pos_min: int = 1) -> bool:
        return (
            len(self.positive_distances) >= pos_min
            and len(self.negative_distances) >= neg_min
        )


class TrainingDataset(Dataset):
    def __init__(
        self,
        data: List[List[Dict[str, Any]]],
        tokenizer,
        max_seq_length: int,
        threshold_pos: float,
        threshold_neg: float,
        threshold_hard_neg: float,
        samples_from_single_anchor: int,
        k_negatives: int,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.threshold_hard_neg = threshold_hard_neg
        self.k_negatives = k_negatives

        self.encoded_data_grouped = []

        self.ds_statements = []
        for file in data:
            for theorem in file:
                self.ds_statements.append(theorem["statement"])

        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            analyzer="word",
        )
        self.vectorizer.fit(self.ds_statements)

        self.items_dict = {}

        item_index = 0
        self.biggest_index = 0
        for file in tqdm(data, desc=f"Precomputing items in files", leave=False):
            file_items = []
            for theorem in file:
                item = DatasetItem(
                    theorem["statement"],
                    theorem["proofString"],
                    item_index,
                    tokenizer,
                    max_seq_length,
                    threshold_pos,
                    threshold_neg,
                    threshold_hard_neg,
                    self.vectorizer,
                )
                file_items.append(item)
                self.items_dict[item_index] = item
                self.biggest_index = item_index
                item_index += 1
            self.encoded_data_grouped.append(file_items)

        self._calculate_distances(samples_from_single_anchor)

    """
    Calculates the distance between all pairs of proofs
    @param max_samples_from_id: How many positive and negative samples to take from each proof
    """

    def _calculate_distances(self, max_samples_from_id: int):
        logger.info("Preparing distances between proofs")

        file_indices = list(range(len(self.encoded_data_grouped)))
        for file_index, file in enumerate(self.encoded_data_grouped):
            for ds_item in tqdm(
                file, desc=f"Calculating distances for file #{file_index}", leave=False
            ):
                # Stage 1: Calculate distances to all statements from
                # the same file

                for other_ds_item in file:
                    if ds_item.idx == other_ds_item.idx:
                        continue
                    dist = proof_distance(
                        ds_item.proof_str,
                        other_ds_item.proof_str,
                    )
                    assert self.items_dict.get(other_ds_item.idx) is not None
                    ds_item.add_distance(other_ds_item.idx, dist)

                # Stage 2: Calculate distances to k randomly chosen
                # statements in other files
                for _ in range(max_samples_from_id):
                    other_file_indices = [
                        file for file in file_indices if file != file_index
                    ]
                    other_file_index = random.choice(other_file_indices)

                    other_statement: DatasetItem = random.choice(
                        self.encoded_data_grouped[other_file_index]
                    )
                    dist = proof_distance(
                        other_statement.proof_str,
                        ds_item.proof_str,
                    )

                    assert self.items_dict.get(other_statement.idx) is not None
                    ds_item.add_distance(other_statement.idx, dist)

    def __len__(self):
        # Infinite length (due to step-based training)
        return int(1e9)

    def __getitem__(self, index: int):
        while True:
            anchor_idx = random.choice(list(self.items_dict.keys()))
            anchor_item = self.items_dict[anchor_idx]

            if anchor_item.has_enough_distances(neg_min=self.k_negatives):
                break

        pos_idx, pos_dist = random.choice(anchor_item.positive_distances)
        pos_item = self.items_dict[pos_idx]

        neg_samples = random.sample(anchor_item.negative_distances, self.k_negatives)
        neg_indices, neg_distances = zip(*neg_samples)
        neg_items = [self.items_dict[neg_idx] for neg_idx in neg_indices]

        return {
            "input_ids_anchor": anchor_item.input_ids,
            "attention_mask_anchor": anchor_item.attention_mask,
            "input_ids_positive": pos_item.input_ids,
            "attention_mask_positive": pos_item.attention_mask,
            "input_ids_negatives": stack(
                [neg_item.input_ids for neg_item in neg_items]
            ),
            "attention_mask_negatives": stack(
                [neg_item.attention_mask for neg_item in neg_items]
            ),
            "anchor_idx": anchor_idx,
            "positive_idx": pos_idx,
            "negative_idxs": neg_indices,
            "positive_distance": pos_dist,
            "negative_distances": neg_distances,
        }


class ValidationDataset(Dataset):
    def __init__(
        self,
        data: List[List[Dict[str, Any]]],
        tokenizer,
        max_seq_length: int,
        threshold_pos: float,
        threshold_neg: float,
        threshold_hard_neg: float,
        samples_from_single_anchor: int,
        k_negatives: int,
        query_size_in_eval: int,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.threshold_hard_neg = threshold_hard_neg
        self.k_negatives = k_negatives
        self.query_size_in_eval = query_size_in_eval

        self.encoded_data_grouped = []
        self.items_dict = {}

        self.ds_statements = []
        for file in data:
            for theorem in file:
                self.ds_statements.append(theorem["statement"])

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            analyzer="word",
        )
        self.vectorizer.fit(self.ds_statements)

        item_index = 0
        self.biggest_index = 0
        for file in tqdm(data, desc=f"Precomputing items in files", leave=False):
            file_items = []
            for theorem in file:
                item = DatasetItem(
                    theorem["statement"],
                    theorem["proofString"],
                    item_index,
                    tokenizer,
                    max_seq_length,
                    threshold_pos,
                    threshold_neg,
                    threshold_hard_neg,
                    self.vectorizer,
                )
                file_items.append(item)
                self.items_dict[item_index] = item
                self.biggest_index = item_index
                item_index += 1
            self.encoded_data_grouped.append(file_items)

        self._calculate_distances(samples_from_single_anchor)

    """
    Calculates the distance between all pairs of proofs
    @param max_samples_from_id: How many positive and negative samples to take from each proof
    """

    def _calculate_distances(self, max_samples_from_id: int):
        logger.info("Preparing distances between proofs")

        file_indices = list(range(len(self.encoded_data_grouped)))
        for file_index, file in enumerate(self.encoded_data_grouped):
            for ds_item in tqdm(
                file, desc=f"Calculating distances for file #{file_index}", leave=False
            ):
                # Stage 1: Calculate distances to all statements from
                # the same file

                for other_ds_item in file:
                    if ds_item.idx == other_ds_item.idx:
                        continue
                    dist = proof_distance(
                        ds_item.proof_str,
                        other_ds_item.proof_str,
                    )
                    ds_item.add_distance(other_ds_item.idx, dist)

                # Stage 2: Calculate distances to k randomly chosen
                # statements in other files
                for _ in range(max_samples_from_id):
                    other_file_indices = [
                        file for file in file_indices if file != file_index
                    ]
                    other_file_index = random.choice(other_file_indices)

                    other_statement: DatasetItem = random.choice(
                        self.encoded_data_grouped[other_file_index]
                    )
                    dist = proof_distance(
                        other_statement.proof_str,
                        ds_item.proof_str,
                    )
                    ds_item.add_distance(other_statement.idx, dist)

    def __len__(self):
        # Infinite length (due to step-based training)
        return int(1e9)

    def __getitem__(self, index: int):
        anchor_idx: int = index % (self.biggest_index + 1)
        anchor_item = self.items_dict[anchor_idx]
        while True:
            if anchor_item.has_enough_distances(neg_min=self.query_size_in_eval - 1):
                break

            anchor_idx = random.choice(list(self.items_dict.keys()))
            anchor_item = self.items_dict[anchor_idx]

        # Choose the fraction of positive samples between 0.2 and 0.5
        pos_fraction = random.uniform(0.1, 0.5)
        number_of_pos_samples = min(
            int(self.query_size_in_eval * pos_fraction),
            len(anchor_item.positive_distances),
        )
        number_of_neg_samples = self.query_size_in_eval - number_of_pos_samples

        pos_samples: List[Tuple[int, float]] = random.sample(
            anchor_item.positive_distances, number_of_pos_samples
        )
        neg_samples: List[Tuple[int, float]] = random.sample(
            anchor_item.negative_distances, number_of_neg_samples
        )

        all_samples = pos_samples + neg_samples
        random.shuffle(all_samples)

        query_indices, distances = zip(*all_samples)
        other_items = [self.items_dict[index] for index in query_indices]

        return {
            "input_ids_anchor": anchor_item.input_ids,
            "attention_mask_anchor": anchor_item.attention_mask,
            "input_ids_others": stack(
                [other_item.input_ids for other_item in other_items]
            ),
            "attention_mask_others": stack(
                [other_item.attention_mask for other_item in other_items]
            ),
            "anchor_idx": anchor_idx,
            "other_idxs": query_indices,
            "distances": distances,
        }


class RankingDataset:
    def __init__(
        self,
        max_seq_length: int,
        tokenizer,
        all_statements_path: str,
        reference_premises_path: str,
    ):
        data = load_single_file_json_dataset(all_statements_path)
        logger.info("Loaded dataset with %d statements", len(data))
        self.statements: List[str] = [d["statement"] for d in data]
        self.proofs: List[str] = [d["proofString"] for d in data]

        def encode_statement(st: str):
            return tokenizer(
                st,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )

        self.encoded_statements = [
            {**encode_statement(s), "original": s} for s in self.statements
        ]

        reference_promises = json.load(open(reference_premises_path, "r"))
        self.reference_premises = reference_promises["references"]

        self.cliques = [
            [{**encode_statement(st), "original": st} for st in clique["clique"]]
            for clique in self.reference_premises
        ]

    def string_similarity(self, s1: str, s2: str):
        # Jaccard similarity

        s1_set = set(s1.split())
        s2_set = set(s2.split())

        intersection = len(s1_set.intersection(s2_set))
        union = len(s1_set.union(s2_set))

        return intersection / union if union > 0 else 0.0

    def test_get_accuracy_score(self, top_k: int = 6) -> float:
        scores = []

        for clique in self.cliques:
            for anchor in clique:
                similarity_scores = [
                    (self.string_similarity(anchor["original"], x["original"]), i)
                    for i, x in enumerate(self.encoded_statements)
                ]
                similarity_scores.sort(reverse=True, key=lambda x: x[0])

                indices = [i for _, i in similarity_scores]

                top_k_predicted = [
                    self.encoded_statements[i]["original"] for i in indices[:top_k]
                ]

                clique_others = [
                    x["original"] for x in clique if x["original"] != anchor["original"]
                ]

                score = len(set(clique_others) & set(top_k_predicted)) / min(
                    len(clique_others), top_k
                )

                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    @torch.no_grad()
    def get_accuracy_score(self, model: nn.Module, device, top_k: int = 6) -> float:
        model.eval()
        scores = []

        for clique in self.cliques:
            for anchor in clique:
                anchor_ids = anchor["input_ids"].squeeze(0).to(device)
                anchor_mask = anchor["attention_mask"].squeeze(0).to(device)

                emb_anchor = model(anchor_ids.unsqueeze(0), anchor_mask.unsqueeze(0))
                emb_anchor = F.normalize(emb_anchor, dim=-1)

                other_inputs = [
                    (x["input_ids"].squeeze(0), x["attention_mask"].squeeze(0))
                    for x in self.encoded_statements
                    if x["original"] != anchor["original"]
                ]

                if not other_inputs:
                    continue

                input_ids_others = torch.stack([ids for ids, _ in other_inputs]).to(
                    device
                )
                attn_others = torch.stack([mask for _, mask in other_inputs]).to(device)

                emb_others = model(input_ids_others, attn_others)
                emb_others = F.normalize(emb_others, dim=-1)

                sims = (emb_anchor @ emb_others.T).squeeze(0)
                sorted_indices = sims.argsort(descending=True)

                top_k_predicted = [
                    self.encoded_statements[i]["original"]
                    for i in sorted_indices[:top_k]
                ]

                clique_others = [
                    x["original"] for x in clique if x["original"] != anchor["original"]
                ]

                score = len(set(clique_others) & set(top_k_predicted)) / min(
                    len(clique_others), top_k
                )

                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any


def visualize_dataset(dataset: Any, save_dir: str = "dataset_visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid")

    dist_records = []
    anchor_records = []
    for item in dataset.items_dict.values():
        pos_ds = [d for _, d in item.positive_distances]
        neg_ds = [d for _, d in item.negative_distances]
        if not pos_ds or not neg_ds:
            continue

        dist_records += [{"distance": d, "type": "positive"} for d in pos_ds]
        dist_records += [{"distance": d, "type": "negative"} for d in neg_ds]

        proof_len = len(split_tactics(item.proof_str))
        hypo_count = item.original_statement.count(":")
        anchor_records.append(
            {
                "length": proof_len,
                "hypotheses": hypo_count,
                "mean_pos": np.mean(pos_ds),
                "mean_neg": np.mean(neg_ds),
            }
        )

    df_dist = pd.DataFrame(dist_records)
    df_anchor = pd.DataFrame(anchor_records)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_dist,
        x="distance",
        hue="type",
        bins=30,
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.6,
    )

    for t in ["positive", "negative"]:
        q = [0.25, 0.5, 0.75]
        qs = df_dist[df_dist.type == t].distance.quantile(q)
        for qi, v in zip(q, qs):
            plt.axvline(v, linestyle="--", label=f"{t} Q{int(qi*100)}", alpha=0.7)
    plt.title("Proof Distance Distribution with KDE and Quartiles")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distance_kde_quartiles.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        x="type", y="distance", data=df_dist, inner="quartile", scale="width"
    )
    plt.title("Violin Plot of Proof Distances")
    plt.xlabel("Pair Type")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distance_violin.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.ecdfplot(data=df_dist, x="distance", hue="type")
    plt.title("Empirical CDF of Proof Distances")
    plt.xlabel("Distance")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distance_ecdf.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_anchor, x="length", y="mean_pos", alpha=0.6)
    sns.regplot(
        data=df_anchor, x="length", y="mean_pos", scatter=False, color="red", ci=95
    )
    plt.title("Proof Length vs. Mean Positive Distance")
    plt.xlabel("Proof Length (# tactics)")
    plt.ylabel("Mean Positive Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "length_vs_mean_pos.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_anchor, x="hypotheses", y="mean_neg", alpha=0.6)
    sns.regplot(
        data=df_anchor, x="hypotheses", y="mean_neg", scatter=False, color="red", ci=95
    )
    plt.title("Hypotheses Count vs. Mean Negative Distance")
    plt.xlabel("Hypotheses Count")
    plt.ylabel("Mean Negative Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hypotheses_vs_mean_neg.png"))
    plt.close()

    sns.pairplot(
        df_anchor, vars=["length", "hypotheses", "mean_pos", "mean_neg"], corner=True
    )
    plt.suptitle("Pairplot of Anchor Summary Statistics", y=1.02)
    plt.savefig(os.path.join(save_dir, "anchor_pairplot.png"))
    plt.close()

    ex_file = os.path.join(save_dir, "sample_examples.md")
    with open(ex_file, "w") as f:
        keys = list(dataset.items_dict.keys())
        random.shuffle(keys)
        for idx in keys[:20]:
            item = dataset.items_dict[idx]
            f.write(f"### Anchor #{idx}\n")
            f.write(f"Statement: `{item.original_statement}`  \n")
            f.write(f"Proof: ```\n{item.proof_str}\n```  \n")
            if item.positive_distances:
                j, d = random.choice(item.positive_distances)
                f.write(
                    f"- Positive Example (d={d:.3f}): `{dataset.items_dict[j].original_statement}`\n"
                )
            if item.negative_distances:
                k, d = random.choice(item.negative_distances)
                f.write(
                    f"- Negative Example (d={d:.3f}): `{dataset.items_dict[k].original_statement}`\n"
                )
            f.write("\n")

    print(f"All visualizations saved under {save_dir}")
