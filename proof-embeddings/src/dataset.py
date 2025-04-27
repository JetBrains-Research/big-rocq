import json
import torch.nn as nn

from torch.utils.data import Dataset
from torch import stack
import torch.nn.functional as F
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import logging
from dataclasses import dataclass

from .data import load_single_file_json_dataset
from .similarity import proof_distance
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    anchor: int
    positive: int
    negative: int
    positive_distance: float
    negative_distance: float

    def __getitem__(self, keys):
        return iter(getattr(self, k) for k in keys)

    def __hash__(self):
        return hash((self.anchor, self.positive, self.negative))


class PrecomputedDistances:
    anchor: int
    positive_distances: List[Tuple[int, float]]
    negative_distances: List[Tuple[int, float]]

    def __init__(self, anchor: int):
        self.anchor = anchor
        self.positive_distances = []
        self.negative_distances = []

    def has_enough_distances(self, k: int, enough_pos: int = 1) -> bool:
        return len(self.positive_distances) >= enough_pos and len(self.negative_distances) >= k


class TheoremDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_seq_length: int,
        threshold_pos: float,
        threshold_neg: float,
        samples_from_single_anchor: int,
        k_negatives: int,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.k_negatives = k_negatives

        self.statements: List[str] = [d["statement"] for d in data]
        self.proofs: List[str] = [d["proofString"] for d in data]

        self.encoded_statements = [
            tokenizer(
                s,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            for s in self.statements
        ]

        self.distances = self._calculate_distances(samples_from_single_anchor)

    """
    Calculates the distance between all pairs of proofs
    @param max_samples_from_id: How many positive and negative samples to take from each proof
    """
    def _calculate_distances(
        self,
        max_samples_from_id: int
    ) -> Dict[int, PrecomputedDistances]:
        logger.info("Preparing distances between proofs")

        dist_dict = {}

        i_indices = list(range(len(self.data)))
        random.shuffle(i_indices)

        j_indices = list(range(len(self.data)))
        random.shuffle(j_indices)

        for i in tqdm(i_indices, desc="Calculating distances", leave=False):
            possible_js = random.sample(j_indices, min(max_samples_from_id, len(j_indices)))
            dist_dict[i] = PrecomputedDistances(i)

            for j in possible_js:
                if i == j:
                    continue
                dist = proof_distance(self.proofs[i], self.proofs[j])

                # To avoid that (i, j) and (j, i) have different
                # distances just in case precision goes brrr
                dist = round(dist, 4)

                if dist <= self.threshold_pos:
                    dist_dict[i].positive_distances.append((j, dist))
                elif dist >= self.threshold_neg:
                    dist_dict[i].negative_distances.append((j, dist))

        return dist_dict

    def __len__(self):
        # Infinite length (due to step-based training)
        return int(1e9)

    def __getitem__(self, index: int):
        while True:
            anchor_idx: int = random.choice(list(self.distances.keys()))
            anchor_distances = self.distances[anchor_idx]

            if anchor_distances.has_enough_distances(self.k_negatives):
                break

        pos_idx, pos_dist = random.choice(anchor_distances.positive_distances)
        neg_samples = random.sample(anchor_distances.negative_distances, self.k_negatives)
        neg_indices, neg_distances = zip(*neg_samples)

        anchor_enc = self.encoded_statements[anchor_idx]
        pos_encodings = self.encoded_statements[pos_idx]
        neg_encodings = [self.encoded_statements[neg_idx] for neg_idx in neg_indices]

        return {
            "input_ids_anchor": anchor_enc["input_ids"].squeeze(0),
            "attention_mask_anchor": anchor_enc["attention_mask"].squeeze(0),
            "input_ids_positive": pos_encodings["input_ids"].squeeze(0),
            "attention_mask_positive": pos_encodings["attention_mask"].squeeze(0),
            "input_ids_negatives": stack([neg["input_ids"].squeeze(0) for neg in neg_encodings]),
            "attention_mask_negatives": stack([neg["attention_mask"].squeeze(0) for neg in neg_encodings]),
            "anchor_idx": anchor_idx,
            "positive_idx": pos_idx,
            "negative_idxs": neg_indices,
            "positive_distance": pos_dist,
            "negative_distances": neg_distances
        }


    def pprintTriplet(self, triplet: Triplet):
        def removeNewlines(s: str):
            return s.replace("\n", " ")

        print(f"""
        Anchor Statement: {removeNewlines(self.statements[triplet.anchor])}
        Positive Statement: {removeNewlines(self.statements[triplet.positive])}
        Negative Statement: {removeNewlines(self.statements[triplet.negative])}
        
        Anchor Proof: {removeNewlines(self.proofs[triplet.anchor])}
        Positive Proof: {removeNewlines(self.proofs[triplet.positive])}
        Distance to Positive Proof: {triplet.positive_distance}
        
        Anchor Proof: {removeNewlines(self.proofs[triplet.anchor])}
        Negative Proof: {removeNewlines(self.proofs[triplet.negative])}
        Distance to Negative Proof: {triplet.negative_distance}
        """)


class ValidationDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_seq_length: int,
        threshold_pos: float,
        threshold_neg: float,
        samples_from_single_anchor: int,
        k_negatives: int,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.k_negatives = k_negatives

        self.statements: List[str] = [d["statement"] for d in data]
        self.proofs: List[str] = [d["proofString"] for d in data]

        self.encoded_statements = [
            tokenizer(
                s,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            for s in self.statements
        ]

        self.distances = self._calculate_distances(samples_from_single_anchor)

    """
    Calculates the distance between all pairs of proofs
    @param max_samples_from_id: How many positive and negative samples to take from each proof
    """
    def _calculate_distances(
        self,
        max_samples_from_id: int
    ) -> Dict[int, PrecomputedDistances]:
        logger.info("Preparing distances between proofs")

        dist_dict = {}

        i_indices = list(range(len(self.data)))
        random.shuffle(i_indices)

        j_indices = list(range(len(self.data)))
        random.shuffle(j_indices)

        for i in tqdm(i_indices, desc="Calculating distances", leave=False):
            possible_js = random.sample(j_indices, min(max_samples_from_id, len(j_indices)))
            dist_dict[i] = PrecomputedDistances(i)

            for j in possible_js:
                if i == j:
                    continue
                dist = proof_distance(self.proofs[i], self.proofs[j])

                # To avoid that (i, j) and (j, i) have different
                # distances just in case precision goes brrr
                dist = round(dist, 4)

                if dist <= self.threshold_pos:
                    dist_dict[i].positive_distances.append((j, dist))
                elif dist >= self.threshold_neg:
                    dist_dict[i].negative_distances.append((j, dist))

        return dist_dict

    def __len__(self):
        # Infinite length (due to step-based training)
        return int(1e9)

    def __getitem__(self, index: int, max_samples_for_ranking: int = 20):
        while True:
            anchor_idx: int = random.choice(list(self.distances.keys()))
            anchor_distances = self.distances[anchor_idx]

            if anchor_distances.has_enough_distances(max_samples_for_ranking - 1):
                break

        # Choose the fraction of positive samples between 0.2 and 0.5
        pos_fraction = random.uniform(0.1, 0.5)
        number_of_pos_samples = min(int(max_samples_for_ranking * pos_fraction), len(anchor_distances.positive_distances))
        number_of_neg_samples = max_samples_for_ranking - number_of_pos_samples

        pos_samples = random.sample(anchor_distances.positive_distances, number_of_pos_samples)
        neg_samples = random.sample(anchor_distances.negative_distances, number_of_neg_samples)

        all_samples = pos_samples + neg_samples
        random.shuffle(all_samples)

        indices, distances = zip(*all_samples)
        anchor_enc = self.encoded_statements[anchor_idx]
        other_encodings = [self.encoded_statements[idx] for idx in indices]

        return {
            "input_ids_anchor": anchor_enc["input_ids"].squeeze(0),
            "attention_mask_anchor": anchor_enc["attention_mask"].squeeze(0),
            "input_ids_others": stack([other["input_ids"].squeeze(0) for other in other_encodings]),
            "attention_mask_others": stack([other["attention_mask"].squeeze(0) for other in other_encodings]),
            "anchor_idx": anchor_idx,
            "other_idxs": indices,
            "distances": distances,
        }


class PairTheoremDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_seq_length: int,
        threshold_pos: float,
        threshold_neg: float,
        _samples_from_single_anchor: int
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg

        self.statements: List[str] = [d["statement"] for d in data]
        self.proofs: List[str] = [d["proofString"] for d in data]

        self.encoded_statements = [
            tokenizer(
                s,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            for s in self.statements
        ]

        self.proof_distances = self._calculate_distances(100)
        self.dataset_samples = self._create_pairs_dataset()

    def _create_pairs_dataset(self) -> List[Dict[str, Any]]:
        negative_pairs = set()
        positive_pairs = set()

        for i in range(len(self.proof_distances)):
            distances = self.proof_distances[i]

            for j, dist in distances:
                if dist <= self.threshold_pos:
                    positive_pairs.add((i, j, dist))
                else:
                    negative_pairs.add((i, j, dist))

        positive_pairs = list(positive_pairs)
        negative_pairs = list(negative_pairs)

        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)
        min_class_size = min(len(positive_pairs), len(negative_pairs))

        positive_pairs = positive_pairs[:min_class_size]
        negative_pairs = negative_pairs[:min_class_size]

        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)

        pairs = [self.__contrastive_item(p) for p in pairs]
        return pairs

    def __contrastive_item(self, distanced_pair: Tuple[int, int, float]):
        i, j, dist = distanced_pair

        input_i = self.encoded_statements[i]
        input_j = self.encoded_statements[j]

        if dist <= self.threshold_pos:
            label = 1
        else:
            label = 0

        input_ids_i = input_i["input_ids"].squeeze(0)
        attention_mask_i = input_i["attention_mask"].squeeze(0)

        input_ids_j = input_j["input_ids"].squeeze(0)
        attention_mask_j = input_j["attention_mask"].squeeze(0)

        return {
            "input_ids_i": input_ids_i,
            "attention_mask_i": attention_mask_i,
            "input_ids_j": input_ids_j,
            "attention_mask_j": attention_mask_j,
            "dist": dist,
            "label": label,
            "idx_i": i,
            "idx_j": j
        }

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index: int):
        return self.dataset_samples[index]

    def _calculate_distances(
        self,
        max_samples_from_id: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        logger.info("Preparing distances between proofs")

        dist_dict = {}

        i_indices = list(range(len(self.data)))
        random.shuffle(i_indices)

        j_indices = list(range(len(self.data)))
        random.shuffle(j_indices)

        for i in tqdm(i_indices, desc="Calculating distances", leave=False):
            possible_js = random.sample(j_indices, min(max_samples_from_id, len(j_indices)))
            for j in possible_js:
                if i == j:
                    continue
                dist = proof_distance(self.proofs[i], self.proofs[j])

                # To avoid that (i, j) and (j, i) have different
                # distances just in case precision goes brrr
                dist = round(dist, 4)

                dist_dict.setdefault(i, [])
                dist_dict[i].append((j, dist))

        return dist_dict


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
            {**encode_statement(s), "original": s}
            for s in self.statements
        ]

        reference_promises = json.load(open(reference_premises_path, "r"))
        self.reference_premises = reference_promises["references"]

        self.cliques = [
            [{**encode_statement(st), "original": st} for st in clique["clique"]]
            for clique in self.reference_premises
        ]

    @torch.no_grad()
    def get_accuracy_score(
        self,
        model: nn.Module,
        device,
        top_k: int = 6
    ) -> float:
        model.eval()
        scores = []

        for clique in self.cliques:
            for anchor in clique:
                anchor_ids = anchor["input_ids"].squeeze(0).to(device)
                anchor_mask = anchor["attention_mask"].squeeze(0).to(device)

                emb_anchor = model(anchor_ids.unsqueeze(0), anchor_mask.unsqueeze(0))  # [1, D]
                emb_anchor = F.normalize(emb_anchor, dim=-1)

                other_inputs = [
                    (x["input_ids"].squeeze(0), x["attention_mask"].squeeze(0))
                    for x in self.encoded_statements
                    if x["original"] != anchor["original"]
                ]

                if not other_inputs:
                    continue

                input_ids_others = torch.stack([ids for ids, _ in other_inputs]).to(device)
                attn_others = torch.stack([mask for _, mask in other_inputs]).to(device)

                emb_others = model(input_ids_others, attn_others)  # [N, D]
                emb_others = F.normalize(emb_others, dim=-1)

                sims = (emb_anchor @ emb_others.T).squeeze(0)  # [N]
                sorted_indices = sims.argsort(descending=True)

                top_k_predicted = [
                    self.encoded_statements[i]["original"]
                    for i in sorted_indices[:top_k]
                ]

                clique_others = [x["original"] for x in clique if x["original"] != anchor["original"]]

                score = len(
                    set(clique_others) & set(top_k_predicted)
                ) / min(len(clique_others), top_k)

                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0


def visualize_dataset(dataset, save_dir="dataset_visualizations", sample_size=1000):
    os.makedirs(save_dir, exist_ok=True)

    proof_lengths = []
    proof_distances_pos = []
    proof_distances_neg = []
    hypotheses_counts = []

    distances = dataset.distances
    for anchor_id, precomputed in distances.items():
        for j, dist in precomputed.positive_distances:
            proof_distances_pos.append(dist)
        for j, dist in precomputed.negative_distances:
            proof_distances_neg.append(dist)

    sample_statements = random.sample(dataset.statements, min(sample_size, len(dataset.statements)))
    for stmt in sample_statements:
        tokenized = dataset.tokenizer(stmt, truncation=True, padding=False)
        proof_lengths.append(len(tokenized["input_ids"]))
        # Hypotheses roughly counted by parentheses ( (x : A) )
        hypotheses_count = stmt.count(":")
        hypotheses_counts.append(hypotheses_count)

    plt.figure(figsize=(8,6))
    plt.hist(proof_distances_pos, bins=30, alpha=0.7, label='Positive Distances')
    plt.hist(proof_distances_neg, bins=30, alpha=0.7, label='Negative Distances')
    plt.legend()
    plt.title("Distribution of Proof Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "distance_distribution.png"))
    plt.close()

    plt.figure(figsize=(8,6))
    sns.histplot(proof_lengths, bins=30, kde=True)
    plt.title("Statement Token Lengths")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "statement_lengths.png"))
    plt.close()

    plt.figure(figsize=(8,6))
    sns.histplot(hypotheses_counts, bins=30, kde=True)
    plt.title("Hypotheses per Statement")
    plt.xlabel("Hypotheses count (rough estimate)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "hypotheses_counts.png"))
    plt.close()

    with open(os.path.join(save_dir, "examples.txt"), "w") as f:
        anchors = random.sample(list(distances.keys()), min(20, len(distances)))
        for anchor_id in anchors:
            anchor_stmt = dataset.statements[anchor_id]
            positives = distances[anchor_id].positive_distances
            negatives = distances[anchor_id].negative_distances
            if positives and negatives:
                pos_idx, pos_dist = random.choice(positives)
                neg_idx, neg_dist = random.choice(negatives)

                pos_stmt = dataset.statements[pos_idx]
                neg_stmt = dataset.statements[neg_idx]

                f.write(f"ANCHOR: {anchor_stmt}\n")
                f.write(f"  POSITIVE (d={pos_dist:.4f}): {pos_stmt}\n")
                f.write(f"  NEGATIVE (d={neg_dist:.4f}): {neg_stmt}\n")
                f.write("\n")

    print(f"Visualizations and examples saved to {save_dir}")

