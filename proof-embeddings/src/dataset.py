from os import abort

from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random
import logging
from dataclasses import dataclass

from .similarity import proof_distance

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


class TheoremDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_seq_length: int,
        threshold_pos: float,
        threshold_neg: float,
        samples_from_single_anchor: int
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

        self.proof_distances = self._calculate_distances(samples_from_single_anchor)
        self.dataset_samples = self._create_triplets_dataset()

    def _create_triplets_dataset(self) -> List[Dict[str, Any]]:
        triplets = set()
        i_indices = list(range(len(self.proof_distances)))

        for i in tqdm(i_indices, desc="Creating triplets", leave=False):
            distances = self.proof_distances[i]
            pos_samples = [(j, dist) for j, dist in distances if dist <= self.threshold_pos]
            neg_samples = [(j, dist) for j, dist in distances if dist >= self.threshold_neg]

            max_samples = min(len(pos_samples), len(neg_samples))

            pos_samples = pos_samples[:max_samples]
            neg_samples = neg_samples[:max_samples]

            # for (pos, dist_pos) in pos_samples:
            #     for (neg, dist_neg) in neg_samples:
            #         triplets.add(Triplet(i, pos, neg, dist_pos, dist_neg))

            for j in range(max_samples):
                pos, dist_pos = pos_samples[j]
                neg, dist_neg = neg_samples[j]

                triplet = Triplet(i, pos, neg, dist_pos, dist_neg)
                triplets.add(triplet)

        triplets = list(triplets)
        random.shuffle(triplets)

        triplets = [self.__triplet_item(t) for t in triplets]
        return triplets

    """
    Calculates the distance between all pairs of proofs
    @param max_samples_from_id: How many positive and negative samples to take from each proof
    """
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

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index: int):
        return self.dataset_samples[index]

    def __triplet_item(self, triplet: Triplet):
        input_anchor = self.encoded_statements[triplet.anchor]
        input_pos_sample = self.encoded_statements[triplet.positive]
        input_neg_sample = self.encoded_statements[triplet.negative]

        input_ids_anchor = input_anchor["input_ids"].squeeze(0)
        attention_mask_anchor = input_anchor["attention_mask"].squeeze(0)
        input_ids_pos_sample = input_pos_sample["input_ids"].squeeze(0)
        attention_mask_pos_sample = input_pos_sample["attention_mask"].squeeze(0)
        input_ids_neg_sample = input_neg_sample["input_ids"].squeeze(0)
        attention_mask_neg_sample = input_neg_sample["attention_mask"].squeeze(0)

        return {
            "input_ids_anchor": input_ids_anchor,
            "attention_mask_anchor": attention_mask_anchor,
            "input_ids_pos_sample": input_ids_pos_sample,
            "attention_mask_pos_sample": attention_mask_pos_sample,
            "input_ids_neg_sample": input_ids_neg_sample,
            "attention_mask_neg_sample": attention_mask_neg_sample,
            "idx_anchor": triplet.anchor,
            "idx_pos_sample": triplet.positive,
            "idx_neg_sample": triplet.negative,
            "dist_pos": triplet.positive_distance,
            "dist_neg": triplet.negative_distance
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


class ValidationTheoremDataset(Dataset):
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
        pairs = set()

        for i in range(len(self.proof_distances)):
            distances = self.proof_distances[i]

            for j, dist in distances:
                pairs.add((i, j, dist))

        pairs = list(pairs)
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