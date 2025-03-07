from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import random

from .similarity import proof_distance


class TheoremDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_seq_length: int,
        threshold_pos: float,
        threshold_neg: float,
        mode: str = "train"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.mode = mode

        self.statements = [d["statement"] for d in data]
        self.proofs = [d["proof"] for d in data]

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

        if self.mode == "train":
            self.pairs = self._create_pairs()
        else:
            self.pairs = [(i, i) for i in range(len(data))]

    def _create_pairs(self) -> List[Tuple[int, int, float]]:
        """
        Creates (i, j, distance_ij) tuples
        """
        indices = list(range(len(self.data)))
        random.shuffle(indices)

        pairs = []
        for i in range(len(self.data)):
            possible_js = random.sample(indices, min(10, len(indices)))
            for j in possible_js:
                if i == j:
                    continue
                dist = proof_distance(self.proofs[i], self.proofs[j])
                pairs.append((i, j, dist))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        i, j, dist = self.pairs[index]

        input_i = self.encoded_statements[i]
        input_j = self.encoded_statements[j]

        if dist <= self.threshold_pos:
            label = 1
        elif dist >= self.threshold_neg:
            label = 0
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
