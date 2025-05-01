import torch
import torch.nn.functional as F
from torch import nn


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        anchor: [batch_size, embedding_dim]
        positive: [batch_size, embedding_dim]
        negatives: [batch_size, num_negatives, embedding_dim]
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        positive_scores = (anchor * positive).sum(dim=-1, keepdim=True)
        negative_scores = torch.einsum("bd,bnd->bn", anchor, negatives)

        logits = torch.cat([positive_scores, negative_scores], dim=1)

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits / self.temperature, labels)

        return loss
