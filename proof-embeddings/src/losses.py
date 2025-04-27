import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    """
    L = y * 1/2 * d^2 + (1-y) * 1/2 * max(0, margin - d)^2,
        where d - Euclidean distance between embeddings
    y=1 for positive, y=0 for negative.
    """
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, emb_i, emb_j, labels):
        distances = torch.norm(emb_i - emb_j, p=2, dim=1)
        pos_term = labels * 0.5 * distances.pow(2)
        neg_term = (1 - labels) * 0.5 * torch.clamp(self.margin - distances, min=0).pow(2)
        loss = pos_term + neg_term
        return loss.mean()


class TripletMarginLoss(nn.Module):
    """
    dist(a,p) + margin < dist(a,n)
    """
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


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

        positive_scores = (anchor * positive).sum(dim=-1, keepdim=True)  # [B,1]
        negative_scores = torch.einsum('bd,bnd->bn', anchor, negatives)  # [B,num_negatives]

        logits = torch.cat([positive_scores, negative_scores], dim=1)  # [B, 1 + num_negatives]

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits / self.temperature, labels)

        return loss
