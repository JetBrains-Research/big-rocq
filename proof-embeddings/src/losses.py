import torch
import torch.nn as nn


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
