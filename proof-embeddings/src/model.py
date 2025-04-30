import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel
import logging

logger = logging.getLogger(__name__)


class BERTStatementEmbedder(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int = 768):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(p=0.1)

        bert_hidden_size = self.bert.config.hidden_size
        if bert_hidden_size != embedding_dim:
            self.projection = nn.Linear(bert_hidden_size, embedding_dim)
        else:
            self.projection = None

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :]

        cls_embed = self.dropout(cls_embed)

        if self.projection is not None:
            cls_embed = self.projection(cls_embed)
        return cls_embed


class DistanceClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, emb_x, emb_y):
        combined_emb = torch.cat([emb_x, emb_y], dim=1)
        logit = self.classifier(combined_emb)
        return logit


class LightweightTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_layers: int = 1, num_heads: int = 4,
                 dim_feedforward: int = None, max_len: int = 128):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        assert input_ids.max().item() < self.embed.num_embeddings, \
            f"Found token index {input_ids.max().item()} >= vocab size {self.embed.num_embeddings}"

        B, L = input_ids.shape
        token_embeds = self.embed(input_ids)

        pos_indices = torch.arange(0, L, device=input_ids.device,
                                   dtype=torch.long) % self.positional_embed.num_embeddings
        pos_embeds = self.positional_embed(pos_indices)
        x = token_embeds + pos_embeds

        if attention_mask is not None:
            pad_mask = attention_mask == 0
        else:
            pad_mask = None

        encoded = self.encoder(x, src_key_padding_mask=pad_mask)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).type_as(encoded)
            masked_enc = encoded * mask
            sum_enc = masked_enc.sum(dim=1)
            lengths = mask.sum(dim=1)
            lengths = torch.clamp(lengths, min=1e-9)
            sentence_embeds = sum_enc / lengths
        else:
            sentence_embeds = encoded.mean(dim=1)

        return sentence_embeds
