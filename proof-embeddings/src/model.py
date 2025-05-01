import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel
import logging

logger = logging.getLogger(__name__)


class RocqStatementEmbedder(nn.Module):
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
