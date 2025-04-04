import torch
import torch.nn as nn
from transformers import BertModel
import logging

logger = logging.getLogger(__name__)


class BERTStatementEmbedder(nn.Module):
    def __init__(self, model_name: str, freeze_bert: bool = False, embedding_dim: int = 768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        if bert_hidden_size != embedding_dim:
            self.projection = nn.Linear(bert_hidden_size, embedding_dim)
        else:
            self.projection = None

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        if self.projection is not None:
            cls_embed = self.projection(cls_embed)
        return cls_embed


class RulerModel(nn.Module):
    def __init__(self, model_name: str, freeze_bert: bool = False, embedding_dim: int = 768):
        super().__init__()
        self.bert = BERTStatementEmbedder(model_name, freeze_bert, embedding_dim)
        self.distance_layer = nn.Linear(2 * embedding_dim, 1)

    def forward(self, input_ids_x, attention_mask_x, input_ids_y, attention_mask_y):
        emb_x = self.bert(input_ids_x, attention_mask_x)
        emb_y = self.bert(input_ids_y, attention_mask_y)

        combined_emb = torch.cat((emb_x, emb_y), dim=1)
        distance = self.distance_layer(combined_emb)

        return distance