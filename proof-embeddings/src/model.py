import torch.nn as nn
from transformers import BertModel


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
