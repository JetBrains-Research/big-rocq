import logging
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

logger = logging.getLogger(__name__)

class SentenceEmbedder:
    def __init__(self, model_name: str, max_seq_length: int = 128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seq_length = max_seq_length
        logger.info(f"Loading model `{model_name}` on {self.device}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        ).to(self.device)
        out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def distance(self, a: str, b: str) -> float:
        e1 = self.embed(a)
        e2 = self.embed(b)

        sim = F.cosine_similarity(e1, e2).item()
        return float(1.0 - sim)
