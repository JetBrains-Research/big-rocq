import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
import logging

from app.model import SentenceEmbedder

logger = logging.getLogger(__name__)
cfg = OmegaConf.load("config.yaml")
embedder: SentenceEmbedder = SentenceEmbedder(cfg.model_name, cfg.max_seq_length)
router = APIRouter()


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    application = FastAPI(
        title="RocqStar Ranker Service",
        description="Embed two Rocq statements via the RocqStar embeddings and return their cosine‐based distance",
        version="0.1.0",
    )
    application.include_router(router)
    return application


class DistanceRequest(BaseModel):
    statement1: str
    statement2: str


class DistanceResponse(BaseModel):
    distance: float


@router.post("/distance", response_model=DistanceResponse)
async def compute_distance(req: DistanceRequest):
    global embedder
    if embedder is None:
        logger.warning("Model not loaded")
        raise HTTPException(503, "Model not loaded; please try again later")
    dist = embedder.distance(req.statement1, req.statement2)
    logger.info(
        f"Distance `{dist:.4f}` for input lengths ({len(req.statement1)},{len(req.statement2)})"
    )
    return DistanceResponse(distance=dist)


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level="info",
        reload=False,
    )
