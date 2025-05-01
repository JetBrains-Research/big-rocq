import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from omegaconf import OmegaConf

from .model import SentenceEmbedder

logger = logging.getLogger(__name__)
router = APIRouter()


class StartResponse(BaseModel):
    status: str
    message: str

class DistanceRequest(BaseModel):
    statement1: str
    statement2: str

class DistanceResponse(BaseModel):
    distance: float

def get_config() -> object:
    return OmegaConf.load("config.yaml")

embedder: SentenceEmbedder = None

@router.get("/start", response_model=StartResponse)
async def start(config = Depends(get_config)):
    global embedder
    if embedder is None:
        embedder = SentenceEmbedder(config.model_name, config.max_seq_length)
    return StartResponse(status="ok", message="model loaded")

@router.post("/distance", response_model=DistanceResponse)
async def compute_distance(req: DistanceRequest):
    global embedder
    if embedder is None:
        logger.warning("Distance requested before /start")
        raise HTTPException(503, "Model not loaded; call GET /start first")
    dist = embedder.distance(req.statement1, req.statement2)
    logger.info(f"Distance `{dist:.4f}` for input lengths ({len(req.statement1)},{len(req.statement2)})")
    return DistanceResponse(distance=dist)

@router.post("/stop", response_model=StartResponse)
async def stop():
    logger.info("Shutdown requested via /stop")
    os._exit(0)
