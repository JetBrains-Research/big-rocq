import logging
import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from .api import router


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    application = FastAPI(
        title="Proof‐Similarity Service",
        description="Embed two statements via RoBERTa and return their cosine‐based distance",
        version="0.1.0",
    )
    application.include_router(router)
    return application


app = create_app()

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    uvicorn.run(
        "app.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level="info",
        reload=False,
    )
