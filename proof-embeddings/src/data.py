import json
import logging
import os
from typing import List, Dict, Any


logger = logging.getLogger(__name__)


def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    logger.info(f"Loading dataset from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records from {path}")
    return data


def split_dataset(data: List[Dict[str, Any]], train_ratio: float, val_ratio: float, test_ratio: float):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-7, "Ratios must sum to 1."

    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size
    
    logger.info(f"Splitting dataset into {train_size} training samples, {val_size} validation samples, and {test_size} testing samples")
    if (train_size <= 1) or (val_size <= 1) or (test_size <= 1):
        logger.error("One of the splits is <= 1. Check the ratios. Length minimum 2 is required to compute correlations.")

    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    return train_data, val_data, test_data
