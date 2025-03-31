import json
import logging
import os
from typing import List, Dict, Any


logger = logging.getLogger(__name__)


"""
Recursively load all JSON files in a directory and return
the concatenated dataset.
"""
def load_dataset(root: str) -> List[Dict[str, Any]]:
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    
    dataset = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(".json"):
                dataset += load_single_file_json_dataset(os.path.join(root, file))
    
    return dataset

def load_single_file_json_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    logger.info(f"Loading dataset from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "fileSamples" not in data or "filePath" not in data:
        logger.error(f"Invalid JSON file: {path}")
        return []

    dataset = data["fileSamples"]
    logger.info(f"Loaded {len(dataset)} records from {data['filePath']}")
    return dataset


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
