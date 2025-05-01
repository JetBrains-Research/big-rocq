import json
import logging
import os
from typing import List, Dict, Any, Tuple
import random

from numpy import random

logger = logging.getLogger(__name__)


"""
Recursively load all JSON files in a directory and return
the concatenated dataset.
"""


def load_dataset(
    root: str, min_samples_in_file: int
) -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    dataset = {}
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                file_ds = load_single_file_json_dataset(file_path)

                if len(file_ds) >= min_samples_in_file:
                    dataset[f"{root}/{file}"] = file_ds
                else:
                    logger.warning(
                        f"File {file_path} has too little samples: {len(file_ds)}"
                    )

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


def split_dataset(
    data: Dict[str, List[Dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    remove: List[str] = None,
) -> Tuple[
    List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]
]:
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-7
    ), "Ratios must sum to 1."

    files_count = len(data)
    train_size = int(files_count * train_ratio)
    val_size = int(files_count * val_ratio)
    test_size = files_count - train_size - val_size

    logger.info(
        f"Splitting dataset into {train_size} training samples, {val_size} validation samples, and {test_size} testing samples"
    )
    if (train_size <= 1) or (val_size <= 1) or (test_size <= 1 and test_ratio != 0):
        logger.error(
            "One of the splits is <= 1. Check the ratios. Length minimum 2 is required to compute correlations."
        )

    for remove_file in remove:
        if data.get(remove_file):
            data.pop(remove_file)

    data_list = list(data.values())
    random.shuffle(data_list)

    train_data = data_list[:train_size]
    val_data = data_list[train_size : train_size + val_size]
    test_data = data_list[train_size + val_size :]
    return train_data, val_data, test_data
