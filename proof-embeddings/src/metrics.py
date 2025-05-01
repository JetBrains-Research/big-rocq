from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def compute_correlation_scores(
    distances_pred: List[float], distances_true: List[float]
) -> Tuple[float, float, float, float]:
    pearson_corr, p_pearson = pearsonr(distances_pred, distances_true)
    spearman_corr, p_spearman = spearmanr(distances_pred, distances_true)
    return pearson_corr, p_pearson, spearman_corr, p_spearman


def f_scores(
    recalls: dict[int, float],
    precisions: dict[int, float],
    k_values: List[int],
    beta: float = 1.0,
) -> dict[int, float]:
    results = {}
    for k in k_values:
        if recalls[k] + precisions[k] == 0:
            results[k] = 0.0
        else:
            results[k] = (
                (1 + beta**2)
                * (recalls[k] * precisions[k])
                / ((beta**2 * recalls[k]) + precisions[k])
            )

    return results


def recall_at_k(
    predicted_ranked_items: List[int], relevant_items: List[int], k_values: List[int]
) -> dict:
    logger.info(
        f"Relevant items: {relevant_items}, Predicted items: {predicted_ranked_items}"
    )

    results = {}

    if len(relevant_items) == 0:
        for k in k_values:
            results[k] = 0.0
        return results

    for k in k_values:
        first_k_predictions = predicted_ranked_items[:k]
        valid_recs_count = len(set(first_k_predictions).intersection(relevant_items))
        results[k] = float(valid_recs_count) / float(len(relevant_items))

    return results


def precision_at_k(
    predicted_ranked_items: List[int], relevant_items: List[int], k_values: List[int]
) -> dict:
    results = {}

    for k in k_values:
        first_k_predictions = predicted_ranked_items[:k]
        valid_recs_count = len(set(first_k_predictions).intersection(relevant_items))
        results[k] = float(valid_recs_count) / float(k)

    return results
