from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_correlation_scores(
    distances_pred: List[float],
    distances_true: List[float]
) -> Tuple[float, float, float, float]:    
    pearson_corr, p_pearson = pearsonr(distances_pred, distances_true)
    spearman_corr, p_spearman = spearmanr(distances_pred, distances_true)
    return pearson_corr, p_pearson, spearman_corr, p_spearman


def f_scores(
    recalls: dict[int, float],
    precisions: dict[int, float],
    k_values: List[int],
    beta: float = 1.0
) -> dict[int, float]:
    results = {}
    for k in k_values:
        if recalls[k] + precisions[k] == 0:
            results[k] = 0.0
        else:
            results[k] = (1 + beta**2) * (recalls[k] * precisions[k]) / ((beta**2 * recalls[k]) + precisions[k])

    return results


def recall_at_k(
    query2truth: dict,
    query2predictions: dict,
    k_values: List[int]
) -> dict:
    # logger.info(f"Truth: {query2truth}, Predictions: {query2predictions}")
    results = {}
    anchor_count = len(query2truth)

    if anchor_count == 0:
        return {k: 0.0 for k in k_values}

    for k in k_values:
        recall_sum_all_anchors = 0
        empty_recs_count = 0
        for anchor_id, relevant_set in query2truth.items():
            first_k_predictions = query2predictions[anchor_id][:k]
            valid_recs_count = len(set(first_k_predictions).intersection(relevant_set))
            recall_for_anchor = float(valid_recs_count) / float(len(relevant_set)) if len(relevant_set) > 0 else 0.0
            if len(relevant_set) == 0:
                empty_recs_count += 1

            recall_sum_all_anchors += recall_for_anchor

        # Average recall across all anchors
        results[k] = float(recall_sum_all_anchors) / float(anchor_count - empty_recs_count) if anchor_count - empty_recs_count > 0 else 0.0

    return results


def precision_at_k(
    query2truth: dict,
    query2predictions: dict,
    k_values: List[int]
) -> dict:
    results = {}
    anchor_count = len(query2truth)

    if anchor_count == 0:
        return {k: 0.0 for k in k_values}

    for k in k_values:
        precision_sum_all_anchors = 0
        for anchor_id, relevant_set in query2truth.items():
            first_k_predictions = query2predictions[anchor_id][:k]
            valid_recs_count = len(set(first_k_predictions).intersection(relevant_set))
            precision_for_anchor = float(valid_recs_count) / float(k)

            precision_sum_all_anchors += precision_for_anchor

        # Average precision across all anchors
        results[k] = float(precision_sum_all_anchors) / float(anchor_count)

    return results