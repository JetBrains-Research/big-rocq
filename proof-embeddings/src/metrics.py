from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple


def compute_correlation_scores(
    distances_pred: List[float],
    distances_true: List[float]
) -> Tuple[float, float, float, float]:
    pearson_corr, p_pearson = pearsonr(distances_pred, distances_true)
    spearman_corr, p_spearman = spearmanr(distances_pred, distances_true)
    return pearson_corr, p_pearson, spearman_corr, p_spearman


def recall_at_k(
    query2truth: dict,
    query2preds: dict,
    k_values: List[int]
) -> dict:
    results = {}
    n = len(query2truth)

    if n == 0:
        return {k: 0.0 for k in k_values}

    for k in k_values:
        correct = 0
        for q_idx, relevant_set in query2truth.items():
            preds = query2preds[q_idx][:k]
            if len(set(preds).intersection(relevant_set)) > 0:
                correct += 1
        results[k] = float(correct) / float(n)
    return results
