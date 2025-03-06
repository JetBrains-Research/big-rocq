import json
import itertools
import logging
from typing import List
from bm25 import bm25, compute_doc_freqs
import os
from tqdm import tqdm
from Levenshtein import distance
from functools import lru_cache

from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_all_json_data(folder_path: str) -> List[dict]:
    all_data = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Loading {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)

    logger.info(f"Aggregated {len(all_data)} total entries from folder '{folder_path}'.")
    return all_data


def split_proof_into_sentences(proof_text: str) -> List[str]:
    proof_text = proof_text.strip()
    sentences = proof_text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def normalized_string_distance(s1: str, s2: str) -> float:
    """
    Normalized Levenshtein: distance(s1, s2) / max(len(s1), len(s2)).
    """
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0

    return distance(s1, s2) / float(max_len)


@lru_cache(None)
def proof_distance(proof1: str, proof2: str) -> float:
    """
    We treat the proofs as sequences of sentences. Then we apply
    a Levenshtein distance at the sequence level. But:
     - cost of insertion = 1
     - cost of deletion = 1
     - cost of substitution = normalized_string_distance(sentence1, sentence2)
    """

    sents1 = split_proof_into_sentences(proof1)
    sents2 = split_proof_into_sentences(proof2)
    n1, n2 = len(sents1), len(sents2)

    if n1 == 0:
        return float(n2)
    if n2 == 0:
        return float(n1)

    dp = [[0.0]*(n2+1) for _ in range(n1+1)]
    
    for i in range(n1+1):
        dp[i][0] = float(i)
    for j in range(n2+1):
        dp[0][j] = float(j)

    for i in range(1, n1+1):
        for j in range(1, n2+1):
            cost_sub = normalized_string_distance(sents1[i-1], sents2[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1.0,
                dp[i][j-1] + 1.0,
                dp[i-1][j-1] + cost_sub
            )

    return dp[n1][n2]


def normalized_proof_distance(proof1: str, proof2: str) -> float:
    sents1 = split_proof_into_sentences(proof1)
    sents2 = split_proof_into_sentences(proof2)
    n1, n2 = len(sents1), len(sents2)
    max_sents = max(n1, n2)
    if max_sents == 0:
        return 0.0
    return proof_distance(proof1, proof2) / float(max_sents)


def tokenize_statement(stmt: str) -> List[str]:
    return stmt.lower().split()


def build_bm25_matrix(statements: List[List[str]]) -> List[List[float]]:
    """
    Build NxN matrix of BM25 "similarities" for each pair of statements.
      sim(i,j) = 0.5 * [bm25(statements[i], [statements[j]]) 
                        + bm25(statements[j], [statements[i]])]
    To make it symmetric, we average the two BM25 scores.
    """
    n = len(statements)
    if n == 0:
        return []
    
    matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
            elif j > i:
                # We'll treat statements[i] as query vs statements[j] as doc
                # and vice versa, then average
                sim_ij_one_dir = bm25(statements[i], [statements[j]], doc_freqs=compute_doc_freqs([statements[j]]))[0]
                sim_ij_other_dir = bm25(statements[j], [statements[i]], doc_freqs=compute_doc_freqs([statements[i]]))[0]
                sim_ij = 0.5 * (sim_ij_one_dir + sim_ij_other_dir)
                matrix[i][j] = sim_ij
                matrix[j][i] = sim_ij
    return matrix


def main():
    dataset_path = "perfectCorrelation"
    data = load_all_json_data(dataset_path)

    statements_raw = [d["statement"] for d in data]
    proofs = [d["proof"] for d in data]

    tokenized_statements = [tokenize_statement(s) for s in statements_raw]

    logger.info("Computing pairwise BM25 similarities for statements...")
    bm25_matrix = build_bm25_matrix(tokenized_statements)

    n = len(proofs)
    logger.info("Computing pairwise proof distances...")
    proof_dist_matrix = [[0.0]*n for _ in range(n)]
    total_pairs = n*(n-1)//2
    with tqdm(total=total_pairs, desc="Computing pairwise proof distances") as pbar:
        for i in range(n):
            for j in range(i+1, n):
                dist_ij = normalized_proof_distance(proofs[i], proofs[j])
                proof_dist_matrix[i][j] = dist_ij
                proof_dist_matrix[j][i] = dist_ij
                pbar.update(1)

    def distance_to_similarity(d: float) -> float:
        return 1.0 / (1.0 + d)

    statement_sims = []
    proof_sims = []
    for i, j in itertools.combinations(range(n), 2):
        stmt_sim = bm25_matrix[i][j]
        p_dist = proof_dist_matrix[i][j]
        p_sim = distance_to_similarity(p_dist)

        statement_sims.append(stmt_sim)
        proof_sims.append(p_sim)

    pearson_corr, pval_pearson = pearsonr(statement_sims, proof_sims)
    spearman_corr, pval_spearman = spearmanr(statement_sims, proof_sims)

    logger.info(f"Pearson Correlation = {pearson_corr:.4f}, p-value = {pval_pearson:.4g}")
    logger.info(f"Spearman Correlation = {spearman_corr:.4f}, p-value = {pval_spearman:.4g}")

    print("Pearson correlation:", pearson_corr, ", p-value =", pval_pearson)
    print("Spearman correlation:", spearman_corr, ", p-value =", pval_spearman)

if __name__ == "__main__":
    main()

# Results on 1927 IMM theorems (1855701 pairs): 
# Bm25(stetements) correlation with Levenshtein(proofs) distance:
# Pearson correlation: -0.15425089416981363 , p-value = 0.0
# Spearman correlation: -0.17093627617989743 , p-value = 0.0
# Bm25(proofs) correlation with Bm25(proofs) distance:
# Pearson correlation =  0.0293  p-value = 0.0
# Spearman correlation = 0.23972122829525572  p-value = 0.0
# Both results are to be considered as negligible correlation.
