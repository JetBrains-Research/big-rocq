from Levenshtein import distance
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
import numpy as np


def split_tactics(proof_text: str) -> list[str]:
    text = proof_text.replace('\n', ';')
    raw = re.split(r'[.;]+', text)
    return [s.strip() for s in raw if s.strip()]


def statement_distance(s1: str, s2: str, vectorizer) -> float:
    """
    1 - cosine( tfidf(s1), tfidf(s2) ), in [0,1].
    """
    v1 = vectorizer.transform([s1])
    v2 = vectorizer.transform([s2])
    sim = cosine_similarity(v1, v2)[0, 0]
    return 1.0 - float(sim)


def tactics_jaccard(s1: str, s2: str) -> float:
    t1 = set(split_tactics(s1))
    t2 = set(split_tactics(s2))
    if not t1 and not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return 1.0 - (inter / union)


def normalized_string_distance(s1: str, s2: str) -> float:
    """
    distance(s1, s2) / max(len(s1), len(s2)).
    """
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0

    return distance(s1, s2) / float(max_len)


def add_small_noize(dist: float) -> float:
    max_noize = 0.05
    min_dist = 0.001

    if dist < min_dist:
        noise = random.uniform(0.0, max_noize)
        return noise

    return dist


def jitter(dist: float, eps: float = 1e-3) -> float:
    return float(np.clip(dist + random.uniform(-eps, +eps), 0.0, 1.0))


def proof_distance(proof1: str, proof2: str, stmt1: str, stmt2: str, vectorizer) -> float:
    """
    Levenshtein distance at the sequence level, but:
     - cost of insertion = 1
     - cost of deletion = 1
     - cost of substitution = normalized_string_distance(sentence1, sentence2)

    We return the normalized distance: distance(proof1, proof2) / max(len(proof1), len(proof2)).
    """

    sents1 = split_tactics(proof1)
    sents2 = split_tactics(proof2)
    n1, n2 = len(sents1), len(sents2)

    max_len = max(n1, n2)

    if n1 == 0 or n2 == 0:
        return 1.0

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

    proof_dist = dp[n1][n2] / float(max_len)

    stmt_dist = statement_distance(stmt1, stmt2, vectorizer)

    tac_dist = tactics_jaccard(proof1, proof2)

    a, b, g = 0.7, 0.15, 0.15
    composite = a * proof_dist + b * stmt_dist + g * tac_dist
    composite = jitter(composite, eps=1e-3)

    rounded_dist = round(float(np.clip(composite, 0.0, 1.0)), 5)

    return rounded_dist
