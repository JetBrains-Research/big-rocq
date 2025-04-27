from Levenshtein import distance
from typing import List
import random
import re


def split_tactics(proof_text: str) -> list[str]:
    # first normalize newlines to semicolons
    text = proof_text.replace('\n', ';')
    # split on any run of '.' or ';'
    raw = re.split(r'[.;]+', text)
    # strip and filter
    return [s.strip() for s in raw if s.strip()]


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


def jitter(dist: float, eps: float=0.02) -> float:
    return min(1.0, max(0.0, dist + random.uniform(-eps, eps)))


# TODO: When distance = 0 add random noise to the distance
def proof_distance(proof1: str, proof2: str) -> float:
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
    return jitter(proof_dist, eps=0.02)

    # return proof_dist
