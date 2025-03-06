from Levenshtein import distance
from typing import List


def split_proof_into_sentences(proof_text: str) -> List[str]:
    proof_text = proof_text.strip()
    sentences = proof_text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


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


def proof_distance(proof1: str, proof2: str) -> float:
    """
    Levenshtein distance at the sequence level, but:
     - cost of insertion = 1
     - cost of deletion = 1
     - cost of substitution = normalized_string_distance(sentence1, sentence2)

    We return the normalized distance: distance(proof1, proof2) / max(len(proof1), len(proof2)).
    """

    sents1 = split_proof_into_sentences(proof1)
    sents2 = split_proof_into_sentences(proof2)
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

    return dp[n1][n2] / float(max_len)
