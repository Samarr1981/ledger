"""Pure-Python BM25 (Okapi). No numpy, no third-party deps.

Matches the algorithm and defaults of rank_bm25.BM25Okapi so retrieval
results are unchanged by the swap.
"""

from __future__ import annotations

import math

K1 = 1.5
B = 0.75
EPSILON = 0.25


class BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = K1, b: float = B, epsilon: float = EPSILON):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size

        self.doc_freqs: list[dict[str, int]] = []
        doc_counts: dict[str, int] = {}
        for doc in corpus:
            freqs: dict[str, int] = {}
            for word in doc:
                freqs[word] = freqs.get(word, 0) + 1
            self.doc_freqs.append(freqs)
            for word in freqs:
                doc_counts[word] = doc_counts.get(word, 0) + 1

        self.idf: dict[str, float] = {}
        self._calc_idf(doc_counts)

    def _calc_idf(self, doc_counts: dict[str, int]) -> None:
        idf_sum = 0.0
        negative_idfs = []
        for word, freq in doc_counts.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        average_idf = idf_sum / len(self.idf)
        eps = self.epsilon * average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores
