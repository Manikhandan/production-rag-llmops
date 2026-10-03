from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from ragops.types import Chunk, Hit

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HybridIndex:
    """TF-IDF vectors + BM25-style lexical scores, fused with RRF."""

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60) -> None:
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.docs_tokens = [tokenize(c.text) for c in chunks]
        self.df: dict[str, int] = defaultdict(int)
        for tokens in self.docs_tokens:
            for term in set(tokens):
                self.df[term] += 1
        self.n = len(chunks)
        self.avgdl = sum(len(t) for t in self.docs_tokens) / self.n if self.n else 1.0
        vocab = sorted(self.df)
        self.vocab = {term: i for i, term in enumerate(vocab)}
        self.vectors = [self._tfidf(tokens) for tokens in self.docs_tokens]

    def _tfidf(self, tokens: list[str]) -> dict[int, float]:
        counts = Counter(tokens)
        vec = {}
        length = 0.0
        for term, tf in counts.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue
            idf = math.log((self.n + 1) / (self.df[term] + 1)) + 1
            weight = (tf / len(tokens)) * idf if tokens else 0.0
            vec[idx] = weight
            length += weight * weight
        norm = math.sqrt(length) or 1.0
        return {i: w / norm for i, w in vec.items()}

    def _cosine(self, query: dict[int, float], doc: dict[int, float]) -> float:
        if len(query) > len(doc):
            query, doc = doc, query
        return sum(weight * doc.get(idx, 0.0) for idx, weight in query.items())

    def _bm25(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        k1, b = 1.5, 0.75
        counts = Counter(doc_tokens)
        score = 0.0
        dl = len(doc_tokens) or 1
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((self.n - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            denom = tf + k1 * (1 - b + b * dl / self.avgdl)
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        q_tokens = tokenize(query)
        q_vec = self._tfidf(q_tokens)
        vector_ranked = sorted(
            range(self.n), key=lambda i: self._cosine(q_vec, self.vectors[i]), reverse=True
        )
        lexical_ranked = sorted(
            range(self.n), key=lambda i: self._bm25(q_tokens, self.docs_tokens[i]), reverse=True
        )
        fused: dict[int, float] = defaultdict(float)
        for rank, idx in enumerate(vector_ranked, start=1):
            fused[idx] += 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(lexical_ranked, start=1):
            fused[idx] += 1.0 / (self.rrf_k + rank)
        ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
        hits = []
        for rank, idx in enumerate(ordered, start=1):
            lexical = self._bm25(q_tokens, self.docs_tokens[idx])
            vector = self._cosine(q_vec, self.vectors[idx])
            if lexical <= 0 and vector <= 0:
                continue
            hits.append(Hit(chunk=self.chunks[idx], score=fused[idx], rank=rank, method="rrf"))
        return hits
