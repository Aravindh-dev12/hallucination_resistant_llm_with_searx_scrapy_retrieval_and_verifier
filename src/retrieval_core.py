"""Pure hybrid-retrieval algorithms used by the FAISS retriever."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(text)) for text in documents]
        self.lengths = [sum(freq.values()) for freq in self.term_frequencies]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequency.update(frequencies.keys())

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        total_documents = len(self.documents)
        scores: list[float] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length = self.lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (total_documents - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1.0 - self.b
                    + self.b * length / max(self.average_length, 1.0)
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            scores.append(score)
        return scores

    def rank(self, query: str, limit: int) -> list[int]:
        scores = self.score(query)
        return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:limit]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    weights: Sequence[float] | None = None,
    rank_constant: int = 60,
) -> list[tuple[int, float]]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("one fusion weight is required for each ranking")
    scores: defaultdict[int, float] = defaultdict(float)
    for ranking, weight in zip(ranked_lists, weights):
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] += weight / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def deduplicate_documents(
    documents: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    seen: set[str] = set()
    unique = []
    for document in documents:
        identity = str(
            document.get("content_hash")
            or document.get("url")
            or document.get("source")
            or document.get("id")
            or document.get("text", "")
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(document)
    return unique
