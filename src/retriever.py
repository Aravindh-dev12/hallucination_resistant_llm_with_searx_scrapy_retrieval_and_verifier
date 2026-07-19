import json
import os
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from retrieval_core import BM25Index, deduplicate_documents, reciprocal_rank_fusion
from utils import load_config

cfg = load_config()
INDEX = cfg["data"]["index_path"]
META = INDEX + ".meta.json"
EMB = cfg["model"]["embedding"]


class Retriever:
    """Hybrid dense + BM25 retriever with deployment-safe index validation."""

    def __init__(self):
        self.embedder = SentenceTransformer(EMB)
        expected_dimension = self.embedder.get_sentence_embedding_dimension()
        loaded = False

        if os.path.exists(INDEX) and os.path.exists(META):
            try:
                persisted_index = faiss.read_index(INDEX)
                with open(META, "r", encoding="utf-8") as handle:
                    persisted_meta = json.load(handle)
                if (
                    persisted_index.d == expected_dimension
                    and persisted_index.ntotal == len(persisted_meta)
                    and persisted_meta
                ):
                    self.index = persisted_index
                    self.meta = persisted_meta
                    loaded = True
            except (OSError, ValueError, json.JSONDecodeError):
                loaded = False

        if not loaded:
            self.meta = self._load_fallback_documents()
            self.index = self._build_index(self.meta)

        self.lexical = BM25Index(
            [str(document.get("text", "")) for document in self.meta]
        )

    def _build_index(self, documents):
        embeddings = self.embedder.encode(
            [document["text"] for document in documents],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    @staticmethod
    def _load_fallback_documents():
        documents = []
        for root in ("docs", "examples"):
            for path in Path(root).rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore").strip()
                except OSError:
                    continue
                if text:
                    documents.append(
                        {"id": str(path), "text": text[:12000], "source": str(path)}
                    )
        if not documents:
            documents.append(
                {
                    "id": "deployment-fallback",
                    "source": "built-in",
                    "text": (
                        "This system reduces hallucination through retrieval, "
                        "claim-level entailment verification, citations, and safe "
                        "abstention when evidence is insufficient."
                    ),
                }
            )
        return documents

    def retrieve(self, query, k=10):
        candidate_k = min(max(k * 4, k), len(self.meta))
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        _dense_scores, dense_ids = self.index.search(query_embedding, candidate_k)
        dense_ranking = [
            index
            for index in dense_ids[0].tolist()
            if 0 <= index < len(self.meta)
        ]
        lexical_ranking = self.lexical.rank(query, candidate_k)
        fused = reciprocal_rank_fusion(
            [dense_ranking, lexical_ranking],
            weights=cfg["retrieval"].get("fusion_weights", [1.0, 1.0]),
            rank_constant=cfg["retrieval"].get("rrf_rank_constant", 60),
        )
        ranked_documents = []
        fusion_scores = []
        for document_id, score in fused:
            ranked_documents.append(self.meta[document_id])
            fusion_scores.append(score)
        unique = deduplicate_documents(ranked_documents)
        score_by_identity = {
            id(document): score
            for document, score in zip(ranked_documents, fusion_scores)
        }
        selected = list(unique)[:k]
        return selected, [score_by_identity[id(document)] for document in selected]
