import json
import os

import faiss
from sentence_transformers import SentenceTransformer

from retrieval_core import BM25Index, deduplicate_documents, reciprocal_rank_fusion
from utils import load_config

cfg = load_config()
INDEX = cfg["data"]["index_path"]
META = INDEX + ".meta.json"
EMB = cfg["model"]["embedding"]


class Retriever:
    """Hybrid dense + BM25 retriever with reciprocal-rank fusion."""

    def __init__(self):
        if not os.path.exists(INDEX):
            raise FileNotFoundError(f"Index not found at {INDEX}. Build it first.")
        self.index = faiss.read_index(INDEX)
        with open(META, "r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.embedder = SentenceTransformer(EMB)
        self.lexical = BM25Index(
            [str(document.get("text", "")) for document in self.meta]
        )

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
