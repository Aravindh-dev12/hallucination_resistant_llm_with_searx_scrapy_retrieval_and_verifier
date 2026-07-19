import json
import os
from pathlib import Path

import faiss
import requests
from sentence_transformers import SentenceTransformer

from retrieval_core import BM25Index, deduplicate_documents, reciprocal_rank_fusion
from utils import load_config

cfg = load_config()
INDEX = cfg["data"]["index_path"]
META = INDEX + ".meta.json"
EMB = cfg["model"]["embedding"]


class Retriever:
    """Hybrid local retrieval augmented with live SearxNG/Wikipedia evidence."""

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
                    self.index, self.meta = persisted_index, persisted_meta
                    loaded = True
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if not loaded:
            self.meta = self._load_fallback_documents()
            self.index = self._build_index(self.meta)
        self.lexical = BM25Index([str(item.get("text", "")) for item in self.meta])

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
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    documents.append(
                        {"id": str(path), "text": text[:12000], "source": str(path)}
                    )
        return documents or [{
            "id": "deployment-fallback",
            "source": "built-in",
            "text": (
                "This system reduces hallucination through retrieval, claim-level "
                "verification, citations, and safe abstention."
            ),
        }]

    @staticmethod
    def _searx_documents(query):
        base_url = os.getenv("SEARX_URL", "").rstrip("/")
        if not base_url:
            return []
        response = requests.get(
            f"{base_url}/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=8,
            headers={"User-Agent": "HallucinationResistantLLM/1.0"},
        )
        response.raise_for_status()
        documents = []
        for index, result in enumerate(response.json().get("results", [])[:6]):
            text = " ".join(
                part for part in (result.get("title"), result.get("content")) if part
            ).strip()
            if text:
                documents.append({
                    "id": f"searx-{index}",
                    "text": text,
                    "source": result.get("url") or "SearxNG",
                    "url": result.get("url"),
                })
        return documents

    @staticmethod
    def _wikipedia_documents(query):
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 5,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
            },
            timeout=8,
            headers={"User-Agent": "HallucinationResistantLLM/1.0"},
        )
        response.raise_for_status()
        documents = []
        pages = response.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            text = f"{page.get('title', '')}. {page.get('extract', '')}".strip()
            if len(text.split()) >= 8:
                documents.append({
                    "id": f"wikipedia-{page_id}",
                    "text": text[:6000],
                    "source": page.get("fullurl") or "Wikipedia",
                    "url": page.get("fullurl"),
                })
        return documents

    def _online_documents(self, query):
        try:
            documents = self._searx_documents(query)
        except requests.RequestException:
            documents = []
        try:
            documents.extend(self._wikipedia_documents(query))
        except requests.RequestException:
            pass
        seen, unique = set(), []
        for document in documents:
            identity = document.get("url") or document["text"][:160]
            if identity not in seen:
                seen.add(identity)
                unique.append(document)
        return unique

    def retrieve(self, query, k=10):
        candidate_k = min(max(k * 4, k), len(self.meta))
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        _dense_scores, dense_ids = self.index.search(query_embedding, candidate_k)
        dense_ranking = [
            index for index in dense_ids[0].tolist() if 0 <= index < len(self.meta)
        ]
        lexical_ranking = self.lexical.rank(query, candidate_k)
        fused = reciprocal_rank_fusion(
            [dense_ranking, lexical_ranking],
            weights=cfg["retrieval"].get("fusion_weights", [1.0, 1.0]),
            rank_constant=cfg["retrieval"].get("rrf_rank_constant", 60),
        )
        local_documents = [self.meta[document_id] for document_id, _score in fused]
        local_scores = [float(score) for _document_id, score in fused]
        selected = list(deduplicate_documents(local_documents))[:k]
        score_by_id = {
            id(document): score
            for document, score in zip(local_documents, local_scores)
        }
        scores = [score_by_id.get(id(document), 0.0) for document in selected]
        online = self._online_documents(query)
        return online + selected, [1.0] * len(online) + scores
