from retrieval_core import BM25Index, deduplicate_documents, reciprocal_rank_fusion


def test_bm25_recovers_exact_rare_term():
    index = BM25Index(["general language model", "CVE-2026-1234 mitigation"])
    assert index.rank("CVE-2026-1234", 1) == [1]


def test_rank_fusion_combines_dense_and_lexical_results():
    fused = reciprocal_rank_fusion([[0, 1], [1, 2]], weights=[1.0, 1.0])
    assert fused[0][0] == 1


def test_duplicate_sources_are_removed():
    documents = [
        {"url": "https://example.com/a", "text": "one"},
        {"url": "https://example.com/a", "text": "duplicate"},
    ]
    assert len(deduplicate_documents(documents)) == 1
