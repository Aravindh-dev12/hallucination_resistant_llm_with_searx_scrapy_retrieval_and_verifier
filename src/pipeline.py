from retriever import Retriever
from reranker import Reranker
from instruct_generator import InstructGenerator
from verifier_ensemble import VerifierEnsemble
from citation_extractor import extract_citations
from utils import load_config

cfg = load_config()

class Pipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = InstructGenerator()
        self.verifier = VerifierEnsemble()
    def answer(self, query):
        docs, scores = self.retriever.retrieve(query, k=cfg['inference']['k_retrieval'])
        reranked, r_scores = self.reranker.rerank(query, docs, top_k=cfg['inference']['rerank_top_k'])
        topk = reranked[:cfg['inference']['k_retrieval']]
        context = '\n\n'.join([d['text'] for d in topk])
        gen = self.generator.generate(query, context=context, max_tokens=cfg['inference']['max_tokens'], temperature=cfg['inference']['temperature'])
        evidence_texts = [d['text'] for d in topk]
        vscore, per_ev = self.verifier.score(gen, evidence_texts)
        citations = extract_citations(gen, evidence_texts)
        result = {'query': query, 'answer': gen, 'verifier_score': vscore, 'per_evidence': per_ev, 'citations': citations, 'flagged': vscore < cfg['inference']['verifier_threshold']}
        return result
