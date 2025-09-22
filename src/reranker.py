from sentence_transformers import CrossEncoder
from utils import load_config

cfg = load_config()
MODEL = cfg['model']['cross_encoder']

class Reranker:
    def __init__(self, model_name=MODEL):
        self.model = CrossEncoder(model_name)
    def rerank(self, query, docs, top_k=10):
        pairs = [(query, d['text']) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in ranked[:top_k]], [r[1] for r in ranked[:top_k]]
