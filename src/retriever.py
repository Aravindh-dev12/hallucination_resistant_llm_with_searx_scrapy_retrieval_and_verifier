import faiss, json, os
from sentence_transformers import SentenceTransformer
from utils import load_config

cfg = load_config()
INDEX = cfg['data']['index_path']
META = INDEX + '.meta.json'
EMB = cfg['model']['embedding']

class Retriever:
    def __init__(self):
        if not os.path.exists(INDEX):
            raise FileNotFoundError(f'Index not found at {INDEX}. Build it first.')
        self.index = faiss.read_index(INDEX)
        with open(META, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.embedder = SentenceTransformer(EMB)
    def retrieve(self, query, k=10):
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, ids = self.index.search(q, k)
        ids = ids[0].tolist()
        docs = [self.meta[i] for i in ids if i < len(self.meta)]
        return docs, scores[0].tolist()
