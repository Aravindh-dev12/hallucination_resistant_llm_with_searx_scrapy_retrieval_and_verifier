import json, os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import load_config

cfg = load_config()
EMB = cfg['model']['embedding']
DOCS = cfg['data']['docs_jsonl']
INDEX = cfg['data']['index_path']

def load_docs(path):
    docs=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def build():
    docs = load_docs(DOCS)
    model = SentenceTransformer(EMB)
    texts = [d['text'] for d in docs]
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, INDEX)
    with open(INDEX + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print('Index saved to', INDEX)

if __name__ == '__main__':
    build()
