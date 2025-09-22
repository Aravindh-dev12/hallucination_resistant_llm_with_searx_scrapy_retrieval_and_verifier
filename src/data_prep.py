import os, glob, re
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path
from utils import save_jsonl, load_config

cfg = load_config()
RAW = cfg['data']['raw_dir']
OUT = cfg['data']['docs_jsonl']

def clean(text):
    if '<html' in text.lower():
        soup = BeautifulSoup(text, 'html.parser')
        t = soup.get_text(separator='\n')
    else:
        t = text
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def chunk(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i=0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

def process():
    Path(cfg['data']['processed_dir']).mkdir(parents=True, exist_ok=True)
    docs=[]
    files = glob.glob(os.path.join(RAW, '*'))
    for path in tqdm(files):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()
        except Exception:
            continue
        text = clean(raw)
        for idx, c in enumerate(chunk(text)):
            docs.append({'id': f"{os.path.basename(path)}_{idx}", 'text': c, 'source': path})
    save_jsonl(docs, OUT)
    print('Saved', len(docs), 'passages to', OUT)

if __name__ == '__main__':
    process()
