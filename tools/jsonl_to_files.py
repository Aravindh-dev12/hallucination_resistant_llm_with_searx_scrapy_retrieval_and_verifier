# tools/jsonl_to_files.py
import sys, json, os
from pathlib import Path
def convert(jsonl_path, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                j = json.loads(line)
                text = j.get('text') or j.get('snippet') or ''
                url = j.get('url') or f'page_{i}'
                fname = Path(out_dir) / (f"{i}_{Path(url).stem}.txt")
                with open(fname, 'w', encoding='utf-8') as out:
                    out.write(text)
            except Exception:
                continue
    print('Converted jsonl to files in', out_dir)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python tools/jsonl_to_files.py input.jsonl out_dir')
    else:
        convert(sys.argv[1], sys.argv[2])
