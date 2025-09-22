# searx_search.py
import argparse, os, json
from pathlib import Path
import requests

def run_search(queries_file, out_dir='data', max_results=50, searx_url="https://searx.tiekoetter.com/search"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    urls_out = Path(out_dir) / 'searx_urls.txt'

    with open(queries_file, 'r', encoding='utf-8') as f, open(urls_out, 'w', encoding='utf-8') as out:
        for q in [l.strip() for l in f if l.strip()]:
            print('Query:', q)
            params = {
                "q": q,
                "format": "json",
                "language": "en",
                "categories": "general",
            }
            resp = requests.get(searx_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for r in data.get("results", [])[:max_results]:
                url = r.get("url") or r.get("link")
                snippet = r.get("content") or r.get("snippet") or ""
                out.write(json.dumps({"query": q, "url": url, "snippet": snippet}) + "\n")

    print("Saved URLs to", urls_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--out", default="data")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--searx-url", type=str,
                        default="https://searx.tiekoetter.com/search",
                        help="Base URL of Searx instance (must end with /search)")
    args = parser.parse_args()

    run_search(args.queries_file,
               out_dir=args.out,
               max_results=args.max_results,
               searx_url=args.searx_url)
