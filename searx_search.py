import argparse, json, time
import requests
from pathlib import Path

def run_search(queries_file, out_dir="data", max_results=10, api_key="", cx=""):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    urls_out = Path(out_dir) / "searx_urls.txt"

    with open(queries_file, "r", encoding="utf-8") as f, open(urls_out, "w", encoding="utf-8") as out:
        for q in [l.strip() for l in f if l.strip()]:
            print("Query:", q)
            params = {"key": api_key, "cx": cx, "q": q, "num": max_results}
            try:
                resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=30)
                resp.raise_for_status()
                results = resp.json().get("items", [])
                for r in results:
                    out.write(json.dumps({
                        "query": q,
                        "url": r.get("link"),
                        "snippet": r.get("snippet", "")
                    }) + "\n")
                time.sleep(1)  # small delay to avoid quota issues
            except Exception as e:
                print("Error:", e)
    print("Saved URLs to", urls_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--out", default="data")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--api-key", required=True, help="Google API Key")
    parser.add_argument("--cx", required=True, help="Custom Search Engine ID")
    args = parser.parse_args()

    run_search(args.queries_file, args.out, args.max_results, args.api_key, args.cx)
