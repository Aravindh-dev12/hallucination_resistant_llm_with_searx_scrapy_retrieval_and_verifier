Hallucination-Resistant LLM — Searx + Scrapy Production Package

This package implements an advanced RAG + Verification LLM pipeline with a fully open-source large-scale web search pipeline based on Searx and Scrapy.

Quick start:
1. Create venv and install requirements:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. (Optional) Run your own Searx instance or set SEARX_URL in environment.

3. Search & Crawl:
   python searx_search.py --queries-file examples/queries.txt --out data --max-results 50
   scrapy runspider searx_spider.py -a urls_file=data/searx_urls.txt -o data/raw_pages.jsonl
   python tools/jsonl_to_files.py data/raw_pages.jsonl data/sources

4. Process & index:
   python src/data_prep.py
   python src/build_index.py

5. Run UI:
   python src/ui.py

See docs for full instructions and production deployment (Docker + Kubernetes).
