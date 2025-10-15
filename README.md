

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



## Check the Article 
Comprehensive Hallucination-Resistant Transformer Architecture with Advanced Neural Verification, Multi-Question Batch Processing, and Cloud Deployment Creators
Description
This work presents a hallucination-resistant large language model (LLM) system designed for robust, accurate, and user-friendly AI applications. The system integrates Retrieval-Augmented Generation (RAG) with instruction-tuned LLMs refined via LoRA fine-tuning. To ensure factual correctness, an entailment-based verifier checks generated outputs against retrieved evidence.Our architecture includes an external knowledge index using a vector database with dense passage retrieval, grounding the LLM in verified information. The system supports voice input through a high-accuracy ASR model and interacts with users via a flexible chat interface, making it suitable for both text-based and voice-driven applications.All components are containerized using Docker, with microservices for retrieval, generation, and verification managed by Kubernetes. This ensures autoscaling, fault tolerance, and smooth cloud deployment. The modular design allows easy updates of retrieval modules, LLMs, or verification models without major architectural changes.
Extensive evaluations across multi-domain, multi-hop, and domain-specific tasks demonstrate improvements in factual accuracy, response consistency, and reduction of hallucinations, while maintaining fluency and context awareness. Performance analysis shows that the microservice-based deployment efficiently handles high-concurrency scenarios.This framework is suitable for high-stakes applications such as healthcare, legal, education, customer support, and enterprise knowledge management. Future work will focus on adaptive retrieval, continual learning for domain adaptation, and advanced verification models to further enhance reliability and trustworthiness.

https://zenodo.org/records/17212730?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgwNDQ5ODVmLWE4ZDQtNDg4MC1iMzZhLTQzZTE3MWQ1YzA0NCIsImRhdGEiOnt9LCJyYW5kb20iOiI2NTZjNGY1MzY1ZDQ4MTlmNzAzNDJkNWE0MDM4OGM5OCJ9.qWC7FkZHcbdswintggucmUJpClznv4nbdoPHGHTe4IyCBAcKx9nXvp-gYlZu9K-qFwTfvVWTt9Wh-2AmqAsDyg
