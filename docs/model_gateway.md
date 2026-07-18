# Runtime model gateway

The runtime gateway supports two explicit adapter types:

- `transformers`: lazy local Hugging Face loading
- `openai_compatible`: vLLM, SGLang, Ollama proxies or compatible servers

Candidate coding models are disabled by default in
`config/model_backends.yaml`. Enabling a runtime endpoint does not admit its
records into training and does not authorize checkpoint promotion.

For VibeThinker, host a pinned checkpoint behind a local compatible endpoint,
replace placeholder digests in the governed training manifest, complete
provenance and license review, and obtain independent verifier receipts.
ThinkCoder must use an exact, reviewed model identifier before it can be
enabled.

The gateway intentionally contains no checkpoint-promotion method. Promotion
remains exclusively behind `KernelGate`.

Deployment to Hugging Face is handled by the repository workflow and updates
only the existing governed Space. Deployment credentials are supplied only
through the GitHub Actions repository secret.
