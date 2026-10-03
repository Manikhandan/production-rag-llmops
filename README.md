# Production RAG / LLMOps

A retrieval system with a generator behind it, not a notebook that dumps a prompt into an API.

Documents are chunked, indexed with TF-IDF and BM25, fused with reciprocal rank fusion, then passed to a versioned prompt. The default generator is extractive so tests and CI do not need vendor keys. Citations, a groundedness check, token estimates, a small safety filter, and an in-memory cache sit on the API.

## Why this architecture

- **Retrieval and generation are separate modules.** Swapping the provider does not rewrite chunking.
- **Hybrid retrieval is justified here.** Operational docs mix exact identifiers ("hard decline") with paraphrases. Lexical and vector ranks disagree; RRF is a cheap fusion.
- **Extractive provider in CI.** A billed LLM would hide retrieval bugs behind fluent answers. `LLMProvider` is a Protocol: `extractive` (default) or `http` (needs `RAG_LLM_URL` and `RAG_LLM_API_KEY`). The HTTP adapter is a contract; this repository does not execute billed calls.
- **Fail closed on an empty index.** `/ready` reports chunk count.

```
markdown corpus
    → normalize / chunk (overlap)
    → HybridIndex (TF-IDF cosine + BM25 → RRF)
    → prompt template (versioned)
    → generator (extractive default; http adapter is env-configured, not called in CI)
    → citations + groundedness + token estimate
    → FastAPI /v1/ask
```

## Local setup

```bash
pip install -e ".[dev]"
cp .env.example .env
uvicorn ragops.serving:app --port 8000
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question":"How should hard declines be retried?"}'
```

## Tests

```bash
ruff check src tests
pytest -q
```

## Security

- No API keys in the default provider.
- Ask text is length-limited and blocklisted for a couple of injection phrases. That is not a complete safety program.
- Rate limit is a sliding window per client address.

## Trade-offs

- In-process index. Fine for a documentation corpus; not a replacement for a managed vector database at larger scale.
- Whitespace token estimates, not a billed tokenizer.
- No conversation state. Each question is independent so cache keys stay simple.

## What I would improve next

- Persistent index on disk with a content hash of the corpus.
- Wire HttpLLMProvider to a real client with timeouts, retries, and cost counters. This repository does not execute those calls.
- A labeled eval set with recall@k rather than term overlap.
