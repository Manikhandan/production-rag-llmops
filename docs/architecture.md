# Architecture

Clients hit FastAPI. The process holds a HybridIndex built at startup from `corpus/`.
Retrieval, prompt rendering, and generation are separate callables. Evaluation helpers
(`grounded`, `hallucination_flag`) are pure functions used by the API and by tests.
