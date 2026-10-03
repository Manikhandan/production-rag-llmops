from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ragops.config import Settings, get_settings
from ragops.evaluate import hallucination_flag
from ragops.system import RagSystem

BLOCKLIST = ("ignore previous instructions", "exfiltrate")


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class RateLimiter:
    def __init__(self, rate: int = 30, per_seconds: float = 60.0) -> None:
        self.rate = rate
        self.per = per_seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self.events[key]
        while q and now - q[0] > self.per:
            q.popleft()
        if len(q) >= self.rate:
            return False
        q.append(now)
        return True


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    system = RagSystem.build(settings)
    limiter = RateLimiter()
    cache: dict[str, dict] = {}
    app = FastAPI(title="ragops")
    app.state.system = system

    @app.middleware("http")
    async def rid(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        n = len(system.index.chunks)
        return {"ready": n > 0, "chunks": n}

    @app.post("/v1/ask")
    def ask(body: AskIn, request: Request):
        ip = request.client.host if request.client else "local"
        if not limiter.allow(ip):
            raise HTTPException(429, "rate limit")
        q = body.question.strip()
        lowered = q.lower()
        if any(term in lowered for term in BLOCKLIST):
            raise HTTPException(400, "blocked by safety filter")
        if q in cache:
            payload = dict(cache[q])
            payload["cache_hit"] = True
            return payload
        answer = system.ask(q)
        payload = {
            "answer": answer.text,
            "citations": answer.citations,
            "prompt_version": answer.prompt_version,
            "provider": answer.provider,
            "grounded": answer.grounded,
            "hallucination_flag": hallucination_flag(answer),
            "tokens_in": answer.tokens_in,
            "tokens_out": answer.tokens_out,
            "retrieved": answer.retrieved,
            "cache_hit": False,
            "low_confidence": answer.retrieved == 0,
        }
        cache[q] = payload
        return payload

    return app


app = create_app()
