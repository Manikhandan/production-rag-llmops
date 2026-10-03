from __future__ import annotations

import re

from ragops.prompts import render
from ragops.types import Answer, Hit

WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


class ExtractiveProvider:
    """Deterministic provider: pick the highest-ranked chunk and quote it.

    A real LLM gateway would call a remote model with timeout/retry. This provider
    exists so retrieval, citations, and evaluation can be tested without API keys.
    """

    name = "extractive"

    def generate(self, question: str, hits: list[Hit], prompt_version: str, max_context_chars: int) -> Answer:
        if not hits:
            prompt = render(prompt_version, question, "")
            return Answer(
                text="I do not know based on the indexed documents.",
                citations=[],
                prompt_version=prompt_version,
                provider=self.name,
                retrieved=0,
                grounded=True,
                tokens_in=estimate_tokens(prompt),
                tokens_out=10,
            )
        context_parts = []
        citations = []
        used = 0
        for hit in hits:
            block = f"[{hit.chunk.doc_id}] {hit.chunk.text}"
            if used + len(block) > max_context_chars:
                break
            context_parts.append(block)
            citations.append(
                {
                    "doc_id": hit.chunk.doc_id,
                    "chunk_id": hit.chunk.chunk_id,
                    "title": hit.chunk.title,
                    "score": round(hit.score, 5),
                    "source": hit.chunk.source,
                }
            )
            used += len(block)
        context = "\n\n".join(context_parts)
        prompt = render(prompt_version, question, context)
        best = hits[0].chunk.text
        text = f"{best} [{hits[0].chunk.doc_id}]"
        ctx_tokens = _tokens(context)
        ans_tokens = _tokens(text) - {hits[0].chunk.doc_id.lower()}
        grounded = bool(ans_tokens) and ans_tokens <= ctx_tokens
        return Answer(
            text=text,
            citations=citations,
            prompt_version=prompt_version,
            provider=self.name,
            retrieved=len(hits),
            grounded=grounded,
            tokens_in=estimate_tokens(prompt),
            tokens_out=estimate_tokens(text),
        )
