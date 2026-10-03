from __future__ import annotations

from ragops.types import Answer, Hit


def retrieval_hit(question_terms: set[str], hits: list[Hit]) -> bool:
    blob = " ".join(hit.chunk.text.lower() for hit in hits)
    return any(term in blob for term in question_terms)


def groundedness(answer: Answer) -> bool:
    return answer.grounded


def hallucination_flag(answer: Answer) -> bool:
    return bool(answer.citations) and not answer.grounded
