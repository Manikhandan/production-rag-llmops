from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    ordinal: int


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int
    method: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[dict]
    prompt_version: str
    provider: str
    retrieved: int
    grounded: bool
    tokens_in: int
    tokens_out: int
