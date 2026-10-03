from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragops.config import Settings, get_settings
from ragops.ingest import chunk_document, load_documents
from ragops.providers import get_provider
from ragops.retrieve import HybridIndex
from ragops.types import Answer


@dataclass
class RagSystem:
    index: HybridIndex
    settings: Settings

    @classmethod
    def build(cls, settings: Settings | None = None) -> RagSystem:
        settings = settings or get_settings()
        docs = load_documents(settings.corpus_dir)
        chunks = []
        for doc in docs:
            chunks.extend(chunk_document(doc, settings.chunk_size, settings.chunk_overlap))
        return cls(index=HybridIndex(chunks, rrf_k=settings.rrf_k), settings=settings)

    def ask(self, question: str) -> Answer:
        hits = self.index.search(question, top_k=self.settings.top_k)
        provider = get_provider(
            self.settings.provider,
            llm_url=self.settings.llm_url,
            llm_api_key=self.settings.llm_api_key,
            timeout_s=self.settings.request_timeout_s,
        )
        return provider.generate(
            question, hits, self.settings.prompt_version, self.settings.max_context_chars
        )

    def persist_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"chunks={len(self.index.chunks)}\n")
