from __future__ import annotations

from pathlib import Path

from ragops.types import Document


def load_documents(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        docs.append(
            Document(
                doc_id=path.stem,
                title=title,
                text=text,
                source=str(path),
                metadata={"bytes": path.stat().st_size},
            )
        )
    if not docs:
        return []
    return docs


def chunk_document(doc: Document, size: int, overlap: int) -> list:
    from ragops.types import Chunk

    words = doc.text.split()
    chunks = []
    start = 0
    ordinal = 0
    if size <= overlap:
        raise ValueError("chunk_size must exceed overlap")
    while start < len(words):
        window = words[start : start + size]
        text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{ordinal}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=text,
                source=doc.source,
                ordinal=ordinal,
            )
        )
        ordinal += 1
        start += size - overlap
    return chunks
