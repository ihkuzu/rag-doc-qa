from __future__ import annotations

from pathlib import Path

from .chunking import chunk_pages
from .embedders import Embedder
from .loader import load_path
from .store import Hit, PgVectorStore


def ingest(
    path: str | Path,
    store: PgVectorStore,
    embedder: Embedder,
    max_chars: int = 800,
    overlap_chars: int = 150,
    batch_size: int = 64,
) -> int:
    chunks = chunk_pages(load_path(path), max_chars, overlap_chars)
    # re-ingesting a file replaces its old chunks instead of piling up
    for source in sorted({c.source for c in chunks}):
        store.delete_source(source)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store.add(batch, embedder.embed([c.text for c in batch]))
    return len(chunks)


def retrieve(question: str, store: PgVectorStore, embedder: Embedder, k: int = 5) -> list[Hit]:
    return store.search(embedder.embed([question])[0], k)
