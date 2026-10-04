from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import psycopg

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher is better


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


class PgVectorStore:
    def __init__(self, dsn: str, dimension: int, embedder_name: str):
        self.dimension = dimension
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._setup(embedder_name)

    def _setup(self, embedder_name: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    idx INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({int(self.dimension)}) NOT NULL,
                    UNIQUE (source, page, idx)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(
                "INSERT INTO meta (key, value) VALUES ('embedder', %s) ON CONFLICT DO NOTHING",
                (embedder_name,),
            )
            cur.execute("SELECT value FROM meta WHERE key = 'embedder'")
            stored = cur.fetchone()[0]
        # vectors from two different models cannot be compared with each other
        if stored != embedder_name:
            raise RuntimeError(
                f"this index was built with '{stored}', not '{embedder_name}'; "
                "use an empty database to switch models"
            )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        rows = [
            (c.source, c.page, c.index, c.text, _vector_literal(e))
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (source, page, idx, text, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                ON CONFLICT (source, page, idx)
                DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                """,
                rows,
            )

    def search(self, embedding: Sequence[float], k: int = 5) -> list[Hit]:
        literal = _vector_literal(embedding)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, page, idx, text, 1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, literal, k),
            )
            return [
                Hit(Chunk(source=s, page=p, index=i, text=t), float(score))
                for s, p, i, t, score in cur.fetchall()
            ]

    def delete_source(self, source: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (source,))

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
