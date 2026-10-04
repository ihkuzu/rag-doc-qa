from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, Sequence


class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how",
    "in", "is", "it", "many", "of", "on", "or", "that", "the", "to", "was", "what",
    "when", "which", "with",
}


class HashingEmbedder:
    # works offline and only matches on shared words, so it is meant for tests and demos
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            # cosine distance is undefined for a zero vector
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self._model = model
        self.dimension = int(model.get_sentence_embedding_dimension())
        self.name = f"{model_name.split('/')[-1]}-{self.dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in vector] for vector in vectors]


def get_embedder(name: str | None = None) -> Embedder:
    name = (name or os.getenv("RAGQA_EMBEDDER") or "hashing").lower()
    if name == "hashing":
        return HashingEmbedder()
    if name == "local":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder: {name}")
