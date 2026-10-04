from __future__ import annotations

import os
from dataclasses import asdict
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from .answer import answer
from .embedders import Embedder, get_embedder
from .llm import LLM, LLMError, get_llm
from .store import PgVectorStore

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    question: Question
    k: int = Field(default=4, ge=1, le=10)


class SourceOut(BaseModel):
    number: int
    source: str
    page: int
    score: float
    cited: bool
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


def create_app(store: PgVectorStore, embedder: Embedder, llm: LLM) -> FastAPI:
    app = FastAPI(title="rag-doc-qa")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": store.count()}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        try:
            result = answer(request.question, store, embedder, llm, request.k)
        except LLMError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return AskResponse(
            answer=result.text,
            sources=[SourceOut(**asdict(source)) for source in result.sources],
        )

    return app


def create_app_from_env() -> FastAPI:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("set DATABASE_URL")
    embedder = get_embedder()
    store = PgVectorStore(url, embedder.dimension, embedder.name)
    return create_app(store, embedder, get_llm())
