import pytest
from fastapi.testclient import TestClient

from conftest import FakeLLM
from ragqa.api import create_app
from ragqa.embedders import HashingEmbedder
from ragqa.ingest import ingest
from ragqa.llm import LLMError
from ragqa.store import PgVectorStore


@pytest.fixture
def make_client(clean_db, pdf_factory, tmp_path):
    embedder = HashingEmbedder()
    pdf_factory("handbook.pdf", ["Employees receive thirty vacation days.", "Backups run nightly."])
    store = PgVectorStore(clean_db, embedder.dimension, embedder.name)
    ingest(tmp_path, store, embedder)

    def _make(llm):
        return TestClient(create_app(store, embedder, llm))

    yield _make
    store.close()


def test_health_reports_the_chunk_count(make_client):
    response = make_client(FakeLLM()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 2}


def test_ask_returns_the_answer_with_sources(make_client):
    client = make_client(FakeLLM("Thirty days [1]."))

    response = client.post("/ask", json={"question": "How many vacation days?", "k": 2})

    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Thirty days [1]."
    assert body["sources"][0]["source"] == "handbook.pdf"
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["cited"] is True


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {"question": "q", "k": 0}, {}])
def test_invalid_requests_are_rejected(make_client, payload):
    assert make_client(FakeLLM()).post("/ask", json=payload).status_code == 422


def test_model_failure_returns_502(make_client):
    class Broken:
        def complete(self, system, prompt):
            raise LLMError("model is down")

    response = make_client(Broken()).post("/ask", json={"question": "anything"})

    assert response.status_code == 502
    assert "model is down" in response.json()["detail"]
