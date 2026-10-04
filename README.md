# rag-doc-qa

Question answering over PDF documents using retrieval-augmented generation (RAG).

Language models do not know the content of your documents. This project splits
PDFs into passages, stores them in a vector index, retrieves the passages that
are relevant to a question, and lets a language model answer using only those
passages, with the page it relied on.

**Status: work in progress.** Ingestion and retrieval work; answer generation
and evaluation are next.

## Roadmap

- [x] PDF loading with page numbers
- [x] Sentence-aware chunking with overlap
- [x] Embeddings and storage in Postgres (pgvector)
- [x] Semantic search from the command line
- [ ] Question answering API (FastAPI) with source citations
- [ ] Evaluation set: measure how often the right passage is retrieved
- [ ] Docker setup and CI

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests start a temporary Postgres with pgvector through the `pgserver`
package, so no database setup is needed to run them.

### Try it on your own PDFs

Start a Postgres with pgvector (a compose file will follow):

```bash
docker run -d --name ragqa-db -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

Index some PDFs and search them:

```bash
python -m ragqa ingest data/pdfs
python -m ragqa search "how often must the pump be inspected?" -k 3
```

Look at how the documents are split before indexing:

```bash
python -m ragqa chunks data/pdfs --max-chars 800 --overlap 150 --show 2
```

### Embedding models

| `RAGQA_EMBEDDER` | What it does |
| --- | --- |
| `hashing` (default) | Offline and deterministic, matches on shared words only. Good for tests and demos. |
| `local` | Sentence embeddings from `all-MiniLM-L6-v2`. Install with `pip install -e ".[local]"`, downloads the model on first use. |

The index remembers which embedder built it and refuses to mix vectors from
different models.

## Design notes

- **Chunk size.** Chunks are capped at `max_chars` and built from whole sentences,
  so a passage rarely starts or ends mid-thought. A small overlap is carried from
  one chunk into the next so an answer sitting on a boundary is still found.
- **Page tracking.** Every chunk remembers its file and page, which is what makes
  cited answers possible.
- **Similarity.** Search uses cosine distance with an HNSW index in pgvector.
- **Re-ingesting.** Indexing a file again replaces its old chunks.
- **Scanned PDFs.** Pages without a text layer are skipped. OCR is out of scope for now.

## Project layout

```
src/ragqa/
  loader.py      PDF -> pages
  chunking.py    pages -> chunks
  embedders.py   text -> vectors
  store.py       pgvector storage and search
  ingest.py      loader + chunking + embeddings + store
  cli.py         command line
tests/
```
