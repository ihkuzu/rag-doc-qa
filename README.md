# rag-doc-qa

Question answering over PDF documents using retrieval-augmented generation (RAG).

Large language models do not know the content of your documents. This project
splits PDFs into passages, stores them in a vector index, retrieves the passages
that are relevant to a question, and lets a language model answer using only
those passages, with the page it relied on.

**Status: work in progress.** The ingestion part is done; retrieval and answering
are next.

## Roadmap

- [x] PDF loading with page numbers
- [x] Sentence-aware chunking with overlap
- [ ] Embeddings and storage in Postgres (pgvector)
- [ ] Retrieval and question answering API (FastAPI) with source citations
- [ ] Evaluation set: measure how often the right passage is retrieved
- [ ] Docker setup and CI

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Put some PDFs into `data/pdfs/` and inspect how they are split:

```bash
python -m ragqa chunks data/pdfs --max-chars 800 --overlap 150 --show 2
```

## Design notes

- **Chunk size.** Chunks are capped at `max_chars` and built from whole sentences,
  so a passage rarely starts or ends mid-thought. A small overlap is carried from
  one chunk into the next so an answer sitting on a boundary is still found.
- **Page tracking.** Every chunk remembers its file and page, which is what makes
  cited answers possible later.
- **Scanned PDFs.** Pages without a text layer are skipped. OCR is out of scope for now.

## Project layout

```
src/ragqa/
  loader.py     PDF -> pages
  chunking.py   pages -> chunks
  cli.py        command line tools
tests/
```
