from __future__ import annotations

import argparse
import os
from collections import Counter

from .answer import answer
from .chunking import chunk_pages
from .embedders import get_embedder
from .ingest import ingest, retrieve
from .llm import LLMError, get_llm
from .loader import load_path
from .store import PgVectorStore


def _open_store(args: argparse.Namespace):
    url = args.database_url or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("set DATABASE_URL or pass --database-url")
    embedder = get_embedder(args.embedder)
    return PgVectorStore(url, embedder.dimension, embedder.name), embedder


def _cmd_chunks(args: argparse.Namespace) -> int:
    pages = load_path(args.path)
    if not pages:
        print("No extractable text found.")
        return 1

    chunks = chunk_pages(pages, args.max_chars, args.overlap)
    per_file = Counter(chunk.source for chunk in chunks)
    for source, count in sorted(per_file.items()):
        print(f"{source}: {count} chunks")
    print(f"total: {len(chunks)} chunks from {len(pages)} pages")

    for chunk in chunks[: args.show]:
        print(f"\n--- {chunk.source} p.{chunk.page} #{chunk.index} ({len(chunk.text)} chars)")
        print(chunk.text)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    store, embedder = _open_store(args)
    with store:
        added = ingest(args.path, store, embedder, args.max_chars, args.overlap)
        print(f"indexed {added} chunks ({store.count()} in the database)")
    return 0 if added else 1


def _cmd_search(args: argparse.Namespace) -> int:
    store, embedder = _open_store(args)
    with store:
        hits = retrieve(args.question, store, embedder, args.k)
    if not hits:
        print("No results.")
        return 1
    for rank, hit in enumerate(hits, start=1):
        snippet = hit.chunk.text[:200] + ("..." if len(hit.chunk.text) > 200 else "")
        print(f"{rank}. {hit.score:.3f}  {hit.chunk.source} p.{hit.chunk.page}")
        print(f"   {snippet}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    store, embedder = _open_store(args)
    with store:
        try:
            result = answer(args.question, store, embedder, get_llm(args.llm, args.model), args.k)
        except LLMError as error:
            print(f"error: {error}")
            return 1
    print(result.text)
    if result.sources:
        print("\nSources:")
    for source in result.sources:
        mark = "*" if source.cited else " "
        print(f" {mark} [{source.number}] {source.source} p.{source.page} ({source.score:.3f})")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("ragqa.api:create_app_from_env", factory=True, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragqa")
    sub = parser.add_subparsers(dest="command", required=True)

    sizes = argparse.ArgumentParser(add_help=False)
    sizes.add_argument("--max-chars", type=int, default=800)
    sizes.add_argument("--overlap", type=int, default=150)

    db = argparse.ArgumentParser(add_help=False)
    db.add_argument("--database-url", help="defaults to $DATABASE_URL")
    db.add_argument("--embedder", choices=["hashing", "local"], help="defaults to $RAGQA_EMBEDDER")

    chunks = sub.add_parser("chunks", parents=[sizes], help="show how PDFs are split")
    chunks.add_argument("path", help="a PDF file or a directory of PDFs")
    chunks.add_argument("--show", type=int, default=0, metavar="N", help="print the first N chunks")
    chunks.set_defaults(func=_cmd_chunks)

    ingest_cmd = sub.add_parser("ingest", parents=[sizes, db], help="index PDFs in the database")
    ingest_cmd.add_argument("path", help="a PDF file or a directory of PDFs")
    ingest_cmd.set_defaults(func=_cmd_ingest)

    search = sub.add_parser("search", parents=[db], help="find the passages closest to a question")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=5, help="number of results")
    search.set_defaults(func=_cmd_search)

    ask = sub.add_parser("ask", parents=[db], help="answer a question from the indexed documents")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=4, help="passages given to the model")
    ask.add_argument("--llm", choices=["ollama"], help="defaults to $RAGQA_LLM")
    ask.add_argument("--model", help="defaults to $RAGQA_MODEL")
    ask.set_defaults(func=_cmd_ask)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
