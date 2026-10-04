"""Command line helpers."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .chunking import chunk_pages
from .loader import load_directory, load_pdf


def _cmd_chunks(args: argparse.Namespace) -> int:
    target = Path(args.path)
    pages = load_pdf(target) if target.is_file() else load_directory(target)
    if not pages:
        print("No extractable text found.")
        return 1

    chunks = chunk_pages(pages, args.max_chars, args.overlap)
    per_file = Counter(chunk.source for chunk in chunks)
    for source, count in sorted(per_file.items()):
        print(f"{source}: {count} chunks")
    print(f"total: {len(chunks)} chunks from {len(pages)} pages")

    if args.show:
        for chunk in chunks[: args.show]:
            print(f"\n--- {chunk.source} p.{chunk.page} #{chunk.index} ({len(chunk.text)} chars)")
            print(chunk.text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragqa")
    sub = parser.add_subparsers(dest="command", required=True)

    chunks = sub.add_parser("chunks", help="split PDFs into chunks and print statistics")
    chunks.add_argument("path", help="a PDF file or a directory of PDFs")
    chunks.add_argument("--max-chars", type=int, default=800)
    chunks.add_argument("--overlap", type=int, default=150)
    chunks.add_argument("--show", type=int, default=0, metavar="N", help="print the first N chunks")
    chunks.set_defaults(func=_cmd_chunks)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
