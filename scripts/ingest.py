"""Build the handbook indexes from data/handbook/*.md.

    python scripts/ingest.py                # chunks.jsonl + Pinecone vector index
    python scripts/ingest.py --keyword-only # chunks.jsonl only (BM25 is built from it at load time)

Steps:
1. Parse every handbook file on clause headings (1 chunk = 1 citable clause). Files with no clause
   IDs (e.g. marketing brochures) are excluded; clause headings with no rule text are skipped.
2. Write data/index/chunks.jsonl, the single source of truth for both indexes.
3. Embed every chunk (multilingual-e5-large, input_type=passage) and upsert to Pinecone with
   id = clause_id and the chunk metadata. Vectors whose clause no longer exists are deleted.

Exit code: 0 on success, 2 on a configuration or handbook error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.handbook import CHUNKS_PATH, HANDBOOK_DIR, HandbookError, parse_handbook_dir, write_chunks  # noqa: E402


def _pinecone_metadata(chunk) -> dict:
    # Pinecone metadata cannot hold null, so superseded_by is "" when the clause is current.
    return {
        "document": chunk.document,
        "clause_id": chunk.clause_id,
        "effective_date": chunk.effective_date,
        "superseded_by": chunk.superseded_by or "",
        "is_superseded": chunk.is_superseded,
        "title": chunk.title,
        "text": chunk.text,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--handbook-dir", type=Path, default=HANDBOOK_DIR)
    ap.add_argument("--out", type=Path, default=CHUNKS_PATH)
    ap.add_argument("--keyword-only", action="store_true", help="skip the Pinecone vector index")
    args = ap.parse_args(argv)

    try:
        report = parse_handbook_dir(args.handbook_dir)
    except HandbookError as exc:
        print(f"handbook error: {exc}", file=sys.stderr)
        return 2
    if not report.chunks:
        print(f"no clauses found in {args.handbook_dir}", file=sys.stderr)
        return 2

    write_chunks(report.chunks, args.out)
    docs = sorted({c.document for c in report.chunks})
    superseded = [c.clause_id for c in report.chunks if c.is_superseded]
    print(f"parsed {len(report.chunks)} clause chunks from {docs}", file=sys.stderr)
    print(f"  superseded clauses: {superseded}", file=sys.stderr)
    print(f"  skipped headings without rule text: {report.skipped_headings}", file=sys.stderr)
    print(f"  excluded files (no clause IDs): {report.excluded_files}", file=sys.stderr)
    print(f"  wrote {args.out}", file=sys.stderr)
    print("keyword index: BM25 over the same chunks (built in memory from chunks.jsonl at load time)", file=sys.stderr)

    if args.keyword_only:
        print("vector index: skipped (--keyword-only)", file=sys.stderr)
        return 0

    from src.vectorstore import PineconeStore, VectorStoreError

    try:
        store = PineconeStore()
        store.ensure_index()
        store.upsert([(c.clause_id, c.search_text, _pinecone_metadata(c)) for c in report.chunks])
        stale_ids = sorted(store.list_ids() - {c.clause_id for c in report.chunks})
        store.delete(stale_ids)
        count = store.wait_for_count(len(report.chunks))
    except VectorStoreError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    print(
        f"vector index: Pinecone {store.index_name!r} namespace={store.namespace!r} vectors={count} "
        f"(expected {len(report.chunks)}; removed stale ids {stale_ids})",
        file=sys.stderr,
    )
    if count != len(report.chunks):
        print("warning: vector count does not match chunk count yet (index may still be catching up)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
