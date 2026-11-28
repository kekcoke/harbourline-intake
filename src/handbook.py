"""Handbook parsing: one chunk = one citable clause.

A clause starts at a heading ``### <clause_id> <title>`` and runs until the next heading of any
level. The line after the heading must be ``Effective: YYYY-MM-DD | Superseded by: <id or —>``.
A heading with no rule text is skipped (never indexed on its own). A file with no clause headings
(e.g. a marketing brochure) is excluded from the corpus.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANDBOOK_DIR = ROOT / "data" / "handbook"
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.jsonl"

CLAUSE_ID_RE = r"\d+\.\d+[a-z]?"
_CLAUSE_HEADING_RE = re.compile(rf"^###\s+({CLAUSE_ID_RE})\b[ \t]*(.*)$")
_ANY_HEADING_RE = re.compile(r"^#{1,6}\s")
_META_RE = re.compile(
    r"^Effective:\s*(\d{4}-\d{2}-\d{2})\s*\|\s*Superseded by:\s*(" + CLAUSE_ID_RE + r"|—|-|none)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    document: str
    clause_id: str
    effective_date: str
    superseded_by: str | None
    text: str
    title: str = ""

    @property
    def is_superseded(self) -> bool:
        return self.superseded_by is not None

    @property
    def search_text(self) -> str:
        """Text used for both keyword and vector indexing (title + rule text)."""
        return f"{self.title}. {self.text}" if self.title else self.text

    def to_dict(self) -> dict:
        return asdict(self)


class HandbookError(ValueError):
    pass


@dataclass
class ParseReport:
    chunks: list[Chunk]
    skipped_headings: list[str]  # clause headings with no rule text
    excluded_files: list[str]  # files with no clause IDs


def parse_document(path: Path) -> tuple[list[Chunk], list[str]]:
    """Parse one handbook file. Returns (chunks, clause_ids skipped for having no rule text)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[Chunk] = []
    skipped: list[str] = []
    i = 0
    while i < len(lines):
        m = _CLAUSE_HEADING_RE.match(lines[i])
        if not m:
            i += 1
            continue
        clause_id, title = m.group(1), m.group(2).strip()
        j = i + 1
        body: list[str] = []
        while j < len(lines) and not _ANY_HEADING_RE.match(lines[j]):
            body.append(lines[j])
            j += 1
        i = j
        body_lines = [b.strip() for b in body if b.strip()]
        if not body_lines:
            skipped.append(clause_id)
            continue
        meta = _META_RE.match(body_lines[0])
        if not meta:
            raise HandbookError(f"{path.name} clause {clause_id}: expected 'Effective: YYYY-MM-DD | Superseded by: …' line")
        text = " ".join(body_lines[1:]).strip()
        if not text:
            skipped.append(clause_id)
            continue
        sup = meta.group(2)
        chunks.append(
            Chunk(
                document=path.name,
                clause_id=clause_id,
                effective_date=meta.group(1),
                superseded_by=None if sup in ("—", "-") or sup.lower() == "none" else sup,
                text=text,
                title=title,
            )
        )
    return chunks, skipped


def parse_handbook_dir(directory: Path = HANDBOOK_DIR) -> ParseReport:
    chunks: list[Chunk] = []
    skipped: list[str] = []
    excluded: list[str] = []
    for path in sorted(directory.glob("*.md")):
        file_chunks, file_skipped = parse_document(path)
        if not file_chunks:
            excluded.append(path.name)
            continue
        chunks.extend(file_chunks)
        skipped.extend(f"{path.name}#{c}" for c in file_skipped)
    validate_chunks(chunks)
    return ParseReport(chunks=chunks, skipped_headings=skipped, excluded_files=excluded)


def validate_chunks(chunks: list[Chunk]) -> None:
    ids = [c.clause_id for c in chunks]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise HandbookError(f"duplicate clause_id(s): {dupes}")
    known = set(ids)
    for c in chunks:
        if c.superseded_by is not None and c.superseded_by not in known:
            raise HandbookError(f"clause {c.clause_id} superseded_by unknown clause {c.superseded_by}")
        if c.superseded_by == c.clause_id:
            raise HandbookError(f"clause {c.clause_id} cannot supersede itself")


def write_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/ingest.py` first.")
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
