"""Pinecone-backed dense index + hosted reranker (Pinecone Inference).

Embeddings: ``multilingual-e5-large`` (1024-d, cosine). Passages are embedded with
``input_type=passage`` and queries with ``input_type=query``.
Reranker: ``bge-reranker-v2-m3``.

Retrieval only depends on the small ``DenseBackend`` protocol below, so tests swap in a fake.
"""

from __future__ import annotations

import os
import time
from typing import Protocol, Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

EMBED_MODEL = "multilingual-e5-large"
EMBED_DIM = 1024
RERANK_MODEL = "bge-reranker-v2-m3"
_EMBED_BATCH = 90  # Pinecone Inference caps a single embed call at 96 inputs


class DenseBackend(Protocol):
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (clause_id, cosine similarity) pairs, best first."""

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document, in input order."""


class VectorStoreError(RuntimeError):
    pass


class PineconeStore:
    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        cloud: str | None = None,
        region: str | None = None,
        namespace: str = "handbook",
    ):
        from pinecone import Pinecone  # lazy: keyword-only mode and tests never need it

        api_key = (api_key or os.getenv("PINECONE_API_KEY", "")).strip()
        if not api_key or api_key.startswith("your_"):
            raise VectorStoreError("PINECONE_API_KEY is not set; add it to .env (see .env.example).")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "harbourline-handbook")
        self.cloud = cloud or os.getenv("PINECONE_CLOUD", "aws")
        self.region = region or os.getenv("PINECONE_REGION", "us-east-1")
        self.namespace = namespace
        self._index = None

    # ---- index lifecycle (ingest) -------------------------------------------------------

    def ensure_index(self) -> None:
        from pinecone import ServerlessSpec

        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

    @property
    def index(self):
        if self._index is None:
            if not self.pc.has_index(self.index_name):
                raise VectorStoreError(f"Pinecone index {self.index_name!r} not found; run `python scripts/ingest.py`.")
            self._index = self.pc.Index(name=self.index_name)
        return self._index

    def embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = list(texts[start : start + _EMBED_BATCH])
            res = self.pc.inference.embed(
                model=EMBED_MODEL, inputs=batch, parameters={"input_type": input_type, "truncate": "END"}
            )
            out.extend(list(item.values) for item in res.data)
        return out

    def upsert(self, records: list[tuple[str, str, dict]]) -> None:
        """records: (id, text to embed, metadata). Metadata values must be str/num/bool/list[str]."""
        vectors = self.embed([text for _, text, _ in records], input_type="passage")
        payload = [{"id": rid, "values": vec, "metadata": meta} for (rid, _, meta), vec in zip(records, vectors)]
        for start in range(0, len(payload), 100):
            self.index.upsert(vectors=payload[start : start + 100], namespace=self.namespace)

    def list_ids(self) -> set[str]:
        ids: set[str] = set()
        for page in self.index.list(namespace=self.namespace):
            # SDK 10.x yields ListResponse pages (.vectors = [ListItem(id=…)]); older SDKs yield id lists.
            if hasattr(page, "vectors"):
                ids.update(item.id for item in page.vectors)
            elif isinstance(page, (list, tuple)):
                ids.update(page)
            else:
                ids.add(str(page))
        return ids

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self.index.delete(ids=list(ids), namespace=self.namespace)

    def vector_count(self) -> int:
        stats = self.index.describe_index_stats()
        ns = (stats.namespaces or {}).get(self.namespace)
        return int(ns.vector_count) if ns else 0

    def wait_for_count(self, expected: int, timeout_s: float = 60.0) -> int:
        deadline = time.monotonic() + timeout_s
        count = self.vector_count()
        while count != expected and time.monotonic() < deadline:
            time.sleep(2)
            count = self.vector_count()
        return count

    # ---- query time ---------------------------------------------------------------------

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        vector = self.embed([query], input_type="query")[0]
        res = self.index.query(vector=vector, top_k=top_k, namespace=self.namespace, include_metadata=False)
        return [(m.id, float(m.score)) for m in res.matches]

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        res = self.pc.inference.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[{"id": str(i), "text": d} for i, d in enumerate(documents)],
            rank_fields=["text"],
            top_n=len(documents),
            return_documents=False,
        )
        scores = [0.0] * len(documents)
        for item in res.data:
            scores[item.index] = float(item.score)
        return scores
