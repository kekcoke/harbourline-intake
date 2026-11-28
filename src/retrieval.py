"""Standalone handbook retrieval. No LLM anywhere in this module.

    retrieve(query, k=5, mode="hybrid" | "keyword", include_superseded=None, requester=None)

Order of operations:
1. Access boundary: a request for another customer's records returns 0 chunks before any index
   is touched (``blocked_reason`` explains why).
2. Candidates: BM25 (keyword) and, in hybrid mode, Pinecone dense search over the same chunks.
3. Superseded filter: chunks with ``superseded_by`` set are dropped unless the query explicitly
   asks about historical rules (or ``include_superseded=True``).
4. Hybrid fusion: normalise each signal (BM25 / max, dense min-max) and take a weighted sum. If the top two fused
   scores are within ``RERANK_MARGIN``, rerank the top ``RERANK_POOL`` with the hosted reranker.
5. Return the top ``k`` with every component score attached.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Sequence

from src.handbook import Chunk, load_chunks
from src.vectorstore import DenseBackend

Mode = Literal["hybrid", "keyword"]

TOP_K = 5
CANDIDATES = 20  # per-signal candidate pool before fusion
HYBRID_ALPHA = 0.5  # weight on the dense signal; 1 - alpha on BM25
RERANK_MARGIN = 0.10  # rerank when fused #1 - #2 < margin (scores are in [0, 1])
RERANK_POOL = 10
# Relevance gate used by the coverage handler: a chunk "clears the relevance threshold" when its
# cross-encoder (bge-reranker-v2-m3) score is >= this. It is a noise floor, not a sufficiency test:
# measured scores for relevant and irrelevant chunks overlap above it (see docs/retrieval-report.md),
# so whether a clearing chunk actually answers the query is judged by the model and its citations
# are then checked in code. Dense e5 cosines were too compressed to threshold at all.
RELEVANCE_THRESHOLD = 0.01


# --------------------------------------------------------------------------------------
# Access boundary
# --------------------------------------------------------------------------------------

POLICY_ID_RE = re.compile(r"\bHL-\d{6}\b")
CLAIM_ID_RE = re.compile(r"\bCLM-\d{4}-\d{5}\b")

_RECORD_NOUNS = r"(?:claims?|claim\s+notes|notes|polic(?:y|ies)|records?|account|details|plan|file|history|status)"
_THIRD_PARTY = (
    r"(?:his|her|their|someone\s+else'?s|another\s+(?:customer|person|member|policyholder)'?s?|"
    r"(?:my\s+)?(?:neighbou?r|friend|colleague|brother|sister|mother|father|mum|mom|dad|husband|wife|"
    r"partner|son|daughter|cousin|boss|employee|tenant|landlord)'s)"
)
_THIRD_PARTY_RE = re.compile(rf"\b{_THIRD_PARTY}\s+(?:\w+\s+){{0,2}}{_RECORD_NOUNS}\b", re.IGNORECASE)


@dataclass(frozen=True)
class Requester:
    """The authenticated customer making the request (identity comes from the session, not the text)."""

    customer_id: str
    policy_numbers: frozenset[str] = frozenset()
    claim_ids: frozenset[str] = frozenset()


def check_access(query: str, requester: Requester | None = None) -> str | None:
    """Return a reason to block retrieval, or None. Pure function, no I/O."""
    m = _THIRD_PARTY_RE.search(query)
    if m:
        return f"third_party_record_request: {m.group(0)!r}"
    if requester is not None:
        foreign_policies = sorted(set(POLICY_ID_RE.findall(query)) - requester.policy_numbers)
        foreign_claims = sorted(set(CLAIM_ID_RE.findall(query)) - requester.claim_ids)
        if foreign_policies or foreign_claims:
            return f"foreign_identifier: {foreign_policies + foreign_claims} not owned by {requester.customer_id}"
    return None


# --------------------------------------------------------------------------------------
# Historical intent (opt-in to superseded clauses)
# --------------------------------------------------------------------------------------

_HISTORICAL_RE = re.compile(
    r"\b(?:superseded|historical|previous(?:ly)?|former(?:ly)?|used\s+to|no\s+longer|"
    r"old(?:er)?\s+(?:rule|policy|clause|version|handbook|wording|terms)|"
    r"(?:before|prior\s+to)\s+(?:(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+)?(?:19|20)\d{2}|"
    r"(?:rule|policy|clause)\s+(?:was|were)\s+in\s+(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def wants_historical(query: str) -> bool:
    return bool(_HISTORICAL_RE.search(query))


# --------------------------------------------------------------------------------------
# Keyword index (BM25, Okapi)
# --------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:\.[0-9a-z]+)?", re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or our "
    "so that the their there this to was we what when which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text)) if t not in _STOPWORDS]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.lengths):
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # final ranking score (rerank score if reranked, else fused / BM25)
    bm25: float = 0.0
    dense: float | None = None  # cosine similarity; None in keyword mode or if not a dense candidate
    fused: float | None = None
    rerank: float | None = None  # cross-encoder score; also the relevance signal for the gate

    @property
    def clause_id(self) -> str:
        return self.chunk.clause_id

    def to_dict(self, include_text: bool = True) -> dict:
        d = {
            "clause_id": self.chunk.clause_id,
            "document": self.chunk.document,
            "effective_date": self.chunk.effective_date,
            "superseded_by": self.chunk.superseded_by,
            "score": round(self.score, 4),
            "bm25": round(self.bm25, 4),
            "dense": None if self.dense is None else round(self.dense, 4),
            "fused": None if self.fused is None else round(self.fused, 4),
            "rerank": None if self.rerank is None else round(self.rerank, 4),
        }
        if include_text:
            d["title"] = self.chunk.title
            d["text"] = self.chunk.text
        return d


@dataclass
class RetrievalResult:
    query: str
    mode: Mode
    chunks: list[ScoredChunk] = field(default_factory=list)
    blocked_reason: str | None = None
    include_superseded: bool = False
    reranked: bool = False
    superseded_dropped: list[str] = field(default_factory=list)

    @property
    def clause_ids(self) -> list[str]:
        return [c.clause_id for c in self.chunks]

    @property
    def best_dense(self) -> float | None:
        vals = [c.dense for c in self.chunks if c.dense is not None]
        return max(vals) if vals else None

    def relevant(self, threshold: float = RELEVANCE_THRESHOLD) -> list[ScoredChunk]:
        """Chunks whose cross-encoder score clears ``threshold`` (empty if relevance was not scored)."""
        return [c for c in self.chunks if c.rerank is not None and c.rerank >= threshold]

    def to_dict(self, include_text: bool = True) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "blocked_reason": self.blocked_reason,
            "include_superseded": self.include_superseded,
            "reranked": self.reranked,
            "superseded_dropped": self.superseded_dropped,
            "chunks": [c.to_dict(include_text) for c in self.chunks],
        }


# --------------------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------------------


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: (1.0 if hi > 0 else 0.0) for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, dense: DenseBackend | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.by_id = {c.clause_id: c for c in self.chunks}
        self.bm25 = BM25([c.search_text for c in self.chunks])
        self._dense = dense

    @property
    def dense(self) -> DenseBackend:
        if self._dense is None:
            from src.vectorstore import PineconeStore

            self._dense = PineconeStore()
        return self._dense

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        mode: Mode = "hybrid",
        include_superseded: bool | None = None,
        requester: Requester | None = None,
        score_relevance: bool = False,
    ) -> RetrievalResult:
        """Top ``k`` chunks. ``score_relevance`` adds a cross-encoder score to every returned chunk
        (one extra rerank call unless the fusion step already reranked); it never changes the order."""
        if mode not in ("hybrid", "keyword"):
            raise ValueError(f"mode must be 'hybrid' or 'keyword', got {mode!r}")
        if include_superseded is None:
            include_superseded = wants_historical(query)
        result = RetrievalResult(query=query, mode=mode, include_superseded=include_superseded)

        blocked = check_access(query, requester)
        if blocked:
            result.blocked_reason = blocked
            return result

        def allowed(cid: str) -> bool:
            return include_superseded or not self.by_id[cid].is_superseded

        raw_bm25 = dict(zip((c.clause_id for c in self.chunks), self.bm25.scores(query)))
        result.superseded_dropped = sorted(
            cid for cid, s in raw_bm25.items() if s > 0 and not allowed(cid)
        )

        if mode == "keyword":
            ranked = sorted(
                ((cid, s) for cid, s in raw_bm25.items() if s > 0 and allowed(cid)), key=lambda x: (-x[1], x[0])
            )[:k]
            result.chunks = [ScoredChunk(self.by_id[cid], score=s, bm25=s) for cid, s in ranked]
            if score_relevance:
                self._score_relevance(query, result.chunks)
            return result

        bm25_cands = dict(
            sorted(((cid, s) for cid, s in raw_bm25.items() if s > 0 and allowed(cid)), key=lambda x: -x[1])[:CANDIDATES]
        )
        dense_hits = self.dense.search(query, top_k=CANDIDATES + 10)
        dense_all = {cid: s for cid, s in dense_hits if cid in self.by_id}
        result.superseded_dropped = sorted(set(result.superseded_dropped) | {c for c in dense_all if not allowed(c)})
        dense_cands = dict(list((cid, s) for cid, s in dense_all.items() if allowed(cid))[:CANDIDATES])

        # BM25 is divided by its max (min-max would zero out the weakest keyword match even when it
        # matched well); dense cosines are min-max stretched because e5 similarities are compressed.
        top_bm25 = max(bm25_cands.values(), default=0.0)
        n_bm25 = {cid: s / top_bm25 for cid, s in bm25_cands.items()} if top_bm25 > 0 else {}
        n_dense = _minmax(dense_cands)
        fused = {
            cid: HYBRID_ALPHA * n_dense.get(cid, 0.0) + (1 - HYBRID_ALPHA) * n_bm25.get(cid, 0.0)
            for cid in set(bm25_cands) | set(dense_cands)
        }
        order = sorted(fused, key=lambda c: (-fused[c], c))
        scored = [
            ScoredChunk(
                self.by_id[cid], score=fused[cid], bm25=raw_bm25.get(cid, 0.0), dense=dense_all.get(cid), fused=fused[cid]
            )
            for cid in order
        ]

        if len(scored) >= 2 and scored[0].score - scored[1].score < RERANK_MARGIN:
            pool = scored[:RERANK_POOL]
            rr = self.dense.rerank(query, [c.chunk.search_text for c in pool])
            for c, s in zip(pool, rr):
                c.rerank = s
                c.score = s
            scored = sorted(pool, key=lambda c: (-c.score, c.clause_id)) + scored[RERANK_POOL:]
            result.reranked = True

        result.chunks = scored[:k]
        if score_relevance:
            self._score_relevance(query, [c for c in result.chunks if c.rerank is None])
        return result

    def _score_relevance(self, query: str, chunks: list[ScoredChunk]) -> None:
        if not chunks:
            return
        for c, s in zip(chunks, self.dense.rerank(query, [c.chunk.search_text for c in chunks])):
            c.rerank = s


_default: Retriever | None = None


def retrieve(
    query: str,
    k: int = TOP_K,
    mode: Mode = "hybrid",
    include_superseded: bool | None = None,
    requester: Requester | None = None,
    score_relevance: bool = False,
) -> RetrievalResult:
    """Module-level convenience wrapper around a lazily-built default ``Retriever``."""
    global _default
    if _default is None:
        _default = Retriever()
    return _default.retrieve(query, k=k, mode=mode, include_superseded=include_superseded, requester=requester, score_relevance=score_relevance)
