"""Benchmark Keyword-only vs Hybrid retrieval on the labelled queries. No LLM.

    python scripts/eval_retrieval.py

Reads evals/retrieval_queries.jsonl ({id, query, expected_ids, forbidden_ids}) and writes
docs/retrieval-report.md plus runs/week-2/retrieval-eval.json.

A query is a HIT when every expected ID is in the Top 5 and no forbidden ID is.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import (  # noqa: E402
    CANDIDATES,
    HYBRID_ALPHA,
    RELEVANCE_THRESHOLD,
    RERANK_MARGIN,
    RERANK_POOL,
    TOP_K,
    Retriever,
    tokenize,
)
from src.vectorstore import EMBED_MODEL, RERANK_MODEL  # noqa: E402

QUERIES = ROOT / "evals" / "retrieval_queries.jsonl"
REPORT = ROOT / "docs" / "retrieval-report.md"
RAW = ROOT / "runs" / "week-2" / "retrieval-eval.json"

# Off-corpus questions used only to show where the relevance threshold sits (not scored as hits).
NEGATIVES = [
    "What is the excess on a veterinary plan?",
    "Does my plan cover my horse's vet bills?",
    "Does Harbourline cover my car repairs after an accident?",
    "Can I get home contents insurance with Harbourline?",
]


def judge(top: list[str], expected: list[str], forbidden: list[str]) -> tuple[bool, list[str], list[str]]:
    missing = [e for e in expected if e not in top]
    present_forbidden = [f for f in forbidden if f in top]
    return (not missing and not present_forbidden), missing, present_forbidden


def full_rank(retriever: Retriever, query: str, mode: str, include_superseded) -> list[str]:
    return retriever.retrieve(query, k=len(retriever.chunks), mode=mode, include_superseded=include_superseded).clause_ids


def explain(retriever: Retriever, q: dict, mode: str, res, missing: list[str], forbidden: list[str]) -> str:
    parts = []
    ranking = full_rank(retriever, q["query"], mode, res.include_superseded) if missing else []
    bm = dict(zip((c.clause_id for c in retriever.chunks), retriever.bm25.scores(q["query"])))
    for e in missing:
        rank = ranking.index(e) + 1 if e in ranking else None
        where = f"ranked #{rank}" if rank else "not retrieved at all"
        shared = sorted(set(tokenize(q["query"])) & set(tokenize(retriever.by_id[e].search_text)))
        if mode == "keyword" and bm.get(e, 0) == 0:
            parts.append(f"expected {e} {where}: BM25 score 0, no query term appears in the clause (vocabulary mismatch)")
        else:
            parts.append(
                f"expected {e} {where} (BM25={bm.get(e, 0):.2f}; only shared terms: {shared}, "
                f"so generic terms outrank it; no stemming or synonyms)"
            )
    for f in forbidden:
        parts.append(f"forbidden {f} surfaced in the Top 5")
    return "; ".join(parts)


def main() -> int:
    queries = [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever = Retriever()
    rows = []
    for q in queries:
        row = {"id": q["id"], "query": q["query"], "expected": q["expected_ids"], "forbidden": q["forbidden_ids"], "note": q.get("note", "")}
        for mode in ("keyword", "hybrid"):
            res = retriever.retrieve(q["query"], mode=mode, score_relevance=(mode == "hybrid"))
            top = res.clause_ids
            hit, missing, forb = judge(top, q["expected_ids"], q["forbidden_ids"])
            row[mode] = {
                "top5": top,
                "hit": hit,
                "include_superseded": res.include_superseded,
                "reranked": res.reranked,
                "why_missed": None if hit else explain(retriever, q, mode, res, missing, forb),
            }
            if mode == "hybrid":
                row[mode]["relevance"] = {c.clause_id: round(c.rerank, 4) for c in res.chunks if c.rerank is not None}
                row[mode]["dense"] = {c.clause_id: round(c.dense, 4) for c in res.chunks if c.dense is not None}
        rows.append(row)

    negatives = []
    for nq in NEGATIVES:
        res = retriever.retrieve(nq, mode="hybrid", score_relevance=True)
        negatives.append({
            "query": nq,
            "top5": res.clause_ids,
            "max_relevance": max((c.rerank for c in res.chunks), default=0.0),
            "max_dense": res.best_dense,
            "clears_threshold": [c.clause_id for c in res.relevant()],
        })

    n = len(rows)
    kw_hits = sum(r["keyword"]["hit"] for r in rows)
    hy_hits = sum(r["hybrid"]["hit"] for r in rows)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps({"rows": rows, "negatives": negatives}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- report -------------------------------------------------------------------------
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L: list[str] = []
    L += [
        "# Retrieval Report — Week 2 (Keyword-only vs Hybrid)",
        "",
        f"_Generated by `python scripts/eval_retrieval.py` on {gen}. No LLM is involved in any number below._",
        "",
        "## Summary",
        "",
        "| Mode | Top-5 hits | Hit rate |",
        "|---|---|---|",
        f"| Keyword-only (BM25) | {kw_hits}/{n} | {kw_hits / n:.0%} |",
        f"| Hybrid (BM25 + Pinecone dense, fused; rerank when close) | {hy_hits}/{n} | {hy_hits / n:.0%} |",
        "",
        "A query is a **hit** only when every expected clause ID is in the Top 5 **and** no forbidden ID is. "
        "Forbidden IDs are superseded clauses that must not surface (4.2, 7.4).",
        "",
        "## Setup",
        "",
        f"- Corpus: `data/index/chunks.jsonl`, {len(retriever.chunks)} clause chunks (1 chunk = 1 clause) from the fictional handbook. The brochure has no clause IDs and is excluded.",
        "- Keyword: Okapi BM25 (k1=1.5, b=0.75), with lower-casing, stop-word removal and no stemming.",
        f"- Dense: Pinecone serverless index, `{EMBED_MODEL}` (1024-d, cosine). Passages use `input_type=passage` and queries use `input_type=query`.",
        f"- Fusion: over each signal's top {CANDIDATES} candidates, BM25 is divided by its max and dense cosine is min-max scaled (e5 cosines are compressed). Score = {HYBRID_ALPHA}·dense + {1 - HYBRID_ALPHA}·BM25.",
        f"- Rerank: if fused #1 − #2 < {RERANK_MARGIN}, the top {RERANK_POOL} are reordered by `{RERANK_MODEL}`.",
        "- Superseded clauses are filtered out before ranking, unless the query explicitly asks about an earlier rule (q06).",
        "",
        "## Per-query results",
        "",
        "| ID | Query | Expected | Forbidden | Keyword Top 5 | KW | Hybrid Top 5 | Hybrid |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        kw, hy = r["keyword"], r["hybrid"]
        L.append(
            f"| {r['id']} | {r['query']} | {', '.join(r['expected'])} | {', '.join(r['forbidden']) or '—'} | "
            f"{', '.join(kw['top5']) or '(none)'} | {'✅' if kw['hit'] else '❌'} | "
            f"{', '.join(hy['top5'])}{' (reranked)' if hy['reranked'] else ''} | {'✅' if hy['hit'] else '❌'} |"
        )
    L += ["", "## Misses", ""]
    any_miss = False
    for mode, label in (("keyword", "Keyword-only"), ("hybrid", "Hybrid")):
        misses = [r for r in rows if not r[mode]["hit"]]
        L.append(f"### {label}: {len(misses)} miss(es)")
        L.append("")
        if not misses:
            L.append("None.")
        for r in misses:
            any_miss = True
            L.append(f"- **{r['id']}**: \"{r['query']}\" ({r['note']}). {r[mode]['why_missed']}.")
        L.append("")
    L += [
        "## Relevance threshold calibration",
        "",
        f"The coverage handler's evidence gate uses the cross-encoder score (`{RERANK_MODEL}`). A chunk *clears the threshold* when its score is ≥ **{RELEVANCE_THRESHOLD}**.",
        "",
        "| Query (labelled) | Expected | Expected clause relevance | Expected clause dense cosine |",
        "|---|---|---|---|",
    ]
    for r in rows:
        rel = r["hybrid"]["relevance"]
        dn = r["hybrid"]["dense"]
        L.append(
            f"| {r['id']} | {', '.join(r['expected'])} | "
            f"{', '.join(f'{e}={rel[e]:.3f}' if e in rel else f'{e}=n/a' for e in r['expected'])} | "
            f"{', '.join(f'{e}={dn[e]:.3f}' if e in dn else f'{e}=n/a' for e in r['expected'])} |"
        )
    L += ["", "| Off-corpus query | Top 5 | Max relevance | Max dense cosine | Clears threshold |", "|---|---|---|---|---|"]
    for ng in negatives:
        L.append(
            f"| {ng['query']} | {', '.join(ng['top5'])} | {ng['max_relevance']:.3f} | "
            f"{(ng['max_dense'] or 0):.3f} | {', '.join(ng['clears_threshold']) or 'none'} |"
        )
    L += [
        "",
        "**Findings**",
        "",
        "- **Dense cosine can't be used as a threshold.** The e5 similarities sit in a narrow 0.75–0.91 band, and off-corpus questions (car repairs, home contents) score as high as on-topic ones.",
        "- **Cross-encoder scores separate better, but still not cleanly.**",
        "  - Some true matches score low: paraphrases like \"skin doctor\" and \"therapist\", and the superseded-trap and historical queries.",
        "  - At the same time, some off-corpus questions have a loosely related clause above those scores.",
        f"- **So the threshold ({RELEVANCE_THRESHOLD}) is a noise floor, not a sufficiency test.**",
        "  - If no current clause clears it, the handler returns `missing` deterministically, with no LLM call.",
        "  - If a clause clears it, the model judges whether it actually answers the question. CONTEXT §4 requires both conditions for `enough`.",
        "  - Every citation is then checked in code. It must be in the retrieved Top 5, must clear the threshold and must be current. No superseded clause may be named or quoted.",
        "- **Example: \"What is the excess on a veterinary plan?\"** clears the floor only through 3.2 (Hospital excess). The model returned `missing`, `[]` (see `runs/week-2/answers.jsonl`), so this acceptance case depends on the model's judgment, not on the score gate alone.",
        "",
    ]
    below = [
        (r["id"], r["query"], e, r["hybrid"]["relevance"].get(e))
        for r in rows
        for e in r["expected"]
        if r["hybrid"]["relevance"].get(e) is not None and r["hybrid"]["relevance"][e] < RELEVANCE_THRESHOLD
    ]
    if below:
        L += [
            "- **Known false negatives at the gate.** Retrieval found the expected clause for the queries below, but its relevance score is below the floor. The coverage handler would answer `missing` (decline, no guess) for this wording:",
        ]
        L += [f"  - {qid} \"{q}\": {e} relevance {v:.3f}" for qid, q, e, v in below]
        L += ["  This is the safe failure direction. Lowering the floor to catch them would also let more off-corpus questions reach the model."]
    L += ["", "Raw per-query scores: `runs/week-2/retrieval-eval.json`."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"keyword {kw_hits}/{n}  hybrid {hy_hits}/{n}  misses={'yes' if any_miss else 'none'} → {REPORT.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
