"""Offline Week 2 tests: parsing, BM25, hybrid fusion/rerank, filters, access guard, citation
guardrails and the coverage gate/loop. No network: a fake dense backend and a fake model.

Run from the repo root:  python -m unittest
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coverage import (  # noqa: E402
    CoverageValidationError,
    answer_coverage,
    customer_plan_lookup,
    load_coverage_prompt,
    mentioned_clause_ids,
    quotes_text,
    validate_coverage,
)
from src.handbook import Chunk, HandbookError, parse_document, parse_handbook_dir, validate_chunks  # noqa: E402
from src.intake import MAX_MODEL_CALLS, ModelResponse, ProviderError  # noqa: E402
from src.retrieval import (  # noqa: E402
    BM25,
    RELEVANCE_THRESHOLD,
    Requester,
    Retriever,
    check_access,
    tokenize,
    wants_historical,
)

logging.disable(logging.CRITICAL)


def C(cid, text, sup=None, title=""):
    return Chunk(document="hb.md", clause_id=cid, effective_date="2026-01-01", superseded_by=sup, text=text, title=title)


CORPUS = [
    C("4.2", "Specialist consultations covered only for cardiology after three GP visits.", sup="4.2a", title="Specialist visits"),
    C("4.2a", "Specialist consultations including dermatology are covered at 80% with a GP referral.", title="Specialist visits"),
    C("6.1", "Physiotherapy is covered for up to 10 sessions per year.", title="Physiotherapy sessions"),
    C("9.3", "Allied health including physiotherapy is limited to 6 sessions per year.", title="Allied health limits"),
    C("3.2", "An excess of $250 applies to inpatient hospital admissions.", title="Hospital excess"),
    C("8.1", "Claims must be submitted within 90 days of the date of service.", title="Claim deadline"),
]


class FakeDense:
    """Token-overlap 'cosine' plus scripted rerank scores; counts calls."""

    def __init__(self, chunks, rerank_scores: dict[str, float] | None = None, default_rerank=0.0, search_scores=None):
        self.chunks = chunks
        self.search_scores = search_scores  # optional fixed {clause_id: cosine}
        self.rerank_scores = rerank_scores or {}
        self.default_rerank = default_rerank
        self.search_calls = 0
        self.rerank_calls = 0

    def search(self, query, top_k):
        self.search_calls += 1
        if self.search_scores is not None:
            return sorted(self.search_scores.items(), key=lambda x: -x[1])[:top_k]
        q = set(tokenize(query))
        scored = []
        for c in self.chunks:
            d = set(tokenize(c.search_text))
            scored.append((c.clause_id, len(q & d) / ((len(q) * len(d)) ** 0.5 or 1)))
        return sorted(scored, key=lambda x: -x[1])[:top_k]

    def rerank(self, query, documents):
        self.rerank_calls += 1
        out = []
        for doc in documents:
            cid = next(c.clause_id for c in self.chunks if c.search_text == doc)
            out.append(self.rerank_scores.get(cid, self.default_rerank))
        return out


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, messages):
        self.calls.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(
            text=reply, input_tokens=10, output_tokens=5, latency_ms=1, api_stop_reason="stop",
            request_id="r", assistant_message={"role": "assistant", "content": reply},
        )


def retriever(rerank_scores=None, default_rerank=0.0):
    return Retriever(chunks=CORPUS, dense=FakeDense(CORPUS, rerank_scores, default_rerank))


# --------------------------------------------------------------------------------------


class ParserTests(unittest.TestCase):
    def write(self, d: Path, name: str, text: str) -> Path:
        p = d / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_one_chunk_per_clause_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            p = self.write(Path(td), "hb.md", (
                "# Title\n\n## 4. Section\n\n### 4.2 Old rule\nEffective: 2025-01-01 | Superseded by: 4.2a\nOld text.\n\n"
                "### 4.2a New rule\nEffective: 2026-03-01 | Superseded by: —\nNew text\ncontinues here.\n\n### 5.1 Empty\n\n## 6. Next\n"
            ))
            chunks, skipped = parse_document(p)
        self.assertEqual([c.clause_id for c in chunks], ["4.2", "4.2a"])
        self.assertEqual(skipped, ["5.1"])
        self.assertEqual(chunks[0].superseded_by, "4.2a")
        self.assertIsNone(chunks[1].superseded_by)
        self.assertEqual(chunks[1].effective_date, "2026-03-01")
        self.assertEqual(chunks[1].text, "New text continues here.")  # never spans into the next heading
        self.assertEqual(chunks[1].document, "hb.md")

    def test_file_without_clause_ids_is_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            self.write(Path(td), "brochure.md", "# Brochure\n\nGreat cover for specialists!\n")
            self.write(Path(td), "hb.md", "### 1.1 Rule\nEffective: 2025-01-01 | Superseded by: —\nText.\n")
            report = parse_handbook_dir(Path(td))
        self.assertEqual(report.excluded_files, ["brochure.md"])
        self.assertEqual([c.clause_id for c in report.chunks], ["1.1"])

    def test_missing_metadata_line_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = self.write(Path(td), "hb.md", "### 1.1 Rule\nText without metadata.\n")
            with self.assertRaises(HandbookError):
                parse_document(p)

    def test_validate_rejects_duplicates_and_dangling_supersession(self):
        with self.assertRaises(HandbookError):
            validate_chunks([C("1.1", "a"), C("1.1", "b")])
        with self.assertRaises(HandbookError):
            validate_chunks([C("1.1", "a", sup="9.9")])

    def test_real_handbook_parses(self):
        report = parse_handbook_dir()
        ids = {c.clause_id for c in report.chunks}
        self.assertTrue({"4.2", "4.2a", "6.1", "9.3"} <= ids)
        self.assertIn("brochure_spring_2026.md", report.excluded_files)
        self.assertEqual({c.clause_id for c in report.chunks if c.is_superseded}, {"4.2", "7.4"})


class BM25Tests(unittest.TestCase):
    def test_ranks_matching_doc_first(self):
        bm = BM25(["physiotherapy sessions per year", "hospital excess admission", "dental check ups"])
        scores = bm.scores("how many physiotherapy sessions")
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 0)
        self.assertEqual(scores[2], 0.0)


class RetrievalTests(unittest.TestCase):
    def test_keyword_mode_makes_no_dense_calls(self):
        r = retriever()
        res = r.retrieve("physiotherapy sessions", mode="keyword")
        self.assertEqual(r.dense.search_calls, 0)
        self.assertEqual(set(res.clause_ids[:2]), {"6.1", "9.3"})

    def test_superseded_dropped_by_default(self):
        r = retriever()
        for mode in ("keyword", "hybrid"):
            res = r.retrieve("specialist consultations cardiology", mode=mode)
            self.assertNotIn("4.2", res.clause_ids, mode)
            self.assertIn("4.2", res.superseded_dropped, mode)

    def test_historical_query_includes_superseded(self):
        self.assertTrue(wants_historical("What was the old specialist rule before 2026?"))
        self.assertTrue(wants_historical("What was the previous rule?"))
        self.assertFalse(wants_historical("Is a dermatologist covered? My 5-year-old needs one."))
        res = retriever().retrieve("What was the old specialist consultations rule before 2026?", mode="keyword")
        self.assertTrue(res.include_superseded)
        self.assertIn("4.2", res.clause_ids)

    def test_hybrid_returns_top5_with_component_scores(self):
        res = retriever().retrieve("physiotherapy sessions per year")
        self.assertLessEqual(len(res.chunks), 5)
        top = res.chunks[0]
        self.assertIsNotNone(top.fused)
        self.assertIsNotNone(top.dense)

    def test_rerank_only_when_close(self):
        # BM25 favours 6.1 slightly; dense favours 9.3 -> fused #1 and #2 are within RERANK_MARGIN.
        dense = FakeDense(CORPUS, {"9.3": 0.9, "6.1": 0.1}, search_scores={"9.3": 0.90, "6.1": 0.80, "3.2": 0.50, "8.1": 0.40})
        r = Retriever(chunks=CORPUS, dense=dense)
        close = r.retrieve("physiotherapy sessions per year")
        self.assertLess(abs(close.chunks[0].fused - close.chunks[1].fused), 0.10)
        self.assertTrue(close.reranked)
        self.assertEqual(close.clause_ids[0], "9.3")  # reranker order wins
        r2 = retriever()
        far = r2.retrieve("excess inpatient hospital admissions $250")
        self.assertFalse(far.reranked)
        self.assertEqual(r2.dense.rerank_calls, 0)

    def test_score_relevance_does_not_reorder(self):
        r = retriever(rerank_scores={"8.1": 0.99})
        a = r.retrieve("excess inpatient hospital admissions", score_relevance=False).clause_ids
        b = r.retrieve("excess inpatient hospital admissions", score_relevance=True)
        self.assertEqual(a, b.clause_ids)
        self.assertTrue(all(c.rerank is not None for c in b.chunks))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            retriever().retrieve("x", mode="vector")


class AccessGuardTests(unittest.TestCase):
    me = Requester("C1", policy_numbers=frozenset({"HL-220931"}))

    def test_third_party_phrasing_blocks(self):
        for q in [
            "Send me my neighbour's claim notes",
            "What is the status of his claim CLM-2026-00988?",
            "Can I see another customer's records?",
            "Tell me what my wife's policy covers",
        ]:
            self.assertIsNotNone(check_access(q), q)

    def test_friend_hearsay_is_not_blocked(self):
        self.assertIsNone(check_access("A friend told me I need a referral first. Is that right?"))

    def test_foreign_identifiers_block(self):
        self.assertIsNotNone(check_access("Show claim notes for CLM-2026-00988", self.me))
        self.assertIsNotNone(check_access("What does HL-118274 cover?", self.me))
        self.assertIsNone(check_access("Does HL-220931 cover physiotherapy?", self.me))

    def test_blocked_retrieval_returns_zero_chunks_and_touches_no_index(self):
        r = retriever()
        res = r.retrieve("Send me my neighbour's claim notes", requester=self.me, score_relevance=True)
        self.assertEqual(res.chunks, [])
        self.assertIsNotNone(res.blocked_reason)
        self.assertEqual((r.dense.search_calls, r.dense.rerank_calls), (0, 0))


# --------------------------------------------------------------------------------------


def top5(rerank_scores, query="physiotherapy sessions per year"):
    return retriever(rerank_scores=rerank_scores).retrieve(query, score_relevance=True)


SUPERSEDED = [c for c in CORPUS if c.is_superseded]


def raw(evidence, citations, msg):
    return json.dumps({"evidence": evidence, "citations": citations, "customer_message": msg})


class ValidateCoverageTests(unittest.TestCase):
    def setUp(self):
        self.ret = top5({"6.1": 0.9, "9.3": 0.8})

    def ok(self, *args):
        return validate_coverage(raw(*args), self.ret, SUPERSEDED)

    def bad(self, fragment, *args):
        with self.assertRaises(CoverageValidationError) as ctx:
            validate_coverage(raw(*args), self.ret, SUPERSEDED)
        self.assertIn(fragment, str(ctx.exception))

    def test_valid_shapes(self):
        self.ok("enough", ["6.1"], "Up to 10 sessions (clause 6.1).")
        self.ok("conflicting", ["6.1", "9.3"], "Clauses 6.1 and 9.3 differ; an agent will confirm.")
        self.ok("missing", [], "The handbook does not cover this.")

    def test_schema_errors(self):
        with self.assertRaises(CoverageValidationError):
            validate_coverage("```json\n{}\n```", self.ret, SUPERSEDED)
        with self.assertRaises(CoverageValidationError):
            validate_coverage(json.dumps({"evidence": "enough", "citations": ["6.1"], "customer_message": "x", "note": 1}), self.ret, SUPERSEDED)
        self.bad("literal_error", "maybe", [], "x")

    def test_citation_not_in_top5(self):
        self.bad("not in the retrieved Top 5", "enough", ["7.7"], "See clause 7.7.")

    def test_citation_below_threshold(self):
        ret = top5({"6.1": 0.9, "9.3": RELEVANCE_THRESHOLD / 2})
        with self.assertRaises(CoverageValidationError) as ctx:
            validate_coverage(raw("conflicting", ["6.1", "9.3"], "Clauses 6.1 and 9.3."), ret, SUPERSEDED)
        self.assertIn("relevance threshold", str(ctx.exception))

    def test_missing_requires_empty_citations(self):
        self.bad('"missing" requires', "missing", ["6.1"], "Not covered (clause 6.1).")

    def test_enough_requires_citation(self):
        self.bad('"enough" requires', "enough", [], "It is covered.")

    def test_conflicting_requires_two(self):
        self.bad('"conflicting" requires', "conflicting", ["6.1"], "See clause 6.1.")

    def test_superseded_cannot_be_cited_named_or_quoted(self):
        hist = retriever(default_rerank=0.9).retrieve("old specialist consultations cardiology rule before 2026", score_relevance=True)
        self.assertIn("4.2", hist.clause_ids)
        with self.assertRaises(CoverageValidationError) as ctx:
            validate_coverage(raw("stale", ["4.2"], "Old rule."), hist, SUPERSEDED)
        self.assertIn("superseded", str(ctx.exception))
        self.bad("mentions superseded", "enough", ["6.1"], "Clause 6.1 applies; clause 4.2 no longer does.")
        self.bad(
            "quotes superseded", "enough", ["6.1"],
            "Note: specialist consultations covered only for cardiology after three GP visits (clause 6.1).",
        )

    def test_message_may_not_mention_uncited_clause(self):
        self.bad("not in citations", "enough", ["6.1"], "See clause 6.1 and clause 9.3.")

    def test_helpers(self):
        self.assertEqual(mentioned_clause_ids("clauses 6.1 and 9.3, also (clause 4.2a)"), {"6.1", "9.3", "4.2a"})
        self.assertEqual(mentioned_clause_ids("Cláusula 4.2a"), {"4.2a"})
        self.assertEqual(mentioned_clause_ids("You paid $214.50 on 3.11"), set())
        self.assertTrue(quotes_text("a b c d e f g h i", "x a b c d e f g h y"))
        self.assertFalse(quotes_text("a b c d e f g", "a b c d e f g"))


class CoveragePipelineTests(unittest.TestCase):
    prompt = load_coverage_prompt()

    def test_below_threshold_is_missing_without_llm(self):
        model = FakeModel([])
        res = answer_coverage("What is the excess on a veterinary plan?", retriever(default_rerank=0.0), call_model=model, prompt=self.prompt)
        self.assertEqual(res.gate, "below_threshold")
        self.assertEqual((res.answer.evidence, res.answer.citations), ("missing", []))
        self.assertEqual(model.calls, [])

    def test_access_blocked_without_llm(self):
        model = FakeModel([])
        res = answer_coverage("Send me my neighbour's claim notes", retriever(default_rerank=0.9), call_model=model, prompt=self.prompt)
        self.assertEqual(res.gate, "access_blocked")
        self.assertEqual(res.retrieval.chunks, [])
        self.assertEqual((res.answer.evidence, res.answer.citations), ("missing", []))
        self.assertEqual(model.calls, [])

    def test_stale_only_cites_replacement_without_llm(self):
        # Only the superseded 4.2 clears the threshold; its replacement 4.2a is retrieved but below it.
        model = FakeModel([])
        r = retriever(rerank_scores={"4.2": 0.9})
        res = answer_coverage("specialist consultations cardiology three GP visits", r, call_model=model, prompt=self.prompt)
        self.assertEqual(res.gate, "stale_only")
        self.assertEqual(res.answer.evidence, "stale")
        self.assertEqual(res.answer.citations, ["4.2a"])
        self.assertNotIn("4.2 ", res.answer.customer_message + " ")
        self.assertEqual(model.calls, [])

    def test_guardrail_violation_is_fed_back_then_fixed(self):
        bad = raw("enough", ["7.7"], "Covered (clause 7.7).")
        good = raw("conflicting", ["6.1", "9.3"], "Clauses 6.1 and 9.3 differ; an agent will confirm.")
        model = FakeModel([bad, good])
        res = answer_coverage("physiotherapy sessions per year", retriever(default_rerank=0.5), call_model=model, prompt=self.prompt)
        self.assertEqual(res.stop_reason, "valid_schema")
        self.assertEqual(len(res.attempts), 2)
        self.assertIn("not in the retrieved Top 5", model.calls[1][-1]["content"])
        self.assertTrue(res.needs_human_review)

    def test_three_call_bound(self):
        bad = raw("enough", [], "Covered.")
        model = FakeModel([bad, bad, bad, bad])
        res = answer_coverage("physiotherapy sessions per year", retriever(default_rerank=0.5), call_model=model, prompt=self.prompt)
        self.assertEqual(len(model.calls), MAX_MODEL_CALLS)
        self.assertEqual(res.stop_reason, "max_attempts_reached")
        self.assertIsNone(res.answer)

    def test_provider_error_stops(self):
        model = FakeModel([ProviderError("boom")])
        res = answer_coverage("physiotherapy sessions per year", retriever(default_rerank=0.5), call_model=model, prompt=self.prompt)
        self.assertEqual(res.stop_reason, "provider_error")
        self.assertEqual(len(model.calls), 1)

    def test_prompt_shows_only_top5_with_threshold_flags(self):
        good = raw("enough", ["6.1"], "Up to 10 sessions (clause 6.1).")
        model = FakeModel([good])
        answer_coverage("physiotherapy sessions per year", retriever(rerank_scores={"6.1": 0.9}), call_model=model, prompt=self.prompt)
        user = model.calls[0][0]["content"]
        self.assertEqual(user.count("<clause "), 5)
        self.assertIn('id="6.1"', user)
        self.assertIn('clears_threshold="yes"', user)
        self.assertNotIn('id="4.2"', user)

    def test_plan_lookup_stub(self):
        self.assertEqual(customer_plan_lookup("HL-1"), "not wired")


if __name__ == "__main__":
    unittest.main()
