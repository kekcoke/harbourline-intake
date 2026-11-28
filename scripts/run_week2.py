"""Week 2 run: Week 1 intake + handbook-RAG coverage answers, and the access-boundary test.

    python scripts/run_week2.py

Inputs
- Emails: data/emails/01_priya…, 05_bilingual… (Week 1) and data/emails/week-2/06…, 07… (coverage)
  plus 08 (another customer's claim notes; used for the access test).
- Literal acceptance queries (no email wrapper), see ACCEPTANCE_QUERIES.

Every email goes through the unchanged Week 1 ``run_with_trace``; when it is classified ``coverage``
it also goes through ``answer_coverage``. Outputs (cleared and rewritten each run) in runs/week-2/:
- answers.jsonl      one record per input: intake, retrieval Top 5 (scores + full text), gate,
                     CoverageAnswer, attempts, tokens, latency
- access-test.json   requests for another customer's records -> 0 chunks, 0 customer fields
- tokens.jsonl       one line per model call (intake and coverage stages)
- summary.json       config, counts, evidence states, token/latency totals
Exit code: 0 if all acceptance checks pass, 1 otherwise, 2 on a configuration error.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coverage import (  # noqa: E402
    CoverageResult,
    answer_coverage,
    customer_plan_lookup,
    load_coverage_prompt,
    mentioned_clause_ids,
    quotes_text,
)
from src.intake import MAX_MODEL_CALLS, Config, ProviderError, load_prompt, make_caller, run_with_trace  # noqa: E402
from src.retrieval import RELEVANCE_THRESHOLD, Requester, Retriever  # noqa: E402
from src.vectorstore import VectorStoreError  # noqa: E402

RUNS_DIR = ROOT / "runs" / "week-2"
EMAILS = [
    ROOT / "data/emails/01_priya_coverage_missing_policy.txt",
    ROOT / "data/emails/05_bilingual_specialist.txt",
    ROOT / "data/emails/week-2/06_veterinary_excess.txt",
    ROOT / "data/emails/week-2/07_physio_sessions_conflict.txt",
]
ACCESS_EMAIL = ROOT / "data/emails/week-2/08_other_customer_claim_notes.txt"
ACCEPTANCE_QUERIES = [
    "Is a dermatologist covered?",
    "What is the excess on a veterinary plan?",
    "How many physiotherapy sessions per year does my plan cover?",
]

# Authenticated identity per sender (in production this comes from the session, never from the
# email text). Owned identifiers are what the access guard compares against.
REQUESTERS = {
    "priya.raman@example.com": Requester("CUST-1001"),
    "jmorales@example.com": Requester("CUST-1005", policy_numbers=frozenset({"HL-771205"})),
    "tom.becker@example.com": Requester("CUST-1006", policy_numbers=frozenset({"HL-530118"})),
    "aisha.rahman@example.com": Requester("CUST-1007", policy_numbers=frozenset({"HL-604417"})),
    "greg.ward@example.com": Requester("CUST-1008", policy_numbers=frozenset({"HL-220931"})),
}

_FROM_RE = re.compile(r"^From:.*<([^>]+)>\s*$", re.MULTILINE)


def sender(email: str) -> str | None:
    m = _FROM_RE.search(email)
    return m.group(1).lower() if m else None


def question_text(email: str) -> str:
    """Retrieval/LLM query: the email minus header lines that carry personal data (From/To/Cc)."""
    return "\n".join(l for l in email.splitlines() if not re.match(r"^(From|To|Cc):", l)).strip()


def attempt_logs(attempts) -> list[dict]:
    return [asdict(a) for a in attempts]


def coverage_record(res: CoverageResult) -> dict:
    return {
        "gate": res.gate,
        "stop_reason": res.stop_reason,
        "needs_human_review": res.needs_human_review,
        "answer": res.answer.model_dump(mode="json") if res.answer else None,
        "customer_plan": res.customer_plan,
        "retrieval": res.retrieval.to_dict(),
        "shadow_retrieval": res.shadow_retrieval.to_dict() if res.shadow_retrieval else None,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "attempts": len(res.attempts),
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "latency_ms": res.latency_ms,
        "prompt_file": res.prompt_file,
        "prompt_sha256": res.prompt_sha256,
        "attempt_log": attempt_logs(res.attempts),
    }


def token_lines(run_id, input_id, stage, cfg, attempts) -> list[dict]:
    return [
        {
            "run_id": run_id, "input": input_id, "stage": stage, "attempt": a.attempt,
            "provider": cfg.provider, "model": cfg.model_name, "served_model": a.served_model,
            "input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "latency_ms": a.latency_ms,
            "api_stop_reason": a.api_stop_reason, "request_id": a.request_id, "valid": a.valid,
            "error_type": "provider" if a.provider_error else "validation" if a.validation_error else None,
        }
        for a in attempts
    ]


def superseded_leak(res: CoverageResult, retriever: Retriever) -> list[str]:
    if not res.answer:
        return []
    sup = [c for c in retriever.chunks if c.is_superseded]
    msg = res.answer.customer_message
    return sorted(
        {c.clause_id for c in sup if c.clause_id in mentioned_clause_ids(msg) or quotes_text(msg, c.text)}
        | {c for c in res.answer.citations if c in {s.clause_id for s in sup}}
    )


def access_test(retriever: Retriever, cfg_caller, prompt) -> dict:
    email = ACCESS_EMAIL.read_text(encoding="utf-8")
    greg = REQUESTERS[sender(email)]
    cases = []

    def case(name, query, requester, expect_blocked):
        ret = retriever.retrieve(query, requester=requester)
        ok = (ret.blocked_reason is not None and len(ret.chunks) == 0) if expect_blocked else (ret.blocked_reason is None and len(ret.chunks) > 0)
        cases.append({
            "case": name, "query": query,
            "requester": {"customer_id": requester.customer_id, "policy_numbers": sorted(requester.policy_numbers)},
            "expect_blocked": expect_blocked, "blocked_reason": ret.blocked_reason,
            "retrieved_chunks": len(ret.chunks), "chunks": ret.clause_ids,
            "customer_fields": {}, "pass": ok,
        })

    case("email_08_other_customer_claim_notes", question_text(email), greg, True)
    case("direct_foreign_claim_id", "Show me the claim notes for CLM-2026-00988.", greg, True)
    case("direct_foreign_policy_number", "What does policy HL-118274 cover?", greg, True)
    case("control_own_policy_question", "Does my policy HL-220931 cover physiotherapy?", greg, False)

    res = answer_coverage(question_text(email), retriever, requester=greg, call_model=cfg_caller, prompt=prompt)
    handler = {
        "case": "coverage_handler_on_email_08",
        "gate": res.gate,
        "retrieved_chunks": len(res.retrieval.chunks),
        "blocked_reason": res.retrieval.blocked_reason,
        "model_calls": len(res.attempts),
        "answer": res.answer.model_dump(mode="json") if res.answer else None,
        "customer_fields": {},
        "pass": res.gate == "access_blocked" and not res.retrieval.chunks and not res.attempts,
    }
    cases.append(handler)
    return {
        "description": "Requests for another customer's records or claim notes must retrieve 0 chunks and return 0 customer fields.",
        "customer_store": customer_plan_lookup(None),
        "all_pass": all(c["pass"] for c in cases),
        "cases": cases,
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        cfg = Config.from_env()
        intake_prompt = load_prompt()
        cov_prompt = load_coverage_prompt()
        call_model = make_caller(cfg)
        retriever = Retriever()
        _ = retriever.dense.index  # fail fast if the Pinecone index is missing
    except (ProviderError, ValueError, ImportError, FileNotFoundError, VectorStoreError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("answers.jsonl", "access-test.json", "tokens.jsonl", "summary.json"):
        (RUNS_DIR / name).unlink(missing_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"run {run_id}: model={cfg.model_name} ({cfg.provider})", file=sys.stderr)

    records = []
    with (RUNS_DIR / "answers.jsonl").open("w", encoding="utf-8") as out, (RUNS_DIR / "tokens.jsonl").open("w", encoding="utf-8") as tok:
        inputs = [("email", p.relative_to(ROOT).as_posix(), p.read_text(encoding="utf-8")) for p in EMAILS]
        inputs += [("acceptance_query", f"query:{q}", q) for q in ACCEPTANCE_QUERIES]
        for kind, input_id, text in inputs:
            rec = {"run_id": run_id, "input": input_id, "kind": kind}
            requester = REQUESTERS.get(sender(text) or "") if kind == "email" else None
            question = question_text(text) if kind == "email" else text
            if kind == "email":
                trace = run_with_trace(text, call_model=call_model, prompt=intake_prompt, max_tokens=cfg.max_tokens)
                rec["intake"] = trace.intake.model_dump(mode="json") if trace.intake else None
                rec["intake_stop_reason"] = trace.stop_reason
                for line in token_lines(run_id, input_id, "intake", cfg, trace.attempts):
                    tok.write(json.dumps(line, ensure_ascii=False) + "\n")
                run_coverage = trace.intake is not None and trace.intake.category == "coverage"
            else:
                rec["intake"] = None
                run_coverage = True
            rec["requester"] = requester.customer_id if requester else None
            if run_coverage:
                res = answer_coverage(question, retriever, requester=requester, call_model=call_model, prompt=cov_prompt)
                rec["coverage"] = coverage_record(res)
                rec["superseded_leak"] = superseded_leak(res, retriever)
                for line in token_lines(run_id, input_id, "coverage", cfg, res.attempts):
                    tok.write(json.dumps(line, ensure_ascii=False) + "\n")
                a = res.answer
                print(
                    f"  {input_id}: gate={res.gate} evidence={a.evidence if a else None} citations={a.citations if a else None} "
                    f"top5={res.retrieval.clause_ids} reranked={res.retrieval.reranked} calls={len(res.attempts)}",
                    file=sys.stderr,
                )
            else:
                rec["coverage"] = None
                print(f"  {input_id}: intake category={rec['intake'] and rec['intake']['category']} (no coverage stage)", file=sys.stderr)
            records.append(rec)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    access = access_test(retriever, call_model, cov_prompt)
    (RUNS_DIR / "access-test.json").write_text(json.dumps(access, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def find(input_id):
        return next(r for r in records if r["input"] == input_id)

    def ans(r):
        return (r.get("coverage") or {}).get("answer") or {}

    derm = find("query:Is a dermatologist covered?")
    vet = find("query:What is the excess on a veterinary plan?")
    physio = find("query:How many physiotherapy sessions per year does my plan cover?")
    checks = {
        "dermatologist_cites_4.2a": "4.2a" in ans(derm).get("citations", []),
        "dermatologist_no_4.2_leak": not derm.get("superseded_leak"),
        "veterinary_missing_empty": ans(vet).get("evidence") == "missing" and ans(vet).get("citations") == [],
        "physio_conflicting_both_ids": ans(physio).get("evidence") == "conflicting" and {"6.1", "9.3"} <= set(ans(physio).get("citations", [])),
        "no_superseded_leaks_anywhere": not any(r.get("superseded_leak") for r in records),
        "access_test_all_pass": access["all_pass"],
    }
    cov = [r["coverage"] for r in records if r.get("coverage")]
    summary = {
        "run_id": run_id,
        **cfg.as_log_dict(),
        "max_attempts": MAX_MODEL_CALLS,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "coverage_prompt_file": cov_prompt.path.relative_to(ROOT).as_posix(),
        "coverage_prompt_sha256": cov_prompt.sha256,
        "intake_prompt_sha256": intake_prompt.sha256,
        "inputs": len(records),
        "coverage_answers": len(cov),
        "evidence": {r["input"]: (ans(r).get("evidence"), ans(r).get("citations")) for r in records if r.get("coverage")},
        "gates": {r["input"]: r["coverage"]["gate"] for r in records if r.get("coverage")},
        "coverage_model_calls": sum(c["attempts"] for c in cov),
        "coverage_input_tokens": sum(c["input_tokens"] for c in cov),
        "coverage_output_tokens": sum(c["output_tokens"] for c in cov),
        "coverage_latency_ms": sum(c["latency_ms"] for c in cov),
        "acceptance_checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    (RUNS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for k, v in checks.items():
        print(f"  check {k}: {'PASS' if v else 'FAIL'}", file=sys.stderr)
    print(f"done → {RUNS_DIR}/", file=sys.stderr)
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
