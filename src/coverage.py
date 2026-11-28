"""Week 2 coverage handler: handbook retrieval -> deterministic gate -> cited answer.

Public API
----------
- ``CoverageAnswer``                    strict output schema (CONTEXT.md §4).
- ``validate_coverage(raw, retrieval)`` schema + citation guardrails against the retrieved Top 5.
- ``answer_coverage(question, ...)``    full pipeline; returns a ``CoverageResult`` trace.
- ``customer_plan_lookup(...)``         Week 3 stub, always ``"not wired"``.

Pipeline
--------
1. Hybrid retrieval with relevance scores (``src/retrieval.py``). No LLM.
2. Deterministic gate, no LLM call:
   - access boundary hit           -> ``missing``, ``[]``, decline message
   - nothing clears the threshold  -> check superseded clauses: if one clears -> ``stale`` citing its
                                      current replacement (only if that is in the Top 5); else ``missing``, ``[]``
3. Otherwise the model answers from the Top 5 (``prompts/coverage_v1.md``); every response is
   validated in code and failures are fed back, at most ``MAX_MODEL_CALLS`` calls in total.

Week 1 (``src/intake.py``) is reused for config/providers only; its loop and schema are unchanged.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.handbook import CLAUSE_ID_RE, Chunk
from src.intake import (
    MAX_MODEL_CALLS,
    Attempt,
    CallModel,
    Config,
    ProviderError,
    make_caller,
)
from src.retrieval import RELEVANCE_THRESHOLD, Requester, RetrievalResult, Retriever

ROOT = Path(__file__).resolve().parent.parent
COVERAGE_PROMPT_PATH = ROOT / "prompts" / "coverage_v1.md"

Evidence = Literal["enough", "missing", "conflicting", "stale"]
Gate = Literal["llm", "access_blocked", "below_threshold", "stale_only"]
StopReason = Literal["valid_schema", "max_attempts_reached", "provider_error", "gate"]

MAX_MESSAGE_CHARS = 1500
QUOTE_NGRAM = 8  # this many consecutive words shared with superseded text counts as quoting it


# --------------------------------------------------------------------------------------
# Schema + guardrails
# --------------------------------------------------------------------------------------


class CoverageAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    evidence: Evidence
    citations: list[str] = Field(max_length=5)
    customer_message: str = Field(max_length=MAX_MESSAGE_CHARS)

    @field_validator("citations")
    @classmethod
    def _check_citations(cls, v: list[str]) -> list[str]:
        for cid in v:
            if not re.fullmatch(CLAUSE_ID_RE, cid):
                raise ValueError(f"citation {cid!r} is not a clause ID like '4.2a'")
        if len(set(v)) != len(v):
            raise ValueError("citations must not contain duplicates")
        return v

    @field_validator("customer_message")
    @classmethod
    def _check_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_message must not be empty")
        return v


class CoverageValidationError(ValueError):
    """``str(err)`` is the exact text fed back to the model."""


_CLAUSE_MENTION_RE = re.compile(
    rf"\bclauses?\s+((?:{CLAUSE_ID_RE})(?:\s*(?:,|and|y|&)\s*(?:clause\s+)?(?:{CLAUSE_ID_RE}))*)", re.IGNORECASE
)
_ID_RE = re.compile(rf"(?<![\d.]){CLAUSE_ID_RE}(?![\d.])")


def mentioned_clause_ids(message: str) -> set[str]:
    """Clause IDs referenced as 'clause X' / 'clauses X and Y' (or 'cláusula X')."""
    found: set[str] = set()
    text = re.sub(r"\bcl[aá]usulas?\b", "clause", message, flags=re.IGNORECASE)
    for m in _CLAUSE_MENTION_RE.finditer(text):
        found.update(_ID_RE.findall(m.group(1)))
    return found


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$%]+", text.lower())


def quotes_text(message: str, source: str, n: int = QUOTE_NGRAM) -> bool:
    msg, src = _words(message), _words(source)
    if len(src) < n or len(msg) < n:
        return False
    grams = {tuple(src[i : i + n]) for i in range(len(src) - n + 1)}
    return any(tuple(msg[i : i + n]) in grams for i in range(len(msg) - n + 1))


def validate_coverage(
    raw: str,
    retrieval: RetrievalResult,
    superseded: list[Chunk],
    threshold: float = RELEVANCE_THRESHOLD,
) -> CoverageAnswer:
    """Validate a raw model response against the schema and the citation guardrails.

    ``retrieval`` is the Top 5 the model was shown; ``superseded`` is every superseded clause in the
    corpus (none may be cited, named, or quoted).
    """
    if raw is None or not raw.strip():
        raise CoverageValidationError("Empty response: expected a single JSON object.")
    try:
        ans = CoverageAnswer.model_validate_json(raw.strip())
    except ValidationError as exc:
        lines = [f"{exc.error_count()} validation error(s) for CoverageAnswer:"]
        for err in exc.errors(include_url=False):
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"- {loc}: {err['msg']} [type={err['type']}]")
        raise CoverageValidationError("\n".join(lines)) from exc

    by_id = {c.clause_id: c for c in retrieval.chunks}
    superseded_ids = {c.clause_id for c in superseded}
    errors: list[str] = []

    not_retrieved = [c for c in ans.citations if c not in by_id]
    if not_retrieved:
        errors.append(f"citations {not_retrieved} are not in the retrieved Top 5 {list(by_id)}; cite only retrieved clauses")
    cited_superseded = [c for c in ans.citations if c in superseded_ids]
    if cited_superseded:
        errors.append(f"citations {cited_superseded} are superseded clauses; cite the current replacement instead")
    below = [
        c for c in ans.citations if c in by_id and (by_id[c].rerank is None or by_id[c].rerank < threshold)
    ]
    if below:
        errors.append(f"citations {below} do not clear the relevance threshold; cite only clauses marked clears_threshold=\"yes\"")

    if ans.evidence == "missing" and ans.citations:
        errors.append('evidence "missing" requires "citations": []')
    if ans.evidence == "enough" and not ans.citations:
        errors.append('evidence "enough" requires at least one citation')
    if ans.evidence == "conflicting" and len(ans.citations) < 2:
        errors.append('evidence "conflicting" requires citing both (all) disagreeing clause IDs')

    mentioned = mentioned_clause_ids(ans.customer_message)
    leaked = sorted(mentioned & superseded_ids)
    if leaked:
        errors.append(f"customer_message mentions superseded clause(s) {leaked}; never mention superseded clauses")
    uncited = sorted(mentioned - set(ans.citations) - superseded_ids)
    if uncited:
        errors.append(f"customer_message mentions clause(s) {uncited} that are not in citations")
    for c in superseded:
        if quotes_text(ans.customer_message, c.text):
            errors.append(f"customer_message quotes superseded clause {c.clause_id}; never quote superseded text")

    if errors:
        raise CoverageValidationError("Citation guardrail violation(s):\n" + "\n".join(f"- {e}" for e in errors))
    return ans


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^<!--\s*section:\s*([a-z_]+)\s*-->\s*$", re.MULTILINE)


@dataclass(frozen=True)
class CoveragePrompt:
    path: Path
    sha256: str
    system: str
    user_template: str
    retry_template: str

    def user(self, question: str, clauses: str) -> str:
        return self.user_template.replace("{{QUESTION}}", question.strip()).replace("{{CLAUSES}}", clauses)

    def retry(self, error: str) -> str:
        return self.retry_template.replace("{{VALIDATION_ERROR}}", error)


def load_coverage_prompt(path: Path = COVERAGE_PROMPT_PATH) -> CoveragePrompt:
    text = path.read_text(encoding="utf-8")
    matches = list(_SECTION_RE.finditer(text))
    sections = {
        m.group(1): text[m.end() : (matches[i + 1].start() if i + 1 < len(matches) else len(text))].strip()
        for i, m in enumerate(matches)
    }
    for name, marker in (("system", None), ("user", "{{QUESTION}}"), ("retry", "{{VALIDATION_ERROR}}")):
        if not sections.get(name):
            raise ValueError(f"{path} is missing section {name!r}")
        if marker and marker not in sections[name]:
            raise ValueError(f"{path}: section {name!r} must contain {marker}")
    if "{{CLAUSES}}" not in sections["user"]:
        raise ValueError(f"{path}: user section must contain {{{{CLAUSES}}}}")
    return CoveragePrompt(
        path=path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        system=sections["system"],
        user_template=sections["user"],
        retry_template=sections["retry"],
    )


def render_clauses(retrieval: RetrievalResult, threshold: float = RELEVANCE_THRESHOLD) -> str:
    out = []
    for c in retrieval.chunks:
        ch = c.chunk
        status = f'superseded" replaced_by="{ch.superseded_by}' if ch.is_superseded else "current"
        clears = "yes" if c.rerank is not None and c.rerank >= threshold else "no"
        out.append(
            f'<clause id="{ch.clause_id}" effective="{ch.effective_date}" status="{status}" '
            f'clears_threshold="{clears}">\n{ch.title}. {ch.text}\n</clause>'
        )
    return "\n".join(out) if out else "(no clauses retrieved)"


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------

DECLINE_ACCESS = (
    "Thank you for getting in touch. For privacy reasons we can only discuss a policy or claim with the "
    "policyholder or someone they have authorised in writing, so we can't share another customer's records. "
    "A Harbourline support agent will follow up with you."
)
MISSING_MESSAGE = (
    "Thank you for your question. Our policy handbook doesn't include a rule that answers this, so we don't "
    "want to guess. A Harbourline support agent will review your question and follow up — if you can share "
    "more detail about the treatment or service you mean, that will help them."
)
STALE_MESSAGE = (
    "Thank you for your question. The rule that matched it has been replaced by a newer version{cite}. "
    "A Harbourline support agent will confirm the current terms for you."
)


def customer_plan_lookup(policy_number: str | None) -> str:
    """Customer plan/status lookup: deferred to Week 3."""
    return "not wired"


@dataclass
class CoverageResult:
    question: str
    answer: CoverageAnswer | None
    gate: Gate
    stop_reason: StopReason
    retrieval: RetrievalResult
    shadow_retrieval: RetrievalResult | None = None  # superseded-inclusive check (stale gate)
    attempts: list[Attempt] = field(default_factory=list)
    prompt_file: str = ""
    prompt_sha256: str = ""
    customer_plan: str = "not wired"

    @property
    def needs_human_review(self) -> bool:
        return self.answer is None or self.answer.evidence in ("conflicting", "stale") or self.gate == "access_blocked"

    @property
    def input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.attempts)

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.attempts)

    @property
    def latency_ms(self) -> int:
        return sum(a.latency_ms for a in self.attempts)


def answer_coverage(
    question: str,
    retriever: Retriever,
    requester: Requester | None = None,
    call_model: CallModel | None = None,
    prompt: CoveragePrompt | None = None,
    threshold: float = RELEVANCE_THRESHOLD,
) -> CoverageResult:
    prompt = prompt or load_coverage_prompt()
    try:
        prompt_file = str(prompt.path.relative_to(ROOT))
    except ValueError:
        prompt_file = str(prompt.path)
    superseded = [c for c in retriever.chunks if c.is_superseded]

    ret = retriever.retrieve(question, mode="hybrid", requester=requester, score_relevance=True)
    base = dict(question=question, retrieval=ret, prompt_file=prompt_file, prompt_sha256=prompt.sha256)

    # Gate 1: access boundary — nothing retrieved, nothing generated.
    if ret.blocked_reason:
        return CoverageResult(
            answer=CoverageAnswer(evidence="missing", citations=[], customer_message=DECLINE_ACCESS),
            gate="access_blocked", stop_reason="gate", **base,
        )

    # Gate 2: nothing clears the relevance threshold.
    relevant_current = [c for c in ret.relevant(threshold) if not c.chunk.is_superseded]
    if not relevant_current:
        shadow = ret
        if not ret.include_superseded:
            shadow = retriever.retrieve(
                question, mode="hybrid", requester=requester, include_superseded=True, score_relevance=True
            )
        relevant_old = [c for c in shadow.relevant(threshold) if c.chunk.is_superseded]
        if relevant_old:
            in_top = set(shadow.clause_ids)
            replacements = sorted({c.chunk.superseded_by for c in relevant_old if c.chunk.superseded_by in in_top})
            cite = " (" + ", ".join(f"clause {r}" for r in replacements) + ")" if replacements else ""
            return CoverageResult(
                answer=CoverageAnswer(evidence="stale", citations=replacements, customer_message=STALE_MESSAGE.format(cite=cite)),
                gate="stale_only", stop_reason="gate", shadow_retrieval=shadow, **base,
            )
        return CoverageResult(
            answer=CoverageAnswer(evidence="missing", citations=[], customer_message=MISSING_MESSAGE),
            gate="below_threshold", stop_reason="gate", shadow_retrieval=shadow if shadow is not ret else None, **base,
        )

    # LLM: answer from the Top 5 with validated citations.
    if call_model is None:
        call_model = make_caller(Config.from_env())
    result = CoverageResult(answer=None, gate="llm", stop_reason="max_attempts_reached", **base)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt.user(question, render_clauses(ret, threshold))}]
    for n in range(1, MAX_MODEL_CALLS + 1):
        try:
            resp = call_model(prompt.system, messages)
        except ProviderError as exc:
            result.attempts.append(Attempt(attempt=n, valid=False, provider_error=str(exc)))
            result.stop_reason = "provider_error"
            return result
        attempt = Attempt(
            attempt=n, valid=False, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms, api_stop_reason=resp.api_stop_reason, request_id=resp.request_id,
            served_model=resp.served_model,
        )
        result.attempts.append(attempt)
        try:
            result.answer = validate_coverage(resp.text, ret, superseded, threshold)
        except CoverageValidationError as exc:
            attempt.raw_response = resp.text
            attempt.validation_error = str(exc)
            if n < MAX_MODEL_CALLS:
                messages.append(resp.assistant_message)
                messages.append({"role": "user", "content": prompt.retry(str(exc))})
            continue
        attempt.valid = True
        result.stop_reason = "valid_schema"
        return result
    return result
