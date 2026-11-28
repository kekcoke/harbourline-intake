# Harbourline Email Intake — Week 1 Intake + Week 2 Handbook RAG

Turns raw customer support emails into schema-validated `Intake` objects with draft replies or clarifying questions for Harbourline support agents. Autonomy is **Draft only**: nothing is sent, paid, adjudicated, or written to a ticketing system. See [`engagement/solution-brief.md`](engagement/solution-brief.md).

## Setup (clean checkout)

Requires Python 3.10+ (tested on 3.14). Dependencies are pinned to exact versions in `requirements.txt` (`anthropic==2.12.0`, `openai==3.57.0`, `pydantic==2.13.4`, `python-dotenv==1.2.4`, and for Week 2 `pinecone==10.3.0`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY (default model) or ANTHROPIC_API_KEY + MODEL_NAME=claude-sonnet-5-5
# Week 2: also set PINECONE_API_KEY
```

## Run all samples

```bash
python scripts/run_samples.py
```

This runs every `data/emails/*.txt` through `intake.run_with_trace()`, prints one schema-valid `Intake` JSON object per email to **stdout** (progress goes to stderr), and regenerates `runs/week-1/`. Exit code is `0` when every email validated, `1` if any failed, `2` on a configuration error.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `MODEL_NAME` | `deepseek-flash` | Provider inferred from prefix: `deepseek-*` → DeepSeek, `claude-*` → Anthropic. Alternative: `claude-sonnet-5-5`. |
| `MAX_TOKENS` | `2048` | Max output tokens per model call. |
| `TEMPERATURE` | `0.0` | Sent to DeepSeek (with thinking disabled so it takes effect). **Logged but not sent to Claude** — the current Claude API (version `2026-10-21`, `anthropic` SDK ≥ 2.0) rejects `temperature` on every model. Each run records `temperature_sent`. |
| `EFFORT` | `low` | Claude only (`output_config.effort`). |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek's OpenAI-compatible endpoint. |
| `REQUEST_TIMEOUT_S` | `60` | Per-request timeout for both providers. The SDK default is 10 min. A timeout ends that email as `provider_error`. |
| `DEEPSEEK_API_KEY` | — | Required when `MODEL_NAME` is `deepseek-*`. |
| `ANTHROPIC_API_KEY` | — | Required when `MODEL_NAME` is `claude-*`. |

Two limits are code constants in `src/intake.py`, deliberately not configurable:
- `MAX_MODEL_CALLS = 3`: model calls per email.
- `MAX_EMAIL_CHARS = 20_000`: longer emails, and empty ones, are rejected before any model call.

## How it works

```
email ──► prompts/intake_v1.md (system + user) ──► model ──► validate_intake()
                                                     ▲            │ fail
                                                     └── exact validation error (retry section)
stop at the first valid object, or after 3 model calls — whichever comes first
```

- **`src/intake.py`**: `Intake` (strict Pydantic model), `validate_intake(raw)`, `run(email) -> Intake` (raises `IntakeFailed` with the trace on failure), and `run_with_trace(email) -> IntakeTrace` (tokens, latency, attempts, stop reason).
- **`prompts/intake_v1.md`**: the only prompt. It is loaded from disk at runtime and split into `system`, `user` and `retry` sections. Its sha256 is logged with every run.
- **Validation rules:**
  - The response must be a single JSON object, with no markdown fences and no surrounding text.
  - Exactly the five keys are allowed, with no type coercion and exact enums.
  - `confidence` must be between 0.0 and 1.0.
  - `missing_fields` must be unique snake_case names, at most 5.
  - `customer_message` must be non-empty and at most 1500 characters.
  - `next_action == "ask"` if and only if `missing_fields` is non-empty.
  - `category == "other"` requires an empty `missing_fields`.
- **Stop reasons:**
  - `valid_schema`: an object validated.
  - `max_attempts_reached`: 3 calls were made and none validated.
  - `provider_error`: an API error, timeout or Claude refusal. The loop stops immediately, because retrying with a validation error wouldn't help.
  - `invalid_input`: the email was empty or over `MAX_EMAIL_CHARS`. No model call was made.
- **Truncation hint:** if a failed response was cut off at `MAX_TOKENS` (provider stop reason `max_tokens` or `length`), the error fed back to the model starts with a note saying so.
- SDK auto-retries are disabled, so 3 model calls means at most 3 HTTP requests.
- On Claude Opus 5.5, Sonnet 5.5 and Fable 5.x, server-side refusal fallback (`fallbacks: "default"`) is enabled.

## Outputs: `runs/week-1/`

| File | Contents |
|---|---|
| `<email-stem>.json` | The validated `Intake` for that email. |
| `<email-stem>.failed.json` | Only when an email fails: stop reason, any `input_error`, plus every attempt's raw model response and validation/provider error. |
| `run_log.jsonl` | One line per email: `outcome` (`valid`/`failed`), `stop_reason`, `attempts`, `input_tokens`, `output_tokens`, `latency_ms`, `input_error`, `validation_errors`, prompt file + sha256, and a per-attempt `attempt_log`. |
| `tokens.jsonl` | One flat line per model call: `email_file`, `attempt`, `provider`, `model`, `served_model`, `input_tokens`, `output_tokens`, `latency_ms`, `api_stop_reason`, `request_id`, `valid`, and `error_type` (`validation`/`provider`/null). |
| `summary.json` | Run config (model, provider, max_tokens, temperature, temperature_sent, effort, request_timeout_s), counts, and token/latency totals. |

Each run clears and rewrites these files.

## Week 2: Handbook RAG with strict citations

Coverage questions are answered from the Harbourline policy handbook. Each answer either cites clauses or returns an explicit refusal state. Customer plan and status data is still a stub that returns `"not wired"` (Week 3). Live billing is out of scope.

> The handbook in `data/handbook/` is **synthetic and fictional**. It was written to exercise supersession (4.2 → 4.2a, 7.4 → 7.4a), a conflicting active pair (6.1 vs 9.3) and a gap (no veterinary cover).

### Commands

```bash
python scripts/ingest.py              # build the indexes: data/index/chunks.jsonl + Pinecone vectors
python scripts/ingest.py --keyword-only   # chunks only (BM25 is built from chunks.jsonl at load time)
python scripts/eval_retrieval.py      # Keyword-only vs Hybrid benchmark → docs/retrieval-report.md (no LLM)
python scripts/run_week2.py           # intake + coverage answers → runs/week-2/ (exit 0 = all acceptance checks pass)
```

### Ingestion (`scripts/ingest.py`, `src/handbook.py`)

- **Chunking:** one chunk per clause heading `### <clause_id> <title>`. A chunk never spans two clauses, and a heading with no rule text is skipped. Files with no clause IDs are excluded (the marketing brochure).
- **Chunk metadata:** `document`, `clause_id`, `effective_date`, `superseded_by` (or `null`) and `text` (the full clause text), plus `title`.
- **Validation:** clause IDs must be unique, and every `superseded_by` must point to a clause that exists.
- **Keyword index:** Okapi BM25 (`src/retrieval.py`).
- **Vector index:** Pinecone serverless `harbourline-handbook`, embedded with `multilingual-e5-large`. The ID is the `clause_id`. Vectors whose clause no longer exists are deleted. Both indexes cover the same `chunks.jsonl`.

### Retrieval (`src/retrieval.py`, no LLM)

`retrieve(query, k=5, mode="hybrid"|"keyword", include_superseded=None, requester=None, score_relevance=False)` returns the Top 5 with `bm25`, `dense`, `fused` and `rerank` scores. It runs these steps in order:

1. **Access boundary.** Retrieval returns 0 chunks and sets a `blocked_reason`, without touching any index, when:
   - the query asks for someone else's records or claim notes ("his claim", "my neighbour's claim"), or
   - it names an `HL-`/`CLM-` identifier the authenticated requester doesn't own.
2. **Superseded filter.** Clauses with `superseded_by` set are dropped unless the query explicitly asks about an earlier rule ("old rule", "before 2026", "previously").
3. **Hybrid fusion.** BM25 is divided by its max and dense cosine is min-max scaled. Fused = 0.5·dense + 0.5·BM25. When fused #1 − #2 < 0.10, the top 10 are reranked with `bge-reranker-v2-m3`. `mode="keyword"` runs BM25 only, for benchmarking.

### Coverage answers (`src/coverage.py`, `prompts/coverage_v1.md`)

Output: `{"evidence": "enough|missing|conflicting|stale", "citations": [...], "customer_message": "..."}`.

- **Gate (deterministic, no LLM call):**
  - If access is blocked, the answer is `missing` with `[]` and a privacy decline.
  - If no current clause clears the relevance threshold (cross-encoder score ≥ `RELEVANCE_THRESHOLD = 0.01`), and a superseded clause does, the answer is `stale`. It cites the active replacement if that is in the Top 5.
  - If nothing clears the threshold at all, the answer is `missing` with `[]`.
- **Model step.** Otherwise the model answers from the Top 5. Each clause is marked current or superseded, and whether it clears the threshold.
- **Guardrails, checked in code.** A failure is fed back to the model, up to 3 calls in total. The rules are:
  - Citations must be in the retrieved Top 5, clear the threshold, and be current.
  - `enough` needs at least 1 citation, `conflicting` at least 2, and `missing` needs `[]`.
  - The message may mention only cited clauses.
  - The message may never name or quote a superseded clause. Quoting means sharing 8 or more consecutive words with it.
- **Human review.** `conflicting`, `stale` and access-blocked results are flagged `needs_human_review`.
- **Missing policy number.** A coverage email without a policy number still gets the general cited rule. Week 1's `Intake` keeps asking for `policy_number` so an agent can confirm plan-specific details.

The relevance threshold is a noise floor, not a sufficiency test. The calibration data in `docs/retrieval-report.md` shows relevant and irrelevant cross-encoder scores overlapping above it. Whether a clause that clears it actually answers the question is the model's judgment, and its citations are then checked in code.

### Outputs: `runs/week-2/`

| File | Contents |
|---|---|
| `answers.jsonl` | One record per input. It has the Week 1 intake, the retrieval Top 5 with scores and full clause text, the gate, the `CoverageAnswer`, attempts, tokens and latency, and `superseded_leak`. |
| `access-test.json` | Requests for another customer's records. Each one has 0 chunks, `customer_fields: {}` and a pass flag, and there is a control case that is not blocked. |
| `summary.json` | Config, prompt hashes, evidence per input, token and latency totals, and the acceptance checks. |
| `tokens.jsonl` | One line per model call (`stage`: `intake` or `coverage`). |
| `retrieval-eval.json` | Raw benchmark scores (written by `eval_retrieval.py`). |

## Tests

```bash
python -m unittest
```

These are offline tests with a fake provider and a fake dense backend, so no API key or network is needed. The Week 1 tests (`tests/test_intake.py`) cover:
- every validation rule
- the hard 3-call bound
- the exact validation error being fed back on retry
- the immediate stop on a provider error
- the input guards (empty or oversized email → 0 calls, `invalid_input`)
- the truncation hint
- `served_model` recording

The Week 2 tests (`tests/test_week2.py`) cover:
- clause parsing and the brochure exclusion
- BM25 ranking
- hybrid fusion and the rerank trigger
- the superseded filter and historical detection
- the access guard
- every citation guardrail
- the coverage gate (no LLM call for `missing`, `stale` or a blocked request)
- guardrail feedback and the 3-call bound

## Repo layout

```
src/intake.py              Week 1 pipeline, schema, validation, providers
src/handbook.py            clause parser + chunks.jsonl I/O
src/retrieval.py           BM25, hybrid fusion/rerank, superseded filter, access guard (no LLM)
src/vectorstore.py         Pinecone index + Inference embed/rerank
src/coverage.py            CoverageAnswer, citation guardrails, gate + bounded loop
prompts/intake_v1.md       Week 1 intake prompt
prompts/coverage_v1.md     Week 2 coverage prompt
scripts/run_samples.py     Week 1 sample runner → runs/week-1/
scripts/ingest.py          builds data/index/chunks.jsonl + Pinecone vectors
scripts/eval_retrieval.py  Keyword vs Hybrid benchmark → docs/retrieval-report.md
scripts/run_week2.py       Week 2 runner → runs/week-2/
data/emails/*.txt          5 Week 1 sample emails (fictional)
data/emails/week-2/*.txt   Week 2 fixtures: veterinary, physio conflict, other customer's claim
data/handbook/             fictional policy handbook + excluded brochure
evals/retrieval_queries.jsonl  10 labelled retrieval queries
evals/cases.jsonl          placeholder for Week 4 evals
engagement/solution-brief.md
tests/test_intake.py, tests/test_week2.py
```

## Scope

Out of scope (through Week 2):
- sending email
- paying or adjudicating claims
- ticket, CRM or database writes, and customer plan/status lookup (Week 3; stubbed as `"not wired"`)
- live billing or premium data
- any web UI or server
