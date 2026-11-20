# Harbourline Email Intake — Week 1 Pilot

Turns raw customer support emails into schema-validated `Intake` objects with draft replies or clarifying questions for Harbourline support agents. Autonomy is **Draft only**: nothing is sent, paid, adjudicated, or written to a ticketing system. See [`engagement/solution-brief.md`](engagement/solution-brief.md).

## Setup (clean checkout)

Requires Python 3.10+. Dependencies are pinned to exact versions in `requirements.txt` (`anthropic==2.12.0`, `openai==3.57.0`, `pydantic==2.13.4`, `python-dotenv==1.2.4`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY (default model) or ANTHROPIC_API_KEY + MODEL_NAME=claude-sonnet-5-5
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

## Tests

```bash
python -m unittest
```

These are offline tests with a fake provider, so no API key is needed. They cover:
- every validation rule
- the hard 3-call bound
- the exact validation error being fed back on retry
- the immediate stop on a provider error
- the input guards (empty or oversized email → 0 calls, `invalid_input`)
- the truncation hint
- `served_model` recording

## Repo layout

```
src/intake.py              pipeline, schema, validation, providers
prompts/intake_v1.md       the single versioned prompt
scripts/run_samples.py     sample runner → runs/week-1/
data/emails/*.txt          5 sample emails (fictional)
evals/cases.jsonl          placeholder for Week 4 evals
engagement/solution-brief.md
tests/test_intake.py
```

## Week 1 scope

Out of scope:
- sending email
- paying or adjudicating claims
- ticket, CRM or database writes (Week 3)
- policy handbook retrieval, RAG or embeddings (Week 2)
- any web UI or server
