# Project Brief: Harbourline Support Email Intake (Week 1)

## Problem Statement
Harbourline customer support receives unstructured emails containing coverage questions, billing disputes, claim status requests, and general messages. A raw email is not a ticket—critical identifiers (such as `policy_number`) are often missing, wording varies across languages, and customers frequently repeat conflicting statements they heard previously.

## Target User
- **Harbourline Support Agent**: Reviews incoming emails, asks customers for missing identifiers when needed, and sends final replies.

## Week 1 Goal
Build the smallest reliable intake pipeline (`one email in -> one validated Intake object out`) that prepares structured triage data and a draft customer message for a support agent without fabricating policy rules or executing external actions.

## Functional Requirements
1. **Typed Intake Interface (`src/intake.py`)**:
   - Expose `run(email: str) -> Intake` returning a strictly typed object with:
     - `category`: `"coverage" | "billing" | "claim" | "other"`
     - `missing_fields`: `list[str]` (e.g., `["policy_number"]`)
     - `next_action`: `"ask" | "draft"`
     - `customer_message`: `str` (clarifying question or proposed draft reply)
     - `confidence`: `float` (`0.0` to `1.0`)
2. **Schema Validation & Bounded Loop**:
   - Validate every model response in code (`validate_intake`).
   - On validation failure, pass the validation error back to the model and retry.
   - Stop immediately when validation succeeds OR when **3 model calls** have been made (whichever comes first). Log the stop reason and never drop validation failures.
3. **Versioned Prompt & Environment Config**:
   - Store the sole prompt in `prompts/intake_v1.md` and load it from disk at runtime.
   - Read model configuration (`MODEL_NAME`, `MAX_TOKENS`, `TEMPERATURE`, API key) from environment variables.
4. **Sample Execution & Telemetry (`scripts/run_samples.py`)**:
   - Read all `data/emails/*.txt` files, run `intake.run()`, print one schema-valid JSON object per file, and record input tokens, output tokens, latency (ms), attempt count, and stop reason under `runs/week-1/`.

## Acceptance Test Scenarios (5 Sample Emails)
1. **Priya (Coverage, missing policy number)**: Specialist/dermatologist question without policy number $\rightarrow$ `category: "coverage"`, `missing_fields` includes `"policy_number"`, `next_action: "ask"`, no fabricated coverage/referral rules.
2. **Billing inquiry**: Includes policy number and dollar amount $\rightarrow$ `category: "billing"`, `next_action: "draft"`.
3. **Claims status**: Includes claim ID $\rightarrow$ `category: "claim"`.
4. **Thank-you note**: No actionable request $\rightarrow$ `category: "other"`.
5. **Bilingual specialist inquiry**: Two languages asking about a specialist $\rightarrow$ `category: "coverage"`.

## Out of Scope (Strict Exclusions)
- **No sending emails** automatically (autonomy is **Draft** only).
- **No paying or adjudicating claims**.
- **No external ticket/CRM writes** (Week 3).
- **No policy handbook retrieval / RAG** (Week 2).
- **No web UI**.