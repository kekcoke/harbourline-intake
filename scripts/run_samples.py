"""Run the intake pipeline across all sample emails in data/emails/*.txt.

stdout: one schema-valid Intake JSON object per successfully processed email (one per line).
stderr: progress and failure summaries.
runs/week-1/: per-email Intake JSON, <stem>.failed.json for failures, run_log.jsonl, summary.json.
Exit code: 0 if every email produced a valid Intake, 1 otherwise.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intake import (  # noqa: E402
    MAX_MODEL_CALLS,
    Config,
    IntakeTrace,
    ProviderError,
    load_prompt,
    make_caller,
    run_with_trace,
)

EMAILS_DIR = ROOT / "data" / "emails"
RUNS_DIR = ROOT / "runs" / "week-1"
RUN_LOG = RUNS_DIR / "run_log.jsonl"
SUMMARY = RUNS_DIR / "summary.json"


def _clear_previous_outputs() -> None:
    """Remove only the files this script writes, so stale results never linger."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for path in [*RUNS_DIR.glob("*.json"), RUN_LOG]:
        if path.is_file():
            path.unlink()


def _log_record(run_id: str, email_path: Path, cfg: Config, trace: IntakeTrace) -> dict:
    return {
        "run_id": run_id,
        "email_file": email_path.name,
        "provider": cfg.provider,
        "model": cfg.model_name,
        "outcome": trace.outcome,
        "stop_reason": trace.stop_reason,
        "attempts": len(trace.attempts),
        "max_attempts": MAX_MODEL_CALLS,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "latency_ms": trace.latency_ms,
        "validation_errors": [a.validation_error for a in trace.attempts if a.validation_error],
        "provider_errors": [a.provider_error for a in trace.attempts if a.provider_error],
        "prompt_file": trace.prompt_file,
        "prompt_sha256": trace.prompt_sha256,
        "attempt_log": [asdict(a) for a in trace.attempts],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    email_paths = sorted(EMAILS_DIR.glob("*.txt"))
    if not email_paths:
        print(f"No sample emails found in {EMAILS_DIR}", file=sys.stderr)
        return 1

    try:  # fail fast on a bad MODEL_NAME, a missing API key or a missing SDK
        cfg = Config.from_env()
        prompt = load_prompt()
        call_model = make_caller(cfg)
    except (ProviderError, ValueError, ImportError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    _clear_previous_outputs()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"run {run_id}: {len(email_paths)} email(s), model={cfg.model_name} ({cfg.provider})", file=sys.stderr)

    records = []
    with RUN_LOG.open("w", encoding="utf-8") as log_file:
        for email_path in email_paths:
            email = email_path.read_text(encoding="utf-8")
            trace = run_with_trace(email, call_model=call_model, prompt=prompt)
            record = _log_record(run_id, email_path, cfg, trace)
            records.append(record)
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

            if trace.intake is not None:
                out = trace.intake.model_dump(mode="json")
                (RUNS_DIR / f"{email_path.stem}.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps(out, ensure_ascii=False))
                status = "valid"
            else:
                failure = {
                    "email_file": email_path.name,
                    "outcome": "failed",
                    "stop_reason": trace.stop_reason,
                    "attempts": [
                        {
                            "attempt": a.attempt,
                            "raw_response": a.raw_response,
                            "validation_error": a.validation_error,
                            "provider_error": a.provider_error,
                            "api_stop_reason": a.api_stop_reason,
                            "request_id": a.request_id,
                        }
                        for a in trace.attempts
                    ],
                }
                (RUNS_DIR / f"{email_path.stem}.failed.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                status = "FAILED"

            print(
                f"  {email_path.name}: {status} stop={trace.stop_reason} attempts={len(trace.attempts)} "
                f"in={trace.input_tokens} out={trace.output_tokens} latency={trace.latency_ms}ms",
                file=sys.stderr,
            )

    valid = sum(r["outcome"] == "valid" for r in records)
    summary = {
        "run_id": run_id,
        **cfg.as_log_dict(),
        "max_attempts": MAX_MODEL_CALLS,
        "prompt_file": prompt.path.relative_to(ROOT).as_posix(),
        "prompt_sha256": prompt.sha256,
        "emails": len(records),
        "valid": valid,
        "failed": len(records) - valid,
        "total_model_calls": sum(r["attempts"] for r in records),
        "total_input_tokens": sum(r["input_tokens"] for r in records),
        "total_output_tokens": sum(r["output_tokens"] for r in records),
        "total_latency_ms": sum(r["latency_ms"] for r in records),
        "stop_reasons": {r["email_file"]: r["stop_reason"] for r in records},
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done: {valid}/{len(records)} valid → {RUNS_DIR}/", file=sys.stderr)
    return 0 if valid == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
