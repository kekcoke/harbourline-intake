"""Offline tests for validate_intake and the bounded retry loop. No network, no API keys.

Run from the repo root:  python -m unittest
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_samples as run_samples  # noqa: E402
from src.intake import (  # noqa: E402
    MAX_EMAIL_CHARS,
    Config,
    MAX_MODEL_CALLS,
    Intake,
    IntakeValidationError,
    ModelResponse,
    ProviderError,
    load_prompt,
    provider_for,
    run_with_trace,
    validate_intake,
)

logging.disable(logging.CRITICAL)  # the loop logs every failed attempt; keep test output clean

GOOD = {
    "category": "coverage",
    "missing_fields": ["policy_number"],
    "next_action": "ask",
    "customer_message": "Could you please share your policy number?",
    "confidence": 0.9,
}


def dumps(**overrides) -> str:
    return json.dumps({**GOOD, **overrides})


class ValidateIntakeTests(unittest.TestCase):
    def test_accepts_good_object(self):
        intake = validate_intake(dumps())
        self.assertIsInstance(intake, Intake)
        self.assertEqual(intake.category, "coverage")

    def test_accepts_draft_without_missing_fields(self):
        validate_intake(dumps(category="billing", missing_fields=[], next_action="draft"))

    def test_accepts_integer_confidence_bounds(self):
        validate_intake(dumps(confidence=1))
        validate_intake(dumps(confidence=0))

    def assertRejects(self, raw: str, fragment: str):
        with self.assertRaises(IntakeValidationError) as ctx:
            validate_intake(raw)
        self.assertIn(fragment, str(ctx.exception))

    def test_rejects_empty(self):
        self.assertRejects("", "Empty response")
        self.assertRejects("   \n", "Empty response")

    def test_rejects_markdown_fence(self):
        self.assertRejects("```json\n" + dumps() + "\n```", "json_invalid")

    def test_rejects_trailing_prose(self):
        self.assertRejects(dumps() + "\nHope this helps!", "json_invalid")

    def test_rejects_non_object(self):
        self.assertRejects(json.dumps([GOOD]), "model_type")

    def test_rejects_extra_key(self):
        self.assertRejects(dumps(priority="high"), "priority")

    def test_rejects_missing_key(self):
        data = dict(GOOD)
        del data["confidence"]
        self.assertRejects(json.dumps(data), "confidence")

    def test_rejects_bad_enums(self):
        self.assertRejects(dumps(category="Coverage"), "category")
        self.assertRejects(dumps(next_action="send"), "next_action")

    def test_rejects_type_coercion(self):
        self.assertRejects(dumps(confidence="0.9"), "confidence")
        self.assertRejects(dumps(missing_fields="policy_number"), "missing_fields")

    def test_rejects_confidence_out_of_range(self):
        self.assertRejects(dumps(confidence=1.2), "confidence")
        self.assertRejects(dumps(confidence=-0.1), "confidence")

    def test_rejects_bad_missing_field_names(self):
        self.assertRejects(dumps(missing_fields=["Policy Number"]), "snake_case")
        self.assertRejects(dumps(missing_fields=["policy_number", "policy_number"]), "duplicates")

    def test_rejects_empty_or_long_message(self):
        self.assertRejects(dumps(customer_message="   "), "customer_message")
        self.assertRejects(dumps(customer_message="x" * 1501), "customer_message")

    def test_rejects_inconsistent_action(self):
        self.assertRejects(dumps(missing_fields=[], next_action="ask"), "'ask' but missing_fields is empty")
        self.assertRejects(dumps(next_action="draft"), "must be 'ask'")

    def test_rejects_other_with_missing_fields(self):
        self.assertRejects(dumps(category="other"), "category 'other'")


class FakeModel:
    """Scripted stand-in for a provider. Records every call it receives."""

    def __init__(self, outputs, stop_reason="stop"):
        self.outputs = list(outputs)
        self.stop_reason = stop_reason
        self.calls: list[list[dict]] = []

    def __call__(self, system, messages):
        self.calls.append([dict(m) for m in messages])
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return ModelResponse(
            text=out,
            input_tokens=100,
            output_tokens=20,
            latency_ms=5,
            api_stop_reason=self.stop_reason,
            request_id=f"req_{len(self.calls)}",
            assistant_message={"role": "assistant", "content": out},
            served_model="fake-model-1",
        )


class RetryLoopTests(unittest.TestCase):
    def setUp(self):
        self.prompt = load_prompt()

    def test_valid_first_try(self):
        fake = FakeModel([dumps()])
        trace = run_with_trace("email", call_model=fake, prompt=self.prompt)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(trace.stop_reason, "valid_schema")
        self.assertEqual(trace.outcome, "valid")
        self.assertEqual((trace.input_tokens, trace.output_tokens, trace.latency_ms), (100, 20, 5))

    def test_never_exceeds_three_calls(self):
        fake = FakeModel(["not json"] * 10)
        trace = run_with_trace("email", call_model=fake, prompt=self.prompt)
        self.assertEqual(MAX_MODEL_CALLS, 3)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(trace.stop_reason, "max_attempts_reached")
        self.assertEqual(trace.outcome, "failed")
        self.assertIsNone(trace.intake)
        self.assertEqual(len(trace.attempts), 3)
        for a in trace.attempts:
            self.assertEqual(a.raw_response, "not json")
            self.assertTrue(a.validation_error)

    def test_exact_error_is_fed_back(self):
        fake = FakeModel(["not json", dumps(confidence=5), dumps()])
        trace = run_with_trace("email", call_model=fake, prompt=self.prompt)
        self.assertEqual(trace.stop_reason, "valid_schema")
        self.assertEqual(len(fake.calls), 3)
        # Turn 2: [user email, assistant raw, user retry containing attempt-1 error]
        second = fake.calls[1]
        self.assertEqual([m["role"] for m in second], ["user", "assistant", "user"])
        self.assertEqual(second[1]["content"], "not json")
        self.assertIn(trace.attempts[0].validation_error, second[2]["content"])
        # Turn 3 carries the attempt-2 error verbatim.
        self.assertIn(trace.attempts[1].validation_error, fake.calls[2][-1]["content"])
        self.assertIn("confidence", trace.attempts[1].validation_error)

    def test_invalid_then_valid(self):
        fake = FakeModel(["```json\n{}\n```", dumps()])
        trace = run_with_trace("email", call_model=fake, prompt=self.prompt)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(trace.stop_reason, "valid_schema")
        self.assertEqual(len(trace.errors), 1)

    def test_provider_error_stops_immediately(self):
        fake = FakeModel([ProviderError("boom"), dumps()])
        trace = run_with_trace("email", call_model=fake, prompt=self.prompt)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(trace.stop_reason, "provider_error")
        self.assertEqual(trace.attempts[0].provider_error, "boom")

    def test_served_model_recorded(self):
        trace = run_with_trace("email", call_model=FakeModel([dumps()]), prompt=self.prompt)
        self.assertEqual(trace.attempts[0].served_model, "fake-model-1")


class InputGuardTests(unittest.TestCase):
    def setUp(self):
        self.prompt = load_prompt()

    def assertRejectedInput(self, email: str, fragment: str):
        fake = FakeModel([dumps()])
        trace = run_with_trace(email, call_model=fake, prompt=self.prompt)
        self.assertEqual(len(fake.calls), 0)
        self.assertEqual(trace.stop_reason, "invalid_input")
        self.assertEqual(trace.outcome, "failed")
        self.assertEqual(trace.attempts, [])
        self.assertIn(fragment, trace.input_error)
        self.assertIn(fragment, trace.errors[0])

    def test_empty_email_rejected(self):
        self.assertRejectedInput("", "empty")
        self.assertRejectedInput("  \n\t ", "empty")

    def test_oversized_email_rejected(self):
        self.assertRejectedInput("x" * (MAX_EMAIL_CHARS + 1), "limit")

    def test_email_at_limit_accepted(self):
        fake = FakeModel([dumps()])
        trace = run_with_trace("x" * MAX_EMAIL_CHARS, call_model=fake, prompt=self.prompt)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(trace.stop_reason, "valid_schema")


class TruncationHintTests(unittest.TestCase):
    def setUp(self):
        self.prompt = load_prompt()

    def test_truncated_response_gets_hint(self):
        fake = FakeModel(['{"category": "cov', dumps()], stop_reason="length")
        trace = run_with_trace("email", call_model=fake, prompt=self.prompt, max_tokens=256)
        self.assertEqual(trace.stop_reason, "valid_schema")
        error = trace.attempts[0].validation_error
        self.assertTrue(error.startswith("Response was cut off at the MAX_TOKENS limit (256)"))
        self.assertIn("json_invalid", error)
        self.assertIn(error, fake.calls[1][-1]["content"])

    def test_no_hint_when_not_truncated(self):
        fake = FakeModel(["not json", dumps()], stop_reason="stop")
        trace = run_with_trace("email", call_model=fake, prompt=self.prompt, max_tokens=256)
        self.assertNotIn("cut off", trace.attempts[0].validation_error)


# Scripted replies per email for the runner log tests. Each list has one more reply than
# the loop may use, so consuming it would prove the 3-call ceiling was exceeded.
CASE_A_REPLIES = ["not json", dumps(), "UNUSED-4th"]
CASE_B_REPLIES = [
    "not json",
    "```json\n" + dumps() + "\n```",
    dumps(confidence=5),
    "UNUSED-4th",
]


class ScriptedByEmail:
    """Fake provider that picks a FakeModel based on which test email is being processed."""

    def __init__(self, scripts: dict[str, list[str]]):
        self.models = {marker: FakeModel(replies) for marker, replies in scripts.items()}

    def __call__(self, system, messages):
        first_user = messages[0]["content"]
        for marker, model in self.models.items():
            if marker in first_user:
                return model(system, messages)
        raise AssertionError("unscripted email")


class RunnerLogTests(unittest.TestCase):
    """Case A (recover on attempt 2) and Case B (3 failures) through the real runner + log writers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # removed in tearDown
        root = Path(self.tmp.name)
        emails = root / "emails"
        emails.mkdir()
        (emails / "a_recovers.txt").write_text("CASE-A: what does my plan cover?", encoding="utf-8")
        (emails / "b_exhausts.txt").write_text("CASE-B: billing question", encoding="utf-8")
        self.runs = root / "runs"
        self.fake = ScriptedByEmail({"CASE-A": CASE_A_REPLIES, "CASE-B": CASE_B_REPLIES})
        cfg = Config(model_name="deepseek-flash", max_tokens=2048, temperature=0.0, effort="low",
                     deepseek_base_url="https://api.deepseek.com")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.records = run_samples.run_suite(
                sorted(emails.glob("*.txt")), self.runs, cfg, load_prompt(), self.fake, "test-run"
            )
        self.log = {r["email_file"]: r for r in map(json.loads, (self.runs / "run_log.jsonl").read_text().splitlines())}
        self.tokens = [json.loads(l) for l in (self.runs / "tokens.jsonl").read_text().splitlines()]

    def tearDown(self):
        self.tmp.cleanup()

    def assertRetriesCarryErrors(self, model: FakeModel, record: dict):
        attempts = record["attempt_log"]
        for i, call in enumerate(model.calls[1:], start=1):
            prev = attempts[i - 1]
            self.assertEqual(call[-2], {"role": "assistant", "content": prev["raw_response"]})
            self.assertIn(prev["validation_error"], call[-1]["content"])
            self.assertEqual(call[-1]["role"], "user")

    def assertTotals(self, record: dict):
        for key in ("input_tokens", "output_tokens", "latency_ms"):
            self.assertEqual(record[key], sum(a[key] for a in record["attempt_log"]))
            self.assertGreater(record[key], 0)

    def test_case_a_recovers_on_attempt_2(self):
        model = self.fake.models["CASE-A"]
        rec = self.log["a_recovers.txt"]
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.outputs, ["UNUSED-4th"])
        self.assertEqual((rec["outcome"], rec["stop_reason"], rec["attempts"]), ("valid", "valid_schema", 2))
        self.assertEqual(len(rec["validation_errors"]), 1)
        self.assertIn("json_invalid", rec["validation_errors"][0])
        self.assertEqual(rec["attempt_log"][0]["raw_response"], "not json")
        self.assertIsNone(rec["attempt_log"][1]["raw_response"])
        self.assertTrue(rec["attempt_log"][1]["valid"])
        self.assertRetriesCarryErrors(model, rec)
        self.assertTotals(rec)
        validate_intake((self.runs / "a_recovers.json").read_text())
        self.assertFalse((self.runs / "a_recovers.failed.json").exists())

    def test_case_b_hits_three_call_ceiling(self):
        model = self.fake.models["CASE-B"]
        rec = self.log["b_exhausts.txt"]
        self.assertEqual(len(model.calls), MAX_MODEL_CALLS)
        self.assertEqual(model.outputs, ["UNUSED-4th"])
        self.assertEqual((rec["outcome"], rec["stop_reason"], rec["attempts"]), ("failed", "max_attempts_reached", 3))
        self.assertEqual(len(rec["validation_errors"]), 3)
        self.assertEqual([a["raw_response"] for a in rec["attempt_log"]], CASE_B_REPLIES[:3])
        self.assertIn("less_than_equal", rec["validation_errors"][2])
        self.assertRetriesCarryErrors(model, rec)
        self.assertTotals(rec)
        failed = json.loads((self.runs / "b_exhausts.failed.json").read_text())
        self.assertEqual(failed["stop_reason"], "max_attempts_reached")
        self.assertEqual([a["raw_response"] for a in failed["attempts"]], CASE_B_REPLIES[:3])
        self.assertTrue(all(a["validation_error"] for a in failed["attempts"]))
        self.assertFalse((self.runs / "b_exhausts.json").exists())

    def test_tokens_jsonl_and_summary(self):
        self.assertEqual(len(self.tokens), 5)
        self.assertEqual([(t["email_file"], t["attempt"], t["error_type"]) for t in self.tokens], [
            ("a_recovers.txt", 1, "validation"), ("a_recovers.txt", 2, None),
            ("b_exhausts.txt", 1, "validation"), ("b_exhausts.txt", 2, "validation"), ("b_exhausts.txt", 3, "validation"),
        ])
        for t in self.tokens:
            self.assertGreater(t["input_tokens"], 0)
            self.assertGreater(t["latency_ms"], 0)
        summary = json.loads((self.runs / "summary.json").read_text())
        self.assertEqual((summary["valid"], summary["failed"], summary["total_model_calls"]), (1, 1, 5))


class PromptAndConfigTests(unittest.TestCase):
    def test_prompt_sections_load(self):
        p = load_prompt()
        self.assertIn("JSON", p.system)
        self.assertIn("json", p.system)  # DeepSeek JSON mode requires the lowercase word
        self.assertNotIn("{{EMAIL}}", p.user("hello"))
        self.assertIn("hello", p.user("hello"))
        self.assertIn("ERR", p.retry("ERR"))
        self.assertEqual(len(p.sha256), 64)

    def test_prompt_files(self):
        # Week 1 intake uses only intake_v1.md; Week 2 adds coverage_v1.md for the RAG stage.
        self.assertEqual(sorted(f.name for f in (ROOT / "prompts").iterdir()), ["coverage_v1.md", "intake_v1.md"])
        self.assertEqual(load_prompt().path.name, "intake_v1.md")

    def test_provider_inference(self):
        self.assertEqual(provider_for("deepseek-flash"), "deepseek")
        self.assertEqual(provider_for("claude-sonnet-5-5"), "anthropic")
        with self.assertRaises(ValueError):
            provider_for("gpt-4o")


if __name__ == "__main__":
    unittest.main()
