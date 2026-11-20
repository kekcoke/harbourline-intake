"""Offline tests for validate_intake and the bounded retry loop. No network, no API keys.

Run from the repo root:  python -m unittest
"""

from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intake import (  # noqa: E402
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

    def __init__(self, outputs):
        self.outputs = list(outputs)
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
            api_stop_reason="stop",
            request_id=f"req_{len(self.calls)}",
            assistant_message={"role": "assistant", "content": out},
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


class PromptAndConfigTests(unittest.TestCase):
    def test_prompt_sections_load(self):
        p = load_prompt()
        self.assertIn("JSON", p.system)
        self.assertIn("json", p.system)  # DeepSeek JSON mode requires the lowercase word
        self.assertNotIn("{{EMAIL}}", p.user("hello"))
        self.assertIn("hello", p.user("hello"))
        self.assertIn("ERR", p.retry("ERR"))
        self.assertEqual(len(p.sha256), 64)

    def test_only_one_prompt_file(self):
        self.assertEqual([f.name for f in (ROOT / "prompts").iterdir()], ["intake_v1.md"])

    def test_provider_inference(self):
        self.assertEqual(provider_for("deepseek-flash"), "deepseek")
        self.assertEqual(provider_for("claude-sonnet-5-5"), "anthropic")
        with self.assertRaises(ValueError):
            provider_for("gpt-4o")


if __name__ == "__main__":
    unittest.main()
