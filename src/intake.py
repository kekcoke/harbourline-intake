"""Core intake pipeline: one email in, one validated Intake object out.

Public API
----------
- ``Intake``                 strictly typed result schema (Pydantic v2).
- ``validate_intake(raw)``   deterministic schema + business-rule check on a raw model response.
- ``run(email)``             returns a validated ``Intake`` or raises ``IntakeFailed``.
- ``run_with_trace(email)``  same code path, but returns an ``IntakeTrace`` with per-attempt
                             token counts, latency, raw responses, errors and the stop reason.

The model is called at most ``MAX_MODEL_CALLS`` (3) times per email. The prompt lives only in
``prompts/intake_v1.md`` and is loaded from disk on every run.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

try:  # python-dotenv is in requirements.txt; tolerate its absence so plain env vars still work.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

log = logging.getLogger("harbourline.intake")

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "intake_v1.md"

# Hard bound on model calls per email. Deliberately a constant, not an env var.
MAX_MODEL_CALLS = 3

# Input guard: emails that are empty or longer than this are rejected before any model call.
MAX_EMAIL_CHARS = 20_000

# Provider stop reasons that mean the response was cut off by MAX_TOKENS.
_TRUNCATED_STOP_REASONS = {"max_tokens", "length"}

StopReason = Literal["valid_schema", "max_attempts_reached", "provider_error", "invalid_input"]


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    model_name: str
    max_tokens: int
    temperature: float
    effort: str
    deepseek_base_url: str
    request_timeout_s: float = 60.0

    @property
    def provider(self) -> Literal["deepseek", "anthropic"]:
        return provider_for(self.model_name)

    @property
    def temperature_sent(self) -> bool:
        # The Claude API (version 2026-10-21, anthropic SDK >= 2.0) rejects `temperature`,
        # so it is only sent on the DeepSeek path.
        return self.provider == "deepseek"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model_name=os.getenv("MODEL_NAME", "deepseek-flash").strip(),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            effort=os.getenv("EFFORT", "low").strip(),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
            request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "60")),
        )

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "temperature_sent": self.temperature_sent,
            "effort": self.effort if self.provider == "anthropic" else None,
            "request_timeout_s": self.request_timeout_s,
        }


def provider_for(model_name: str) -> Literal["deepseek", "anthropic"]:
    if model_name.startswith("deepseek-"):
        return "deepseek"
    if model_name.startswith("claude-"):
        return "anthropic"
    raise ValueError(
        f"Cannot infer provider from MODEL_NAME={model_name!r}; expected a 'deepseek-*' or 'claude-*' model."
    )


# --------------------------------------------------------------------------------------
# Schema + deterministic validation
# --------------------------------------------------------------------------------------

_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_MISSING_FIELDS = 5
MAX_MESSAGE_CHARS = 1500


class Intake(BaseModel):
    """One triaged email. Strict: no coercion, no extra keys, all fields required."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    category: Literal["coverage", "billing", "claim", "other"]
    missing_fields: list[str] = Field(max_length=MAX_MISSING_FIELDS)
    next_action: Literal["ask", "draft"]
    customer_message: str = Field(max_length=MAX_MESSAGE_CHARS)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("missing_fields")
    @classmethod
    def _check_missing_fields(cls, v: list[str]) -> list[str]:
        for name in v:
            if not _FIELD_NAME_RE.match(name):
                raise ValueError(f"missing_fields item {name!r} must be a snake_case identifier like 'policy_number'")
        if len(set(v)) != len(v):
            raise ValueError("missing_fields must not contain duplicates")
        return v

    @field_validator("customer_message")
    @classmethod
    def _check_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_message must not be empty")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "Intake":
        if self.next_action == "ask" and not self.missing_fields:
            raise ValueError("next_action is 'ask' but missing_fields is empty; use 'draft' or list the missing fields")
        if self.next_action == "draft" and self.missing_fields:
            raise ValueError("missing_fields is non-empty but next_action is 'draft'; it must be 'ask'")
        if self.category == "other" and self.missing_fields:
            raise ValueError("category 'other' must have an empty missing_fields list")
        return self


class IntakeValidationError(ValueError):
    """Raised by ``validate_intake``. ``str(err)`` is the exact text fed back to the model."""


def validate_intake(raw: str) -> Intake:
    """Deterministically validate a raw model response against ``Intake``.

    The response must be a single JSON object and nothing else (markdown fences are *not*
    stripped). Raises ``IntakeValidationError`` with a human/model-readable message.
    """
    if raw is None or not raw.strip():
        raise IntakeValidationError("Empty response: expected a single JSON object.")
    try:
        intake = Intake.model_validate_json(raw.strip())
    except ValidationError as exc:
        raise IntakeValidationError(_format_validation_error(exc)) from exc
    # allow_inf_nan already rejects these; keep an explicit guard so the rule is visible here.
    if not math.isfinite(intake.confidence):  # pragma: no cover
        raise IntakeValidationError("confidence must be a finite number between 0.0 and 1.0")
    return intake


def _format_validation_error(exc: ValidationError) -> str:
    lines = [f"{exc.error_count()} validation error(s) for Intake:"]
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- {loc}: {err['msg']} [type={err['type']}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Prompt loading (the only prompt text lives in prompts/intake_v1.md)
# --------------------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^<!--\s*section:\s*([a-z_]+)\s*-->\s*$", re.MULTILINE)
_REQUIRED_SECTIONS = ("system", "user", "retry")


@dataclass(frozen=True)
class Prompt:
    path: Path
    sha256: str
    system: str
    user_template: str
    retry_template: str

    def user(self, email: str) -> str:
        return self.user_template.replace("{{EMAIL}}", email.strip())

    def retry(self, validation_error: str) -> str:
        return self.retry_template.replace("{{VALIDATION_ERROR}}", validation_error)


def load_prompt(path: Path = PROMPT_PATH) -> Prompt:
    text = path.read_text(encoding="utf-8")
    matches = list(_SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[m.group(1)] = text[m.end() : end].strip()
    missing = [s for s in _REQUIRED_SECTIONS if not sections.get(s)]
    if missing:
        raise ValueError(f"{path} is missing required section(s): {', '.join(missing)}")
    if "{{EMAIL}}" not in sections["user"] or "{{VALIDATION_ERROR}}" not in sections["retry"]:
        raise ValueError(f"{path}: user section needs {{{{EMAIL}}}} and retry section needs {{{{VALIDATION_ERROR}}}}")
    return Prompt(
        path=path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        system=sections["system"],
        user_template=sections["user"],
        retry_template=sections["retry"],
    )


# --------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    api_stop_reason: str | None
    request_id: str | None
    # Provider-native assistant turn to append before a retry message.
    assistant_message: dict[str, Any]
    # Model that actually produced the response (differs from MODEL_NAME after a Claude fallback).
    served_model: str | None = None


class ProviderError(RuntimeError):
    """The provider call failed (HTTP/API error, timeout, refusal). Not retried."""


# call_model(system, messages) -> ModelResponse
CallModel = Callable[[str, list[dict[str, Any]]], ModelResponse]

# Claude models that accept server-side refusal fallback (`fallbacks: "default"`).
_ANTHROPIC_FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your_"):
        raise ProviderError(f"{name} is not set; add it to .env (see .env.example).")
    return value


def make_deepseek_caller(cfg: Config) -> CallModel:
    import openai  # lazy: only needed on the DeepSeek path

    client = openai.OpenAI(
        api_key=_require_env("DEEPSEEK_API_KEY"),
        base_url=cfg.deepseek_base_url,
        max_retries=0,  # every HTTP request counts toward MAX_MODEL_CALLS
        timeout=cfg.request_timeout_s,
    )

    def call(system: str, messages: list[dict[str, Any]]) -> ModelResponse:
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=cfg.model_name,
                messages=[{"role": "system", "content": system}, *messages],
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},  # temperature has no effect in thinking mode
                stream=False,
            )
        except openai.APIError as exc:  # status, connection and timeout errors; bugs still raise
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = round((time.perf_counter() - start) * 1000)
        if not resp.choices:
            raise ProviderError(f"DeepSeek returned no choices (id={getattr(resp, 'id', None)})")
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage  # Optional in the SDK types
        return ModelResponse(
            text=text,
            input_tokens=(usage.prompt_tokens or 0) if usage is not None else 0,
            output_tokens=(usage.completion_tokens or 0) if usage is not None else 0,
            latency_ms=latency_ms,
            api_stop_reason=choice.finish_reason,
            request_id=getattr(resp, "id", None),
            assistant_message={"role": "assistant", "content": text},
            served_model=getattr(resp, "model", None),
        )

    return call


def make_anthropic_caller(cfg: Config) -> CallModel:
    import anthropic  # lazy: only needed on the Claude path

    _require_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(
        max_retries=0,  # every HTTP request counts toward MAX_MODEL_CALLS
        timeout=cfg.request_timeout_s,
    )
    use_fallback = cfg.model_name in _ANTHROPIC_FALLBACK_MODELS

    def call(system: str, messages: list[dict[str, Any]]) -> ModelResponse:
        # No `temperature`: rejected by the current Claude API. Thinking is adaptive by default.
        kwargs: dict[str, Any] = dict(
            model=cfg.model_name,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": cfg.effort},
        )
        start = time.perf_counter()
        try:
            if use_fallback:
                resp = client.beta.messages.create(**kwargs, betas=[_FALLBACK_BETA], fallbacks="default")
            else:
                resp = client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = round((time.perf_counter() - start) * 1000)
        request_id = getattr(resp, "_request_id", None)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ProviderError(f"Model refused the request (category={category}, request_id={request_id})")
        served_model = getattr(resp, "model", None)
        if any(block.type == "fallback" for block in resp.content):
            log.warning("refusal fallback: %s declined, served by %s (request_id=%s)", cfg.model_name, served_model, request_id)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ModelResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
            api_stop_reason=resp.stop_reason,
            request_id=request_id,
            # Echo the full content (incl. thinking blocks) unchanged on retry.
            assistant_message={"role": "assistant", "content": [b.to_dict() for b in resp.content]},
            served_model=served_model,
        )

    return call


def make_caller(cfg: Config) -> CallModel:
    return make_deepseek_caller(cfg) if cfg.provider == "deepseek" else make_anthropic_caller(cfg)


# --------------------------------------------------------------------------------------
# Bounded validation loop
# --------------------------------------------------------------------------------------


@dataclass
class Attempt:
    attempt: int
    valid: bool
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    api_stop_reason: str | None = None
    request_id: str | None = None
    served_model: str | None = None
    raw_response: str | None = None
    validation_error: str | None = None
    provider_error: str | None = None


@dataclass
class IntakeTrace:
    intake: Intake | None
    stop_reason: StopReason
    attempts: list[Attempt] = field(default_factory=list)
    prompt_file: str = ""
    prompt_sha256: str = ""
    input_error: str | None = None

    @property
    def outcome(self) -> Literal["valid", "failed"]:
        return "valid" if self.intake is not None else "failed"

    @property
    def input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.attempts)

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.attempts)

    @property
    def latency_ms(self) -> int:
        return sum(a.latency_ms for a in self.attempts)

    @property
    def errors(self) -> list[str]:
        errs = [self.input_error] if self.input_error else []
        return errs + [e for a in self.attempts for e in (a.validation_error, a.provider_error) if e]


class IntakeFailed(RuntimeError):
    def __init__(self, trace: IntakeTrace):
        self.trace = trace
        last = trace.errors[-1] if trace.errors else "unknown error"
        super().__init__(f"Intake failed ({trace.stop_reason}) after {len(trace.attempts)} call(s): {last}")


def check_email(email: str) -> str | None:
    """Return a reason the email must not be sent to the model, or None if it is acceptable."""
    if not isinstance(email, str) or not email.strip():
        return "Email is empty."
    if len(email) > MAX_EMAIL_CHARS:
        return f"Email is {len(email)} characters; the limit is {MAX_EMAIL_CHARS}."
    return None


def _truncation_hint(resp: ModelResponse, max_tokens: int | None) -> str:
    if resp.api_stop_reason not in _TRUNCATED_STOP_REASONS:
        return ""
    limit = f" ({max_tokens})" if max_tokens else ""
    return f"Response was cut off at the MAX_TOKENS limit{limit}; return only the compact JSON object.\n"


def run_with_trace(
    email: str,
    call_model: CallModel | None = None,
    prompt: Prompt | None = None,
    max_tokens: int | None = None,
) -> IntakeTrace:
    """Run the bounded validate-and-retry loop and return the full trace.

    ``max_tokens`` is only used to word the truncation hint; defaults to the env config.
    """
    prompt = prompt or load_prompt()
    try:
        prompt_file = str(prompt.path.relative_to(ROOT))
    except ValueError:
        prompt_file = str(prompt.path)
    trace = IntakeTrace(intake=None, stop_reason="max_attempts_reached", prompt_file=prompt_file, prompt_sha256=prompt.sha256)

    # Input guard: reject before building a client or spending a model call.
    input_error = check_email(email)
    if input_error:
        log.error("rejected input: %s", input_error)
        trace.input_error = input_error
        trace.stop_reason = "invalid_input"
        return trace

    if call_model is None or max_tokens is None:
        cfg = Config.from_env()
        max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        if call_model is None:
            call_model = make_caller(cfg)

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt.user(email)}]

    for n in range(1, MAX_MODEL_CALLS + 1):
        try:
            resp = call_model(prompt.system, messages)
        except ProviderError as exc:
            log.error("attempt %d/%d: provider error: %s", n, MAX_MODEL_CALLS, exc)
            trace.attempts.append(Attempt(attempt=n, valid=False, provider_error=str(exc)))
            trace.stop_reason = "provider_error"
            return trace

        attempt = Attempt(
            attempt=n,
            valid=False,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms,
            api_stop_reason=resp.api_stop_reason,
            request_id=resp.request_id,
            served_model=resp.served_model,
        )
        trace.attempts.append(attempt)
        try:
            trace.intake = validate_intake(resp.text)
        except IntakeValidationError as exc:
            error = _truncation_hint(resp, max_tokens) + str(exc)
            attempt.raw_response = resp.text
            attempt.validation_error = error
            log.warning("attempt %d/%d failed validation: %s", n, MAX_MODEL_CALLS, error)
            if n < MAX_MODEL_CALLS:
                messages.append(resp.assistant_message)
                messages.append({"role": "user", "content": prompt.retry(error)})
            continue

        attempt.valid = True
        trace.stop_reason = "valid_schema"
        return trace

    trace.stop_reason = "max_attempts_reached"
    return trace


def run(email: str) -> Intake:
    """Process a single raw email and return a validated Intake object.

    Raises ``IntakeFailed`` (carrying the full trace) if no valid object was produced.
    """
    trace = run_with_trace(email)
    if trace.intake is None:
        raise IntakeFailed(trace)
    return trace.intake
