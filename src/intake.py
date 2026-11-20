"""Core intake pipeline: one email in, one validated Intake object out."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "intake_v1.md"
MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "600"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))


def run(email: str):
    """Process a single raw email and return a validated Intake object."""
    raise NotImplementedError("Scaffold intake schema, validation tool, and 3-call loop.")