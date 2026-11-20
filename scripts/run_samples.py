"""Run the intake pipeline across all sample emails in data/emails/*.txt."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EMAILS_DIR = ROOT / "data" / "emails"
RUNS_DIR = ROOT / "runs" / "week-1"


def main() -> None:
    raise NotImplementedError("Scaffold sample runner, JSON output printing, and token/latency logging.")


if __name__ == "__main__":
    main()