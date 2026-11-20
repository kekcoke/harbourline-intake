# Harbourline Email Intake — Week 1 Pilot

Turns raw customer support emails into schema-validated `Intake` objects with draft replies or clarifying questions for Harbourline support agents.

## Run Sample Suite
python scripts/run_samples.py

## Setup (Clean Checkout)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API key to .env
