# SupportPearlz

Retrieval-augmented customer support knowledge agent for **Pearlz Home Systems** (AquaPearl / AeroPearl), built with Python and LangChain.

> **Safety:** SupportPearlz answers only from Pearlz documentation. It is **not** a medical device advisor and must not be used for clinical decisions. Do not invent health claims or certifications.

## Prerequisites

- Python **3.10–3.12** (developed on 3.12)
- An OpenAI API key (or OpenAI-compatible provider key)

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env` and set `OPENAI_API_KEY`. Never commit `.env`.

### Config smoke check

```bash
python -m src.config
```

Prints loaded settings with secrets masked. Missing `OPENAI_API_KEY` exits with a clear error.

### Tests (Phase 0)

```bash
pytest tests/test_config.py
```

## Project status

| Area | Status |
|------|--------|
| Config, logging, skeleton | Done (Phase 0) |
| Knowledge base (12 docs) | Done (Phase 1) |
| Ingestion loaders + chunking | Done (Phases 2–3) |
| Vector store / RAG / CLI | Done (Phases 4–7) |
| Evaluation harness | Done (Phase 8) — see `evaluation/results/latest.md` |
| Streamlit UI | Done (Phase 9) |

```bash
pytest
python -m src.ingestion.build_index --rebuild   # first time / after KB changes
python -m src.app_cli chat
streamlit run app.py
python -m evaluation.run_eval
```

See `docs/architecture.md` and `docs/implementation_report.md` for the living plan and progress.

## Environment variables

See `.env.example` for the full table (models, chunking, retrieval `k`, thresholds, LangChain cache path, streaming flag, medical disclaimer, logging).

## Licence / academic note

Course assignment AAI–407 / PRAC–03. Knowledge-base content is fictional.
