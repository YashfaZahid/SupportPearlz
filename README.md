# SupportPearlz

SupportPearlz is a retrieval-augmented support agent for a fictional home-systems brand (Pearlz / AquaPearl / AeroPearl). It answers only from the product knowledge base, cites the passage it used, and refuses questions the documents do not cover.

## What it does

- Loads markdown, CSV, PDF, and DOCX from `data/knowledge_base/`
- Chunks documents with metadata preserved for later filtering
- Embeds chunks into Chroma
- Retrieves with dense search, MMR, or a hybrid of BM25 and vectors
- Reranks a wider candidate set down to the passages that go to the model
- Streams an answer with citations in a Streamlit UI and a CLI
- Scores a fixed question set through `evaluation/run_eval.py`

The knowledge base is fictional product documentation: warranty, shipping, pricing, troubleshooting, firmware notes, and a service agreement. Do not treat it as a real company's policy.

## Run

Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set OPENAI_API_KEY
python -m src.ingestion.build_index --rebuild
streamlit run app.py
```

CLI and checks:

```bash
python -m src.app_cli chat
python -m src.config        # prints settings with secrets masked
pytest
python -m evaluation.run_eval
```

## Layout

```text
src/ingestion/     Loaders, chunking, index build
src/retrieval/     Dense, hybrid BM25, rerank, relevance gate
src/chains/        Prompts, citations, memory, RAG chain
src/app_streamlit.py
evaluation/        Question set and latest results
docs/architecture.md
```

Docker Compose is included for a container run. On Streamlit Community Cloud, use `app.py` as the entrypoint and put the same keys in Secrets. Prefer Python 3.12 there.

## Limits

If the documents do not contain the answer, the agent should say so. It is not a medical advisor and must not invent certifications or health claims.
