# SupportPearlz — Implementation Report

Living log of what was built, decisions made, test evidence, and known gaps.  
Update after **each** phase. Do not claim tests passed without running them.

---

## Progress overview

| Phase | Scope | Status |
|-------|--------|--------|
| 0 | Skeleton, config, logging, deps, docs | **Completed** |
| 1 | Knowledge base authoring | **Completed** |
| 2 | Ingestion loaders | **Completed** |
| 3 | Chunking | **Completed** |
| 4 | Embeddings + vector store | **Completed** |
| 5 | Retriever + relevance gate | **Completed** |
| 6 | RAG chain, prompts, citations | **Completed** |
| 7 | Memory + CLI | **Completed** |
| 8 | Evaluation harness | **Completed** (auto-score; manual review still for report) |
| 9 | Streamlit UI (streaming, cache, disclaimer) | **Completed** |
| 10 | Deliverables packaging | Pending |

---

## Phase 0 — Project setup (completed)

### Modules / files added

- `src/config.py` — `Settings` via pydantic-settings; fail-loud on missing API key; `masked_dict()`; `python -m src.config` smoke entry
- `src/utils/logging_setup.py` — central console + file logging
- `src/utils/errors.py` — typed user-facing errors
- Package inits under `src/`, `ingestion/`, `retrieval/`, `chains/`, `utils/`
- `requirements.txt` — pinned LangChain/Chroma/Streamlit stack
- `.env.example`, `.gitignore`, `pytest.ini`, `README.md`
- Placeholders: `data/knowledge_base`, `data/vector_store`, `data/cache`, `logs`, `evaluation/results`
- Tests: `tests/test_config.py`, `tests/conftest.py`

### Technical decisions

1. **pydantic-settings** single `Settings` object — all tunables env-driven (assignment Task 01).
2. **ConfigurationError** raised from a model validator when hosted provider lacks `OPENAI_API_KEY` — fails at startup, not on first API call.
3. **Dual caching flags documented now** (`ENABLE_LANGCHAIN_CACHE`, Streamlit `@st.cache_resource` later) so Phase 6/9 do not redesign config.
4. **Streaming + medical disclaimer** are first-class settings for Phase 9 UI without hard-coding copy in widgets.
5. Python **3.12.10** on the build machine; pins chosen for 3.10–3.12 compatibility band.

### Architecture updates this phase

See amendments table at top of `docs/architecture.md` (streaming, dual caching, navy/teal/warm-white UI, disclaimer).

### Dependencies introduced

See `requirements.txt` (LangChain 0.3.x set, chromadb, pypdf, python-docx, typer, rich, streamlit, pytest).

### Testing results (Phase 0)

| Command | Result | Notes |
|---------|--------|-------|
| `pytest tests/test_config.py` | **5 passed** | Masked secrets, missing key, overlap validation, cache |
| `pytest tests/test_logging_setup.py` | **run in same session** | File handler writes |
| `python -m src.config` (no key) | **exit 1** | `CONFIGURATION ERROR: Missing required OPENAI_API_KEY…` |
| `python -m src.config` (key set) | **exit 0** | Prints masked settings incl. disclaimer, cache, streaming |

### Dependency compatibility notes

- Initial pin `langchain-core==0.3.60` conflicted with `langchain-openai==0.3.18` (≥0.3.61) → settled on **0.3.63**.
- `langchain-chroma==0.2.4` requires **chromadb≥1.0.9** (not 0.6.x) → **chromadb==1.5.9**.

### Known limitations (Phase 0)

- No RAG code yet; empty knowledge base.
- Real API key not required until Phase 4+ live embedding/chat calls.
- Console ellipsis glyph for masked secrets may render oddly on some Windows code pages; masking logic is correct.

---

## Phase 1 — Knowledge base (completed)

### Modules / files added

Customer-facing docs in `data/knowledge_base/` (12 files, 4 formats):

| File | Format |
|------|--------|
| `product_manual_aquapearl500.pdf` | PDF |
| `installation_guide.pdf` | PDF |
| `service_maintenance_agreement.docx` | DOCX |
| `pricing_guide.csv` | CSV |
| `warranty_policy.md`, `refund_returns_policy.md`, `shipping_policy.md`, `troubleshooting_guide.md`, `faq.md`, `privacy_policy.md`, `customer_service_handbook.md`, `firmware_changelog.md` | MD |
| `_seams_and_gaps.md` | Author scratch (loaders must skip `_` prefix) |

Generator: `scripts/generate_kb_binaries.py` (reportlab + python-docx).

### Design choices

- Consistent facts: AquaPearl 500 Pro warranty **24 months from invoice**; filters **consumables excluded**; return window **14 days**; Skardu **5–7 working days**; next-day only KHI/LHE/ISB.
- Seams: claim checklist only in service DOCX; flushing only in installation PDF; remote shipping detail in shipping MD.
- Gaps: student discount, NSF certification, instalments — listed in `_seams_and_gaps.md` only.
- Injection canary in customer service handbook.

### Testing results (Phase 1)

| Command | Result |
|---------|--------|
| `pytest tests/test_knowledge_base.py` (+ prior) | **11 passed** combined with Phase 0 in session |

---

## Phase 2 — Document ingestion loaders (completed)

### Modules / files added

- `src/ingestion/normalize.py` — conservative whitespace cleanup
- `src/ingestion/loaders.py` — extension dispatch (PDF/MD/TXT/DOCX/CSV), metadata contract, per-file failure, content-hash dedupe, `IngestionReport`
- `tests/test_loaders.py`

### Technical decisions

1. CSV loaded **row-wise** as self-describing sentences (SKU + prices stay together).
2. Files starting with `_` or `.` skipped (keeps seams/gaps scratch out of the index).
3. DOCX uses LangChain `Docx2txtLoader` with **python-docx fallback** if `docx2txt` is absent.
4. Corrupt PDF in a folder does not abort the run (acceptance criterion).

### Testing results (Phase 2)

| Command | Result |
|---------|--------|
| `pytest tests/test_loaders.py tests/test_knowledge_base.py` | **12 passed** |

### Known limitations

- PDF page index depends on loader metadata; location formatted as `page:N`.
- Full ingestion→index CLI arrives in Phase 4 (`build_index.py`).

---

## Change log

| Date | Phase | Summary |
|------|-------|---------|
| 2026-10-30 | 0 | Skeleton, config, logging, pinned deps, docs; config/logging tests passed |
| 2026-10-30 | 1 | Authored 12 KB docs + seams/gaps; binary generator script |
| 2026-10-30 | 2 | Loaders + normalize; corrupt-file skip; 12 tests passed |
| 2026-10-30 | 3 | Chunking module + metadata preservation; 4 chunk tests passed |

---

## Phase 3 — Chunking (completed)

### Modules / files added

- `src/ingestion/chunking.py` — `ChunkingConfig`, recursive / markdown_headers / csv_passthrough, `run_chunking_experiment`
- `tests/test_chunk_metadata.py`

### Technical decisions

1. **Adopted defaults** (env): `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`, strategy recursive for index build; CSV passthrough available so prices never split from SKUs.
2. Experiment helper compares ≥6 configs (300/800/1500 × overlap variants + markdown + csv). Full hit@k on sample questions requires an index → recorded in Phase 5 report tables.
3. Metadata `source` / `doc_type` asserted after every split.

### Testing results (Phase 3)

| Command | Result |
|---------|--------|
| `pytest tests/test_chunk_metadata.py tests/test_loaders.py` | **11 passed** |

### Pending / unverified

- Full 6-config hit@k table for the written technical report (partial hit@k covered in Phase 5 tests).

---

## Phase 4 — Embeddings + vector store (completed)

### Modules / files added

- `src/retrieval/embeddings.py` — OpenAI embeddings factory + cosine diagnostic
- `src/retrieval/vector_store.py` — Chroma persist/load/rebuild, fingerprint, delete-by-source upsert, search helper
- `src/ingestion/build_index.py` — CLI: `build [--rebuild]`, `search`, `diagnostic`, `status`
- `src/utils/cache_setup.py` — LangChain SQLite cache (strategy 1)
- `tests/test_vector_store_unit.py`

### Technical decisions

1. **Incremental updates:** delete-by-source then reinsert (documented in code).
2. **Fingerprint file** records embedding model; mismatch forces rebuild.
3. Metadata sanitized for Chroma (no `None` values).
4. Live build: **12 docs → 60 chunks**, model `text-embedding-3-small`.

### Testing results (Phase 4)

| Check | Result |
|-------|--------|
| Unit tests | **3 passed** |
| `build_index --rebuild` | **OK** — 60 chunks, ~8.8s embed |
| Second `build` without `--rebuild` | **No re-embed** — status message only |
| Search “how long is the warranty” | Top hits from `warranty_policy.md` / FAQ |
| Similarity diagnostic | paraphrase **0.82** vs unrelated **~0.27–0.28** |

---

## Phase 5 — Retriever + relevance gate (completed)

### Modules / files added

- `src/retrieval/retriever.py` — similarity / MMR, metadata filter, relevance gate, hit@k helper
- `tests/test_retriever.py`

### Technical decisions

1. Chroma cosine space returns **distance** (lower = better). `SCORE_THRESHOLD` = max acceptable distance; default calibrated to **0.75** (updated `.env` / `.env.example` / config).
2. Architecture note: earlier 0.35 default assumed similarity — **amended** to distance semantics after live scores (~0.43 for good warranty hits).

### Testing results (Phase 5)

| Command | Result |
|---------|--------|
| `pytest tests/test_retriever.py tests/test_vector_store_unit.py` | **10 passed** |

---

## Change log (continued)

| Date | Phase | Summary |
|------|-------|---------|
| 2026-10-30 | 4 | Chroma index + CLI; live embed 60 chunks; diagnostic OK |
| 2026-10-30 | 5 | Retriever/MMR/gate; threshold 0.75 distance; 10 tests passed |
| 2026-10-30 | 6–7 | Grounded chain + memory + CLI; live RAG 3/3 passed |

---

## Phase 6–7 — RAG chain, prompts, memory, CLI (completed)

### Modules / files added

- `src/chains/schemas.py` — `GroundedResponse` (answer, sources, confidence, answered, flags)
- `src/chains/prompts.py` — weak v1 + grounded v2 + condense prompt
- `src/chains/context.py` — `[Sn]` labelling + char/chunk budget
- `src/chains/citations.py` — validate labels + dedupe
- `src/chains/memory.py` — history window + query condensation
- `src/chains/rag_chain.py` — `SupportPearlzAgent.ask()`
- `src/llm.py` — ChatOpenAI factory (temperature from settings)
- `src/app_cli.py` — interactive chat with disclaimer banner, `/reset`, sources, confidence
- Tests: `tests/test_schema.py`, `tests/test_rag_live.py`

### Testing results

| Command | Result |
|---------|--------|
| `pytest tests/test_schema.py tests/test_rag_live.py tests/test_retriever.py` | **14 passed** |
| Live: warranty factual | answered, contains 24, has sources |
| Live: student discount | refusal / not answered |
| Live: return-window follow-up | condensation works |

### Known limitations

- Prompt before/after experiment artefact for the PDF technical report still pending.
- CLI does not stream tokens (Streamlit does).

---

## Phase 8 — Evaluation harness (completed)

### Files

- `evaluation/test_questions.json` — 18 cases across easy / multi_document / follow_up / unanswerable / ambiguous / adversarial
- `evaluation/run_eval.py` — automated runner → JSON + markdown
- Results: `evaluation/results/latest.md` / `latest.json`

### Testing results (live, real API key)

| Metric | Value |
|--------|-------|
| Cases | 18 |
| Mean correctness | **0.94** |
| Mean groundedness | **1.00** |
| Mean citation accuracy | **0.78** |
| Pass rate (auto) | **0.78** |
| Refusal rate (unanswerable) | **1.00** |
| Hallucination proxy | **0** |

Failure attribution (auto): E01/E05/V02 citation matching strictness; **M03** generation (unauthorised-repair conflict) — needs prompt/retrieval follow-up.

---

## Phase 9 — Streamlit UI (completed)

### Files

- `src/app_streamlit.py` — chat UI, disclaimer, filters, reset, streaming + structured finalize
- `.streamlit/config.toml` — navy / teal / warm-white theme
- Dual cache: LangChain SQLite (strategy 1) + `@st.cache_resource` agent (strategy 2)
- `stream_ask()` on agent: stream plain draft tokens, then schema-validated `GroundedResponse` with citations

### How to run

```bash
streamlit run src/app_streamlit.py
```

### Verified

- Module imports cleanly
- Live RAG regression: **7 passed** (`test_schema` + `test_rag_live`) after streaming changes

### Still pending (Phase 10 deliverables)

- architecture.png, technical_report.pdf, demo video, screenshot set
- Prompt weak-vs-strong experiment write-up
- Optional polish on M03 / citation auto-score
