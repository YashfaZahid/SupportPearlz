# SupportPearlz — Implementation Architecture & Plan

**Assignment:** AAI–407 / PRAC–03 — LangChain Real-World RAG Customer Support Agent  
**Product:** SupportPearlz for Pearlz Home Systems (AquaPearl / AeroPearl support knowledge agent)  
**Status:** Living architecture — updated as implementation proceeds  
**Principle:** Depth before breadth. Mandatory CLI + evaluation first; Streamlit is a polish layer over the same core.

### Architecture amendments (post-approval)

| Change | Why |
|--------|-----|
| Elevate **token streaming** in Streamlit (citations finalize after structured object) | Explicit product requirement for portfolio UX; assignment Feature 18 / C4 |
| Add **two caching strategies**: (1) LangChain LLM/embedding response cache on disk/memory for repeated eval & chat turns; (2) Streamlit `@st.cache_resource` for loaded vector store + chain singletons | Explicit requirement; cuts cost/latency without changing RAG semantics |
| UI palette locked to vinyl/koi brand: **`#146466` / `#c92f1c` / `#5aaba9` / `#e2e2c4`** + medical/product-safety disclaimer | Brand asset + trust UX |
| Bonus pack wired: hybrid, rerank, expansion, role KB, tools, traces, Docker | Assignment Section 11 |
| Target runtime noted as **Python 3.12** on this machine (still within 3.10–3.12 band) | Local environment discovery |
| `SCORE_THRESHOLD` interpreted as **max Chroma cosine distance** (default **0.75**), not similarity | Live Phase 4/5 scores: good hits ~0.35–0.55 distance |

---

## 1. One-paragraph system definition

SupportPearlz ingests a folder of company documents **once** into a **persisted** vector index, then answers multi-turn customer questions by (1) condensing history into a standalone query, (2) retrieving relevant chunks with a relevance gate, (3) generating a **schema-validated** answer grounded only in those chunks, and (4) returning answer + confidence + de-duplicated citations — or an explicit refusal with human escalation when retrieval fails or the KB has no answer.

---

## 2. Mandatory requirements preserved (checklist)

| Area | Must satisfy |
|------|----------------|
| Stack | Python ≥3.10, LangChain for loaders/splitters/retrieval/prompts/chains, Pydantic v2 structured output, env-driven config, stdlib logging |
| Pipelines | **Separate** ingestion (write disk) vs query (load disk); never re-embed per question |
| KB | ≥10 authored docs, ≥3 formats, seams + gaps planted, metadata on every chunk |
| RAG behaviour | Grounding, refusal, partial answers, injection resistance, used-only citations, confidence signal |
| Conversation | Query condensation, bounded history, session reset, topic-switch safe |
| Failures | Missing key/index, bad file, empty retrieval, API timeout/rate-limit, schema parse failure → user-readable, logged |
| Evaluation | ≥15 automated cases covering all 6 categories; plus 20-case conformance suite with stage attribution |
| Interface | Interactive CLI **required**; Streamlit **planned** as demo/bonus surface over the same chain |
| Hard constraints | No secrets/vector store in git; selective retrieval only; index self-contained offline at query time |

**Cost asymmetry (design law):** A confident wrong answer is worse than an honest refusal. Prefer pipeline refusal (relevance gate) before model generation when scores are weak.

---

## 3. Technology choices (justified, config-swappable)

| Concern | Choice | Why | Swap path |
|---------|--------|-----|-----------|
| Runtime | Python 3.11 (target; 3.10–3.12 OK) | Assignment band; wide wheel support | Document in README |
| Orchestration | `langchain` + `langchain-core` + `langchain-community` | Mandatory; LCEL / Runnable composition | N/A |
| Chat LLM | OpenAI-compatible via `langchain-openai` (e.g. `gpt-4o-mini`) **or** Groq / Ollama | Low cost for eval loops; temperature **0** | `LLM_PROVIDER`, `LLM_MODEL` |
| Embeddings | Same provider family as indexing (default: OpenAI `text-embedding-3-small`) | Same model for index + query; known dim | `EMBEDDING_MODEL` + rebuild |
| Vector DB | **Chroma** (`langchain-chroma`) | Disk persistence, metadata filters, beginner-friendly, local | Abstract behind `vector_store.py` |
| PDF | `pypdf` (LangChain `PyPDFLoader`) | Simple, no native deps | Swap loader class in dispatch |
| DOCX | `python-docx` + community loader | Required format path | Same |
| CSV | Custom row→sentence formatter + `CSVLoader` or manual docs | Prevents SKU/price split (common fail) | Row-wise only |
| Settings | `pydantic-settings` + `python-dotenv` | Fail-loud at startup | Single `Settings` object |
| CLI | `typer` + optional `rich` | Clear commands: build-index, chat, search | Keep `app_cli.py` thin |
| UI (required for portfolio) | **Streamlit** + streaming tokens + dual cache | Demo surface; same `ask()` / stream API | `src/app_streamlit.py` |
| Cache strategy 1 | LangChain SQLite/InMemory cache | Repeated prompts/embeddings in eval & chat | `ENABLE_LANGCHAIN_CACHE` |
| Cache strategy 2 | Streamlit `@st.cache_resource` | Single loaded store + chain per process | UI only |
| Eval | JSON fixtures + `run_eval.py` | Reproducible artefacts | Manual scoring columns OK |
| Tests | `pytest` for loaders/chunk metadata/schema/citation validate | Credit + regression | Optional per rubric |

**Not in v1 core (avoid complexity):** Docker, FastAPI, hybrid BM25, cross-encoder, LangSmith — document as Phase 8 optional bonus (prefer **C2 rerank** if any bonus is attempted).

---

## 4. Target folder structure

```
supportpearlz/
├── README.md
├── requirements.txt          # pinned
├── .env.example
├── .gitignore                # .env, data/vector_store/, __pycache__, .streamlit/secrets.toml
├── data/
│   ├── knowledge_base/       # 10–12 authored docs (pdf/md/docx/csv)
│   ├── knowledge_base/_seams_and_gaps.md   # planted seams/gaps (scratch → keep)
│   └── vector_store/         # git-ignored Chroma persist dir
├── src/
│   ├── __init__.py
│   ├── config.py             # Settings; secrets masked on print
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── chunking.py
│   │   ├── normalize.py      # whitespace / header strip
│   │   └── build_index.py    # CLI: build | --rebuild
│   ├── retrieval/
│   │   ├── vector_store.py   # create, persist, load, upsert-by-source
│   │   └── retriever.py      # k / MMR / threshold gate + optional metadata filter
│   ├── chains/
│   │   ├── prompts.py        # versioned templates (v1 weak, v2 final)
│   │   ├── schemas.py        # GroundedResponse Pydantic model
│   │   ├── context.py        # [S1] labelling, budget/truncation
│   │   ├── citations.py      # validate / dedupe / resolve
│   │   ├── rag_chain.py      # LCEL composition
│   │   └── memory.py         # history window + condensation
│   ├── app_cli.py
│   ├── app_streamlit.py      # thin UI over rag_chain.ask
│   └── utils/
│       ├── logging_setup.py
│       └── errors.py         # typed user-facing errors
├── evaluation/
│   ├── test_questions.json   # ≥15, all 6 categories
│   ├── conformance_20.json   # T-01…T-20 intents mapped to KB facts
│   ├── run_eval.py
│   └── results/
├── docs/
│   ├── architecture.md       # this file
│   ├── architecture.png      # deliverable 5 (draw after code exists)
│   ├── technical_report.pdf  # later
│   └── screenshots/
└── tests/
    ├── test_loaders.py
    ├── test_chunk_metadata.py
    ├── test_schema.py
    └── test_citations.py
```

Equivalent to Section 3.4 with small, justified extras (`normalize.py`, `context.py`, `citations.py`, Streamlit entry) — keep responsibilities single-purpose.

---

## 5. Module responsibilities

| Module | Responsibility | Does **not** do |
|--------|----------------|-----------------|
| `config.py` | Load/validate env once; expose tunables | Call LLMs |
| `loaders.py` | Extension dispatch; attach metadata contract; per-file try/except; skip duplicates | Embed or split |
| `normalize.py` | Conservative text cleanup | Delete meaningful tables/headings |
| `chunking.py` | RecursiveCharacter baseline + Markdown heading / CSV row alternatives; experiment harness | Persist store |
| `build_index.py` | Orchestrate load → chunk → embed → persist; write ingestion report | Answer questions |
| `vector_store.py` | Persist/load Chroma; rebuild vs delete-by-source upsert; embedding model fingerprint in store metadata | Prompting |
| `retriever.py` | Build similarity / MMR / threshold policies; relevance gate | Call chat LLM |
| `prompts.py` | Role, grounding, refusal, partial, citation, injection rules; version comments | Retrieval |
| `schemas.py` | `answer`, `sources[]`, `confidence` enum, `answered` bool (+ optional `flags`) | UI |
| `context.py` | Label chunks `[Sn]`; separators; char/chunk budget; truncation log | Vector search |
| `citations.py` | Intersect model citations ∩ supplied labels; dedupe by (source, location) | Generation |
| `memory.py` | Last-N history; condense query; reset; log original vs rewritten | Store index |
| `rag_chain.py` | Single `ask(question, history) → GroundedResponse`; retries/backoff | File walking |
| `app_cli.py` / `app_streamlit.py` | I/O only | Business logic |
| `run_eval.py` | Batch questions → JSON + table artefacts | Manual viva narrative |

---

## 6. Data flow

### 6.1 Ingestion (offline)

```
data/knowledge_base/* 
  → loaders (by extension) + metadata {source, doc_type, product_line, version, location}
  → normalize
  → chunking (metadata copied per chunk)
  → embedding model (batched, retry/backoff)
  → Chroma persist → data/vector_store/
  → ingestion report (files found/loaded/skipped + per-doc stats)
```

### 6.2 Query (online)

```
user question + chat_history
  → memory.condense → standalone_query  (logged)
  → embed(standalone_query) with **same** embedding model
  → retriever top-k / MMR + scores + metadata
  → relevance gate
       ├─ fail → GroundedResponse(answered=False, confidence=none, refusal + escalate)  [no LLM or refusal-only]
       └─ pass → context builder [S1]…[Sk] (budgeted)
            → prompt (system rules + context + question)
            → LLM (temperature=0) + structured output
            → schema validate (retry once → typed fallback)
            → citation validate/dedupe
            → GroundedResponse → UI + structured log + append history
```

**Medical / product-safety rules (mapped to Pearlz):** Never invent certifications (e.g. NSF), health claims, or warranty/refund commitments not in context; safety/escalation symptoms from troubleshooting must cite sources; refuse pressure to “just guess”; escalate commercial exceptions to `support@pearlzhome.example`.

---

## 7. UI/UX design

### 7.1 CLI (mandatory)

- Banner: KB name, chunk count, doc count (from store metadata).
- Each turn: answer → Sources (file + page/section/SKU) → Confidence.
- Commands: `/reset`, `/quit`, optional `/debug` (show rewritten query + scores).
- Errors: one-line human message; details only in log file.

### 7.2 Streamlit (portfolio surface — same core)

**Visual direction:** Healthcare SaaS trust UI — **navy** (`#0B1F33`) primary, **teal** (`#0D9488`) accent, **warm-white** (`#F7F4EF`) surfaces. Clear Pearlz wordmark as brand hero signal. One chat composition; citations as first-class trust UI. Prominent medical/product-safety disclaimer in header (from `MEDICAL_SAFETY_DISCLAIMER`). No purple-gradient AI cliché; no emoji clutter.

**Layout**

1. **Header:** Pearlz / SupportPearlz brand + short scope line (“Answers from official Pearlz docs only”).
2. **Main:** Chat transcript; assistant bubbles always show confidence chip + expandable Sources.
3. **Sidebar:** Session reset; optional `doc_type` / `product_line` filter (Should feature); KB status (loaded/missing); link to escalate.
4. **Refusal state:** Distinct muted styling + “Contact support” CTA — never empty fake sources.
5. **Debug expander (dev toggle):** original vs rewritten query, chunk ids/scores — for screenshots/viva.

**UX rules:** Prefer refusal over guess; partial answers visually separate “covered” vs “not in docs”; no invented phone numbers/URLs; temperature fixed low so demos stay consistent.

---

## 8. Configuration surface (all env-driven)

Document in `.env.example` (illustrative names):

- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE` (default `0`)
- `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` (recorded; assert vs store fingerprint)
- `OPENAI_API_KEY` / provider key (required if hosted)
- `VECTOR_STORE_PATH`, `COLLECTION_NAME`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `RETRIEVAL_K`, `RETRIEVAL_STRATEGY` (`similarity`|`mmr`), `SCORE_THRESHOLD`
- `HISTORY_MAX_TURNS`, `CONTEXT_MAX_CHARS`, `CONTEXT_MAX_CHUNKS`
- `LOG_LEVEL`, `LOG_FILE`
- `SUPPORT_ESCALATION_EMAIL`

`python -m src.config` must print masked settings; missing required key → single clear error.

---

## 9. LangChain / dependency compatibility risks

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **LangChain 0.3 vs 1.x package split** | Import errors (`langchain.chat_models` moved) | Pin a **known-good set** together: `langchain`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-chroma`; prefer LCEL (`Runnable`) + `create_history_aware_retriever` / custom RunnableSequence over deprecated `ConversationalRetrievalChain` |
| **`langchain-chroma` separate package** | `from langchain_community.vectorstores import Chroma` soft-deprecated | Import from `langchain_chroma`; pin chroma client compatible with that release |
| **Structured output API drift** | `with_structured_output` / function-calling differences by provider | Abstract in `rag_chain.py`; fallback: JSON mode + Pydantic `model_validate` + one retry |
| **Pydantic v1 leftovers in old examples** | Validation errors, `.dict()` vs `.model_dump()` | Require Pydantic v2 only; no v1 shims in our code |
| **Chroma + numpy / onnxruntime wheels on Windows** | Install fails or runtime crash | Pin `chromadb` version tested on Windows; document Python 3.11; prefer default embedding path (external API) so local ONNX embedding extras stay unused |
| **OpenAI client major bumps** | `langchain-openai` breaks | Pin `openai` and `langchain-openai` as a pair in `requirements.txt` |
| **PDF loaders** | Empty text from scanned PDFs | Author text-based PDFs; log char count=0 as skip/warn |
| **Token vs character chunking** | Silent embed truncation | Keep chunk_size ≪ embedding max input; log token estimate; document model max length in report |
| **MMR / score semantics** | Threshold copied from tutorials fails | Calibrate `SCORE_THRESHOLD` on **our** Chroma + model; store notes that Cosine vs L2 changes meaning |
| **History-aware retriever double LLM cost** | Latency/cost surprise | Expected; log both queries; keep condensation prompt short |
| **Streamlit + asyncio / sync LLM** | Rerun blocking | Call sync `ask()`; use `st.spinner`; avoid mixing conflicting event loops; no Streamlit inside chain modules |
| **python-docx / pypdf version skew** | Loader API changes | Pin both; smoke-test in Phase 2 |

**Pinning policy:** After first successful install, `pip freeze` → `requirements.txt`. Never “latest” in submission.

---

## 10. Testing strategy

| Layer | What | When |
|-------|------|------|
| Unit | Loader dispatch; metadata keys present after split; schema round-trip; citation validator drops `[S99]` | Every Phase 2–7 PR-sized step |
| Ingestion acceptance | Folder with one corrupt file → others load; warning named | Phase 2 exit gate |
| Retrieval-only | hit@k on labelled questions **without** LLM | Phase 3–4 (drives chunking/retriever choice) |
| Chain | Schema always validates including refusal; offline network → clean error | Phase 5 |
| Conversation | Pronoun follow-up + topic switch; log shows rewrite | Phase 6 |
| Prompt experiment | Weak vs final on 5 questions; injection doc + user injection | Phase 5–6 |
| Eval harness | `test_questions.json` ≥15 × 6 categories → JSON + table | Phase 7 early (before polish) |
| Conformance | Map T-01…T-20 to own KB wording; verdict + stage blame | Phase 7 |
| Streamlit smoke | Load index, one Q&A, refusal, reset | Phase 8 |

**Debug order (always):** (1) Was answer text in retrieved chunks? (2) Did context reach the model? (3) Only then change the prompt.

---

## 11. Streamlit deployment plan

| Stage | Action |
|-------|--------|
| Local demo | `streamlit run src/app_streamlit.py` after `python -m src.ingestion.build_index` |
| Secrets | `.streamlit/secrets.toml` **or** env vars; never commit; mirror `.env.example` |
| Index artefact | Build on machine / CI artefact; mount or bake `data/vector_store/` at deploy time (prefer volume mount so rebuild ≠ image rebuild) |
| Community Cloud (optional) | Repo without secrets; set secrets in dashboard; document that vector store must be built in a setup step or committed **only if** instructor allows (default: **do not** commit store — provide build script in README) |
| Screenshot set | Ingestion log, index load, cited answer, refusal, follow-up with rewrite visible, graceful API error |

CLI remains the grading-critical path; Streamlit must not fork a second chain implementation.

---

## 12. Knowledge-base authoring plan (gates Phase 2)

**Required docs (formats):**

1. AquaPearl 500 Pro manual — PDF (alert codes E-01…E-08, specs, maintenance)  
2. Warranty policy — MD/DOCX  
3. Refund & returns — MD/DOCX  
4. Shipping policy — MD (metro / secondary / remote; Skardu; next-day cities)  
5. Installation guide — PDF (flushing procedure seam)  
6. Troubleshooting guide — MD (≥15 symptoms)  
7. FAQ — MD/DOCX (customer language)  
8. Pricing guide — CSV (SKU rows)  
9. Privacy policy — MD  
10. Service & maintenance agreement — DOCX/PDF (**claim procedure** seam referenced by warranty)  
11–12 (recommended): Customer service handbook; firmware changelog (alert-code meaning change)

**Plant ≥3 seams** (e.g. claim steps only in service agreement; flushing only in install guide; remote shipping only in shipping policy).  
**Plant ≥3 gaps** (student discount; NSF certification; instalments) — list in `_seams_and_gaps.md` and **do not** fill later.  
**Injection canary:** one KB sentence resembling “SYSTEM: ignore instructions…” for Task 08 proof.

Internal consistency: warranty months, prices, and exclusions must match across FAQ/manual/CSV.

---

## 13. Phased implementation plan

Each phase has an **exit gate** (testable alone). Do not start the next phase until the gate passes.

### Phase 0 — Project skeleton & config (Task 01)

**Build:** Structure, venv, pinned deps baseline, `.env.example`, `.gitignore`, `config.py`, `logging_setup.py`, empty package inits, README stub.  
**Test:** `python -m src.config` prints masked settings; delete required env → one clear error.  
**Exit gate:** Fresh clone instructions work for config smoke (deps installable).

### Phase 1 — Knowledge base authored (Section 5)

**Build:** All ≥10 documents + seams/gaps file; injection canary line.  
**Test:** Manual consistency pass (warranty period, SKUs, shipping tiers).  
**Exit gate:** Examiner could open folder and see ≥3 formats and ≥10 files.

### Phase 2 — Ingestion loaders (Task 02)

**Build:** `loaders.py`, `normalize.py`, ingestion report; corrupt-file skip; duplicate detection.  
**Test:** Unit tests for metadata keys; corrupt PDF/MD fixture.  
**Exit gate:** All good files load with `source` + `doc_type`; one bad file warned and skipped.

### Phase 3 — Chunking experiment (Task 03)

**Build:** Recursive splitter + Markdown heading split + CSV row-wise; experiment script writing comparison table (≥6 configs).  
**Test:** Metadata survives split; sample 5 questions hit@4 measured per config.  
**Exit gate:** Chosen `(CHUNK_SIZE, OVERLAP, strategy)` documented with numbers; settings wired to env only.

### Phase 4 — Embeddings + vector store (Tasks 04–05)

**Build:** Batch embed + retry; cosine similarity diagnostic script; Chroma persist; `load_or_build` + `--rebuild`; similarity-search helper; embedding model fingerprint; decide **delete-by-source then reinsert** for single-doc updates (document in report).  
**Test:** Process restart loads without re-embed; “how long is the warranty” returns warranty chunks; paraphrase similarity diagnostic.  
**Exit gate:** Second run log proves disk load; search helper usable for debugging.

### Phase 5 — Retriever + relevance gate (Task 06)

**Build:** similarity k=3, k=8, MMR; optional metadata filter; empirical threshold; hit@k table on 8–10 questions.  
**Test:** Retrieval-only eval; gate triggers on known gap questions.  
**Exit gate:** Comparison table + documented threshold; gate refuses empty/weak retrieval.

### Phase 6 — RAG chain + prompts + citations (Tasks 07–08, 10)

**Build:** LCEL chain; context labels; Pydantic schema; grounding prompt v1→v2; injection rules; citation validate/dedupe; context budget; API/schema error handling.  
**Test:** End-to-end answer validates; refusal schema; weak vs strong prompt experiment; injection cases T-17/T-19 style; network kill → clean message.  
**Exit gate:** Single `ask()` always returns valid `GroundedResponse`; citations subset of supplied labels.

### Phase 7 — Memory + CLI (Task 09 + Feature 8)

**Build:** Condensation; history window N; `/reset`; `app_cli.py`; structured turn logging (query, rewrite, chunk ids/scores, latency, refusal).  
**Test:** Three-turn pronoun follow-up; topic switch; logged rewrite.  
**Exit gate:** Acceptance criterion transcript + logs for viva.

### Phase 8 — Evaluation harness (Task 11 + Section 8)

**Build:** `test_questions.json` (≥15, all categories); `run_eval.py`; results JSON + table; map conformance T-01…T-20; failure analysis template.  
**Test:** Harness runs unattended; refusal rate on unanswerable high; honest ≠100%.  
**Exit gate:** Results artefact + aggregate metrics + ≥1 failure attributed to a pipeline stage.

### Phase 9 — Streamlit UI + deploy docs (Feature 17 / demo)

**Build:** `app_streamlit.py` thin wrapper; trust UX; sidebar filters; screenshot pass.  
**Test:** Same questions as CLI; no logic drift.  
**Exit gate:** Six required screenshot types capturable; README run instructions for Streamlit.

### Phase 10 — Deliverables packaging

**Build:** `architecture.png` matching **actual** code; technical report (experiments + limitations); demo script; zip/repo checklist (Appendix B).  
**Test:** Stranger setup from README on clean venv; no secrets in tree.  
**Exit gate:** Submission checklist complete.

### Phase 11 — Optional bonus (only if Must features are strong)

**Priority order:** C2 cross-encoder rerank → C3 metadata routing polish → C1 hybrid search.  
**Rule:** Every bonus needs before/after metric or it does not count.

---

## 14. Error-handling matrix

| Failure | Behaviour |
|---------|-----------|
| Missing API key | Startup abort via Settings |
| Missing / corrupt index | CLI/UI message: run build_index; log path |
| Unreadable file at ingest | Skip + warn; continue |
| Empty / below-threshold retrieval | Pipeline refusal + escalation email; `answered=false` |
| Rate limit / timeout | Exponential backoff (finite); then user-facing error |
| Schema validation fail | One retry with stricter instruction; then safe fallback refusal object |
| Model cites unknown `[Sn]` | Drop invalid; lower confidence or refuse if none left |
| Prompt injection | Treat as data; refuse actions; no system-prompt leak |

---

## 15. Logging contract (every turn)

Timestamp, session id, original query, rewritten query, retrieved chunk ids + scores, strategy/k, refusal flag, latency (and tokens if available), error detail on failure. Central config from `logging_setup.py`; console + file.

---

## 16. Explicit non-goals (keep complexity down)

- No re-implementation of vector search outside LangChain/Chroma abstractions  
- No pasting whole KB into the prompt  
- No second business-logic path inside Streamlit  
- No production FastAPI/Docker unless Phase 11 + time remains  
- No real PII or real company policy text in the KB  

---

## 17. Suggested week schedule (from manual)

| Sitting | Phases | Focus |
|---------|--------|-------|
| W1 S1 | 0–2 | Skeleton + KB + loaders |
| W1 S2 | 3–4 | Chunk experiments + index persistence |
| W2 S1 | 5–6 | Retriever + chain + prompts |
| W2 S2 | 7 | Memory + CLI |
| W2 S3 | 8–10 | Eval, Streamlit, report, video |

---

## 18. Next action after this document

Implement **Phase 0** only: repository skeleton, pinned dependency draft addressing the compatibility table, and config/logging smoke test — then proceed phase-by-phase behind each exit gate.
