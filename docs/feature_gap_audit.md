# Feature gap audit vs AAI–407 / PRAC–03

Palette lock (UI): `#146466` petrol · `#c92f1c` vermillion · `#5aaba9` seafoam · `#e2e2c4` cream.

| # | Requirement | Level | Status | Notes |
|---|-------------|-------|--------|-------|
| F1 | Document ingestion (3+ formats, report) | Must | Done | PDF/MD/DOCX/CSV |
| F2 | Persistent index | Must | Done | Chroma + fingerprint |
| F3 | Semantic retrieval | Must | Done | Embeddings |
| F4 | Grounded generation | Must | Done | Prompt v2 |
| F5 | Source citations (used-only, deduped) | Must | Done | citations.py |
| F6 | Unknown-answer / refusal | Must | Done | Gate + prompt |
| F7 | Partial-answer handling | Must | Done | Prompt rule |
| F8 | Conversation + condensation + reset | Must | Done | memory + CLI/UI |
| F9 | Structured output schema | Must | Done | GroundedResponse |
| F10 | Prompt injection protection | Must | Done | Canary + rules |
| F11 | Error handling | Must | Done | Typed errors |
| F12 | Structured logging | Must | Done | logging_setup |
| F13 | Env-driven config | Must | Done | config.py |
| F14 | Evaluation harness | Must | Done | 18 cases |
| F15 | Confidence signal | Should | Done | high/partial/none |
| F16 | Metadata filtering | Should | Done | UI + retriever |
| F17 | Web / API UI | Could | Done | Streamlit |
| F18 | Streaming responses | Could | Done | stream_ask |
| C1 | Hybrid BM25 + dense | Bonus | **Added** | EnsembleRetriever |
| C2 | Cross-encoder / rerank top-20→4 | Bonus | **Added** | Local fusion reranker |
| C3 | Metadata routing | Bonus | Done | Filters |
| C4 | Streaming | Bonus | Done | |
| C5 | Query expansion / multi-query | Bonus | **Added** | 2–3 variants fuse |
| C6 | Public vs internal KB role | Bonus | **Added** | role flag |
| C7 | Tool / function calling | Bonus | **Added** | mocked order/stock |
| C8 | Tracing / judge eval | Bonus | Partial | JSON turn traces |
| C9 | Caching | Bonus | Done | LC SQLite + st.cache_resource + query cache |
| C10 | Container deploy | Bonus | **Added** | Dockerfile + compose |

Deliverables still soft: architecture.png, technical_report.pdf, demo video, 20-case conformance table (eval covers 18).
