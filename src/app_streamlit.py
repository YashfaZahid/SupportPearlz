"""SupportPearlz Streamlit UI — healthcare SaaS trust surface over the same RAG agent."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when launched as ``streamlit run src/app_streamlit.py``
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.chains.rag_chain import SupportPearlzAgent
from src.chains.schemas import GroundedResponse
from src.config import clear_settings_cache, get_settings
from src.retrieval.vector_store import (
    _resolve_store_path,
    index_exists,
    read_fingerprint,
)
from src.utils.errors import ConfigurationError, GenerationError, IndexError_, SupportPearlzError


st.set_page_config(
    page_title="SupportPearlz",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_cached_agent() -> SupportPearlzAgent:
    """Caching strategy 2: one loaded vector store + agent per process."""
    clear_settings_cache()
    settings = get_settings()
    if not index_exists(settings=settings):
        raise IndexError_(
            "Vector index missing. Run: python -m src.ingestion.build_index --rebuild"
        )
    return SupportPearlzAgent(settings=settings)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id = None


def _render_sources(response: GroundedResponse) -> None:
    if not response.sources:
        st.caption("Sources: none (refusal or uncited)")
        return
    with st.expander("Sources", expanded=True):
        for i, src in enumerate(response.sources, start=1):
            st.markdown(
                f"**[{i}] {src.source}** — {src.location or 'n/a'}\n\n"
                f"> {src.snippet[:220]}"
            )


def main() -> None:
    _init_state()

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        st.error(exc.message, title="Configuration error")
        st.stop()

    with st.sidebar:
        st.markdown("### Pearlz")
        st.caption("Official documentation answers only")
        fp = {}
        try:
            path = _resolve_store_path(settings)
            ready = index_exists(path, settings)
            fp = read_fingerprint(path) or {}
            if ready:
                st.success(
                    f"{fp.get('chunk_count', '?')} chunks indexed",
                    title="Knowledge base ready",
                )
            else:
                st.error("Index not found — build before chatting", title="Knowledge base missing")
        except Exception:
            st.warning("Could not read index status", title="Knowledge base")

        doc_filter = st.selectbox(
            "Filter by document type",
            options=["(any)", "manual", "policy", "faq", "pricing", "guide"],
            index=0,
        )
        product_filter = st.selectbox(
            "Filter by product line",
            options=[
                "(any)",
                "AquaPearl 500 Pro",
                "AquaPearl 300",
                "AeroPearl Mini",
                "general",
            ],
            index=0,
        )
        show_debug = st.toggle("Show retrieval debug", value=False)
        if st.button("Reset session", type="secondary", width="stretch"):
            st.session_state.messages = []
            try:
                get_cached_agent().reset()
            except Exception:
                pass
            st.rerun()

        st.divider()
        st.markdown(f"Escalate: `{settings.support_escalation_email}`")
        st.caption("Cache: LangChain SQLite + Streamlit resource cache")

    st.title("SupportPearlz")
    st.markdown(
        "Customer support knowledge agent for **Pearlz Home Systems** — "
        "grounded answers with citations."
    )
    st.warning(
        settings.medical_safety_disclaimer,
        title="Medical and product-safety notice",
        icon=":material/health_and_safety:",
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                st.caption(f"Confidence: **{meta.get('confidence', 'n/a')}**")
                if meta.get("sources"):
                    with st.expander("Sources"):
                        for i, src in enumerate(meta["sources"], start=1):
                            st.markdown(
                                f"**[{i}] {src.get('source')}** — {src.get('location') or 'n/a'}"
                            )
                if show_debug and meta.get("flags"):
                    st.caption(f"flags: {', '.join(meta['flags'])}")

    prompt = st.chat_input(
        "Ask about warranty, shipping, filters, installation…",
        submit_mode="disable",
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    doc_type = None if doc_filter == "(any)" else doc_filter
    product_line = None if product_filter == "(any)" else product_filter

    with st.chat_message("assistant"):
        try:
            agent = get_cached_agent()
            # Keep one logical session in the cached agent for condensation
            if st.session_state.agent_session_id != agent.session_id:
                st.session_state.agent_session_id = agent.session_id

            stream_box = st.empty()
            draft_parts: list[str] = []
            final: GroundedResponse | None = None

            if settings.enable_streaming:
                for item in agent.stream_ask(
                    prompt, doc_type=doc_type, product_line=product_line
                ):
                    if isinstance(item, GroundedResponse):
                        final = item
                    else:
                        draft_parts.append(item)
                        stream_box.markdown("".join(draft_parts))
            else:
                final = agent.ask(
                    prompt, doc_type=doc_type, product_line=product_line
                )

            if final is None:
                raise GenerationError("No response produced.")

            # Canonical structured answer replaces any streaming draft
            stream_box.markdown(final.answer)
            conf = final.confidence.value
            if final.answered:
                st.success(f"Confidence: {conf}", title="Grounded answer")
            else:
                st.info(
                    f"Confidence: {conf}. Contact {settings.support_escalation_email}",
                    title="Could not answer from documentation",
                )
            _render_sources(final)
            if show_debug and final.flags:
                st.caption(f"flags: {', '.join(final.flags)}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final.answer,
                    "meta": {
                        "confidence": final.confidence.value,
                        "sources": [s.model_dump() for s in final.sources],
                        "flags": final.flags,
                        "answered": final.answered,
                    },
                }
            )
        except (IndexError_, GenerationError, SupportPearlzError) as exc:
            st.error(exc.message, title="Something went wrong")
            st.session_state.messages.append(
                {"role": "assistant", "content": exc.message, "meta": None}
            )


if __name__ == "__main__":
    main()
