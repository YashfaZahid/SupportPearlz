"""SupportPearlz Streamlit UI — dark glassmorphism, Poppins, API-key gate."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

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

# Dark glass + neon magenta (reference aesthetic)
NEON = "#C026D3"
NEON_SOFT = "#E879F9"
GLASS = "rgba(255,255,255,0.06)"
INK = "#F4F4F5"
MUTED = "#A1A1AA"

st.set_page_config(
    page_title="SupportPearlz · Pearlz Home Systems",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)

_GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
  font-family: 'Poppins', sans-serif !important;
}}

/* Hide Streamlit header / chrome */
header[data-testid="stHeader"],
[data-testid="stHeader"] {{
  display: none !important;
  height: 0 !important;
}}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
.stAppDeployButton {{ display: none !important; }}

/* Deep black canvas with soft neon wash */
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(720px 420px at 12% 8%, rgba(192,38,211,0.18), transparent 55%),
    radial-gradient(640px 380px at 88% 92%, rgba(168,85,247,0.12), transparent 50%),
    linear-gradient(165deg, #050506 0%, #0A0A0B 45%, #0E0E10 100%) !important;
}}
[data-testid="stAppViewContainer"] > .main {{
  background: transparent !important;
}}
.block-container {{
  padding-top: 1.5rem !important;
  max-width: 920px;
}}

/* Floating glass sidebar */
section[data-testid="stSidebar"] {{
  background: transparent !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
  background: rgba(18, 18, 20, 0.72) !important;
  backdrop-filter: blur(22px) saturate(140%);
  -webkit-backdrop-filter: blur(22px) saturate(140%);
  margin: 0.85rem 0.55rem 0.85rem 0.75rem !important;
  padding: 1.1rem 0.85rem 1.25rem !important;
  border-radius: 1.65rem !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  box-shadow:
    0 24px 60px rgba(0,0,0,0.45),
    inset 0 1px 0 rgba(255,255,255,0.06);
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h4 {{
  color: {INK} !important;
  opacity: 1 !important;
}}
section[data-testid="stSidebar"] [data-testid="stCaption"] {{
  color: {MUTED} !important;
}}
section[data-testid="stSidebar"] hr {{
  border-color: rgba(255,255,255,0.08) !important;
}}

.sp-side-card {{
  position: relative;
  background: {GLASS};
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 1rem;
  padding: 0.85rem 0.95rem;
  margin: 0.4rem 0 0.85rem 0;
  backdrop-filter: blur(10px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}}
.sp-side-card::before {{
  content: "";
  position: absolute;
  left: 0;
  top: 18%;
  bottom: 18%;
  width: 2px;
  border-radius: 999px;
  background: {NEON};
  box-shadow: 0 0 12px rgba(192,38,211,0.85);
}}
.sp-side-rail {{
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
  margin: 0.65rem 0 1rem 0;
}}

/* Sidebar controls */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] {{
  background-color: rgba(255,255,255,0.05) !important;
  border-color: rgba(255,255,255,0.12) !important;
  border-radius: 0.75rem !important;
}}
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  color: {INK} !important;
  border-radius: 0.85rem !important;
}}
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg, {NEON}, #9333EA) !important;
  color: #fff !important;
  border: none !important;
  font-weight: 600 !important;
  border-radius: 0.85rem !important;
  box-shadow: 0 0 22px rgba(192,38,211,0.35);
}}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button {{
  color: {MUTED} !important;
  border-radius: 0.65rem !important;
}}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
  color: {INK} !important;
  background: rgba(192,38,211,0.22) !important;
  box-shadow: inset 0 0 0 1px rgba(192,38,211,0.55), 0 0 16px rgba(192,38,211,0.25);
}}

/* Main glass panel */
[data-testid="stMain"] {{
  background: transparent !important;
}}
[data-testid="stMainBlockContainer"] {{
  background: rgba(16, 16, 18, 0.55);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 1.5rem;
  padding: 1.35rem 1.5rem 1.75rem !important;
  margin-top: 0.85rem;
  box-shadow:
    0 28px 70px rgba(0,0,0,0.4),
    inset 0 1px 0 rgba(255,255,255,0.05);
}}

/* Chat bubbles — no human/bot avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"] [data-testid="stImage"],
[data-testid="stChatMessage"] img,
[data-testid="stChatMessage"] [class*="avatar"] {{
  display: none !important;
  width: 0 !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}}
[data-testid="stChatMessage"] {{
  background: rgba(255,255,255,0.045) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 1.1rem !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.25);
  padding: 0.9rem 1rem !important;
  margin-bottom: 0.75rem !important;
  gap: 0 !important;
}}
[data-testid="stChatMessage"] > div:first-child:has(img),
[data-testid="stChatMessage"] > div:first-child:has(svg) {{
  display: none !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
  background: linear-gradient(135deg, rgba(192,38,211,0.18), rgba(255,255,255,0.04)) !important;
  border-color: rgba(192,38,211,0.35) !important;
  box-shadow: 0 0 24px rgba(192,38,211,0.12);
}}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span {{
  color: {INK} !important;
}}

/* Chat input dock */
[data-testid="stBottom"] > div {{
  background: transparent !important;
}}
[data-testid="stChatInput"] {{
  background: rgba(20,20,22,0.88) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 1.35rem !important;
  box-shadow: 0 16px 48px rgba(0,0,0,0.45), 0 0 0 1px rgba(192,38,211,0.08);
  backdrop-filter: blur(16px);
}}
[data-testid="stChatInput"] textarea {{
  font-family: 'Poppins', sans-serif !important;
  color: {INK} !important;
}}

.sp-hero-copy {{
  font-family: 'Poppins', sans-serif;
  color: {MUTED};
  font-size: 0.92rem;
  line-height: 1.55;
  max-width: 42rem;
  margin: 0 0 1rem 0;
}}
.sp-safety {{
  background: rgba(192,38,211,0.10);
  border: 1px solid rgba(192,38,211,0.28);
  border-left: 3px solid {NEON};
  border-radius: 0.95rem;
  padding: 0.8rem 1rem;
  margin: 0 0 1.2rem 0;
  color: {INK};
  font-size: 0.84rem;
  line-height: 1.5;
  box-shadow: 0 0 28px rgba(192,38,211,0.08);
}}
.sp-safety strong {{
  display: block;
  color: {NEON_SOFT};
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.2rem;
}}

/* Gate — glass card */
.sp-gate {{
  width: min(420px, 92vw);
  padding: 1.75rem 1.5rem 1.35rem;
  border-radius: 1.35rem;
  background: rgba(18,18,20,0.78);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 28px 70px rgba(0,0,0,0.5), 0 0 40px rgba(192,38,211,0.12);
  backdrop-filter: blur(20px);
}}
.sp-gate h1 {{
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  font-size: 1.3rem;
  margin: 0 0 0.4rem 0;
  color: {INK};
}}
.sp-gate p {{
  color: {MUTED};
  font-size: 0.9rem;
  line-height: 1.5;
  margin-bottom: 0.75rem;
}}
</style>
"""


def _inject_chrome() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id = None
    if "api_key_unlocked" not in st.session_state:
        st.session_state.api_key_unlocked = False
    if "session_api_key" not in st.session_state:
        st.session_state.session_api_key = ""


def _key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def get_cached_agent(key_fp: str) -> SupportPearlzAgent:
    """Caching strategy 2 — keyed by API key fingerprint so unlock refreshes cleanly."""
    _ = key_fp
    clear_settings_cache()
    settings = get_settings()
    if not index_exists(settings=settings):
        raise IndexError_(
            "Vector index missing. Run: python -m src.ingestion.build_index --rebuild"
        )
    return SupportPearlzAgent(settings=settings)


def _apply_session_api_key(api_key: str) -> None:
    os.environ["OPENAI_API_KEY"] = api_key.strip()
    clear_settings_cache()
    get_cached_agent.clear()


def _confidence_badge(confidence: str, answered: bool) -> None:
    if not answered:
        bg, fg, border = "rgba(192,38,211,0.25)", NEON_SOFT, NEON
        label = "Needs escalation"
    elif confidence == "high":
        bg, fg, border = "rgba(192,38,211,0.18)", NEON_SOFT, "rgba(192,38,211,0.45)"
        label = f"Confidence · {confidence}"
    else:
        bg, fg, border = "rgba(255,255,255,0.05)", MUTED, "rgba(255,255,255,0.12)"
        label = f"Confidence · {confidence}"
    st.markdown(
        f'<span style="display:inline-block;margin-top:0.45rem;padding:0.28rem 0.7rem;'
        f'border-radius:0.65rem;background:{bg};color:{fg};border:1px solid {border};'
        f'font-size:0.78rem;font-weight:600;font-family:Poppins,sans-serif;'
        f'box-shadow:0 0 16px rgba(192,38,211,0.15);">'
        f"{label}</span>",
        unsafe_allow_html=True,
    )


def _render_sources(response: GroundedResponse) -> None:
    if not response.sources:
        st.caption("Sources: none (refusal or tool clarification)")
        return
    with st.expander("Sources used", expanded=True):
        for i, src in enumerate(response.sources, start=1):
            st.markdown(
                f"**[{i}] {src.source}** — `{src.location or 'n/a'}`\n\n"
                f"> {src.snippet[:220]}"
            )


def render_api_key_gate() -> None:
    """Unlock page — no app access until a key is submitted this session."""
    _inject_chrome()
    _left, mid, _right = st.columns([1, 1.15, 1])
    with mid:
        st.markdown('<div class="sp-gate">', unsafe_allow_html=True)
        st.markdown(
            "<h1>Enter your API key</h1>"
            "<p>SupportPearlz needs an OpenAI-compatible API key for embeddings and chat. "
            "The key stays in this browser session only.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("api_key_gate", clear_on_submit=False):
            api_key = st.text_input(
                "OpenAI API key",
                type="password",
                placeholder="sk-…",
                help="Used for chat + embeddings this session.",
            )
            submitted = st.form_submit_button(
                "Unlock SupportPearlz", type="primary", width="stretch"
            )

        if submitted:
            key = (api_key or "").strip()
            if len(key) < 20:
                st.error("Please enter a valid API key.", title="Key required")
                return
            try:
                _apply_session_api_key(key)
                settings = get_settings()
                _ = settings.openai_api_key
                st.session_state.session_api_key = key
                st.session_state.api_key_unlocked = True
                st.session_state.messages = []
                st.rerun()
            except ConfigurationError as exc:
                st.error(exc.message, title="Configuration error")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc), title="Could not unlock")


def render_chat_app() -> None:
    _inject_chrome()
    _init_state()
    _apply_session_api_key(st.session_state.session_api_key)

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        st.session_state.api_key_unlocked = False
        st.error(exc.message, title="Configuration error")
        st.stop()

    key_fp = _key_fingerprint(st.session_state.session_api_key)

    with st.sidebar:
        st.markdown('<div class="sp-side-card">', unsafe_allow_html=True)
        try:
            path = _resolve_store_path(settings)
            ready = index_exists(path, settings)
            fp = read_fingerprint(path) or {}
            if ready:
                st.markdown(
                    f"**Knowledge base**  \n"
                    f"{fp.get('chunk_count', '?')} chunks  \n"
                    f"`{fp.get('embedding_model', 'embeddings')}`"
                )
            else:
                st.markdown(
                    "**Index missing**  \n"
                    "Run `python -m src.ingestion.build_index --rebuild`"
                )
        except Exception:
            st.markdown("**Knowledge base**  \nStatus unavailable")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Session")
        role = st.segmented_control(
            "Access role",
            options=["customer", "agent"],
            default="customer",
            help="Customers cannot retrieve internal handbook chunks.",
        )
        doc_filter = st.selectbox(
            "Document type",
            options=["(any)", "manual", "policy", "faq", "pricing", "guide"],
            index=0,
        )
        product_filter = st.selectbox(
            "Product line",
            options=[
                "(any)",
                "AquaPearl 500 Pro",
                "AquaPearl 300",
                "AeroPearl Mini",
                "general",
            ],
            index=0,
        )
        show_debug = st.toggle("Show debug flags", value=False)

        st.markdown('<div class="sp-side-rail"></div>', unsafe_allow_html=True)

        if st.button("Reset chat", type="secondary", width="stretch"):
            st.session_state.messages = []
            try:
                get_cached_agent(key_fp).reset()
            except Exception:
                pass
            st.rerun()

        if st.button("Lock & change API key", type="primary", width="stretch"):
            st.session_state.api_key_unlocked = False
            st.session_state.session_api_key = ""
            st.session_state.messages = []
            get_cached_agent.clear()
            clear_settings_cache()
            st.rerun()

        st.markdown('<div class="sp-side-card">', unsafe_allow_html=True)
        st.markdown(
            f"**Escalate**  \n`{settings.support_escalation_email}`  \n\n"
            "Hybrid · rerank · tools · streaming"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<p class="sp-hero-copy">Grounded answers from official Pearlz documentation — '
        "citations, confidence scores, and honest refusals when the knowledge base is silent.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sp-safety"><div><strong>Safety notice</strong>'
        f"{settings.medical_safety_disclaimer}</div></div>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="ㅤ"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                _confidence_badge(meta.get("confidence", "n/a"), meta.get("answered", False))
                if meta.get("sources"):
                    with st.expander("Sources"):
                        for i, src in enumerate(meta["sources"], start=1):
                            st.markdown(
                                f"**[{i}] {src.get('source')}** — {src.get('location') or 'n/a'}"
                            )
                if show_debug and meta.get("flags"):
                    st.caption(f"flags: {', '.join(meta['flags'])}")

    prompt = st.chat_input(
        "Ask about warranty, E-03 beeps, shipping to Skardu, order 88213…",
        submit_mode="disable",
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="ㅤ"):
        st.markdown(prompt)

    doc_type = None if doc_filter == "(any)" else doc_filter
    product_line = None if product_filter == "(any)" else product_filter
    access_role = role or "customer"

    with st.chat_message("assistant", avatar="ㅤ"):
        try:
            agent = get_cached_agent(key_fp)
            agent.set_role(access_role)
            if st.session_state.agent_session_id != agent.session_id:
                st.session_state.agent_session_id = agent.session_id

            stream_box = st.empty()
            draft_parts: list[str] = []
            final: GroundedResponse | None = None

            with st.spinner("Retrieving grounded context…"):
                if settings.enable_streaming:
                    for item in agent.stream_ask(
                        prompt,
                        doc_type=doc_type,
                        product_line=product_line,
                        access_role=access_role,
                    ):
                        if isinstance(item, GroundedResponse):
                            final = item
                        else:
                            draft_parts.append(item)
                            stream_box.markdown("".join(draft_parts))
                else:
                    final = agent.ask(
                        prompt,
                        doc_type=doc_type,
                        product_line=product_line,
                        access_role=access_role,
                    )

            if final is None:
                raise GenerationError("No response produced.")

            stream_box.markdown(final.answer)
            _confidence_badge(final.confidence.value, final.answered)
            if not final.answered:
                st.info(
                    f"Human escalation: {settings.support_escalation_email}",
                    title="Not covered in documentation",
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


def main() -> None:
    _init_state()
    if not st.session_state.api_key_unlocked or not st.session_state.session_api_key:
        render_api_key_gate()
        return
    render_chat_app()


if __name__ == "__main__":
    main()
