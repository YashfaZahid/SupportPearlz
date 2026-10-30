"""Composed RAG chain: condense → retrieve → gate → generate → validate citations."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_chroma import Chroma
from src.chains.citations import validate_and_dedupe_citations
from src.chains.context import format_context
from src.chains.memory import condense_question, window_history
from src.chains.prompts import PROMPT_VERSION, build_grounded_prompt
from src.chains.schemas import GroundedResponse
from src.config import Settings, get_settings
from src.llm import get_chat_model
from src.retrieval.retriever import retrieve
from src.retrieval.vector_store import load_vector_store
from src.utils.cache_setup import setup_langchain_cache
from src.utils.errors import GenerationError, IndexError_, SupportPearlzError
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _refusal_message(settings: Settings) -> str:
    return (
        "I could not find this in the Pearlz documentation, so I cannot confirm an answer. "
        f"Please contact {settings.support_escalation_email} for help from a human agent."
    )


class SupportPearlzAgent:
    """Stateful session wrapper around the RAG pipeline."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        vector_store: Chroma | None = None,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        setup_logging(level=self.settings.log_level, log_file=self.settings.log_file)
        setup_langchain_cache(self.settings)
        self.session_id = session_id or str(uuid.uuid4())
        self.history: list[BaseMessage] = []
        self.vector_store = vector_store or load_vector_store(settings=self.settings)
        self.llm = get_chat_model(self.settings, streaming=False)
        self.stream_llm = get_chat_model(self.settings, streaming=True)
        self.prompt = build_grounded_prompt(
            self.settings.support_escalation_email,
            self.settings.medical_safety_disclaimer,
        )
        self._structured = self.llm.with_structured_output(GroundedResponse)

    def reset(self) -> None:
        self.history.clear()
        logger.info("session_reset session_id=%s", self.session_id)

    def ask(
        self,
        question: str,
        *,
        chat_history: Sequence[BaseMessage] | Sequence[tuple[str, str]] | None = None,
        doc_type: str | None = None,
        product_line: str | None = None,
    ) -> GroundedResponse:
        """End-to-end grounded answer. Always returns a schema-valid object."""
        started = time.perf_counter()
        history = window_history(
            chat_history if chat_history is not None else self.history,
            settings=self.settings,
        )
        refusal_flag = False
        rewritten = question
        try:
            rewritten = condense_question(
                question, history, self.llm, settings=self.settings
            )

            filter_dict = {}
            if doc_type:
                filter_dict["doc_type"] = doc_type
            if product_line:
                filter_dict["product_line"] = product_line

            retrieval = retrieve(
                self.vector_store,
                rewritten,
                settings=self.settings,
                filter_dict=filter_dict or None,
            )

            if not retrieval.relevant:
                refusal_flag = True
                response = GroundedResponse.refusal(
                    _refusal_message(self.settings),
                    flags=["relevance_gate", retrieval.gate_reason],
                )
            else:
                labeled = format_context(retrieval.chunks, settings=self.settings)
                history_text = _history_as_text(history)
                response = self._generate(
                    question=question,
                    context=labeled.text,
                    chat_history=history_text,
                    supplied=labeled.labels,
                )
                if labeled.truncated:
                    response.flags = list(response.flags) + ["context_truncated"]

            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "turn session_id=%s latency_ms=%s refusal=%s prompt=%s "
                "original=%r rewritten=%r answered=%s confidence=%s",
                self.session_id,
                latency_ms,
                refusal_flag or not response.answered,
                PROMPT_VERSION,
                question,
                rewritten,
                response.answered,
                response.confidence.value,
            )

            # Update internal history for CLI/Streamlit sessions
            if chat_history is None:
                self.history.append(HumanMessage(content=question))
                self.history.append(AIMessage(content=response.answer))
                self.history = window_history(self.history, settings=self.settings)

            return response

        except IndexError_:
            raise
        except SupportPearlzError:
            raise
        except Exception as exc:
            logger.exception("Generation failure session_id=%s", self.session_id)
            raise GenerationError(
                "The assistant could not complete this request. Please try again shortly.",
                detail=str(exc),
            ) from exc

    def stream_ask(
        self,
        question: str,
        *,
        doc_type: str | None = None,
        product_line: str | None = None,
    ) -> Iterator[str | GroundedResponse]:
        """Yield answer token strings, then a final ``GroundedResponse``.

        Citations are held until the structured object completes (Feature 18 / C4).
        """
        history = window_history(self.history, settings=self.settings)
        rewritten = condense_question(
            question, history, self.llm, settings=self.settings
        )
        filter_dict = {}
        if doc_type:
            filter_dict["doc_type"] = doc_type
        if product_line:
            filter_dict["product_line"] = product_line

        retrieval = retrieve(
            self.vector_store,
            rewritten,
            settings=self.settings,
            filter_dict=filter_dict or None,
        )
        if not retrieval.relevant:
            response = GroundedResponse.refusal(
                _refusal_message(self.settings),
                flags=["relevance_gate", retrieval.gate_reason],
            )
            yield response.answer
            self.history.append(HumanMessage(content=question))
            self.history.append(AIMessage(content=response.answer))
            self.history = window_history(self.history, settings=self.settings)
            yield response
            return

        labeled = format_context(retrieval.chunks, settings=self.settings)
        history_text = _history_as_text(history)

        # Stream a plain-language draft (not JSON), then finalize with structured schema.
        if self.settings.enable_streaming:
            stream_messages = [
                (
                    "system",
                    "You are SupportPearlz. Answer only from CONTEXT. Be concise. "
                    "Do not output JSON. If the context is insufficient, say you could not find it.",
                ),
                (
                    "human",
                    f"CONTEXT:\n{labeled.text}\n\nQUESTION:\n{question}\n\nWrite the customer-facing answer only.",
                ),
            ]
            for chunk in self.stream_llm.stream(stream_messages):
                text = getattr(chunk, "content", None) or ""
                if text:
                    yield text

        response = self._generate(
            question=question,
            context=labeled.text,
            chat_history=history_text,
            supplied=labeled.labels,
        )
        if labeled.truncated:
            response.flags = list(response.flags) + ["context_truncated"]

        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=response.answer))
        self.history = window_history(self.history, settings=self.settings)
        yield response

    def _generate(
        self,
        *,
        question: str,
        context: str,
        chat_history: str,
        supplied: dict,
    ) -> GroundedResponse:
        payload = {
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }
        try:
            raw = self._structured.invoke(self.prompt.invoke(payload))
            response = _coerce_response(raw)
        except Exception as first_exc:
            logger.warning("Structured output failed once: %s — retrying", first_exc)
            try:
                raw = self._structured.invoke(self.prompt.invoke(payload))
                response = _coerce_response(raw)
            except Exception as second_exc:
                logger.error("Structured output failed after retry: %s", second_exc)
                return GroundedResponse.refusal(
                    _refusal_message(self.settings),
                    flags=["schema_validation_failed"],
                )

        return validate_and_dedupe_citations(response, supplied)


def _coerce_response(raw: object) -> GroundedResponse:
    if isinstance(raw, GroundedResponse):
        return raw
    if isinstance(raw, dict):
        return GroundedResponse.model_validate(raw)
    raise TypeError(f"Unexpected structured output type: {type(raw)!r}")


def _history_as_text(history: Sequence[BaseMessage]) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def ask(question: str, chat_history: Sequence | None = None) -> GroundedResponse:
    """Functional entry point used by thin UIs."""
    agent = SupportPearlzAgent()
    return agent.ask(question, chat_history=chat_history or [])
