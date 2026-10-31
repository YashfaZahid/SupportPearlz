"""Composed RAG chain: condense → expand → retrieve → gate → tools → generate."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Iterator
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_chroma import Chroma

from src.chains.citations import validate_and_dedupe_citations
from src.chains.context import format_context
from src.chains.expansion import expand_queries, fuse_documents
from src.chains.memory import condense_question, window_history
from src.chains.prompts import PROMPT_VERSION, build_grounded_prompt
from src.chains.schemas import Confidence, GroundedResponse, SourceCitation
from src.chains.tools import (
    TOOLS,
    check_stock,
    detect_tool_intent,
    extract_order_id,
    extract_sku,
    get_order_status,
)
from src.config import Settings, get_settings
from src.llm import get_chat_model
from src.retrieval.retriever import ScoredChunk, retrieve, retrieve_with_scores
from src.retrieval.vector_store import load_vector_store
from src.utils.cache_setup import setup_langchain_cache
from src.utils.errors import GenerationError, IndexError_, SupportPearlzError
from src.utils.logging_setup import setup_logging
from src.utils.tracing import append_trace

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
        access_role: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        setup_logging(level=self.settings.log_level, log_file=self.settings.log_file)
        setup_langchain_cache(self.settings)
        self.session_id = session_id or str(uuid.uuid4())
        self.access_role = access_role or self.settings.default_access_role
        self.history: list[BaseMessage] = []
        self.vector_store = vector_store or load_vector_store(settings=self.settings)
        self.llm = get_chat_model(self.settings, streaming=False)
        self.stream_llm = get_chat_model(self.settings, streaming=True)
        self.prompt = build_grounded_prompt(
            self.settings.support_escalation_email,
            self.settings.medical_safety_disclaimer,
        )
        self._structured = self.llm.with_structured_output(GroundedResponse)
        self._query_cache: dict[str, GroundedResponse] = {}

    def reset(self) -> None:
        self.history.clear()
        logger.info("session_reset session_id=%s", self.session_id)

    def set_role(self, role: str) -> None:
        self.access_role = role if role in {"customer", "agent"} else "customer"

    def ask(
        self,
        question: str,
        *,
        chat_history: Sequence[BaseMessage] | Sequence[tuple[str, str]] | None = None,
        doc_type: str | None = None,
        product_line: str | None = None,
        access_role: str | None = None,
    ) -> GroundedResponse:
        """End-to-end grounded answer. Always returns a schema-valid object."""
        started = time.perf_counter()
        role = access_role or self.access_role
        history = window_history(
            chat_history if chat_history is not None else self.history,
            settings=self.settings,
        )
        refusal_flag = False
        rewritten = question
        try:
            # C7 tool routing for order/stock questions
            if self.settings.enable_tools:
                tool_response = self._maybe_tool_answer(question)
                if tool_response is not None:
                    self._commit_history(question, tool_response, chat_history)
                    append_trace(
                        {
                            "session_id": self.session_id,
                            "kind": "tool",
                            "question": question,
                            "answer": tool_response.answer,
                        }
                    )
                    return tool_response

            cache_key = None
            if self.settings.enable_query_cache and not history:
                cache_key = hashlib.sha256(
                    f"{role}|{doc_type}|{product_line}|{question.lower().strip()}".encode()
                ).hexdigest()
                if cache_key in self._query_cache:
                    logger.info("query_cache_hit key=%s", cache_key[:10])
                    return self._query_cache[cache_key]

            rewritten = condense_question(
                question, history, self.llm, settings=self.settings
            )

            filter_dict: dict = {}
            if doc_type:
                filter_dict["doc_type"] = doc_type
            if product_line:
                filter_dict["product_line"] = product_line

            retrieval = self._retrieve_expanded(
                rewritten, filter_dict=filter_dict or None, role=role
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
                "turn session_id=%s latency_ms=%s refusal=%s prompt=%s role=%s "
                "original=%r rewritten=%r answered=%s confidence=%s",
                self.session_id,
                latency_ms,
                refusal_flag or not response.answered,
                PROMPT_VERSION,
                role,
                question,
                rewritten,
                response.answered,
                response.confidence.value,
            )
            append_trace(
                {
                    "session_id": self.session_id,
                    "kind": "rag",
                    "role": role,
                    "question": question,
                    "rewritten": rewritten,
                    "answered": response.answered,
                    "confidence": response.confidence.value,
                    "latency_ms": latency_ms,
                    "sources": [s.model_dump() for s in response.sources],
                    "flags": response.flags,
                }
            )

            if cache_key and response.answered:
                self._query_cache[cache_key] = response

            self._commit_history(question, response, chat_history)
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
        access_role: str | None = None,
    ) -> Iterator[str | GroundedResponse]:
        """Yield answer token strings, then a final ``GroundedResponse``."""
        role = access_role or self.access_role
        if self.settings.enable_tools:
            tool_response = self._maybe_tool_answer(question)
            if tool_response is not None:
                yield tool_response.answer
                self._commit_history(question, tool_response, None)
                yield tool_response
                return

        history = window_history(self.history, settings=self.settings)
        rewritten = condense_question(
            question, history, self.llm, settings=self.settings
        )
        filter_dict: dict = {}
        if doc_type:
            filter_dict["doc_type"] = doc_type
        if product_line:
            filter_dict["product_line"] = product_line

        retrieval = self._retrieve_expanded(
            rewritten, filter_dict=filter_dict or None, role=role
        )
        if not retrieval.relevant:
            response = GroundedResponse.refusal(
                _refusal_message(self.settings),
                flags=["relevance_gate", retrieval.gate_reason],
            )
            yield response.answer
            self._commit_history(question, response, None)
            yield response
            return

        labeled = format_context(retrieval.chunks, settings=self.settings)
        history_text = _history_as_text(history)

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

        self._commit_history(question, response, None)
        yield response

    def _retrieve_expanded(self, query: str, *, filter_dict, role: str):
        cfg = self.settings
        if not cfg.enable_query_expansion:
            return retrieve(
                self.vector_store,
                query,
                settings=cfg,
                filter_dict=filter_dict,
                role=role,
            )

        variants = expand_queries(query, self.llm)
        doc_lists = []
        for variant in variants:
            chunks = retrieve_with_scores(
                self.vector_store,
                variant,
                settings=cfg,
                k=cfg.rerank_candidates if cfg.enable_rerank else cfg.retrieval_k,
                filter_dict=filter_dict,
            )
            # apply role filter manually for expansion path
            docs = [c.document for c in chunks]
            if role == "customer":
                docs = [
                    d
                    for d in docs
                    if str(d.metadata.get("access_level", "public")) != "internal"
                ]
            doc_lists.append(docs)

        fused = fuse_documents(doc_lists, limit=cfg.rerank_candidates)
        from src.retrieval.rerank import rerank_documents

        if cfg.enable_rerank:
            fused = rerank_documents(query, fused, top_n=cfg.retrieval_k)
        else:
            fused = fused[: cfg.retrieval_k]

        # Score fused set for gate
        scored = retrieve_with_scores(
            self.vector_store,
            query,
            settings=cfg,
            strategy="similarity",
            k=max(cfg.retrieval_k, len(fused) or 1),
            filter_dict=filter_dict,
        )
        score_map = {
            (c.document.metadata.get("source"), c.document.page_content[:160]): c.score
            for c in scored
        }
        chunks = [
            ScoredChunk(
                document=d,
                score=score_map.get((d.metadata.get("source"), d.page_content[:160]), 0.5),
                rank=i,
            )
            for i, d in enumerate(fused, start=1)
        ]
        from src.retrieval.retriever import relevance_gate, RetrievalResult

        relevant, reason = relevance_gate(chunks, settings=cfg)
        return RetrievalResult(
            query=query,
            chunks=chunks,
            strategy="hybrid+expand" if cfg.enable_hybrid_search else "expand",
            relevant=relevant,
            gate_reason=reason,
            best_score=min((c.score for c in chunks), default=None),
        )

    def _maybe_tool_answer(self, question: str) -> GroundedResponse | None:
        intent = detect_tool_intent(question)
        if intent == "none":
            return None
        if intent == "order":
            order_id = extract_order_id(question)
            if not order_id:
                return GroundedResponse(
                    answer=(
                        "I can look up order status, but I need your order number. "
                        f"Or contact {self.settings.support_escalation_email}."
                    ),
                    sources=[],
                    confidence=Confidence.partial,
                    answered=False,
                    flags=["tool_missing_order_id"],
                )
            text = get_order_status.invoke({"order_id": order_id})
            return GroundedResponse(
                answer=text,
                sources=[
                    SourceCitation(
                        label="TOOL",
                        source="get_order_status",
                        location=f"order_id:{order_id}",
                        snippet=text[:180],
                    )
                ],
                confidence=Confidence.high,
                answered=True,
                flags=["tool_order_status"],
            )
        if intent == "stock":
            sku = extract_sku(question)
            if not sku:
                return GroundedResponse(
                    answer=(
                        "I can check stock for a SKU (for example AP500-SED). "
                        "Which SKU should I look up?"
                    ),
                    sources=[],
                    confidence=Confidence.partial,
                    answered=False,
                    flags=["tool_missing_sku"],
                )
            text = check_stock.invoke({"sku": sku})
            return GroundedResponse(
                answer=text,
                sources=[
                    SourceCitation(
                        label="TOOL",
                        source="check_stock",
                        location=f"sku:{sku}",
                        snippet=text[:180],
                    )
                ],
                confidence=Confidence.high,
                answered=True,
                flags=["tool_stock"],
            )
        return None

    def _commit_history(
        self,
        question: str,
        response: GroundedResponse,
        chat_history: Sequence | None,
    ) -> None:
        if chat_history is None:
            self.history.append(HumanMessage(content=question))
            self.history.append(AIMessage(content=response.answer))
            self.history = window_history(self.history, settings=self.settings)

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


# Re-export tools list for viva / docs
AVAILABLE_TOOLS = TOOLS
