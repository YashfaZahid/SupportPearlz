"""C8 lightweight JSON turn tracing (local file; LangSmith-compatible fields)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def append_trace(event: dict[str, Any], *, path: Path | None = None) -> None:
    target = path or (PROJECT_ROOT / "logs" / "traces.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logger.debug("trace_written keys=%s", list(event.keys()))
