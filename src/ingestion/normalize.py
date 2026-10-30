"""Conservative text normalisation for ingested documents."""

from __future__ import annotations

import re

_MULTI_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
# Repeated running headers/footers often look like short identical lines; keep light touch.
_FORM_FEED = re.compile(r"\f+")


def normalize_text(text: str) -> str:
    """Collapse noisy whitespace without destroying tables/headings.

    - Replace form feeds with newlines
    - Collapse runs of spaces/tabs
    - Cap consecutive blank lines at two
    - Strip leading/trailing whitespace on the whole document
    """
    if not text:
        return ""
    cleaned = _FORM_FEED.sub("\n", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTI_WS.sub(" ", line).rstrip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _MULTI_NL.sub("\n\n", cleaned)
    return cleaned.strip()
