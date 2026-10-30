"""Central logging configuration for SupportPearlz."""

from __future__ import annotations

import logging
from pathlib import Path


_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = "logs/supportpearlz.log",
) -> None:
    """Configure root logging once: console + optional file handlers.

    Safe to call multiple times; subsequent calls are no-ops unless
    ``force=True`` is needed in tests (use ``reset_logging_for_tests``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(numeric)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(numeric)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured (level=%s)", level)


def reset_logging_for_tests() -> None:
    """Remove handlers so tests can reconfigure cleanly."""
    global _CONFIGURED
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    _CONFIGURED = False
