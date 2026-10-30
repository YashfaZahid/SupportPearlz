"""Phase 0 — logging setup tests."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.logging_setup import reset_logging_for_tests, setup_logging


def test_setup_logging_writes_file(tmp_path: Path) -> None:
    reset_logging_for_tests()
    log_path = tmp_path / "test.log"
    setup_logging(level="INFO", log_file=log_path)
    logging.getLogger("supportpearlz.test").info("hello-phase0")
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "hello-phase0" in text
    reset_logging_for_tests()
