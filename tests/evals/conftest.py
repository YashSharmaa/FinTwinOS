"""Shared fixtures for the evals test suite: force offline mode, never touch the network."""

from __future__ import annotations

import pytest

from fintwinos.core.config import reset_settings_cache


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch):
    """Every evals test runs fully offline with no API key in scope."""
    monkeypatch.setenv("FINTWIN_OFFLINE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FINTWIN_OPENAI_API_KEY", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()
