"""Shared fixtures for twin_core tests.

The demo runtime is session-scoped (building the full demo book takes a
moment); tests that mutate twin state always build their own empty runtime.
"""

from __future__ import annotations

import pytest

from fintwinos.core.interfaces import TwinRuntime
from fintwinos.twin_core.runtime import build_runtime


@pytest.fixture(scope="session")
def demo_runtime() -> TwinRuntime:
    """One fully loaded demo twin shared by read-only tests."""
    return build_runtime(seed=7, with_demo_data=True)


@pytest.fixture()
def empty_runtime() -> TwinRuntime:
    """A fresh, empty twin for tests that ingest their own envelopes."""
    return build_runtime(seed=7, with_demo_data=False)
