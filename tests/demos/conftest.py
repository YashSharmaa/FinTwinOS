"""Test fixtures for the demos suite.

Two jobs:

1. **Stand-in installation.** The demos build the platform through the four
   contracted entry points. If a sibling module is missing from this checkout
   (it is being developed in parallel), a contract-faithful stand-in from
   ``fintwin_demo_standins`` is installed into ``sys.modules`` for that entry
   point only. When the real modules are present they are used untouched, so
   at integration time this suite exercises the genuine platform.

2. **Hermetic environment.** Every test runs offline, with a temp data
   directory and the execute band disabled — exactly the posture of a fresh
   clone with no API key.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

import fintwinos
import fintwinos.agents
import fintwinos.tools
from fintwinos.core.config import reset_settings_cache


def _missing(module_name: str) -> bool:
    """True if importing ``module_name`` raises ModuleNotFoundError for itself."""
    try:
        importlib.import_module(module_name)
        return False
    except ModuleNotFoundError as exc:
        return exc.name is None or module_name.startswith(exc.name) or exc.name == module_name
    except Exception:
        # A present-but-broken sibling should fail loudly in its own suite,
        # not be silently masked here.
        return False


def _install_standins() -> None:
    import fintwin_demo_standins as standins

    if _missing("fintwinos.twin_core.runtime"):
        pkg = types.ModuleType("fintwinos.twin_core")
        pkg.__path__ = []  # mark as package
        runtime_mod = types.ModuleType("fintwinos.twin_core.runtime")
        runtime_mod.build_runtime = standins.build_runtime
        pkg.runtime = runtime_mod
        sys.modules["fintwinos.twin_core"] = pkg
        sys.modules["fintwinos.twin_core.runtime"] = runtime_mod
        fintwinos.twin_core = pkg

    if _missing("fintwinos.twin_sim"):
        sim_mod = types.ModuleType("fintwinos.twin_sim")
        sim_mod.register_all = standins.register_all
        sys.modules["fintwinos.twin_sim"] = sim_mod
        fintwinos.twin_sim = sim_mod

    if _missing("fintwinos.tools.catalog"):
        catalog_mod = types.ModuleType("fintwinos.tools.catalog")
        catalog_mod.build_default_registry = standins.build_default_registry
        sys.modules["fintwinos.tools.catalog"] = catalog_mod
        fintwinos.tools.catalog = catalog_mod

    if _missing("fintwinos.agents.runtime"):
        agents_mod = types.ModuleType("fintwinos.agents.runtime")
        agents_mod.handle_case = standins.handle_case
        sys.modules["fintwinos.agents.runtime"] = agents_mod
        fintwinos.agents.runtime = agents_mod


_install_standins()


@pytest.fixture(autouse=True)
def demo_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Hermetic, offline, deterministic environment for every demo test."""
    monkeypatch.setenv("FINTWIN_OFFLINE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FINTWIN_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FINTWIN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FINTWIN_EXECUTE_TOOLS_ENABLED", "0")
    reset_settings_cache()
    yield
    reset_settings_cache()
