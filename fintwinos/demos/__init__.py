"""Packaged end-to-end demonstrations — the front door of FinTwinOS.

Every demo module exposes two entry points:

* ``main()`` — the contracted CLI hook (``fintwinos demo <name>``);
* ``run(offline_ok=True, console=None, seed=7, settings=None) -> dict`` —
  programmatic execution returning structured results that tests (and your
  own scripts) can assert on.

All demos build the platform through the contracted entry points
(``build_runtime`` → ``register_all`` → ``build_default_registry`` →
``handle_case``), route every twin access through the governed tool registry,
and run fully offline with deterministic fallbacks when no OpenAI key is
configured.

Created by Yash Sharma (https://www.linkedin.com/in/yashsharmaa/), MIT License.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Demo names accepted by ``fintwinos demo <name>`` and :func:`run_demo`.
DEMO_NAMES: list[str] = [
    "liquidity",
    "aml_triage",
    "analyst_research",
    "customer_ops",
    "day_in_the_life",
]

__all__ = ["DEMO_NAMES", "run_demo"]


def run_demo(name: str, **kwargs: Any) -> dict[str, Any]:
    """Run one packaged demo by name and return its structured results.

    Args:
        name: One of :data:`DEMO_NAMES`.
        **kwargs: Forwarded to the demo's ``run`` (``offline_ok``, ``console``,
            ``seed``, ``settings``).

    Raises:
        ValueError: If ``name`` is not a packaged demo.
    """
    if name not in DEMO_NAMES:
        raise ValueError(f"unknown demo '{name}'; choose from {', '.join(DEMO_NAMES)}")
    module = importlib.import_module(f"fintwinos.demos.{name}")
    return module.run(**kwargs)
