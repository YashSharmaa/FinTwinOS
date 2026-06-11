"""Consolidated bounded offline-RL pipeline across every decision environment.

``run_rl_pipeline`` is the entry point behind the ``fintwinos rl`` CLI command.
It runs the full log -> learn -> off-policy-evaluate -> shadow -> gate loop
(implemented once in :mod:`fintwinos.rl.train_offline_policy`) for *every*
bounded decision environment in :data:`fintwinos.rl.envs.ENV_NAMES`, then folds
the per-environment outcomes into a single JSON-serialisable report and,
optionally, writes it to ``<report_dir>/report.json``.

This is deliberately conservative by construction: the gate judges the *lower
confidence bound* of each candidate's off-policy value, so the aggregate
``verdicts`` tally reflects how many bounded policies would actually be cleared
for shadow or live use — never an unconditional "deploy".

Part of FinTwinOS (MIT) by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fintwinos.core.config import get_settings
from fintwinos.rl.envs import ENV_NAMES
from fintwinos.rl.train_offline_policy import main as run_one

__all__ = ["run_rl_pipeline"]

#: Gate verdicts in display order; also the keys of the aggregate tally.
VERDICTS = ("deploy", "shadow_only", "reject")


def _env_report(result: dict[str, Any]) -> dict[str, Any]:
    """Project one ``train_offline_policy.main`` result to a serialisable summary."""
    return {
        "env": result["env"],
        "verdict": result["verdict"],
        "n_transitions": result["n_transitions"],
        "n_episodes": result["n_episodes"],
        "behaviour_value": result["behaviour_value"],
        "threshold": result["threshold"],
        "ope": {name: ope.as_dict() for name, ope in result["ope"].items()},
        "shadow_buffer": result["shadow_buffer"].as_dict(),
        "shadow_env": result["shadow_env"].as_dict(),
        "training": result["training"],
    }


def run_rl_pipeline(
    seed: int | None = None,
    report_dir: Path | str | None = None,
    *,
    envs: tuple[str, ...] | list[str] | None = None,
    epochs: int = 30,
    n_episodes: int = 50,
) -> dict[str, Any]:
    """Run the bounded offline-RL pipeline over every environment and report.

    Args:
        seed: Master determinism seed. ``None`` resolves to ``Settings.seed``
            (``FINTWIN_SEED``), so a bare ``fintwinos rl`` is fully reproducible.
        report_dir: When given, ``report.json`` is written here (created if
            needed) and its path is returned under ``report_path``.
        envs: Restrict to these environment names; defaults to all of
            :data:`fintwinos.rl.envs.ENV_NAMES`.
        epochs: Conservative Q-learning sweeps per environment.
        n_episodes: Behaviour-policy episodes logged per environment.

    Returns:
        ``{"seed", "envs", "epochs", "n_episodes", "verdicts", "results"}`` —
        where ``verdicts`` is a ``{verdict: count}`` tally and ``results`` maps
        each environment name to its serialisable per-env report. Includes
        ``report_path`` when ``report_dir`` was given.
    """
    resolved_seed = get_settings().seed if seed is None else int(seed)
    env_names = tuple(envs) if envs else ENV_NAMES

    results: dict[str, Any] = {}
    verdicts: dict[str, int] = dict.fromkeys(VERDICTS, 0)
    for name in env_names:
        outcome = run_one(
            env_name=name,
            epochs=epochs,
            n_episodes=n_episodes,
            seed=resolved_seed,
            quiet=True,
        )
        results[name] = _env_report(outcome)
        verdicts[outcome["verdict"]] = verdicts.get(outcome["verdict"], 0) + 1

    report: dict[str, Any] = {
        "seed": resolved_seed,
        "envs": list(env_names),
        "epochs": epochs,
        "n_episodes": n_episodes,
        "verdicts": verdicts,
        "results": results,
    }

    if report_dir is not None:
        out_dir = Path(report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.json"
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["report_path"] = str(out_path)

    return report
