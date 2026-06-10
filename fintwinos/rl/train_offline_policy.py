"""End-to-end bounded offline RL pipeline: log -> learn -> evaluate -> gate.

Wires the whole module together exactly the way a production run would:

1. roll a *stochastic, propensity-logging* behaviour policy through a bounded
   decision env, producing a :class:`~fintwinos.rl.episodes.ReplayBuffer`;
2. fit a :class:`~fintwinos.rl.offline.ConservativeQLearner` on the log;
3. estimate the learned policy's value off-policy (IPS / SNIPS / DR);
4. shadow-run the candidate against the behaviour policy (buffer and env);
5. pass the doubly-robust estimate through
   :func:`~fintwinos.rl.shadow.deployment_gate` against the logged baseline.

Run it from the command line (``python -m fintwinos.rl.train_offline_policy``)
or call :func:`main` from tests and notebooks — it returns the full result
dictionary and is deterministic given ``seed``.

Part of FinTwinOS (MIT) by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from fintwinos.rl.envs import ENV_NAMES, make_env
from fintwinos.rl.episodes import ReplayBuffer, collect_episodes
from fintwinos.rl.offline import ConservativeQLearner
from fintwinos.rl.ope import OPEResult, doubly_robust, ips, snips
from fintwinos.rl.shadow import ShadowRunner, deployment_gate

__all__ = ["behaviour_policy_for", "main"]

#: Exploration mass mixed into every behaviour policy so propensities stay
#: bounded away from zero and importance weights stay finite.
_BEHAVIOUR_EPSILON = 0.3


def _preferred_action(env_name: str, state: np.ndarray, n_actions: int) -> int:
    """Deterministic heuristic core of the behaviour policy, per environment."""
    if env_name == "alert_triage":
        score = float(state[0])
        if score > 0.75:
            return 2  # escalate
        if score > 0.4:
            return 1  # investigate
        return 0  # dismiss
    if env_name == "liquidity_action":
        cash, headroom = float(state[0]), float(state[1])
        if cash > 0.3:
            return 0  # do_nothing
        if headroom > 0.2:
            return 1  # draw_facility
        return 3  # term_funding
    if env_name == "queue_routing":
        pressure = float(state[3])  # queue / SLA buffer ratio
        return min(n_actions - 1, int(pressure * 2.0))
    return 0


def behaviour_policy_for(env: Any):
    """Build the propensity-logging behaviour policy for one env.

    The policy is an epsilon-mixture of a deterministic per-env heuristic and
    the uniform distribution: with probability ``1 - eps + eps/n`` it plays
    the heuristic action, with ``eps/n`` each alternative — and it *reports
    the exact probability* of the action it sampled, which is what makes the
    resulting log usable for off-policy evaluation.

    Returns:
        Callable ``(state, rng) -> (action, behaviour_prob)``.
    """
    n_actions = env.n_actions
    env_name = getattr(env, "name", "")

    def policy(state: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
        preferred = _preferred_action(env_name, state, n_actions)
        probs = np.full(n_actions, _BEHAVIOUR_EPSILON / n_actions)
        probs[preferred] += 1.0 - _BEHAVIOUR_EPSILON
        action = int(rng.choice(n_actions, p=probs))
        return action, float(probs[action])

    return policy


def _render(console: Console, env_name: str, result: dict[str, Any]) -> None:
    """Pretty-print the pipeline outcome as rich tables."""
    table = Table(title=f"Offline policy evaluation — {env_name}")
    table.add_column("Estimator", style="bold")
    table.add_column("Estimate", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("ESS", justify="right")
    for method in ("ips", "snips", "doubly_robust"):
        ope: OPEResult = result["ope"][method]
        table.add_row(
            method,
            f"{ope.estimate:+.4f}",
            f"[{ope.lo_ci:+.4f}, {ope.hi_ci:+.4f}]",
            f"{ope.ess:.1f}",
        )
    console.print(table)

    summary = Table(title="Shadow run and deployment gate")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("logged transitions", str(result["n_transitions"]))
    summary.add_row("behaviour value (per step)", f"{result['behaviour_value']:+.4f}")
    summary.add_row("threshold", f"{result['threshold']:+.4f}")
    summary.add_row("buffer divergence", f"{result['shadow_buffer'].divergence_rate:.1%}")
    summary.add_row("env reward delta / episode", f"{result['shadow_env'].reward_delta:+.3f}")
    verdict = result["verdict"]
    colour = {"deploy": "green", "shadow_only": "yellow", "reject": "red"}[verdict]
    summary.add_row("verdict", f"[{colour}]{verdict}[/{colour}]")
    console.print(summary)


def main(
    env_name: str = "alert_triage",
    epochs: int = 30,
    n_episodes: int = 50,
    seed: int = 7,
    cql_alpha: float = 0.5,
    threshold: float | None = None,
    min_ess: float | None = 30.0,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the full offline pipeline on one bounded decision env.

    Args:
        env_name: One of :data:`fintwinos.rl.envs.ENV_NAMES`.
        epochs: Conservative Q-learning sweeps over the log.
        n_episodes: Behaviour-policy episodes to log.
        seed: Master determinism seed (data, training shuffles, bootstraps).
        cql_alpha: Conservatism weight of the learner.
        threshold: Deployment-gate threshold; defaults to the logged
            behaviour policy's per-step value, so deploying means beating the
            incumbent with confidence.
        min_ess: Effective-sample-size floor applied by the gate.
        quiet: Suppress the rich tables (used by tests).

    Returns:
        Dictionary with the buffer size, training diagnostics, all three
        :class:`~fintwinos.rl.ope.OPEResult` objects, both shadow reports,
        the gate ``verdict`` and the exported Q-table.
    """
    if env_name not in ENV_NAMES:
        raise KeyError(f"unknown env '{env_name}'; available: {', '.join(ENV_NAMES)}")

    env = make_env(env_name, seed=seed)
    behaviour = behaviour_policy_for(env)
    episodes = collect_episodes(env, behaviour, n_episodes=n_episodes, seed=seed)
    buffer = ReplayBuffer.from_episodes(episodes, n_actions=env.n_actions)

    learner = ConservativeQLearner(
        n_actions=env.n_actions, alpha=cql_alpha, gamma=0.95, seed=seed
    )
    training = learner.fit(buffer, epochs=epochs)

    def target_probs(state: np.ndarray) -> np.ndarray:
        return learner.policy_probabilities(state, epsilon=0.05)

    ope_results = {
        "ips": ips(buffer, target_probs, seed=seed),
        "snips": snips(buffer, target_probs, seed=seed),
        "doubly_robust": doubly_robust(buffer, target_probs, seed=seed),
    }

    behaviour_value = buffer.mean_reward()
    gate_threshold = behaviour_value if threshold is None else float(threshold)
    verdict = deployment_gate(
        ope_results["doubly_robust"], gate_threshold, min_ess=min_ess
    )

    greedy = learner.greedy_policy()

    def baseline_deterministic(state: np.ndarray) -> int:
        return _preferred_action(env_name, state, env.n_actions)

    runner = ShadowRunner(baseline_deterministic, greedy, seed=seed)
    shadow_buffer = runner.run_buffer(buffer)
    shadow_env = runner.run_env(make_env(env_name, seed=seed + 1), n_episodes=6)

    result: dict[str, Any] = {
        "env": env_name,
        "n_transitions": len(buffer),
        "n_episodes": n_episodes,
        "training": training,
        "ope": ope_results,
        "behaviour_value": behaviour_value,
        "threshold": gate_threshold,
        "verdict": verdict,
        "shadow_buffer": shadow_buffer,
        "shadow_env": shadow_env,
        "q_table": learner.export_q_table(),
    }
    if not quiet:
        _render(Console(), env_name, result)
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI, not pytest
    import typer

    typer.run(main)
