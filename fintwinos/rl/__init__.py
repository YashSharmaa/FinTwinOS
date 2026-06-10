"""Bounded offline reinforcement learning for FinTwinOS.

This module is the *learning* layer of the twin: it never acts against
production systems. Policies are trained on logged twin episodes
(:mod:`~fintwinos.rl.episodes`), kept pessimistic about unsupported actions
(:mod:`~fintwinos.rl.offline`), evaluated strictly off-policy with confidence
intervals (:mod:`~fintwinos.rl.ope`), exercised in shadow against the
incumbent (:mod:`~fintwinos.rl.shadow`) and only then considered for
deployment — and even that final step happens through the governed
execute-band tooling, never from here.

Everything is pure numpy, deterministic given a seed, and fully offline.

Part of FinTwinOS (MIT) by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from fintwinos.rl.bandits import LinUCB, ThompsonGaussian
from fintwinos.rl.envs import (
    ENV_NAMES,
    AlertTriageEnv,
    BoundedDiscreteEnv,
    LiquidityActionEnv,
    QueueRoutingEnv,
    make_alert_triage_env,
    make_env,
    make_liquidity_action_env,
    make_queue_routing_env,
)
from fintwinos.rl.episodes import (
    Episode,
    ReplayBuffer,
    Transition,
    build_episodes_from_replay,
    collect_episodes,
)
from fintwinos.rl.offline import ConservativeQLearner, StateDiscretiser
from fintwinos.rl.ope import (
    OPEResult,
    doubly_robust,
    effective_sample_size,
    fit_reward_model,
    ips,
    snips,
)
from fintwinos.rl.rewards import BriefWeights, expected_shortfall, risk_sensitive_reward
from fintwinos.rl.shadow import ShadowReport, ShadowRunner, deployment_gate

__all__ = [
    "ENV_NAMES",
    "AlertTriageEnv",
    "BoundedDiscreteEnv",
    "BriefWeights",
    "ConservativeQLearner",
    "Episode",
    "LinUCB",
    "LiquidityActionEnv",
    "OPEResult",
    "QueueRoutingEnv",
    "ReplayBuffer",
    "ShadowReport",
    "ShadowRunner",
    "StateDiscretiser",
    "ThompsonGaussian",
    "Transition",
    "build_episodes_from_replay",
    "collect_episodes",
    "deployment_gate",
    "doubly_robust",
    "effective_sample_size",
    "expected_shortfall",
    "fit_reward_model",
    "ips",
    "make_alert_triage_env",
    "make_env",
    "make_liquidity_action_env",
    "make_queue_routing_env",
    "risk_sensitive_reward",
    "snips",
]
