# FinTwinOS documentation

FinTwinOS is an MIT-licensed, auditable digital-twin operating system for financial
organisations. It keeps a live, federated twin of an institution, books, exposures,
workflows, controls and customer journeys, and exposes that twin through typed
`observe_*` / `simulate_*` / `propose_*` / `execute_*` function-calls, specialised
agent swarms, calibrated simulators and bounded reinforcement learning, all behind a
first-class governance plane.

The central promise: **before a team or a model acts, the organisation can ask the
twin what is likely to happen, what could go wrong, which policies apply, and which
human approvals are mandatory.**

This page is the map. Every document below is self-contained, cross-linked, and kept
consistent with the code in `fintwinos/`, module paths, tool bands, environment
variables and CLI commands named in these pages are the real ones.

## Start here, by audience

| You are | Start with | Then read |
|---|---|---|
| A risk, model-risk or compliance officer | [Governance controls map](governance-controls-map.md) | [Threat model](threat-model.md), [Evaluation & release gates](evaluation.md), [Model cards](model-cards/llm-routing.md) |
| A regulator or supervisor reviewing the system | [Architecture](architecture.md) | [Governance controls map](governance-controls-map.md), [Audit & incident runbook](runbooks/incident-response.md) |
| An operator deploying or running FinTwinOS | [Deployment runbook](runbooks/deployment.md) | [Incident response](runbooks/incident-response.md), [Calibration runbook](runbooks/calibration.md) |
| A developer reading or extending the code | [CONTRACTS.md](../CONTRACTS.md) | [Architecture](architecture.md) |
| A researcher evaluating the claims | [Research basis](research-basis.md) | [Evaluation & release gates](evaluation.md), [Roadmap](roadmap.md) |

## Reference

- **[Architecture](architecture.md)**, the full reference architecture: federated
  twin substrate, data plane, the four-band function-call layer with five governance
  fields, the hierarchical-and-debating agent layer, the staged RL layer, sim-to-real
  calibration, and the governance plane. Includes the system flowchart, the design
  trade-off table, and a component-to-module map.
- **[Threat model](threat-model.md)**, a STRIDE-style threat model of the twin
  itself: ingestion poisoning, prompt injection via documents, tool-schema abuse,
  approval-token theft, audit tampering, simulator gaming and model-routing
  exfiltration, each mapped to concrete mitigations in code.
- **[Governance controls map](governance-controls-map.md)**, SR 11-7, NIST AI RMF
  (plus the Generative AI Profile), NIST IR 8356, DORA, the EU AI Act, FSB and
  BoE/FCA expectations mapped to specific FinTwinOS modules and flags, plus the
  mandatory deployment checklist.
- **[Evaluation & release gates](evaluation.md)**, the layered evaluation stack,
  hard release gates, the four headline experiments, the public benchmark plan, and
  how `fintwinos eval` maps onto all of it.
- **[Research basis](research-basis.md)**, the 2024 – May 2026 research and
  regulatory record the design rests on, with named sources.
- **[Roadmap](roadmap.md)**, the phased delivery plan from 2026-07 onward, including
  the stance on compiling specialist models later.

## Runbooks

- **[Deployment](runbooks/deployment.md)**, hybrid/on-prem deployment, the complete
  environment-variable table, and the staged shadow-mode rollout.
- **[Incident response](runbooks/incident-response.md)**, the kill switch,
  break-glass with countersign, rollback, and forensics from the hash-chained audit
  trail.
- **[Calibration](runbooks/calibration.md)**, recalibrating simulators with
  Kolmogorov–Smirnov and interval-coverage diagnostics.

## Model cards

- **[LLM routing](model-cards/llm-routing.md)**, the task-class-to-model routing
  layer over OpenAI models, with cost metering and the deterministic offline stub.
- **[AML subgraph scorer](model-cards/aml-subgraph-scorer.md)**, the graph-based
  alert-prioritisation scorer and its recall-floor governance.
- **[Simulators](model-cards/simulators.md)**, the market, treasury liquidity,
  compliance-ring and customer-ops simulators, and why none of them is ever treated
  as a faithful world.

## Project documents (repository root)

- [README](../README.md), overview and quickstart.
- [CONTRACTS.md](../CONTRACTS.md), module entry points, hard rules and ownership.
- [SECURITY.md](../SECURITY.md), private vulnerability disclosure policy.
- [AUTHORS.md](../AUTHORS.md), creator and maintainers.
- [LICENSE](../LICENSE), MIT.

## Conventions used throughout

- Module paths are given relative to the repository root, e.g.
  `fintwinos/tools/registry.py`.
- Environment variables use the `FINTWIN_` prefix and are read by
  `fintwinos/core/config.py` (`OPENAI_API_KEY` is also honoured).
- CLI commands are subcommands of the installed `fintwinos` entry point
  (`fintwinos info`, `fintwinos demo`, `fintwinos eval`, `fintwinos serve-tools`,
  `fintwinos export-schemas`, `fintwinos version`).
- Everything documented here runs fully offline with `FINTWIN_OFFLINE=1`: no API
  key, no network, deterministic rule-based fallbacks.

---

FinTwinOS, created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/), MIT License.
