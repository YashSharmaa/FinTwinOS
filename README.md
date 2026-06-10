<div align="center">

<img src="docs/assets/logo.svg" alt="FinTwinOS logo" width="130"/>

# FinTwinOS

**An auditable digital-twin operating system for financial organisations.**

<p>
  <a href="https://github.com/YashSharmaa/FinTwinOS/actions/workflows/ci.yml"><img src="https://github.com/YashSharmaa/FinTwinOS/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/tests-949%20passing-16a34a" alt="tests: 949 passing"/>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white" alt="Python 3.11–3.13"/>
  <img src="https://img.shields.io/badge/license-MIT-F97316" alt="MIT License"/>
  <img src="https://img.shields.io/badge/LLM-OpenAI-412991?logo=openai&logoColor=white" alt="LLM provider: OpenAI"/>
  <img src="https://img.shields.io/badge/lint-ruff-D7FF64" alt="lint: ruff"/>
  <img src="https://img.shields.io/badge/offline--first-FINTWIN__OFFLINE%3D1-EA580C" alt="offline-first"/>
</p>

*Before a team or a model acts, ask the twin what is likely to happen, what could go wrong,*
*which policies apply, and which human approvals are mandatory.*

</div>

---

FinTwinOS keeps a live, federated digital twin of a financial institution — books,
exposures, workflows, controls and customer journeys — and exposes it through **typed
LLM function-calls**, **specialised agent swarms**, **calibrated simulators** and
**bounded reinforcement learning**, all behind first-class governance gates.

This is not another chatbot or a generic agent framework. It is an operating system for
**simulating and steering financial decisions** across risk, trading, compliance,
treasury and customer operations — aggressive on simulation and evaluation,
deliberately conservative on autonomy.

## How it works

<p align="center">
  <img src="docs/assets/architecture-animation.svg" alt="Animated architecture: sources flow into the digital twin, agent swarms rehearse on simulators, proposals pass a default-deny policy gate, and everything lands on a hash-chained audit trail" width="100%"/>
</p>

1. **Observe** — connectors stream events into the twin's graph, time-series and
   document stores; every record carries provenance.
2. **Simulate** — agent swarms rehearse the decision on calibrated simulators that
   return confidence intervals, never bare point estimates.
3. **Propose** — the planner aggregates domain analyses, a red-team critic attacks the
   draft, and the policy gate issues a verdict.
4. **Execute (gated)** — the execute band is default-deny: it needs an explicit policy
   allow-rule, a valid human approval token *and* a global enable flag — and every step
   lands on the tamper-evident audit chain.

## Why

- Financial firms have adopted AI broadly, but full autonomy remains rare and firms
  report only partial understanding of vendor models, rising third-party concentration
  and hidden model risk (BoE/FCA 2024 survey; FSB 2024).
- Finance-agent benchmarks (Finance Agent Benchmark, FinMCP-Bench, SECQUE, BlueFin,
  BFCL V4) show frontier models are still far from dependable on hard financial work.
- The open-source gap is not in agent wrappers — it is in **financially realistic,
  auditable, benchmarked** systems with inspectable orchestration, replaceable model
  routing and hybrid deployment.

## What's inside

| Layer | Modules | What it does |
|---|---|---|
| 🏦 Twin substrate | `fintwinos/twin_core` | Entity resolution, graph + time-series + document stores, event ingestion, replay engine |
| 🎛️ Simulation | `fintwinos/twin_sim` | Agent-based market sim, treasury liquidity, compliance/fraud-ring, customer-ops queues — all with confidence intervals and calibration diagnostics |
| 🔧 Tool layer | `fintwinos/tools` | Typed `observe_*` / `simulate_*` / `propose_*` / `execute_*` tools with risk tier, side-effect class, approval policy and provenance; MCP-style JSON-RPC server |
| 🤖 Agent swarms | `fintwinos/agents` | Planner, sensing, risk, compliance, treasury, customer-ops, critic/red-team agents and a durable case orchestrator |
| 🧠 Models | `fintwinos/models` | OpenAI LLM routing with cost metering and offline stubs; graph AML scoring; time-series forecasters; tabular scorecards |
| 📈 Learning | `fintwinos/rl` | Replay buffers, conservative offline RL, contextual bandits, off-policy evaluation, shadow-mode deployment gating |
| 🛡️ Governance | `fintwinos/policy` | Policy gates, YAML rule packs, maker-checker approvals, dual control, break-glass, kill switch — plus a tamper-evident, hash-chained audit trail |
| ✅ Evaluation | `fintwinos/evals` | Function-call exactness, agent-run consistency, domain metric suites, benchmark adapters, report generation |
| 🔌 Data | `fintwinos/connectors`, `fintwinos/datasets` | EDGAR, CSV/CDC/webhook connectors; synthetic generators for transactions, fraud rings, journeys, market paths; data cards |
| 🎬 Demos | `fintwinos/demos`, `examples/` | Liquidity stress, AML triage, analyst research, customer ops and an end-to-end day-in-the-life — all runnable fully offline |

## Quickstart

```bash
pip install -e ".[dev,server]"

# Run completely offline — no API key, no network:
FINTWIN_OFFLINE=1 fintwinos demo liquidity
FINTWIN_OFFLINE=1 fintwinos demo aml_triage
FINTWIN_OFFLINE=1 fintwinos eval all

# With OpenAI (the default LLM provider):
export OPENAI_API_KEY=sk-...
fintwinos demo day_in_the_life

# Serve the typed tool catalog over MCP-style JSON-RPC:
fintwinos serve-tools

# Inspect configuration and export canonical JSON Schemas:
fintwinos info
fintwinos export-schemas
```

## Safety posture

The execute tool band is **off by default** (`FINTWIN_EXECUTE_TOOLS_ENABLED=0`). Even
when enabled, every execute call needs (1) an explicit policy allow-rule, (2) a valid
human approval token, and (3) passes through the hash-chained audit trail. Observe,
simulate and propose tools are structurally side-effect free. The deployment rule:
**no write-enabled production access until a workflow passes replay, shadow mode, human
review and control testing.**

## Evaluation & release gates

`fintwinos eval all` runs the full evaluation stack offline and checks **8 release
gates** — function-call exactness, tool-hallucination rate, multi-run consistency, and
domain floors for risk, treasury, compliance and customer operations. Gates fail
conservatively: no evidence means no promotion. See
[`docs/evaluation.md`](docs/evaluation.md) for the gate definitions and the headline
experiments.

## Architecture

```mermaid
flowchart LR
    subgraph Sources[Enterprise and external sources]
        A[Core banking and payments]
        B[OMS EMS and treasury]
        C[KYC AML CRM and case tools]
        D[Market data news and filings]
    end
    subgraph Twin[Financial twin substrate]
        E[Event ingestion and entity resolution]
        F[Time series and feature store]
        G[Graph and state store]
        H[Document and policy index]
    end
    subgraph Agents[LLM and ML control layer]
        I[Sensing swarm]
        J[Domain analysis swarms]
        K[Critic and red team swarm]
        L[Planner and execution swarm]
    end
    subgraph Sim[Simulation and learning]
        M[Market and operations simulators]
        N[Scenario engine and stress tests]
        O[Offline RL and off policy evaluation]
        P[Calibration and drift monitors]
    end
    subgraph Guard[Governance and action]
        Q[Human approval gateway]
        R[Policy engine and audit trace]
        S[Write back adapters]
    end
    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    E --> G
    E --> H
    F --> I
    G --> I
    H --> I
    I --> J
    J --> K
    K --> L
    G --> M
    F --> M
    H --> N
    M --> O
    N --> O
    O --> L
    P --> L
    L --> Q
    Q --> R
    R --> S
    S --> A
    S --> B
    S --> C
    S --> E
```

See [`docs/architecture.md`](docs/architecture.md) for the full reference architecture,
[`docs/governance-controls-map.md`](docs/governance-controls-map.md) for the
regulator-facing controls map (SR 11-7, DORA, EU AI Act, NIST AI RMF, NIST IR 8356),
and [`docs/evaluation.md`](docs/evaluation.md) for the evaluation stack and release gates.

## Development

```bash
pip install -e ".[dev,server]"
ruff check fintwinos tests
pytest
```

See [`CONTRACTS.md`](CONTRACTS.md) for module boundaries and entry-point contracts.

## License

MIT © 2026 Yash Sharma. See [LICENSE](LICENSE).

<div align="center">

---

<img src="docs/assets/logo.svg" alt="" width="40"/>

**Created and maintained by Yash Sharma**

<a href="https://www.linkedin.com/in/yashsharmaa/"><img src="https://img.shields.io/badge/LinkedIn-Yash%20Sharma-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn: Yash Sharma"/></a>

</div>
