# Research basis

FinTwinOS is an opinionated synthesis of research and supervisory work published
between 2024 and May 2026. This page names the sources behind each design decision
so the architecture can be challenged on its evidence, not its rhetoric. Where a
quantitative claim appears anywhere in this documentation, it is attributed to the
named source below; FinTwinOS makes no unattributed quantitative claims of its own.

The convergent lesson across all of it: **build with tools, not just prompts; with
state, not just retrieval; with explicit coordination, not just more agents; with
calibrated simulators, not just synthetic logs; and with governance, not just
benchmarks.**

## Digital twins and their security

| Source | Year | What it contributes | Where it shows up in FinTwinOS |
|---|---|---|---|
| NIST digital-twin definition and state-of-the-art work | 2024–2025 | A disciplined definition: a twin is a synchronised, decision-grade virtual representation, not a dashboard | The stateful substrate — graph, time series, documents, replay — in `fintwinos/twin_core/` ([architecture §1](architecture.md#1-the-federated-twin-substrate)) |
| NIST digital-twin standardisation work (ISO 23247-style framing) | 2024–2025 | Twins need canonical entity models and explicit synchronisation interfaces to interoperate | Canonical entities and `EventEnvelope` ingestion in `fintwinos/core/types.py`; exported JSON Schemas (`fintwinos export-schemas`) |
| NIST IR 8356 — security and trust considerations for digital twins | 2024 (current edition) | Twins introduce their own attack surface: ingestion, synchronisation, simulation and write-back channels each need threat modelling | The [threat model](threat-model.md) is structured around exactly those channels (T1, T5, T6, outbox isolation) |
| NIST AI RMF and the Generative AI Profile | 2023 / 2024 | Govern–Map–Measure–Manage discipline, plus GenAI-specific risks (confabulation, information integrity) | The [governance controls map](governance-controls-map.md); schema-validated tool calls and hallucinated-tool-rate tracking |

## The state of AI in financial institutions

| Source | Year | What it contributes | Where it shows up in FinTwinOS |
|---|---|---|---|
| Bank of England / FCA survey of artificial intelligence in UK financial services | 2024 | 75% of firms already use AI and a further 10% plan to; foundation models account for 17% of use cases; only about 2% of use cases are fully autonomous; 46% of firms report only partial understanding of the AI they use; third-party concentration is rising | The conservative-autonomy posture; explicit, inventoriable model routing; the replaceable provider adapter; human oversight as the default, not the exception |
| Financial Stability Board report on the financial stability implications of AI | 2024 | AI amplifies third-party concentration, cyber risk, market correlations and model risk at the system level | Control plane separated from model providers; `FINTWIN_OFFLINE=1` as proof of independence; cost/usage metering for concentration monitoring |

These two documents explain FinTwinOS's central bet: adoption is broad, autonomy
is rare, and understanding is partial — so the scarce, valuable thing is an
auditable system that *earns* autonomy in stages rather than assuming it.

## Benchmarks: why dependability cannot be assumed

| Source | Year | What it contributes | Where it shows up in FinTwinOS |
|---|---|---|---|
| Finance Agent Benchmark | 2025 | 537 real SEC research tasks; the best model scored 46.8% | The filings-research suite in the [benchmark plan](evaluation.md#benchmark-plan); the case for simulate-before-act and human review |
| FinMCP-Bench | 2026 | 613 tool-using tasks over 65 real financial MCP servers | Evaluation of the MCP-style edge (`fintwinos serve-tools`) |
| BlueFin | 2026 | Real finance spreadsheet tasks; strongest models under 50% | The tabular-reasoning suite |
| BFCL V4 (Berkeley Function-Calling Leaderboard) | 2025 (V4) | Multi-turn function calling with memory, hallucination and format-sensitivity measurement | The function-call layer metrics: selection exactness, argument correctness, hallucinated-tool rate |
| SECQUE | 2025 | Expert-written SEC-filings QA: comparison, ratio, risk and insight tasks | The filing-QA suite for analyst agents |
| Elliptic2 | 2024 | Large-scale AML dataset at the *subgraph* level — money laundering is a pattern over entities, not a single transaction | The graph-first compliance design; the [AML subgraph scorer](model-cards/aml-subgraph-scorer.md) |
| MIRAI | 2024 | Event forecasting with tool use | The forecasting suite |
| TimeFound and operational time-series-foundation-model studies | 2025 / 2026 | Time-series foundation models are promising but routing and operational fit matter more than raw model choice | Task-class routing as a first-class layer (`fintwinos/models/llm_routing/router.py`), applied to classical forecasters too |

## Multi-agent orchestration

| Source | Year | What it contributes | Where it shows up in FinTwinOS |
|---|---|---|---|
| Survey of LLM multi-agent collaboration mechanisms | 2025 | Taxonomy of coordination patterns; more agents without explicit coordination does not reliably help | The hierarchical-and-debating pattern with explicit roles, and the [single-agent vs swarm experiment](evaluation.md#1-single-agent-vs-swarm) |
| HPTSA — hierarchical planning with task-specific agents | 2024 | A planner that spawns scoped sub-agents outperforms flat teams on complex multi-step tasks | The planner-decomposes / domain-swarms-execute structure in `fintwinos/agents/` |
| LEMON — learned orchestration specifications | 2025 | Orchestration itself can be specified and learned as an artefact, not improvised per run | Durable orchestration graphs as data, with workflow compilation deferred to the [roadmap](roadmap.md) |
| Dr. MAS — agent-wise reward normalisation for multi-agent RL | 2025 | Stable multi-agent credit assignment requires per-agent reward treatment | The RL layer's per-decision bounded rewards rather than one global scalar (`fintwinos/rl/`) |
| Review of orchestration traces | May 2026 | No public RL method yet handles agent *stopping* decisions well | Stopping/escalation stays rule-based behind the policy gate; it is explicitly **not** delegated to learned policies in this release |
| Controlled study of prompt-only self-orchestration | 2026 | Prompt-only self-orchestration can win on procedural tasks under controlled conditions | The [prompt-only vs durable experiment](evaluation.md#2-prompt-only-vs-durable-orchestration); prompt-only mode permitted for low-risk read-only work only |
| Workflow-compilation work | 2026 | Stable workflows can later be compiled into specialist models | The "compiled specialists later" stance in the [roadmap](roadmap.md) |

## Simulation and sim-to-real

| Source | Year | What it contributes | Where it shows up in FinTwinOS |
|---|---|---|---|
| SYN-DIGITS | 2026 | Synthetic-data pipelines need post-hoc calibration; raw synthetic fidelity claims do not hold | Mandatory `calibration` blocks on every `SimulationResult`; the [calibration runbook](runbooks/calibration.md) |
| Offline domain randomisation | 2025 | Fit simulator parameter distributions to offline data rather than hand-tuning | The calibration procedure: fit against replay episodes, version the assumptions (`assumptions_version`) |
| RL in agent-based market simulation | 2024 | Agent-based market simulators can reproduce stylised facts and flash-crash responses, making them usable RL environments — with care | The agent-based market simulator in `fintwinos/twin_sim/` and its stylised-facts validation ([simulators card](model-cards/simulators.md)) |
| Risk-sensitive RL with Expected Shortfall objectives | 2025 | RL can optimise under tail-risk constraints directly | Hard Expected Shortfall penalties in the RL reward design (`fintwinos/rl/`) |

## Regulation and supervision

| Source | Year | What it contributes | Where it shows up in FinTwinOS |
|---|---|---|---|
| Federal Reserve SR 11-7 — guidance on model risk management | 2011, still the anchor | Model inventories, validation, effective challenge, evidence of use | Inventory-by-construction (tool catalog, configured routes, versioned simulators); the critic swarm; the [controls map](governance-controls-map.md) |
| DORA — EU Digital Operational Resilience Act | Applying from 2025 | ICT risk management, third-party risk, resilience testing, incident handling | Replay-based resilience exercises, kill-switch drills, the [incident-response runbook](runbooks/incident-response.md) |
| EU AI Act | 2024, phased application | Risk-based framework; human oversight, logging and documentation for high-risk AI | Approval tokens, awaiting-human states, the audit chain, this documentation pack |

## How to challenge this

If new evidence contradicts a design decision, the right move is an issue that
cites it. This page is updated when the evidence window moves; claims older than the window
are either re-validated or removed.

---

FinTwinOS — created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) — MIT License.
