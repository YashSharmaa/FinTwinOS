# FinTwinOS product brief (source document)

This is the distilled founding brief for FinTwinOS. The docs in `docs/` are adapted
from it. Citations are listed in `docs/research-basis.md`.

## Concept

FinTwinOS is a vendor-neutral, cloud-and-hybrid deployable platform that keeps a
continuously updated digital twin of a financial firm and exposes that twin through
typed read, simulate, propose and execute functions. Central promise: **before a team
or model acts, the organisation can ask the twin what is likely to happen, what could
go wrong, which policies apply, and which human approvals are mandatory.**

Posture: aggressive on simulation and evaluation, conservative on autonomy. RL
optimises bounded control decisions first — queue routing, simulation budgets, hedging
candidate selection, escalation thresholds, staffing, scenario prioritisation — before
it touches any irreversible act.

## Domain value map

| Function area | Twin mirrors | Swarm does | Value |
|---|---|---|---|
| Risk | Positions, exposures, market states, limits, scenarios, model versions | Simulates shocks, proposes hedges, tests policy changes, explains limit breaches | Faster stress testing, better limit governance, model lineage |
| Trading | Order flow, inventory, venue conditions, market-impact assumptions | Rehearses routing, execution, inventory choices before production | Lower slippage risk, safer strategy iteration, surveillance |
| Compliance | Customer/entity graph, KYC docs, alerts, policy corpus, case history | Prioritises alerts, drafts narratives, finds policy support, simulates false-positive trade-offs | Higher analyst throughput with preserved auditability |
| Treasury | Cash ladders, collateral pools, intraday liquidity, funding spreads | Simulates liquidity scenarios, recommends transfers, prioritises contingency actions | Resilience and funding efficiency under stress |
| Customer ops | Journey state, channel history, documents, complaints, SLA queues | Routes cases, predicts escalations, drafts compliant responses, simulates service policy changes | Lower handling time, fewer SLA breaches |

All five domains share one substrate: a stateful twin (graph + time-series + documents),
not just prompts. AML is a subgraph problem (Elliptic2); trading/treasury need
calibrated scenario engines; risk needs versioned models and counterfactual replay;
customer ops needs calibrated persona and queue simulation.

## Architecture decisions

| Decision | Recommended stance |
|---|---|
| Twin topology | **Federated domain twins** with shared canonical entities (customer, legal entity, account, instrument, trade, case, policy, scenario) |
| Orchestration | **Durable external orchestration graph** for regulated workflows; prompt-only self-orchestration only for low-risk read-only work; workflow compilation later for mature procedures |
| Tool protocol | **Dual layer**: internal canonical tool SDK plus MCP/JSON-RPC-compatible edge |
| Learning strategy | **Staged**: rules → offline RL → shadow → narrow online |
| Simulation model | **Hybrid** rules + agent-based + ML simulators for auditability and calibration |
| Memory design | **Blackboard + local memories** to limit leakage and improve traceability |
| Deployment | **Tenant-isolated hybrid** cloud/on-prem |

Every tool carries five governance fields beyond a normal schema: risk tier, approval
policy, side-effect class, idempotency, provenance requirements. Four bands:
`observe_*`, `simulate_*`, `propose_*`, `execute_*`. Only execute may touch production,
behind human/policy gates.

Agent pattern: hierarchical-and-debating orchestration — planner decomposes, sensing
swarm gathers state and checks freshness, domain swarms work in parallel, critic/red-team
attacks the draft, execution swarm assembles a bounded plan.

RL: supervised/rule baseline first, offline RL second, shadow third, then narrow online
adaptation. Rewards combine business payoff with hard risk and compliance constraints
(e.g. Expected Shortfall penalties, recall floors, fairness guardrails).

Sim-to-real: simulators are never treated as faithful worlds — continuous calibration
via historical replay, offline data fitting, explicit uncertainty. Every simulator
exposes confidence intervals and calibration diagnostics, not point estimates.

Governance plane is first-class: every observation, tool call, policy check, simulation
branch, human approval and write-back event is a tamper-evident audit record.

## Evaluation stack and release gates

| Layer | Core metrics | Release gate |
|---|---|---|
| Function calls | Tool selection exactness, argument correctness, schema validity, hallucinated-tool rate | No write-enabled tools until exactness is stable |
| Agent runs | Task success, multi-run consistency, critic catch rate, human-review rate, latency, unit cost | No production use if reliability collapses across repeats |
| Risk & treasury | PnL impact, Expected Shortfall, limit breaches, funding-cost delta, scenario coverage | No autonomy unless hard-risk constraints never violated in replay and shadow |
| Compliance | Alert precision/recall, analyst minutes saved, policy citation accuracy, narrative completeness | No case-closure automation unless recall floors and policy traceability met |
| Customer ops | First-contact resolution, AHT, complaint escalation rate, SLA breaches | No autonomous regulated actions without strong human-oversight evidence |
| Platform | Drift detection lag, trace completeness, approval latency, break-glass usage, incident recovery time | No scale-out without observable control effectiveness |

Key experiments: single-agent vs swarm; prompt-only vs durable orchestrator;
simulate-before-act ablation; rules/supervised vs offline RL on bounded control tasks.

Benchmark plan: BFCL V4 + FinMCP-Bench (function layer); SECQUE + Finance Agent
Benchmark (filings/research); BlueFin (spreadsheets); Elliptic2 (graph compliance);
calibrated market simulator (trading/treasury); historical case replay + synthetic
persona stress tests with calibration (customer ops).

## Datasets

| Class | Public starter | Notes |
|---|---|---|
| Filings | SEC EDGAR APIs (JSON + XBRL) | Base for analyst/compliance demos |
| AML graphs | Elliptic2 | Subgraph-level suspicious-pattern learning |
| Research tasks | Finance Agent Benchmark | Hard filings research questions |
| Filing QA | SECQUE | Comparison, ratio, risk, insight tasks |
| Tool use | FinMCP-Bench | Real-world financial tool invocation |
| Spreadsheets | BlueFin | Real finance spreadsheet tasks |
| Forecasting | MIRAI | Event forecasting with tools |
| Privileged | Internal transactions, positions, cases, transcripts | Deployment-local only |

## Research basis (2024 – May 2026)

- NIST digital-twin definition/state-of-the-art; NIST DT standardisation (ISO 23247-style);
  NIST IR 8356 (DT security & trust); NIST GenAI RMF profile.
- BoE/FCA AI survey 2024: 75% of firms use AI, 10% plan to; foundation models 17% of use
  cases; only 2% fully autonomous; 46% of firms only partially understand their AI;
  rising third-party concentration.
- FSB 2024: AI amplifies third-party concentration, cyber risk, market correlations,
  model risk.
- Finance Agent Benchmark 2025: best model 46.8% on 537 real SEC research tasks.
  FinMCP-Bench 2026: 613 tool-using tasks over 65 real financial MCPs. BlueFin 2026:
  strongest models under 50% on finance spreadsheets. BFCL V4: multi-turn, memory,
  hallucination, format sensitivity. SECQUE 2025: expert filings QA.
- Elliptic2 2024 (AML subgraphs); TimeFound 2025 + operational TSFM work 2026 (routing
  matters); MIRAI (event forecasting).
- Multi-agent: collaboration survey 2025; HPTSA (planner spawning sub-agents); LEMON
  (learned orchestration specs); Dr. MAS (agent-wise reward normalisation); May 2026
  orchestration-traces review (no public RL method for stopping decisions yet).
- Sim-to-real: SYN-DIGITS 2026 (post-hoc calibration needed); offline domain
  randomisation 2025 (fit simulator distributions to offline data); RL in agent-based
  market simulation 2024 (stylised facts, flash-crash response); risk-sensitive RL 2025
  (Expected Shortfall objectives).
- Regulatory: SR 11-7 (model risk management, effective challenge, inventories);
  DORA (ICT risk, third-party risk, resilience testing, incident handling); EU AI Act
  (risk-based framework, human oversight for high-risk); 2026 controlled study showing
  prompt-only self-orchestration can win on procedural tasks; 2026 workflow-compilation
  work (compile stable workflows into specialist models later).

Convergence: build with tools, not just prompts; with state, not just retrieval; with
explicit coordination, not just more agents; with calibrated simulators, not just
synthetic logs; with governance, not just benchmarks.

## Security and regulatory deployment checklist

| Control area | Implement before production | Main source |
|---|---|---|
| Use-case scoping | Classify workflows by criticality, legal impact, high-risk AI obligations | SR 11-7, EU AI Act |
| Model inventory | Full inventory of LLMs, simulators, classical models, prompts, tools, workflow versions | SR 11-7 |
| Effective challenge | Independent review of model logic, prompt/tool design, simulator assumptions | SR 11-7 |
| Validation | Validate inputs, processes, outputs, documentation, outcomes, monitoring | SR 11-7 |
| Human oversight | Humans can monitor, interpret, override, halt high-risk decisions | EU AI Act, BoE/FCA |
| Data governance | Lineage, retention, PII minimisation, access segregation, dataset cards | BoE/FCA, NIST AI RMF |
| Twin security | Threat-model ingestion, sync, simulation, write-back channels | NIST IR 8356 |
| Third-party risk | Separate control plane from model providers; replaceable adapters; concentration monitoring | BoE/FCA, FSB, DORA |
| Resilience testing | Failover, incident, replay, recovery, penetration-style exercises | DORA |
| Approval controls | Maker-checker, dual control, risk-tier-based write permissions | DORA, SR 11-7, EU AI Act |
| Monitoring & drift | Tool errors, calibration drift, policy breaches, latency, post-release behaviour | SR 11-7, NIST AI RMF |
| Incident handling | Kill switches, rollback, notification logic, forensics-ready traces | DORA, NIST IR 8356 |
| Documentation pack | Model cards, system cards, scenario assumptions, benchmark results, approval matrices | BoE/FCA, SR 11-7, NIST AI RMF |

Deployment rule: **no write-enabled production access until the system passes replay,
shadow mode, human review and control testing at the workflow level.**

## Roadmap

1. Core: canonical twin schema + event models; read-only tools + replay engine.
2. Agents: domain swarms + critic patterns; human approval + policy gates.
3. Learning: simulator calibration + off-policy evaluation; offline RL for bounded decisions.
4. Production: hybrid deployment, hardening, docs; public benchmark harness + demo apps.
5. Later: compiled specialist models once workflows are stable and governance is externalised.

---

*FinTwinOS — created by Yash Sharma (https://www.linkedin.com/in/yashsharmaa/) — MIT License.*
