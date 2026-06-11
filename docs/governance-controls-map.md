# Governance controls map

This page maps supervisory expectations to concrete FinTwinOS modules, flags and
procedures. It is written for model-risk, compliance and audit functions evaluating
the platform, and for engineers who need to know *which code* answers *which
control question*.

Frameworks covered: Federal Reserve **SR 11-7** (model risk management), **NIST AI
RMF** and its **Generative AI Profile**, **NIST IR 8356** (digital-twin security
and trust), **DORA** (EU digital operational resilience), the **EU AI Act**, the
**FSB** 2024 report on AI and financial stability, and the **BoE/FCA** 2024 survey
of AI in UK financial services.

FinTwinOS is an open-source platform, not a compliance certification. The mapping
below shows where each control is *implemented or supported*; the adopting
institution remains responsible for operating the controls and evidencing them to
its supervisors.

## Control map

| Control area | Implement before production | Main source | Where in FinTwinOS |
|---|---|---|---|
| Use-case scoping | Classify workflows by criticality, legal impact, high-risk AI obligations | SR 11-7, EU AI Act | `RiskTier` on every tool and `Decision` (`fintwinos/core/types.py`); per-tool `risk_tier` in `ToolSpec` (`fintwinos/tools/registry.py`); band separation observe/simulate/propose/execute |
| Model inventory | Full inventory of LLMs, simulators, classical models, prompts, tools, workflow versions | SR 11-7 | Configured model tiers in `fintwinos/core/config.py` (`FINTWIN_LLM_MODEL_PRIMARY/FAST/CHEAP`); `registry.public_catalog()` for the complete tool inventory; `runtime.simulators` registry; `assumptions_version` on every `Scenario` and `SimulationResult`; [model cards](model-cards/llm-routing.md) |
| Effective challenge | Independent review of model logic, prompt/tool design, simulator assumptions | SR 11-7 | Critic/red-team swarm in the agent layer (`fintwinos/agents/`); `PolicyGate.check_decision` forcing review of executing decisions (`fintwinos/policy/gates.py`); the four headline experiments in [evaluation](evaluation.md) |
| Validation | Validate inputs, processes, outputs, documentation, outcomes, monitoring | SR 11-7 | JSON Schema validation on every tool call (Draft 2020-12, `fintwinos/tools/registry.py`); `fintwinos eval` suites and release gates (`fintwinos/evals/`); calibration diagnostics on every simulator ([calibration runbook](runbooks/calibration.md)) |
| Human oversight | Humans can monitor, interpret, override, halt high-risk decisions | EU AI Act, BoE/FCA | `ApprovalToken` required on every execute call; `awaiting_human` status from `handle_case` (`fintwinos/agents/runtime.py`); kill switch `FINTWIN_EXECUTE_TOOLS_ENABLED=0`; default policy pack flags high-risk proposals for review |
| Data governance | Lineage, retention, PII minimisation, access segregation, dataset cards | BoE/FCA, NIST AI RMF | `Provenance` on every envelope and tool result (`fintwinos/core/types.py`); `provenance_required` governance field; data cards in `datasets/cards/` (registered in `fintwinos/datasets/registry.py`); offline mode keeps data in-perimeter |
| Twin security | Threat-model ingestion, sync, simulation, write-back channels | NIST IR 8356 | The full [threat model](threat-model.md) (T1–T9), with mitigations mapped to code |
| Third-party risk | Separate control plane from model providers; replaceable adapters; concentration monitoring | BoE/FCA, FSB, DORA | Single provider adapter in `fintwinos/models/llm_routing/client.py`; configuration-driven routing (`fintwinos/models/llm_routing/router.py`); total provider independence via `FINTWIN_OFFLINE=1`; per-call cost/usage metering |
| Resilience testing | Failover, incident, replay, recovery, penetration-style exercises | DORA | Replay engine for episode re-execution (`ReplayEngine`, `fintwinos/core/interfaces.py`); [incident-response runbook](runbooks/incident-response.md) with kill-switch and rollback drills; offline mode as the degraded-mode rehearsal |
| Approval controls | Maker-checker, dual control, risk-tier-based write permissions | DORA, SR 11-7, EU AI Act | `FINTWIN_DUAL_CONTROL_REQUIRED=1` (default on); `ApprovalToken` scoping/expiry/status; execute band default-deny in `PolicyGate`; `min_risk_tier` rule matching |
| Monitoring & drift | Tool errors, calibration drift, policy breaches, latency, post-release behaviour | SR 11-7, NIST AI RMF | Audit events for every failure class (`tool.failed`, `tool.schema_rejected`, `policy.checked`); `calibration_report()` on every simulator; `elapsed_ms` on every `ToolResult`; eval suites re-runnable in CI (`fintwinos eval all`) |
| Incident handling | Kill switches, rollback, notification logic, forensics-ready traces | DORA, NIST IR 8356 | Kill switch + break-glass procedures in the [incident-response runbook](runbooks/incident-response.md); hash-chained `AuditTrail` with `verify()` for forensics (`fintwinos/core/audit.py`) |
| Documentation pack | Model cards, system cards, scenario assumptions, benchmark results, approval matrices | BoE/FCA, SR 11-7, NIST AI RMF | This documentation set: [architecture](architecture.md), [model cards](model-cards/simulators.md), [evaluation](evaluation.md), `assumptions_version` fields, exported canonical schemas (`fintwinos export-schemas`) |

## Framework-by-framework notes

**SR 11-7.** FinTwinOS treats simulators, LLM routes, prompts and tools as models
or model-adjacent artefacts: all are versioned, inventoriable and challengeable.
Effective challenge is structural (the critic swarm and mandatory human review),
not an annual afterthought. The hash-chained audit trail provides the "evidence of
use" dimension validation teams usually struggle to assemble.

**NIST AI RMF + Generative AI Profile.** The Govern/Map/Measure/Manage functions
map respectively to the policy plane (`fintwinos/policy/`), risk tiers and
provenance on every artefact, the evaluation stack (`fintwinos/evals/`), and the
staged deployment rule below. The GenAI Profile's confabulation and
information-integrity risks are addressed by schema-validated tool calls,
hallucinated-tool-rate tracking and provenance requirements.

**NIST IR 8356.** The twin-specific channels it highlights, ingestion,
synchronisation, simulation, write-back, are addressed across the
[threat model](threat-model.md): ingestion is T1, simulation is T6, and write-back
is the outbox-isolated execute band (boundary B4). Synchronisation integrity rests
on envelope provenance and `content_hash()` on every event (the T1 mitigations).

**DORA.** ICT incident handling, resilience testing and third-party concentration
are covered by the incident runbook, replay-based exercises, and the replaceable,
metered provider adapter. The audit trail's append-only JSONL export is designed to
feed the institution's incident-reporting timelines.

**EU AI Act.** For workflows an institution classifies as high-risk, FinTwinOS
supplies the human-oversight machinery (approval tokens, awaiting-human states,
kill switch), logging (audit chain), and technical documentation (this pack, model
cards, exported schemas). Classification of specific use cases remains the
deployer's obligation.

**FSB / BoE-FCA.** Both flag third-party concentration and partial understanding
of vendor models. FinTwinOS's answers: the control plane never lives with the model
provider, offline mode proves independence, routing is explicit configuration, and
the evaluation stack measures the deployed system rather than trusting vendor
claims.

## The deployment rule

> **No write-enabled production access until the system passes replay, shadow mode,
> human review and control testing at the workflow level.**

This rule is operationalised as the following checklist, applied **per workflow**
(not per deployment). The staged rollout mechanics are in the
[deployment runbook](runbooks/deployment.md).

### Stage 1, Replay

- [ ] Workflow runs end-to-end against recorded episodes via the replay engine
      (`runtime.replay`), fully offline (`FINTWIN_OFFLINE=1`).
- [ ] All eval suites for the workflow's layer pass their
      [release gates](evaluation.md#release-gates) (`fintwinos eval all`).
- [ ] Hard risk constraints (Expected Shortfall caps, recall floors) violated zero
      times across the replay corpus.
- [ ] Simulator calibration diagnostics within tolerance
      ([calibration runbook](runbooks/calibration.md)).

### Stage 2, Shadow

- [ ] `FINTWIN_ENVIRONMENT=shadow`, `FINTWIN_SHADOW_MODE=1`,
      `FINTWIN_EXECUTE_TOOLS_ENABLED=0`.
- [ ] Workflow runs on live data in parallel with the incumbent process; decisions
      logged, compared, never acted on.
- [ ] Shadow period covers at least one full business cycle for the workflow
      (month-end for treasury, full alert cycle for compliance, etc.).
- [ ] Divergence between shadow decisions and incumbent decisions reviewed and
      dispositioned by the workflow owner.

### Stage 3, Human review

- [ ] Named approvers and dual-control pairs assigned;
      `FINTWIN_DUAL_CONTROL_REQUIRED=1`.
- [ ] Approver runbook drills completed: granting, refusing, and revoking
      `ApprovalToken`s; recognising `awaiting_human` states.
- [ ] Effective-challenge review of prompts, tool specs and simulator assumptions
      signed off by a function independent of the builders (SR 11-7).
- [ ] Escalation path and kill-switch authority documented and rehearsed
      ([incident-response runbook](runbooks/incident-response.md)).

### Stage 4, Control testing

- [ ] Kill switch drill: flipping `FINTWIN_EXECUTE_TOOLS_ENABLED=0` (and the
      in-process deny-all rule) verifiably blocks execute calls; audited.
- [ ] Approval bypass attempts (missing token, expired token, wrong-subject token)
      verifiably refused and audited (`tool.approval_missing`).
- [ ] Audit chain integrity verified (`AuditTrail.verify()`) and export to
      write-once storage confirmed.
- [ ] Policy default-deny confirmed: an execute tool with no allow rule is blocked
      even with a valid token.
- [ ] Only after all four stages: enable `FINTWIN_EXECUTE_TOOLS_ENABLED=1` for the
      narrowest possible allow-rule set, with the outbox reviewed before any
      downstream relay.

## Evidence generation

Every checklist item above produces machine-readable evidence: eval reports
(`fintwinos eval all --report .fintwinos/eval-report`), audit chain extracts
(JSONL at `FINTWIN_AUDIT_PATH`), calibration reports
(`simulator.calibration_report()`), and the exported canonical schemas
(`fintwinos export-schemas`). Supervisory requests can be answered from artefacts,
not recollection.

---

FinTwinOS, created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/), MIT License.
