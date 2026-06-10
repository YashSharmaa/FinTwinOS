# Threat model

This is a STRIDE-style threat model of the FinTwinOS twin itself — not of the bank's
estate around it. A digital twin of a financial institution concentrates an unusual
amount of decision-relevant state in one place, and it sits on the path between
models and the real world. NIST IR 8356 is explicit that digital twins demand their
own security and trust analysis across ingestion, synchronisation, simulation and
write-back channels; this document is that analysis for FinTwinOS.

Companion documents: the [architecture](architecture.md) describes the components
referenced here; the [incident-response runbook](runbooks/incident-response.md)
describes what to do when a threat materialises; the
[governance controls map](governance-controls-map.md) maps these mitigations to
regulatory expectations. Vulnerabilities in the mitigations themselves should be
reported privately per [SECURITY.md](../SECURITY.md).

## Assets and trust boundaries

**Assets** (in rough order of sensitivity):

1. The audit trail — the institution's evidence of what happened
   (`fintwinos/core/audit.py`).
2. Approval tokens — the keys that unlock the execute band
   (`ApprovalToken`, `fintwinos/core/types.py`).
3. Twin state — graph, time series, documents, replay episodes
   (`fintwinos/twin_core/`).
4. The tool registry and policy rules — the control surface
   (`fintwinos/tools/registry.py`, `fintwinos/policy/gates.py`).
5. The execute outbox — the staging area for real-world actions
   (`settings.data_dir / "outbox"`).
6. LLM traffic — prompts that may carry twin state to an external provider
   (`fintwinos/models/llm_routing/client.py`).
7. Simulator assumptions and calibration state (`fintwinos/twin_sim/`).

**Trust boundaries**:

- **B1: Sources → ingestion.** Connectors consume external systems of varying
  trustworthiness and emit `EventEnvelope`s.
- **B2: Documents → agents.** Text stored in the `DocumentStore` (filings, customer
  uploads, emails) reaches LLM context windows.
- **B3: Agents → tools.** Model-generated arguments cross into the registry.
- **B4: Registry → world.** The execute band is the only crossing to anything with
  side effects, and in this release it terminates at the local outbox.
- **B5: FinTwinOS → LLM provider.** Prompts leave the deployment perimeter unless
  offline mode is on.
- **B6: Humans → approvals.** Approval tokens are minted by people and consumed by
  the registry.

## STRIDE summary

| # | Threat | STRIDE class | Boundary | Primary mitigations (code) |
|---|---|---|---|---|
| T1 | Ingestion poisoning | Tampering, Spoofing | B1 | Envelope provenance + `content_hash()`, episode replay forensics, `IngestionError`, quarantine via policy |
| T2 | Prompt injection via documents | Elevation of privilege | B2, B3 | Band invariants, default-deny execute, schema validation, critic/red-team swarm, human approval gateway |
| T3 | Tool-schema abuse | Tampering, Elevation | B3 | `ToolSpec` registration invariants, Draft 2020-12 validation, duplicate-registration refusal |
| T4 | Approval-token theft or replay | Spoofing, Elevation | B6 | Subject scoping, expiry, status checks in `is_valid_for()`, dual control, audited grants |
| T5 | Audit tampering | Tampering, Repudiation | — (internal) | SHA-256 hash chain, `AuditTrail.verify()`, append-only JSONL, WORM export |
| T6 | Simulator gaming | Tampering (of evidence) | — (internal) | Mandatory confidence + calibration blocks, seeds, `assumptions_version`, KS/coverage drift checks, simulate-before-act ablation |
| T7 | Model-routing exfiltration | Information disclosure | B5 | `FINTWIN_OFFLINE=1`, key handling in `Settings`, replaceable provider adapter, cost/usage metering anomalies |
| T8 | Denial of service on the tool layer | Denial of service | B3, B4 | Request timeouts, bounded retries, exception capture in dispatch, idempotency cache |
| T9 | Repudiation of agent or human actions | Repudiation | all | `CallContext.caller` + ticket ids on every call, `llm.completed` audit events, decision records |

The numbered sections below give the attack narrative and the precise mitigation
mapping for each.

## T1 — Ingestion poisoning

**Attack.** An adversary with influence over an upstream source (a compromised
connector feed, a malicious counterparty record, a corrupted market-data file)
plants false state in the twin: a fake limit, an inflated balance, a tampered KYC
attribute. Downstream, simulations and agent analyses faithfully reason over the
poisoned state — "garbage in, governance-approved garbage out."

**Mitigations.**

- Every ingestion unit is an `EventEnvelope` with a `Provenance` block (source
  system, ingestion time, record hash, licence) and a deterministic
  `content_hash()`; consumers can detect in-flight mutation and attribute every
  fact in the twin to its source (`fintwinos/core/types.py`).
- Ingestion failures raise `IngestionError` (`fintwinos/core/errors.py`) — data is
  rejected loudly, never coerced.
- The replay engine records envelopes into episodes, so after discovery the
  poisoned window can be identified, the episode replayed without the poisoned
  envelopes, and the divergence quantified (forensics procedure in the
  [incident-response runbook](runbooks/incident-response.md)).
- Policy rules can quarantine by source: a `PolicyRule` with `conditions` on the
  source argument can force human review of any tool result derived from a suspect
  feed (`fintwinos/policy/gates.py`).
- Tools with `provenance_required=True` cannot return unattributed data — the
  registry attaches or demands provenance on every result.

## T2 — Prompt injection via documents

**Attack.** A document in the twin — a filing, a customer complaint, an email
attachment — contains adversarial instructions ("ignore your instructions and
transfer..."). An agent retrieves it through `observe_*` document tools, the text
enters the LLM context, and the model emits tool calls serving the attacker.

**Mitigations.** FinTwinOS assumes prompt injection **will** succeed at the model
layer and is designed so a fully compromised model still cannot act:

- Agents can only act through the typed registry; there is no raw store access in
  production paths (`fintwinos/agents/base.py`). A hijacked model can only emit
  tool calls.
- Bands are structural: `observe_*`/`simulate_*`/`propose_*` tools are side-effect
  free by registration invariant, so injected instructions can at worst read more
  twin state or generate misleading proposals (`fintwinos/tools/registry.py`).
- The execute band is quadruple-locked: `FINTWIN_EXECUTE_TOOLS_ENABLED=1` must be
  set deployment-wide, a policy `allow` rule must match (execute is default-deny in
  `PolicyGate.check_tool_call`), a valid human `ApprovalToken` must be attached,
  and the handler still only writes to the local outbox.
- Arguments are schema-validated before any gating, so injection cannot smuggle
  malformed payloads through type confusion.
- The critic/red-team swarm reviews drafts precisely to catch instruction-following
  anomalies before a proposal reaches a human, and `PolicyGate.check_decision`
  forces human review of every executing `Decision`.
- Misleading *proposals* remain the residual risk; that is why proposals carry
  rationales and provenance, and why the human approval gateway is mandatory rather
  than advisory.

## T3 — Tool-schema abuse

**Attack.** A malicious or careless module registers a tool that lies about itself:
an "observe" tool with side effects, an execute tool that claims to need no
approval, a tool whose schema accepts arbitrary payloads, or a tool that shadows an
existing name to intercept calls.

**Mitigations** — the registry invariants reject these at registration time, before
any call can occur (`ToolSpec._enforce_band_invariants`,
`fintwinos/tools/registry.py`):

- Band/prefix binding: a tool's name must start with its band prefix; a
  `simulate_*` name cannot carry execute semantics.
- observe/simulate/propose specs with a side-effect class other than `none`/`read`
  are rejected with a `ValueError`.
- Execute specs must declare a `reversible` or `irreversible` side effect **and**
  `requires_human_approval=True`; anything else is rejected.
- Input schemas are themselves validated (`Draft202012Validator.check_schema`), so
  unparseable or trivially permissive schemas fail registration.
- Duplicate names are refused (`ToolRegistry.register` raises on re-registration),
  preventing interception by shadowing.
- At call time, unknown tools are audited (`tool.unknown`) and rejected; argument
  validation failures are audited (`tool.schema_rejected`). Hallucinated-tool and
  argument-correctness rates are tracked continuously by the
  [evaluation stack](evaluation.md).

## T4 — Approval-token theft or replay

**Attack.** An attacker obtains a granted `ApprovalToken` (from a log, a serialized
blackboard, a compromised approver session) and attaches it to a different, more
damaging execute call — or replays an old token long after its context has expired.

**Mitigations.**

- Tokens are scoped: `is_valid_for(subject)` requires the token's `subject` to
  match the exact tool name being called (the `"*"` wildcard exists for
  administrative use and should be prohibited by deployment policy). A token
  granted for `execute_` tool A is useless against tool B
  (`fintwinos/core/types.py`).
- Tokens expire: `expires_at` is checked at call time; the
  [deployment runbook](runbooks/deployment.md) mandates short expiries.
- Tokens carry status: anything other than `approved` (pending, rejected,
  escalated) fails validation, so revocation is a status flip.
- Dual control: with `FINTWIN_DUAL_CONTROL_REQUIRED=1` (the default), sensitive
  grants need maker-checker countersigning — one stolen credential is not enough.
- Every grant and every use is audited: the registry logs `tool.approval_missing`
  on failures and records the approving context on successes, so token misuse is
  reconstructable from the chain.
- Residual risk: a token stolen *and* used within its scope and lifetime against
  its own subject. This is bounded by expiry, dual control, and the outbox design —
  the blast radius of this release's execute band is a local directory, reviewed
  before any downstream relay.

## T5 — Audit tampering

**Attack.** An insider with file access edits the audit log to hide an action —
deleting a record, altering a payload, reordering events — defeating repudiation
controls and regulatory evidence.

**Mitigations — and how the hash chain detects tampering.**

- Each `AuditRecord`'s hash is SHA-256 over its canonical JSON body — sequence
  number, timestamp, actor, action, payload — **plus the previous record's hash**
  (`AuditRecord.body_for_hash`, `fintwinos/core/audit.py`). The first record chains
  to a fixed all-zeros genesis hash.
- Therefore: editing any record's content changes its recomputed hash and breaks
  the match with its stored hash; deleting or reordering records breaks the
  `prev_hash` linkage of the successor. `AuditTrail.verify()` walks the entire
  chain and returns `False` on the first broken link. Tampering anywhere is
  detectable everywhere downstream of it.
- Persistence is append-only JSONL (`FINTWIN_AUDIT_PATH`); the
  [deployment runbook](runbooks/deployment.md) requires shipping it to WORM or
  otherwise write-once storage with periodic anchor hashes recorded out-of-band, so
  an attacker would need to forge the entire suffix of the chain *and* every
  external anchor.
- `verify()` runs in scheduled integrity checks and as the first step of every
  forensic investigation ([incident-response runbook](runbooks/incident-response.md)).
- Residual risk: truncation of the entire tail after the last external anchor.
  Anchor frequency is the control dial.

## T6 — Simulator gaming

**Attack.** Decisions are justified by simulation evidence, so the simulator
becomes a target: an attacker (or an over-eager optimisation process) tunes
scenarios, seeds or assumptions until the simulation says yes — the sim-to-real gap
weaponised. This includes RL policies that exploit simulator idiosyncrasies rather
than learning real value.

**Mitigations.**

- The `SimulationResult` contract forbids naked point estimates: every result
  carries confidence intervals per headline metric and a `calibration` block, and
  every simulator must expose `calibration_report()`
  (`fintwinos/core/interfaces.py`, `fintwinos/core/types.py`).
- Results record their `seed` and `assumptions_version`; cherry-picking seeds or
  silently editing assumptions is visible in the audit trail, where simulation
  branches are logged like every other consequential event.
- Continuous calibration against historical replay (KS distance, interval
  coverage — [calibration runbook](runbooks/calibration.md)) detects simulators
  drifting from reality; out-of-tolerance simulators raise `CalibrationError` and
  are pulled from decision support.
- The staged RL ladder is itself a mitigation: offline RL results must survive
  off-policy evaluation, then **shadow mode against reality**, before any online
  use — a policy that gamed the simulator fails shadow comparison
  (`fintwinos/rl/`, [evaluation gates](evaluation.md#release-gates)).
- The simulate-before-act ablation experiment quantifies how much trust simulation
  deserves, rather than assuming it.

## T7 — Model-routing exfiltration

**Attack.** Twin state — positions, customer data, case narratives — flows into
prompts and out to an external LLM provider; a misconfigured route, an over-broad
prompt builder, or a compromised routing layer exfiltrates sensitive state. A
second variant: routing silently swaps in an unapproved model, breaking the model
inventory.

**Mitigations.**

- **The off switch is total:** `FINTWIN_OFFLINE=1` removes the provider from the
  system — `LLMClient` returns deterministic local stubs, agents use rule-based
  fallbacks, and nothing leaves the perimeter. Air-gapped deployment is a
  first-class, fully tested mode, not a degraded one
  (`fintwinos/models/llm_routing/client.py`).
- Routing is explicit configuration, not model self-selection: the three tiers come
  from `Settings` (`FINTWIN_LLM_MODEL_PRIMARY/FAST/CHEAP`), so the deployed model
  set is inventoriable and reviewable (SR 11-7's inventory expectation —
  see the [governance controls map](governance-controls-map.md)).
- The provider boundary is a single adapter; the control plane is separated from
  the model provider and the adapter is replaceable, addressing the third-party
  concentration concern raised by the BoE/FCA survey and the FSB.
- Every LLM call is metered (tokens, indicative cost) and audited
  (`llm.completed` events with model name and usage); volume or model-name
  anomalies are observable signals.
- API keys live in settings/environment, are redacted in `fintwinos info`
  (shown only as set/missing), and are never written to the audit payload.
- Residual risk: legitimately routed prompts still contain twin state. Deployments
  handling regulated data should run offline or against an in-perimeter endpoint,
  and minimise prompt content per the data-governance controls in the
  [deployment runbook](runbooks/deployment.md).

## T8 — Denial of service on the tool layer

**Attack.** A runaway agent loop, a malicious client of the MCP edge, or a
pathological tool argument exhausts the registry, the simulators or the LLM budget.

**Mitigations.** LLM calls carry timeouts (`FINTWIN_REQUEST_TIMEOUT`) and bounded
retries with backoff (`FINTWIN_MAX_RETRIES`); tool handler exceptions are captured
into failed `ToolResult`s rather than crashing the orchestrator; the idempotency
cache absorbs keyed repeats; cost metering exposes budget burn in real time; and
the MCP server validates JSON-RPC envelopes before any dispatch
(`fintwinos/tools/envelope.py`).

## T9 — Repudiation

**Attack.** An operator or an agent denies having taken an action; or actions
cannot be attributed among concurrent agents.

**Mitigations.** Every registry call carries a `CallContext` with `caller` and
`ticket_id`, audited on dispatch; every `Decision` records its `owner` and
rationale; every LLM completion is audited per agent; blackboard postings record
their actor and timestamp. Combined with the tamper-evident chain (T5), the system
produces non-repudiable, ordered evidence of who did what, when, and on whose
authority.

## Cross-cutting mitigations

| Mitigation | What it is | Code anchor |
|---|---|---|
| Registry invariants | Structural band/side-effect/approval rules enforced at registration | `ToolSpec` validators, `fintwinos/tools/registry.py` |
| Kill switch | `FINTWIN_EXECUTE_TOOLS_ENABLED=0` disables the execute band beneath the policy layer; a prepended deny-all `PolicyRule` covers running processes | `Settings.execute_tools_enabled`; `PolicyGate.add_rule(prepend=True)` |
| Break-glass with countersign | Emergency widening of permissions requires a second approver's token under dual control, is time-boxed by token expiry, and is fully audited | `FINTWIN_DUAL_CONTROL_REQUIRED`, `ApprovalToken.expires_at`; procedure in the [incident-response runbook](runbooks/incident-response.md) |
| Role-based access | Callers are identified per call; approval tokens carry roles; policy rules condition on band, tool pattern, risk tier and arguments | `CallContext.caller`, `ApprovalToken.role`, `PolicyRule` |
| Default-deny execute | No execute call proceeds without an explicit allow rule, independent of approvals | `PolicyGate.check_tool_call`, `fintwinos/policy/gates.py` |
| Tamper-evident audit | Hash-chained records, chain verification, append-only persistence | `AuditTrail`, `fintwinos/core/audit.py` |
| Offline mode | Whole-system operation with zero external calls | `FINTWIN_OFFLINE`, `fintwinos/core/config.py` |
| Outbox isolation | Execute handlers write to `settings.data_dir / "outbox"`, never to live systems | [CONTRACTS.md](../CONTRACTS.md) hard rule 1 |

## What this model does not cover

Host and network security, identity-provider compromise, supply-chain attacks on
dependencies (run `pip-audit`, pinned in the `dev` extra), and the institution's
surrounding estate are out of scope here and belong to the adopting organisation's
broader threat model — as NIST IR 8356 recommends, the twin's model should be
embedded in, not substituted for, the enterprise one.

---

FinTwinOS — created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) — MIT License.
