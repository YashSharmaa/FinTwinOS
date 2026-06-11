# Deployment runbook

How to deploy FinTwinOS in a tenant-isolated, hybrid cloud/on-prem posture, and how
to walk a workflow through the staged rollout demanded by the
[deployment rule](../governance-controls-map.md#the-deployment-rule): replay →
shadow → human review → control testing.

Audience: platform operators and the risk/compliance partners who sign the stage
gates. Companion runbooks: [incident response](incident-response.md) and
[calibration](calibration.md).

## Deployment model

FinTwinOS is **tenant-isolated hybrid**: one institution per deployment, runnable
entirely on-premises, in a private cloud, or split — with the non-negotiable
invariant that **the control plane (policy gate, approval flow, audit trail) never
leaves the institution's perimeter**, whatever happens with model providers.

Three reference shapes:

1. **Air-gapped / on-prem.** `FINTWIN_OFFLINE=1`. No OpenAI key, no egress. Agents
   run rule-based fallbacks; every demo, eval and simulator works identically.
   This is the right starting shape for any regulated data, and it is a fully
   tested first-class mode, not a degraded one.
2. **Hybrid.** Twin substrate, audit and policy on-prem; LLM calls to OpenAI (or
   any in-perimeter OpenAI-compatible endpoint approved by model risk) through the
   single adapter in `fintwinos/models/llm_routing/client.py`. Prompt content is
   the data-governance review surface (see
   [threat T7](../threat-model.md#t7--model-routing-exfiltration)).
3. **Private cloud.** Everything in the institution's cloud tenancy; identical
   configuration surface.

## Installation

```bash
# Python >= 3.11
pip install -e ".[dev,server]"

# Verify the install and effective configuration (secrets shown as set/missing):
fintwinos version
fintwinos info

# Smoke-test fully offline:
FINTWIN_OFFLINE=1 fintwinos demo liquidity
FINTWIN_OFFLINE=1 fintwinos eval all
```

The `[server]` extra (FastAPI + uvicorn) is required only for the MCP-style tool
server (`fintwinos serve-tools`, default `127.0.0.1:8341`). Keep the server bound
to localhost or behind the institution's own authenticating proxy; the registry's
gating applies on every transport, but network exposure is still the adopter's
perimeter decision.

## Environment variables

All settings are read by `fintwinos/core/config.py` from the environment (prefix
`FINTWIN_`) or a local `.env` file. `OPENAI_API_KEY` is honoured as a fallback for
`FINTWIN_OPENAI_API_KEY`.

| Variable | Default | Purpose |
|---|---|---|
| `FINTWIN_OFFLINE` | `0` | `1` forces offline mode: deterministic LLM stubs, no network, rule-based agent fallbacks |
| `FINTWIN_SEED` | `7` | Global deterministic seed; all randomness flows through `numpy.random.default_rng(seed)` |
| `FINTWIN_DATA_DIR` | `.fintwinos` | Working data directory; the execute outbox lives at `<data_dir>/outbox` |
| `FINTWIN_AUDIT_PATH` | unset | JSONL persistence path for the audit chain; set it in every non-local environment |
| `FINTWIN_ENVIRONMENT` | `local` | `local` \| `shadow` \| `production` — drives the rollout stages below |
| `FINTWIN_EXECUTE_TOOLS_ENABLED` | `0` | Hard off-switch for the execute band; leave `0` until Stage 4 sign-off |
| `FINTWIN_DUAL_CONTROL_REQUIRED` | `1` | Maker-checker countersigning for sensitive approval grants |
| `FINTWIN_SHADOW_MODE` | `1` | Learned/agent decisions are logged and compared, never acted on |
| `FINTWIN_OPENAI_API_KEY` | unset | OpenAI API key (or set `OPENAI_API_KEY`); irrelevant when offline |
| `FINTWIN_LLM_PROVIDER` | `openai` | LLM provider identifier |
| `FINTWIN_LLM_MODEL_PRIMARY` | `gpt-5` | Planning, analysis, critique tier |
| `FINTWIN_LLM_MODEL_FAST` | `gpt-5-mini` | Drafting, extraction tier |
| `FINTWIN_LLM_MODEL_CHEAP` | `gpt-5-nano` | Classification, routing, cheap calls |
| `FINTWIN_LLM_TEMPERATURE` | `0.2` | Sampling temperature for LLM calls |
| `FINTWIN_LLM_MAX_OUTPUT_TOKENS` | `4096` | Output token cap per call |
| `FINTWIN_REQUEST_TIMEOUT` | `60.0` | Per-request timeout (seconds) for LLM calls |
| `FINTWIN_MAX_RETRIES` | `3` | Bounded retries with exponential backoff for transient LLM failures |

Two operational notes:

- **Settings are cached per process** (`get_settings()` is `lru_cache`d). Changing
  an environment variable requires a process restart to take effect — plan
  restarts into any flag change, and see the
  [incident-response runbook](incident-response.md) for the in-process kill path
  that does not wait for one.
- Model tiers must match the institution's **approved model inventory**
  (SR 11-7 — see the [controls map](../governance-controls-map.md)); changing a
  tier is a model change and re-enters the rollout at Stage 1 for affected
  workflows.

## Data directory layout

Under `FINTWIN_DATA_DIR` (default `.fintwinos`):

- `outbox/` — the **only** place execute-band handlers write. Nothing in this
  release writes to a live external system; relaying outbox artefacts downstream
  is a deliberate, human-owned step outside FinTwinOS.
- `eval-report*` — evaluation reports from `fintwinos eval` (configurable via
  `--report`).
- The audit JSONL lives wherever `FINTWIN_AUDIT_PATH` points; keep it on storage
  with write-once or versioned retention, and ship periodic anchor hashes
  out-of-band (see [threat T5](../threat-model.md#t5--audit-tampering)).

## Staged rollout

The stages implement the deployment rule per workflow. The full sign-off checklist
lives in the [governance controls map](../governance-controls-map.md#the-deployment-rule);
this section gives the operator's view.

### Stage 0 — Local

```bash
FINTWIN_OFFLINE=1 FINTWIN_ENVIRONMENT=local fintwinos demo day_in_the_life
FINTWIN_OFFLINE=1 fintwinos eval all
```

Everything deterministic, no key, no network. Use this shape for development,
CI and supervisor walkthroughs.

### Stage 1 — Replay

Same configuration as local, but against recorded production episodes loaded into
the replay engine. Run the workflow's eval suites; archive the reports. Gate:
release gates pass and hard constraints show zero violations across the corpus.

### Stage 2 — Shadow

```bash
FINTWIN_ENVIRONMENT=shadow
FINTWIN_SHADOW_MODE=1
FINTWIN_EXECUTE_TOOLS_ENABLED=0
FINTWIN_AUDIT_PATH=/var/fintwinos/audit/chain.jsonl   # institution-appropriate path
```

The workflow runs on live data in parallel with the incumbent process. Decisions
are produced, audited and compared — never acted on. Hold the stage for at least
one full business cycle of the workflow; review every divergence with the
workflow owner.

### Stage 3 — Human review

No configuration change: this stage is about people. Approvers named,
dual-control pairs assigned, drills run (granting, refusing, revoking
`ApprovalToken`s; handling `awaiting_human` case states), effective-challenge
review signed off, kill-switch authority documented.

### Stage 4 — Control testing, then narrow production

Run the control tests (kill-switch drill, approval-bypass attempts, audit
verification, default-deny confirmation) from the checklist. Only after sign-off:

```bash
FINTWIN_ENVIRONMENT=production
FINTWIN_EXECUTE_TOOLS_ENABLED=1     # plus the narrowest possible policy allow rules
FINTWIN_DUAL_CONTROL_REQUIRED=1
```

Add explicit `allow` rules **only** for the specific `execute_*` tools the
workflow needs — the execute band remains default-deny for everything else
(`fintwinos/policy/gates.py`) — and keep token expiries short. Review the outbox
before any downstream relay.

## Upgrades and rollback

- Pin the installed version; upgrade in `local` → `shadow` order, re-running
  `fintwinos eval all` at each step and diffing reports against the previous
  release.
- Any change to prompts, tool specs, policy packs, simulator assumptions
  (`assumptions_version`) or model tiers is a **model change**: affected workflows
  re-enter at Stage 1 (replay).
- Rollback procedure (including state and audit considerations) is in the
  [incident-response runbook](incident-response.md#rollback).

## Deployment health checklist

- [ ] `fintwinos info` shows the intended environment, flags and model tiers.
- [ ] `FINTWIN_AUDIT_PATH` set; audit chain verifying clean on schedule.
- [ ] `fintwinos eval all` green on the deployed commit; report archived.
- [ ] Simulator calibration within tolerance ([calibration runbook](calibration.md)).
- [ ] Kill-switch and break-glass drills within their rehearsal window
      ([incident response](incident-response.md)).
- [ ] Outbox review process operating; no unreviewed artefacts older than the
      agreed SLA.

---

FinTwinOS — created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) — MIT License.
