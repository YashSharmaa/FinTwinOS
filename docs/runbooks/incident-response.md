# Incident-response runbook

What to do when FinTwinOS misbehaves or is attacked: stopping the system (kill
switch), emergency authority (break-glass with countersign), recovering (rollback),
and proving what happened (forensics from the audit chain). Written to satisfy the
incident-handling expectations of DORA and NIST IR 8356, see the
[governance controls map](../governance-controls-map.md) for the mapping and the
[threat model](../threat-model.md) for the attack scenarios these procedures answer.

Rehearse this runbook. Stage 4 of the
[deployment rule](../governance-controls-map.md#the-deployment-rule) requires the
kill-switch drill before any write-enabled production access, and the
[deployment health checklist](deployment.md#deployment-health-checklist) keeps the
drills inside a rehearsal window.

## Severity classification

| Severity | Definition | Examples | First move |
|---|---|---|---|
| SEV-1 | Execute band implicated: an unapproved, mis-approved or erroneous action reached the outbox, or controls failed a live test | Approval bypass, policy misconfiguration allowing execute, audit chain verification failure | Kill switch, immediately |
| SEV-2 | Decision integrity implicated, no execution: poisoned twin state, prompt-injection reaching proposals, simulator producing wrong evidence | Suspect connector feed, calibration breach driving live decisions | Freeze affected workflows to observe/simulate only |
| SEV-3 | Degradation without integrity loss | LLM provider outage, latency, cost runaway | Fall back to offline mode; throttle |

When in doubt, classify up. A suspected SEV-2 with any execute-band exposure is a
SEV-1.

## The kill switch

The execute band has a hard off-switch beneath the policy layer:
`Settings.execute_tools_enabled`, set by `FINTWIN_EXECUTE_TOOLS_ENABLED`. When it
is `0`, `ToolRegistry._gate_execute` refuses every non-dry-run execute call before
any policy rule or approval token is even consulted, and audits the refusal as
`tool.execute_disabled` (`fintwinos/tools/registry.py`).

**Two-step engagement, do both:**

1. **In-process, immediate (no restart).** Settings are `lru_cache`d per process,
   so flipping the environment variable alone does not affect a running process.
   Prepend a deny-all rule to the live policy gate; the first matching deny wins
   over everything (`fintwinos/policy/gates.py`):

   ```python
   from fintwinos.core.types import ToolBand
   from fintwinos.policy.gates import PolicyRule, RuleEffect

   gate.add_rule(
       PolicyRule(
           name="incident-kill-switch",
           description="SEV incident <ticket>: execute band frozen by <name>",
           effect=RuleEffect.deny,
           bands=[ToolBand.execute],
       ),
       prepend=True,
   )
   ```

2. **Process-level, durable.** Set `FINTWIN_EXECUTE_TOOLS_ENABLED=0` in the
   environment and restart the service processes, so the hard gate holds across
   restarts and new processes.

**Verify engagement:** attempt a dry-run-off execute call with a valid token and
confirm it is refused; confirm the refusal appears in the audit trail
(`tool.execute_disabled` or `policy.checked` with `allowed=false`). Record the
verification in the incident ticket.

For SEV-2 freezes, use the same prepend technique with a deny rule scoped to the
affected workflows' tools (`tools=["propose_hedge_*"]`-style glob patterns) rather
than the whole band.

## Break-glass with countersign

Break-glass is the controlled *widening* of authority during an incident, e.g.
allowing an `execute_*` remediation tool that normal policy would route through a
slower approval chain. It is the inverse of the kill switch and is deliberately
harder to do than to undo.

Requirements (all four, no exceptions):

1. **Dual control.** `FINTWIN_DUAL_CONTROL_REQUIRED=1` stays on. The acting
   operator's `ApprovalToken` must be countersigned by a second, independent
   approver, two tokens for the same `subject`, different `granted_by`, both
   `approved`. One credential is never enough
   (`ApprovalToken`, `fintwinos/core/types.py`).
2. **Tight scope.** Tokens name the exact tool as `subject`, the `"*"` wildcard
   subject is prohibited in break-glass. The enabling policy rule is an `allow`
   scoped to the specific tool pattern, prepended, and named
   `breakglass-<ticket>-<tool>` so it is unmistakable in the audit trail.
3. **Time box.** Every break-glass token sets `expires_at` (15–60 minutes).
   Expiry is enforced at call time by `is_valid_for()`; there is no open-ended
   break-glass.
4. **Audit and review.** The grant, every use (`tool.called` / `tool.completed`
   with the caller and ticket id), and the removal of the rule are all in the
   chain. Break-glass usage is a tracked platform metric
   ([evaluation stack](../evaluation.md#the-evaluation-stack)) and every use gets
   a post-incident review.

**Stand-down:** remove the break-glass rule, let tokens expire (or flip their
status), re-verify the policy default-deny, and note stand-down time in the ticket.

## Rollback

Use when a release, policy pack, prompt set or simulator assumption change is
implicated.

1. Engage the kill switch (above) if the execute band is in any way implicated.
2. Redeploy the previous pinned version; restore the previous policy pack and
   `assumptions_version`. Any such change is a model change, so the affected
   workflows re-enter the [staged rollout](deployment.md#staged-rollout) at replay.
3. **Never roll back the audit trail.** The chain is append-only history; the
   rollback itself must appear in it. If the audit file was affected by the
   incident, start a new chain file and preserve the old one as evidence, do not
   edit it.
4. Twin state: prefer **reconstruction over restoration**, replay the recorded
   episodes (`runtime.replay`) from the last known-good point, excluding any
   envelopes identified as poisoned (see forensics below). This yields a state
   whose lineage is itself auditable.
5. Re-run `fintwinos eval all` and the affected workflow's suites before
   unfreezing; diff against the last green report.

## Forensics from the audit chain

The audit trail (`fintwinos/core/audit.py`) is the system of evidence: every tool
call, policy verdict, simulation branch, approval use and failure class is a
hash-chained record with actor, action, payload and timestamp.

**Step 1, Preserve.** Copy the JSONL file at `FINTWIN_AUDIT_PATH` to evidence
storage immediately; record its SHA-256. Continue operating on a new chain file if
the original is itself suspect.

**Step 2, Verify integrity.**

```python
from fintwinos.core.audit import AuditTrail

trail = AuditTrail.load("/var/fintwinos/audit/chain.jsonl")
assert len(trail) > 0
print("chain intact:", trail.verify())
```

`verify()` walks every record: any edited payload breaks that record's recomputed
hash; any deletion or reordering breaks the successor's `prev_hash` link. If it
returns `False`, bisect to the first broken link, everything before it is still
trustworthy evidence, and the break point itself localises the tampering window
(see [threat T5](../threat-model.md#t5-audit-tampering)). A verification failure
is automatically SEV-1.

**Step 3, Reconstruct the timeline.** Filter by actor and action:

```python
calls      = trail.records(action="tool.called")
verdicts   = trail.records(action="policy.checked")
refusals   = trail.records(action="tool.approval_missing")
schema_rej = trail.records(action="tool.schema_rejected")
llm_calls  = trail.records(action="llm.completed")
```

Useful patterns: `tool.unknown` spikes suggest a hallucinating or probing caller;
`tool.approval_missing` against execute tools shows attempted (and blocked)
actions; `policy.checked` records carry the matched rule names, so a
misconfigured allow rule is identifiable by name; every `ToolResult.audit_ref`
ties an agent-side result back to its exact `tool.called` record.

**Step 4, Correlate with the twin.** Use the replay engine to re-drive the
episode(s) covering the incident window and compare outcomes with and without
suspect envelopes (ingestion-poisoning triage, [threat T1](../threat-model.md#t1-ingestion-poisoning));
check `EventEnvelope.content_hash()` and `Provenance` to attribute suspect facts
to their source; pull the implicated `SimulationResult.seed` and
`assumptions_version` to reproduce any simulation evidence exactly.

**Step 5, Report.** DORA-style incident reporting is the institution's process,
but the artefacts come from here: the preserved chain, the verification result,
the timeline extract, replay comparisons and eval reports. Trace completeness is
a platform metric precisely so this step never starts from zero.

## Notification and post-incident

- Notify per the institution's regulatory obligations (DORA timelines for ICT
  incidents in the EU; equivalent regimes elsewhere). FinTwinOS's role is to make
  the evidence pack assemblable within the deadline.
- Post-incident review covers: which gate or invariant failed or held, whether a
  new `PolicyRule`, eval case or threat-model entry is needed, and whether drills
  need re-running. Every break-glass use and every SEV-1/SEV-2 gets one.
- If the incident revealed a vulnerability in FinTwinOS itself, report it
  privately per [SECURITY.md](../../SECURITY.md).

---

FinTwinOS, created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/), MIT License.
