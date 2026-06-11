# Security policy

FinTwinOS is governance software for financial institutions: a vulnerability in its
gating, audit or approval logic is a vulnerability in someone's control
environment. Please report security issues privately and give us the chance to fix
them before disclosure.

## Reporting a vulnerability

- **Email:** yash.sharma.contact@gmail.com with the subject line
  `[FINTWINOS SECURITY] <short description>`.
- **Do not** open a public GitHub issue, discussion or PR for anything you believe
  is a vulnerability.
- Please include: affected version/commit, a minimal reproduction, the impact as
  you understand it, and whether the issue is already known or exploited anywhere.
  Encrypted reports are welcome, say so in a first plain email and we will
  arrange a key exchange.

You will receive an acknowledgement within **72 hours** and a substantive
assessment within **14 days**. We practise coordinated disclosure: we ask for up to
**90 days** from acknowledgement to ship a fix before public disclosure, and we
will credit reporters in the release notes unless you prefer otherwise. There is no
bug bounty programme.

## Scope, what counts as a vulnerability here

Anything that defeats a documented control is in scope, in particular:

- **Execute-band bypass:** invoking an `execute_*` tool without all three of
  `FINTWIN_EXECUTE_TOOLS_ENABLED=1`, a matching policy `allow` rule, and a valid
  `ApprovalToken` (`fintwinos/tools/registry.py`).
- **Band invariant escapes:** registering or invoking observe/simulate/propose
  tools with real side effects, or evading the band-prefix and side-effect-class
  checks in `ToolSpec`.
- **Approval forgery:** crafting, replaying or widening `ApprovalToken`s beyond
  their subject, expiry or status (`fintwinos/core/types.py`).
- **Audit-chain defeat:** modifying, deleting or reordering `AuditTrail` records
  in a way `verify()` does not detect (`fintwinos/core/audit.py`).
- **Policy-gate confusion:** rule-matching or ordering behaviour that allows what
  the rules deny, including execute default-deny bypasses
  (`fintwinos/policy/gates.py`).
- **Outbox escape:** any path by which the execute band writes outside
  `settings.data_dir / "outbox"`.
- **Offline-mode leaks:** any network egress with `FINTWIN_OFFLINE=1` set.
- **Secret exposure:** API keys appearing in logs, audit payloads or tool results.
- Injection paths through the MCP-style server (`fintwinos serve-tools`) that
  reach gating-relevant state.

Out of scope: vulnerabilities in dependencies (report upstream, but tell us if
FinTwinOS's usage amplifies them), prompt-injection *content* that is correctly
contained by the gates (that containment working is the design, see the
[threat model](docs/threat-model.md)), issues requiring an already-compromised
host, and findings in forks or unsupported versions.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (latest release) | Yes, security fixes |
| `main` (unreleased) | Yes, fixes land here first |
| Older releases | No, please upgrade |

## Hardening guidance for deployers

The deployment-relevant security posture is documented, not folkloric:

- Follow the [deployment runbook](docs/runbooks/deployment.md), especially the
  staged rollout and the environment-variable table. Leave
  `FINTWIN_EXECUTE_TOOLS_ENABLED=0` and `FINTWIN_DUAL_CONTROL_REQUIRED=1` at their
  defaults until Stage 4 sign-off.
- Run `FINTWIN_OFFLINE=1` for any regulated data unless an in-perimeter endpoint
  has been approved by your model-risk function.
- Persist the audit chain (`FINTWIN_AUDIT_PATH`) to write-once storage and anchor
  it out-of-band; verify on schedule
  ([incident-response runbook](docs/runbooks/incident-response.md)).
- Keep `fintwinos serve-tools` on localhost or behind your own authenticating
  proxy.
- Audit your dependency tree: `pip-audit` ships in the `dev` extra.

## Our commitments

- Security-relevant changes to gating, audit or approval logic are called out
  explicitly in release notes.
- The [threat model](docs/threat-model.md) is maintained as code changes; if a
  report invalidates a documented mitigation, fixing the document is part of
  fixing the bug.

---

FinTwinOS, created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/), MIT License.
