# Pull request

## What and why

Summarise the change and the motivation. Link related issues.

## Checklist

- [ ] **Tests**, added/updated pytest coverage and the full suite passes
      offline: `FINTWIN_OFFLINE=1 pytest`.
- [ ] **Lint**, `ruff check .` is clean.
- [ ] **CONTRACTS.md respected**, I only touched my module's owned paths,
      coded against `fintwinos/core/` (never another module's internals), kept
      tool-band invariants (observe/simulate/propose side-effect free; execute
      behind approval + policy + `FINTWIN_EXECUTE_TOOLS_ENABLED`), and routed
      consequential actions through the shared `AuditTrail`.
- [ ] **Offline-first**, every new code path works with `FINTWIN_OFFLINE=1`
      via deterministic rule-based fallbacks; no test hits the network.
- [ ] **Determinism**, all randomness goes through
      `numpy.random.default_rng(seed)`.
- [ ] **No privileged data**, this PR contains no real customer, account,
      trade or other confidential data; fixtures are synthetic or public.
- [ ] **No new dependencies**, or they were explicitly agreed with the
      maintainer first.

## Notes for reviewers

Anything that needs special attention: migration steps, behaviour changes,
follow-ups.
