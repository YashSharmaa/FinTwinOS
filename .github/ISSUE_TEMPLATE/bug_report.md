---
name: Bug report
about: Something in FinTwinOS does not behave as documented
title: "[bug] "
labels: ["bug", "needs-triage"]
assignees: []
---

## Summary

A clear, one-paragraph description of the bug.

## Reproduction

Steps to reproduce, ideally as a runnable snippet. Please reproduce with
`FINTWIN_OFFLINE=1` first, offline runs are deterministic, which makes bugs
far easier to confirm and bisect.

```bash
FINTWIN_OFFLINE=1 fintwinos ...
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the full traceback / log output in a code
block. If a tool call was involved, include the relevant `AuditTrail` records
if you can, they pinpoint exactly which gate or handler misbehaved.

## Environment

- FinTwinOS version (`fintwinos version`):
- Python version:
- OS:
- Install method (pip / docker / compose / helm):
- Offline mode (`FINTWIN_OFFLINE`):

## Privileged data check

- [ ] I confirm this report contains **no real customer, account, trade or
      other privileged data**, only synthetic or public data.
