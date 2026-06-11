---
name: Feature request
about: Propose a new capability, tool, simulator, connector or agent
title: "[feature] "
labels: ["enhancement", "needs-triage"]
assignees: []
---

## Problem

What problem does this solve? Who hits it (risk, treasury, compliance,
operations, engineering) and how often?

## Proposed solution

Describe the capability. If it adds tools, say which band(s) they belong to,
`observe_*`, `simulate_*`, `propose_*` or `execute_*`, and the risk tier you
would assign. Remember the hard rules:

- observe / simulate / propose tools must be side-effect free;
- execute tools always require a human approval token, an explicit policy
  allow-rule and `FINTWIN_EXECUTE_TOOLS_ENABLED=1`;
- everything must work fully offline with deterministic fallbacks.

## Alternatives considered

Other approaches you weighed and why you prefer this one.

## Additional context

Links, sketches, references to CONTRACTS.md sections, related issues.
