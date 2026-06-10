# Model card: LLM routing layer

| | |
|---|---|
| **Artefact** | Task-class-to-model routing over OpenAI chat models, with cost metering and a deterministic offline stub |
| **Modules** | `fintwinos/models/llm_routing/router.py` (`ModelRouter`), `fintwinos/models/llm_routing/client.py` (`LLMClient`) |
| **Type** | Routing and access layer — not a trained model itself; this card covers the layer *and* the governance of the routed third-party models |
| **Version** | Tracks the FinTwinOS release (`fintwinos version`) |
| **Owner** | Platform (model tiers owned by the adopting institution's model-risk function) |

## Intended use

- Route every agent LLM call to the cheapest model tier adequate for its task
  class: `planning` / `analysis` / `critique` → the primary tier (default
  `gpt-5`); `drafting` / `extraction` → the fast tier (default `gpt-5-mini`);
  `classification` / `routing` / `cheap` → the cheap tier (default `gpt-5-nano`).
- Provide a single, replaceable provider boundary: all OpenAI access in FinTwinOS
  flows through `LLMClient.complete()` — retries with exponential backoff, strict
  JSON-schema output, OpenAI function-calling tool definitions, and per-call
  token/cost metering.
- Degrade to a **deterministic offline stub** whenever `FINTWIN_OFFLINE=1` or no
  API key is present: stubs are reproducible (a hash of the prompt), flagged with
  `offline=True`, and agents are contractually required to fall back to
  rule-based logic. The LLM enriches behaviour; it is never a hard dependency.

**Out of scope / misuse:** the routing layer must not be used to bypass the tool
registry (models receive tool *definitions*; calls still dispatch through
`ToolRegistry.call` with full gating); routed models must not make execute-band
decisions autonomously — every executing `Decision` requires human review by
`PolicyGate.check_decision` regardless of which model proposed it.

## Configuration and data

- Tiers are pure configuration: `FINTWIN_LLM_MODEL_PRIMARY`,
  `FINTWIN_LLM_MODEL_FAST`, `FINTWIN_LLM_MODEL_CHEAP`, plus
  `FINTWIN_LLM_TEMPERATURE`, `FINTWIN_LLM_MAX_OUTPUT_TOKENS`,
  `FINTWIN_REQUEST_TIMEOUT`, `FINTWIN_MAX_RETRIES`
  ([full table](../runbooks/deployment.md#environment-variables)). Remapping a
  tier is a model change and re-enters the staged rollout.
- **Training data: none.** FinTwinOS does not train, fine-tune or store gradients
  for the routed models. The data consideration is *prompt content*: prompts may
  carry twin state to the provider, which is the exfiltration surface analysed in
  [threat T7](../threat-model.md#t7--model-routing-exfiltration) and the reason
  offline/in-perimeter deployment is first-class.
- The price table in `router.py` (`DEFAULT_PRICE_TABLE`) is **indicative, for
  budgeting and observability only — never for billing**; it is overridable via
  `ModelRouter(price_table=...)`.

## Metrics

- Per-client lifetime metering: `LLMClient.usage_summary()` reports calls, input
  and output tokens, estimated cost (USD) and offline status.
- Per-call audit: every agent completion appends an `llm.completed` audit record
  with model name, offline flag, usage and cost — making the deployed model mix
  continuously inventoriable (SR 11-7) and anomalies observable.
- Quality of routed models is **not** taken from vendor claims: it is measured by
  the [evaluation stack](../evaluation.md) (function-call exactness, agent-run
  consistency, domain suites) on the deployed configuration.

## Limitations

- Routing is static per task class; it does not adapt to per-prompt difficulty.
  A misdeclared `task_class` on an agent sends work to the wrong tier.
- Cost figures are estimates from a static table; reconcile against provider
  billing before using them in any financial control.
- The offline stub is deterministic but **not** a capability substitute: offline
  runs measure the rule-based floor of the system, not LLM-enriched performance.
  Compare like with like in eval reports (the `offline` flag is recorded).
- Retries handle transience, not outages; sustained provider failure is a SEV-3
  ([incident runbook](../runbooks/incident-response.md#severity-classification))
  with offline mode as the documented fallback.
- Third-party concentration is mitigated, not eliminated: one provider adapter is
  the current surface (BoE/FCA and FSB concern — see the
  [controls map](../governance-controls-map.md)).

## Governance hooks

- Inventory: tiers visible via `fintwinos info`; every call audited with its
  model name.
- Kill paths: `FINTWIN_OFFLINE=1` severs the provider entirely; per-call timeout
  and bounded retries cap blast radius.
- Effective challenge: prompts and routing configuration are reviewable artefacts;
  changes re-enter the [deployment rule](../governance-controls-map.md#the-deployment-rule)
  at replay.

---

FinTwinOS — created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) — MIT License.
