# FinTwinOS module contracts

Every module codes against `fintwinos/core/`, never against another module's internals.
Read `fintwinos/core/types.py`, `fintwinos/core/interfaces.py`, `fintwinos/tools/registry.py`,
`fintwinos/policy/gates.py`, `fintwinos/models/llm_routing/` and `fintwinos/agents/base.py`
before writing any module code.

## Canonical entry points

| Entry point | Provided by | Consumed by |
|---|---|---|
| `fintwinos.twin_core.runtime.build_runtime(seed=7, with_demo_data=True) -> TwinRuntime` | twin_core | tools, agents, rl, evals, demos |
| `fintwinos.twin_sim.register_all(runtime: TwinRuntime) -> None` | twin_sim | tools, rl, demos |
| `fintwinos.tools.catalog.build_default_registry(runtime: TwinRuntime, policy_gate=None, settings=None) -> ToolRegistry` | tools | agents, evals, demos, server |
| `fintwinos.agents.runtime.handle_case(case_id, objective, runtime, registry, llm) -> dict` | agents | demos, evals |
| `fintwinos.demos.<name>.main()` | demos | CLI (`fintwinos demo <name>`) |
| `fintwinos.evals.runner.run_suites(suite, report_prefix) -> dict` | evals | CLI (`fintwinos eval`) |

`handle_case` returns `{"status": "complete" | "awaiting_human" | "blocked", "decision": ..., "policy": ...}`.

## Hard rules

1. **Tool bands.** Tools are named `observe_*`, `simulate_*`, `propose_*`, `execute_*`.
   The registry enforces: observe/simulate/propose are side-effect free; execute always
   requires a human `ApprovalToken`, an explicit policy `allow` rule, AND
   `FINTWIN_EXECUTE_TOOLS_ENABLED=1`. Execute handlers must write to a local outbox
   (`settings.data_dir / "outbox"`), never to a real external system.
2. **Offline-first.** Everything must work with `FINTWIN_OFFLINE=1` (no OpenAI key, no
   network). Agents must implement deterministic rule-based fallbacks when
   `ctx.llm.offline` is true. Tests must never hit the network; network-dependent
   connector tests are marked `@pytest.mark.network` and skipped by default.
3. **Determinism.** All randomness goes through `numpy.random.default_rng(seed)`.
   Simulators accept `seed` and return reproducible `SimulationResult`s with
   `confidence` intervals and a `calibration` block, never bare point estimates.
4. **Audit.** Anything consequential appends to the shared `AuditTrail`
   (`runtime.audit` / `registry.audit`). Never bypass the registry to mutate state in
   production code paths.
5. **Envelopes in, envelopes out.** Connectors emit `EventEnvelope`; the twin ingests
   envelopes via `runtime.ingestor`; the replay engine records them into episodes.
6. **Ownership.** Each module owns only its own directories and `tests/<module>/`.
   Shared foundation files (`fintwinos/core/*`, `fintwinos/tools/{__init__,registry,envelope}.py`,
   `fintwinos/policy/{__init__,gates}.py`, `fintwinos/models/llm_routing/{client,router}.py`,
   `fintwinos/agents/{__init__,base}.py`, `fintwinos/cli.py`, `pyproject.toml`, `README.md`)
   are owned by the orchestrator, extend them via new modules, do not edit them.
7. **Dependencies.** Use only the deps already in `pyproject.toml` (numpy, pandas,
   networkx, httpx, pydantic, jsonschema, pyyaml, typer, rich, openai; fastapi/uvicorn
   under the `server` extra). No torch, no sklearn, no statsmodels, implement the maths
   in numpy. If a dependency is genuinely unavoidable, report it instead of adding it.
8. **Attribution.** FinTwinOS is MIT-licensed, created by Yash Sharma
   (https://www.linkedin.com/in/yashsharmaa/). Keep the credit line in README, docs
   footers and the CLI `version` command intact.
