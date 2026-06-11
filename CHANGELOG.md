# Changelog

All notable changes to FinTwinOS are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `fintwinos.connectors.ingest` (`ingest_path`, `ingest_edgar`) backing the
  `fintwinos ingest` CLI command — drive CSV/TSV, JSONL-CDC and SEC EDGAR sources
  into a twin through the canonical `pump` driver.
- `fintwinos.rl.pipeline.run_rl_pipeline` backing `fintwinos rl` — runs the bounded
  offline-RL log → train → OPE → shadow → gate loop across every decision
  environment and writes a consolidated JSON report.
- `fintwinos.twin_core.replay.verify_replay` backing `fintwinos replay-verify` —
  proves the determinism guarantee by replaying the demo episode into a fresh twin
  and comparing content-hashed snapshots.
- `fintwinos.datasets.registry.load_dataset` backing `fintwinos datasets load`, plus
  an `offline_available` column in `list_datasets()`.
- `observe_market_regime` risk tool — rolling z-score regime detection (calm/stressed
  with hysteresis) over a twin price series, wiring in `models.time_series.regime`.
- Native ingestion routing for SEC EDGAR `filing.*` envelopes into the document store.
- Document store contents are now part of twin snapshots (`take_snapshot` version 2),
  so replay/counterfactual verification covers documents.
- `LLMClient` accepts an audit trail (threaded into the budget guard) and exposes
  `aclose()` / async-context-manager support to release the OpenAI connection pool.
- RBAC enforcement on `ApprovalWorkflow.request`, `BreakGlass.revoke` and (opt-in)
  the kill switch; new `breakglass.revoke` RBAC action.
- `side_effects` matcher on policy rules; the `deny-irreversible-critical` baseline
  rule now correctly targets irreversible side effects only.
- `.github/workflows/release.yml` publishing the `ghcr.io/<owner>/fintwinos-tools-server`
  image on `v*` tags.

## [0.1.0]

Initial public release: an auditable digital-twin operating system for financial
organisations — federated twin substrate, typed observe/simulate/propose/execute
tool catalog, multi-agent orchestration with critics and policy gates, bounded
offline RL, calibrated simulators, governance plane (RBAC, approvals, kill switch,
break-glass) and an offline-first evaluation stack.
