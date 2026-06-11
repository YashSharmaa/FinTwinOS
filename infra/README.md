# Deploying FinTwinOS

FinTwinOS is **offline-first**: every deployment mode below works with zero
secrets and zero network egress (`FINTWIN_OFFLINE=1`), running deterministic
LLM stubs and rule-based agent fallbacks. Adding an `OPENAI_API_KEY` and
setting `FINTWIN_OFFLINE=0` upgrades the same deployment to live LLM calls,
nothing else changes.

The unit of deployment is the **typed tools server**
(`fintwinos serve-tools`), which exposes the governed tool catalog
(observe / simulate / propose / execute bands) over HTTP with a `/healthz`
endpoint for probes.

## Deploy matrix

| Mode | Best for | Command | Secrets needed |
|---|---|---|---|
| **Local (pip)** | Development, demos, evals | `make install && make serve` | None (offline) |
| **Docker** | Single-host trials, air-gapped labs | `make docker-build && docker run --rm -p 8341:8341 ghcr.io/yashsharmaa/fintwinos-tools-server:0.1.0` | None (offline by default) |
| **Compose** | Persistent single-host service with state volume | `cd infra && docker compose up --build` | None; optional `.env` at repo root |
| **Helm / on-prem Kubernetes** | Production, shadow-mode pilots, regulated on-prem clusters | `helm install fintwinos infra/helm/fintwinos` | None; optional `Secret` for the OpenAI key |

### Local (pip)

```bash
make install                 # pip install -e ".[dev,server]"
make test                    # offline, deterministic
make demo DEMO=liquidity     # packaged demos, no key needed
make serve                   # tool catalog on 127.0.0.1:8341
```

### Docker

The image (`infra/Dockerfile`) is a multi-stage build: a builder stage
produces a wheel, and a slim runtime stage installs only that wheel with the
`[server]` extra. It runs as the non-root user `fintwin` (UID/GID 10001),
ships a `HEALTHCHECK` against `/healthz`, defaults to `FINTWIN_OFFLINE=1`,
and stores all local twin state (audit chains, execute-band outbox, eval
reports) under the `/data` volume.

```bash
make docker-build
docker run --rm -p 8341:8341 \
  ghcr.io/yashsharmaa/fintwinos-tools-server:0.1.0                 # offline
docker run --rm -p 8341:8341 \
  -e FINTWIN_OFFLINE=0 -e OPENAI_API_KEY=sk-... \
  ghcr.io/yashsharmaa/fintwinos-tools-server:0.1.0                 # live
```

### Compose

`infra/docker-compose.yml` runs the `tools-server` service with a named
volume for `/data` and reads an optional `.env` from the repo root (the file
is not required, the stack is fully functional without it).

```bash
cd infra
docker compose up --build              # offline, zero secrets
FINTWIN_OFFLINE=0 docker compose up -d # live, key taken from ../.env
```

### Helm / on-prem Kubernetes

`infra/helm/fintwinos` deploys the tools server with liveness/readiness
probes on `/healthz`, a ConfigMap-driven environment, conservative pod
security defaults (non-root, no privilege escalation, all capabilities
dropped) and an optional PVC for `/data`.

```bash
helm install fintwinos infra/helm/fintwinos                  # offline default

# Live mode: provision the key as a Secret, never as a chart value.
kubectl create secret generic fintwinos-openai \
  --from-literal=OPENAI_API_KEY=sk-...
helm upgrade fintwinos infra/helm/fintwinos \
  --set openaiSecret.enabled=true --set env.FINTWIN_OFFLINE="0"
```

Key chart values: `image.*`, `resources`, `env` (any `FINTWIN_*` setting),
`openaiSecret.{enabled,name,key}`, `persistence.existingClaim`,
`service.{type,port}`. See `values.yaml` for the full annotated list.

## Hybrid deployments (regulated environments)

A common pattern for financial institutions is **on-prem twin, outbound LLM
only**:

- The twin runtime, stores, simulators, audit trail and tool registry all run
  inside your perimeter (Compose or Helm). No data leaves except the prompts
  you explicitly allow.
- Start in `FINTWIN_OFFLINE=1` to validate behaviour end-to-end with
  deterministic stubs, then enable live calls for selected environments by
  flipping `FINTWIN_OFFLINE` and injecting the key via your secret manager.
- Keep `FINTWIN_ENVIRONMENT=shadow` and `FINTWIN_SHADOW_MODE=1` until your
  governance forum signs off; the execute band stays hard-off regardless
  unless `FINTWIN_EXECUTE_TOOLS_ENABLED=1` **and** a policy allow-rule
  **and** a human `ApprovalToken` are all present. Even then, execute
  handlers write to the local outbox under the data dir, never directly to a
  production system.
- The audit trail is a hash chain: persist `/data` (Compose volume or PVC) so
  the chain survives restarts and stays verifiable.

## Environment variables

All settings come from `FINTWIN_*` environment variables or a `.env` file
(see `fintwinos/core/config.py`; the OpenAI key is also read from the
conventional `OPENAI_API_KEY`).

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | unset | OpenAI key; unset means offline stubs |
| `FINTWIN_OFFLINE` | `0` (pip) / `1` (docker, compose, helm) | Force deterministic offline mode, no network |
| `FINTWIN_LLM_MODEL_PRIMARY` | `gpt-5` | Planning, analysis, critique |
| `FINTWIN_LLM_MODEL_FAST` | `gpt-5-mini` | Drafting, extraction |
| `FINTWIN_LLM_MODEL_CHEAP` | `gpt-5-nano` | Classification, routing |
| `FINTWIN_LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `FINTWIN_LLM_MAX_OUTPUT_TOKENS` | `4096` | Per-call output cap |
| `FINTWIN_REQUEST_TIMEOUT` | `60.0` | LLM request timeout (seconds) |
| `FINTWIN_MAX_RETRIES` | `3` | LLM retry budget |
| `FINTWIN_SEED` | `7` | Global determinism seed |
| `FINTWIN_DATA_DIR` | `.fintwinos` (`/data` in containers) | Audit trail, outbox, reports |
| `FINTWIN_AUDIT_PATH` | unset | Optional explicit JSONL audit path |
| `FINTWIN_ENVIRONMENT` | `local` | `local` \| `shadow` \| `production` |
| `FINTWIN_EXECUTE_TOOLS_ENABLED` | `0` | Hard off-switch for the execute band |
| `FINTWIN_DUAL_CONTROL_REQUIRED` | `1` | Require dual-control approvals |
| `FINTWIN_SHADOW_MODE` | `1` | Propose-only shadow operation |

---

FinTwinOS is MIT-licensed, created by
[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/).
