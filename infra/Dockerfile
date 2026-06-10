# syntax=docker/dockerfile:1.7
#
# FinTwinOS tools-server image.
#
# Multi-stage build:
#   1. builder  — builds a wheel from the source tree with `python -m build`.
#   2. runtime  — slim image that installs only the wheel (+ the [server] extra),
#                 runs as a non-root user, and serves the typed tool catalog.
#
# The image defaults to FINTWIN_OFFLINE=1 so it starts with zero secrets and no
# network egress: every LLM call returns a deterministic stub and all agents fall
# back to their rule-based paths. Mount an .env / set OPENAI_API_KEY and
# FINTWIN_OFFLINE=0 to enable live LLM calls.
#
# Build (from the repository root):
#   docker build -f infra/Dockerfile -t fintwinos/tools-server:0.1.0 .
# Run:
#   docker run --rm -p 8341:8341 fintwinos/tools-server:0.1.0

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------------- #
# Stage 1: build the wheel                                                    #
# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src

RUN pip install --no-cache-dir build

# Only what the wheel needs — keeps the build context cache-friendly.
COPY pyproject.toml README.md LICENSE ./
COPY fintwinos ./fintwinos

RUN python -m build --wheel --outdir /wheels

# --------------------------------------------------------------------------- #
# Stage 2: slim runtime                                                       #
# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="FinTwinOS tools server" \
      org.opencontainers.image.description="Auditable digital-twin operating system for financial organisations — typed tool catalog server." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Yash Sharma (https://www.linkedin.com/in/yashsharmaa/)"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Offline-first: the container is fully functional with no key and no egress.
    FINTWIN_OFFLINE=1 \
    # Governance defaults: execute band hard-off, shadow mode on.
    FINTWIN_EXECUTE_TOOLS_ENABLED=0 \
    FINTWIN_SHADOW_MODE=1 \
    FINTWIN_DATA_DIR=/data

# Non-root runtime user with a fixed UID/GID (matches the Helm chart's
# podSecurityContext so volume permissions line up on Kubernetes too).
RUN groupadd --gid 10001 fintwin \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin fintwin \
    && mkdir -p /data \
    && chown -R fintwin:fintwin /data

COPY --from=builder /wheels /tmp/wheels
RUN WHEEL="$(ls /tmp/wheels/fintwinos-*.whl)" \
    && pip install --no-cache-dir "${WHEEL}[server]" \
    && rm -rf /tmp/wheels

USER fintwin
WORKDIR /home/fintwin

# Local twin state: audit chains, outbox, eval reports.
VOLUME ["/data"]

EXPOSE 8341

# httpx is a first-party dependency, so the healthcheck needs no curl/wget in
# the slim image. /healthz is served by `fintwinos serve-tools`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import httpx; httpx.get('http://127.0.0.1:8341/healthz', timeout=4.0).raise_for_status()"]

CMD ["fintwinos", "serve-tools", "--host", "0.0.0.0"]
