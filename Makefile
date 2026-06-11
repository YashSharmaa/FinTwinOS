# FinTwinOS developer entry points.
#
# Everything defaults to offline mode where it matters (tests, evals, demos)
# so a fresh clone works with zero secrets. Override PYTHON to target a
# specific interpreter, e.g. `make test PYTHON=.venv/bin/python`.

PYTHON ?= python3
PIP    := $(PYTHON) -m pip
CLI    := $(PYTHON) -m fintwinos.cli

DEMO    ?= liquidity
SUITE   ?= all
HOST    ?= 127.0.0.1
PORT    ?= 8341
VERSION ?= 0.1.0

.DEFAULT_GOAL := help

.PHONY: help install lint test eval demo serve docker-build schemas

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; printf "FinTwinOS make targets:\n\n"} \
		/^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables: PYTHON, DEMO, SUITE, HOST, PORT, VERSION\n"

install: ## Editable install with dev + server extras
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,server]"

lint: ## Ruff over the whole tree (auto-fixes nothing; CI parity)
	$(PYTHON) -m ruff check .

test: ## Full pytest suite, offline and deterministic
	FINTWIN_OFFLINE=1 $(PYTHON) -m pytest -q

eval: ## Run eval suites offline (SUITE=all|<name>); report under .fintwinos/
	FINTWIN_OFFLINE=1 $(CLI) eval $(SUITE) --report .fintwinos/eval-report

demo: ## Run a packaged demo offline (DEMO=liquidity|aml_triage|...)
	FINTWIN_OFFLINE=1 $(CLI) demo $(DEMO)

serve: ## Serve the typed tool catalog (HOST/PORT overridable)
	$(CLI) serve-tools --host $(HOST) --port $(PORT)

docker-build: ## Build the tools-server image from infra/Dockerfile
	docker build -f infra/Dockerfile -t ghcr.io/yashsharmaa/fintwinos-tools-server:$(VERSION) .

schemas: ## Export canonical entity JSON Schemas to ./schemas
	$(CLI) export-schemas --out schemas
