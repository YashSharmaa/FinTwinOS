"""Settings for FinTwinOS.

All values can be supplied via environment variables prefixed ``FINTWIN_`` or a local
``.env`` file. The OpenAI key is also read from the conventional ``OPENAI_API_KEY``.

Offline mode (``FINTWIN_OFFLINE=1``) makes every LLM call return a deterministic stub so
the whole platform — demos, tests, evals — runs with no key and no network access.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINTWIN_", env_file=".env", extra="ignore")

    # --- LLM provider (OpenAI) -------------------------------------------------
    openai_api_key: str | None = Field(default=None, repr=False)
    llm_provider: str = "openai"
    llm_model_primary: str = "gpt-5"        # planning, analysis, critique
    llm_model_fast: str = "gpt-5-mini"      # drafting, extraction
    llm_model_cheap: str = "gpt-5-nano"     # classification, routing, cheap calls
    llm_temperature: float = 0.2            # only sent to models that support it
    llm_reasoning_effort: str = "low"       # gpt-5 / o-series reasoning effort
    llm_max_output_tokens: int = 4096       # includes reasoning tokens on gpt-5
    llm_budget_usd: float = 25.0            # hard per-process spend ceiling; 0 disables
    llm_cache_enabled: bool = True          # content-addressed response cache
    request_timeout: float = 60.0
    max_retries: int = 3

    # --- Runtime ----------------------------------------------------------------
    offline: bool = False
    seed: int = 7
    data_dir: Path = Path(".fintwinos")
    audit_path: Path | None = None
    environment: str = "local"  # local | shadow | production

    # --- Governance defaults ----------------------------------------------------
    execute_tools_enabled: bool = False  # hard off-switch for the execute band
    dual_control_required: bool = True
    shadow_mode: bool = True             # execute calls are intercepted, never invoked
    approval_secret: str | None = Field(default=None, repr=False)  # HMAC key for tokens
    rbac_default_role: str = "analyst"   # role assumed for calls that assert none
    webhook_token: str | None = Field(default=None, repr=False)  # webhook ingest auth

    @model_validator(mode="after")
    def _fallback_openai_key(self) -> Settings:
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        return self

    @property
    def llm_available(self) -> bool:
        return (not self.offline) and bool(self.openai_api_key)

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests to re-read environment variables."""
    get_settings.cache_clear()
