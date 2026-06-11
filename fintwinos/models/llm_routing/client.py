"""Async OpenAI client wrapper used by every agent.

Behaviour:

- Routes each call to a model tier via ``ModelRouter`` (override with ``model=``).
- Supports tool definitions (OpenAI function calling) and strict JSON-schema output.
- Knows the GPT-5 / o-series parameter surface: reasoning models receive
  ``reasoning_effort`` and never receive a non-default ``temperature`` (which they
  reject with a 400).
- Retries transient failures with exponential backoff; permanent request errors
  (authentication, invalid request, permissions) are raised immediately.
- Meters tokens and indicative cost across the client's lifetime, enforces the
  ``Settings.llm_budget_usd`` hard spend ceiling, and serves identical tool-free
  prompts from the content-addressed :class:`ResponseCache`.
- In offline mode (``FINTWIN_OFFLINE=1`` or no API key) returns deterministic stubs, so
  agents must implement rule-based fallbacks and tests never need network access.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from fintwinos.core.config import Settings, get_settings
from fintwinos.models.llm_routing.router import ModelRouter, TaskClass

OFFLINE_MODEL_NAME = "offline-stub"

#: Model-name prefixes that are reasoning models: they take ``reasoning_effort``
#: and reject any non-default ``temperature`` on chat completions.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

_SCHEMA_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def is_reasoning_model(model: str) -> bool:
    """True for GPT-5 / o-series models with the reasoning parameter surface."""
    return model.startswith(REASONING_MODEL_PREFIXES)


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: str = OFFLINE_MODEL_NAME
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    offline: bool = False
    cached: bool = False
    refusal: str | None = None
    truncated: bool = False

    def json_data(self) -> Any:
        """Parse the text as JSON (tolerating markdown fences), None on failure."""
        raw = self.text.strip()
        if raw.startswith("```"):
            raw = _FENCE_RE.sub("", raw).strip()
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        router: ModelRouter | None = None,
        audit: Any | None = None,
    ):
        self.settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings)
        self.audit = audit
        self.offline = not self.settings.llm_available
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self._client: Any = None
        self._budget: Any = None
        self._cache: Any = None
        if not self.offline:
            from openai import AsyncOpenAI  # imported lazily: offline mode needs no SDK state

            self._client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.request_timeout,
            )
            if self.settings.llm_budget_usd > 0:
                from fintwinos.models.llm_routing.budget import BudgetGuard

                # Pass the audit trail through so budget.charged / budget.exceeded
                # records land in the shared audit log when one is supplied.
                self._budget = BudgetGuard(self.settings.llm_budget_usd, audit=audit)
            if self.settings.llm_cache_enabled:
                from fintwinos.models.llm_routing.cache import ResponseCache

                self._cache = ResponseCache(settings=self.settings)

    # ------------------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying AsyncOpenAI client, releasing its connection pool.

        Safe to call in offline mode (a no-op) and more than once. Long-lived
        processes and the eval/demo runners should call this — or use the client
        as an async context manager — to avoid leaking httpx connections.
        """
        client = self._client
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                await close()
            self._client = None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        task: TaskClass = TaskClass.analysis,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        chosen = model or self.router.model_for(task)

        if self.offline:
            return self._offline_response(messages, tools, json_schema)

        # Tool-free calls are content-addressed: an identical prompt is never re-billed.
        cacheable = tools is None and self._cache is not None
        if cacheable:
            cached = self._cache.get(chosen, messages, json_schema)
            if cached is not None:
                return cached.model_copy(update={"cached": True})

        if self._budget is not None:
            self._budget.ensure_available()

        kwargs: dict[str, Any] = {
            "model": chosen,
            "messages": messages,
            "max_completion_tokens": (
                max_output_tokens or self.settings.llm_max_output_tokens
            ),
        }
        if is_reasoning_model(chosen):
            # GPT-5 / o-series: temperature is fixed; effort steers reasoning depth.
            if self.settings.llm_reasoning_effort:
                kwargs["reasoning_effort"] = self.settings.llm_reasoning_effort
        else:
            kwargs["temperature"] = (
                temperature if temperature is not None else self.settings.llm_temperature
            )
        if tools:
            kwargs["tools"] = tools
        if json_schema is not None:
            raw_name = str(json_schema.get("title", "structured_output")) or "structured_output"
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _SCHEMA_NAME_RE.sub("_", raw_name)[:64],
                    "schema": json_schema,
                    "strict": False,
                },
            }

        response = await self._call_with_retries(kwargs)
        wrapped = self._wrap(response, chosen)
        if self._budget is not None:
            self._budget.record(wrapped)
        if cacheable and wrapped.refusal is None and not wrapped.truncated:
            self._cache.put(chosen, messages, wrapped, json_schema)
        return wrapped

    async def _call_with_retries(self, kwargs: dict[str, Any]) -> Any:
        """Retry transient failures only; permanent request errors raise immediately."""
        from openai import (
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
        )

        permanent = (AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError)
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except permanent:
                raise
            except Exception as exc:  # noqa: BLE001 — SDK raises many transient types
                last_error = exc
                if attempt < self.settings.max_retries - 1:
                    await asyncio.sleep(min(2.0 ** attempt, 8.0))
        raise RuntimeError(
            f"LLM call failed after {self.settings.max_retries} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------------------

    def _wrap(self, response: Any, model: str) -> LLMResponse:
        choice = response.choices[0]
        text = choice.message.content or ""
        refusal = getattr(choice.message, "refusal", None) or None
        truncated = getattr(choice, "finish_reason", None) == "length"
        tool_calls: list[dict[str, Any]] = []
        for tc in choice.message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {"_raw": tc.function.arguments}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})

        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        }
        cost = self.router.cost(model, usage["input_tokens"], usage["output_tokens"])
        self.total_cost_usd += cost
        self.total_input_tokens += usage["input_tokens"]
        self.total_output_tokens += usage["output_tokens"]
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            model=model,
            usage=usage,
            cost_usd=cost,
            refusal=refusal,
            truncated=truncated,
        )

    def _offline_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        json_schema: dict[str, Any] | None,
    ) -> LLMResponse:
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        digest = hashlib.sha256(str(last_user).encode("utf-8")).hexdigest()[:12]
        if json_schema is not None:
            text = json.dumps(
                {"offline": True, "stub_id": digest,
                 "note": "deterministic offline stub; use rule-based fallbacks"}
            )
        else:
            snippet = str(last_user)[:160]
            text = f"[offline-stub:{digest}] {snippet}"
        return LLMResponse(text=text, model=OFFLINE_MODEL_NAME, offline=True)

    # ------------------------------------------------------------------------------

    def usage_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
            "offline": self.offline,
        }
        if self._budget is not None:
            summary["budget"] = self._budget.summary()
        if self._cache is not None:
            summary["cache"] = self._cache.stats()
        return summary
