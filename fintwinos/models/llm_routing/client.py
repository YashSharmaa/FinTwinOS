"""Async OpenAI client wrapper used by every agent.

Behaviour:

- Routes each call to a model tier via ``ModelRouter`` (override with ``model=``).
- Supports tool definitions (OpenAI function calling) and strict JSON-schema output.
- Retries transient failures with exponential backoff.
- Meters tokens and indicative cost across the client's lifetime.
- In offline mode (``FINTWIN_OFFLINE=1`` or no API key) returns deterministic stubs, so
  agents must implement rule-based fallbacks and tests never need network access.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from fintwinos.core.config import Settings, get_settings
from fintwinos.models.llm_routing.router import ModelRouter, TaskClass

OFFLINE_MODEL_NAME = "offline-stub"


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: str = OFFLINE_MODEL_NAME
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    offline: bool = False

    def json_data(self) -> Any:
        """Parse the text as JSON, returning None on failure."""
        try:
            return json.loads(self.text)
        except (json.JSONDecodeError, TypeError):
            return None


class LLMClient:
    def __init__(self, settings: Settings | None = None, router: ModelRouter | None = None):
        self.settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings)
        self.offline = not self.settings.llm_available
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self._client: Any = None
        if not self.offline:
            from openai import AsyncOpenAI  # imported lazily: offline mode needs no SDK state

            self._client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.request_timeout,
            )

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

        kwargs: dict[str, Any] = {
            "model": chosen,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.settings.llm_temperature
            ),
            "max_completion_tokens": (
                max_output_tokens or self.settings.llm_max_output_tokens
            ),
        }
        if tools:
            kwargs["tools"] = tools
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "structured_output"),
                    "schema": json_schema,
                    "strict": False,
                },
            }

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                return self._wrap(response, chosen)
            except Exception as exc:  # noqa: BLE001 — SDK raises many transient types
                last_error = exc
                if attempt < self.settings.max_retries - 1:
                    await asyncio.sleep(min(2.0 ** attempt, 8.0))
        raise RuntimeError(f"LLM call failed after {self.settings.max_retries} attempts") from last_error

    # ------------------------------------------------------------------------------

    def _wrap(self, response: Any, model: str) -> LLMResponse:
        choice = response.choices[0]
        text = choice.message.content or ""
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
        return LLMResponse(text=text, tool_calls=tool_calls, model=model, usage=usage, cost_usd=cost)

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
        return {
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
            "offline": self.offline,
        }
