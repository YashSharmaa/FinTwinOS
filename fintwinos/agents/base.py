"""Base classes for FinTwinOS agents.

Design rules every agent must follow:

- All twin access goes through the typed tool registry (never raw store access in
  production paths), so every observation is audited.
- Every agent must work in offline mode via deterministic, rule-based fallbacks;
  the LLM enriches behaviour, it is never a hard dependency.
- Shared state flows through the ``Blackboard``, not hidden module globals, so a
  run's full intermediate state is inspectable and traceable.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings, get_settings
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import ToolResult, utcnow
from fintwinos.models.llm_routing.client import LLMClient, LLMResponse
from fintwinos.models.llm_routing.router import TaskClass
from fintwinos.tools.registry import CallContext, ToolRegistry


class TaskSpec(BaseModel):
    """One step of a plan, owned by exactly one agent."""

    step_id: str
    owner: str
    objective: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    agent: str
    step_id: str | None = None
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.5
    warnings: list[str] = Field(default_factory=list)


class Blackboard:
    """Thread-safe shared memory for one case run, with a full posting history."""

    def __init__(self):
        self._topics: dict[str, list[dict[str, Any]]] = {}
        self._history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def post(self, topic: str, item: Any, actor: str = "unknown") -> None:
        entry = {"topic": topic, "actor": actor, "ts": utcnow().isoformat(), "item": item}
        with self._lock:
            self._topics.setdefault(topic, []).append(entry)
            self._history.append(entry)

    def read(self, topic: str) -> list[Any]:
        with self._lock:
            return [e["item"] for e in self._topics.get(topic, [])]

    def latest(self, topic: str) -> Any | None:
        items = self.read(topic)
        return items[-1] if items else None

    def topics(self) -> list[str]:
        with self._lock:
            return sorted(self._topics)

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)


@dataclass
class AgentContext:
    """Everything an agent may touch during a run."""

    runtime: TwinRuntime | None
    registry: ToolRegistry
    llm: LLMClient
    blackboard: Blackboard = field(default_factory=Blackboard)
    settings: Settings = field(default_factory=get_settings)
    audit: AuditTrail | None = None

    async def tool(self, name: str, arguments: dict[str, Any], caller: str,
                   ticket_id: str | None = None) -> ToolResult:
        return await self.registry.call(
            name, arguments, CallContext(caller=caller, ticket_id=ticket_id)
        )


class BaseAgent(ABC):
    """Subclass per role; implement ``run``. ``ask_llm`` degrades to offline stubs."""

    name: str = "agent"
    role: str = "generic"
    task_class: TaskClass = TaskClass.analysis
    system_prompt: str = (
        "You are a specialist agent inside FinTwinOS, an auditable digital twin of a "
        "financial institution. Be precise, cite the twin state you used, and never "
        "recommend irreversible actions without flagging the required human approvals."
    )

    def __init__(self, name: str | None = None):
        if name is not None:
            self.name = name

    @abstractmethod
    async def run(self, task: TaskSpec, ctx: AgentContext) -> AgentOutput: ...

    async def ask_llm(
        self,
        ctx: AgentContext,
        user_content: str,
        *,
        json_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        response = await ctx.llm.complete(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            task=self.task_class,
            json_schema=json_schema,
            tools=tools,
        )
        if ctx.audit is not None:
            ctx.audit.append(
                self.name,
                "llm.completed",
                {"model": response.model, "offline": response.offline,
                 "usage": response.usage, "cost_usd": response.cost_usd},
            )
        return response

    def output(self, **kwargs: Any) -> AgentOutput:
        kwargs.setdefault("agent", self.name)
        return AgentOutput(**kwargs)
