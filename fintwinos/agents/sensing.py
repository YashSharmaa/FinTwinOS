"""The sensing agent: twin freshness checks and the shared state digest.

Before any domain desk reasons about the twin, the sensing agent establishes *how
trustworthy the twin currently is*: the latest timestamp of every time series, graph
and document counts, replay episode counts, and a staleness flag per series. The
digest is posted to the blackboard (topic ``sensing.digest``) so every downstream
agent reasons from the same snapshot, and staleness warnings surface in the case
record rather than silently biasing the analysis.

The agent prefers registry ``observe_*`` status tools when the tool catalog provides
them (so the read is audited like any other). Freshness probing itself is a pure
read-only walk of the ``TwinRuntime`` protocol surfaces (``timeseries.latest``,
``graph.stats``, ``documents.count``, ``replay.episodes``); it mutates nothing and is
recorded in the audit trail as a sensing event.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fintwinos.agents.base import AgentContext, AgentOutput, BaseAgent, TaskSpec
from fintwinos.core.types import ToolBand, utcnow
from fintwinos.models.llm_routing.router import TaskClass

#: Series whose latest point is older than this are flagged stale.
DEFAULT_STALENESS_HOURS = 48.0

#: Upper bound on the number of series included in the digest, to keep blackboard
#: posts and audit payloads bounded on very large twins.
MAX_SERIES_IN_DIGEST = 64

#: Status-tool discovery patterns, tried in order.
STATUS_TOOL_PATTERNS: tuple[str, ...] = (
    "observe_twin_status",
    "observe_*health*",
    "observe_*status*",
    "observe_*freshness*",
)


def _as_utc(ts: datetime) -> datetime:
    """Treat naive timestamps as UTC; the twin's canonical clock is UTC."""
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


class SensingAgent(BaseAgent):
    """Builds the twin-state digest and flags staleness before the swarm runs."""

    name = "sensing"
    role = "sensing"
    task_class = TaskClass.cheap

    async def run(self, task: TaskSpec, ctx: AgentContext) -> AgentOutput:
        """Probe twin freshness, post the digest to the blackboard, return it."""
        staleness_hours = float(task.inputs.get("staleness_hours", DEFAULT_STALENESS_HOURS))
        warnings: list[str] = []
        tool_results: list[dict[str, Any]] = []
        digest: dict[str, Any] = {
            "as_of": utcnow().isoformat(),
            "staleness_threshold_hours": staleness_hours,
            "series": {},
            "stale_series": [],
            "graph": {},
            "documents": {},
            "episodes": {},
            "tool_digest": {},
        }

        # Prefer audited observe_* status tools when the catalog provides them.
        for spec in self._status_specs(ctx)[:2]:
            result = await ctx.tool(spec.name, {}, caller=self.name, ticket_id=task.step_id)
            record: dict[str, Any] = {
                "tool": result.tool,
                "ok": result.ok,
                "audit_ref": result.audit_ref,
                "error": result.error,
            }
            tool_results.append(record)
            if result.ok and isinstance(result.data, dict):
                digest["tool_digest"][spec.name] = result.data
            elif not result.ok:
                warnings.append(f"status tool '{spec.name}' failed: {result.error}")

        # Direct read-only freshness probe over the runtime protocols.
        if ctx.runtime is not None:
            self._probe_runtime(ctx, digest, staleness_hours, warnings)
        else:
            warnings.append("no twin runtime attached; freshness digest is empty")

        for series_key in digest["stale_series"]:
            entry = digest["series"].get(series_key, {})
            warnings.append(
                f"series '{series_key}' is stale: last point "
                f"{entry.get('age_hours', '?')}h old "
                f"(threshold {staleness_hours:.0f}h)"
            )

        ctx.blackboard.post("sensing.digest", digest, actor=self.name)
        if ctx.audit is not None:
            ctx.audit.append(
                self.name,
                "sensing.digest_posted",
                {
                    "series": len(digest["series"]),
                    "stale_series": len(digest["stale_series"]),
                    "documents": digest["documents"].get("count", 0),
                    "graph": digest["graph"],
                },
            )

        confidence = self._confidence(ctx, digest, warnings)
        summary = (
            f"twin digest: {len(digest['series'])} series "
            f"({len(digest['stale_series'])} stale), "
            f"graph={digest['graph'] or 'unavailable'}, "
            f"documents={digest['documents'].get('count', 'unavailable')}"
        )
        return self.output(
            step_id=task.step_id,
            summary=summary,
            data=digest,
            tool_results=tool_results,
            confidence=confidence,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ internals --

    def _status_specs(self, ctx: AgentContext) -> list[Any]:
        """Discover observe-band status/health tools by pattern, deduplicated."""
        seen: set[str] = set()
        specs: list[Any] = []
        for pattern in STATUS_TOOL_PATTERNS:
            for spec in ctx.registry.list_specs(band=ToolBand.observe, pattern=pattern):
                if spec.name not in seen:
                    seen.add(spec.name)
                    specs.append(spec)
        return specs

    def _probe_runtime(
        self,
        ctx: AgentContext,
        digest: dict[str, Any],
        staleness_hours: float,
        warnings: list[str],
    ) -> None:
        """Read-only freshness walk over the twin stores; failures become warnings."""
        runtime = ctx.runtime
        assert runtime is not None
        now = utcnow()

        try:
            keys = sorted(runtime.timeseries.keys())
        except Exception as exc:
            keys = []
            warnings.append(f"timeseries store unavailable: {exc}")
        if len(keys) > MAX_SERIES_IN_DIGEST:
            warnings.append(
                f"digest truncated to {MAX_SERIES_IN_DIGEST} of {len(keys)} series"
            )
            keys = keys[:MAX_SERIES_IN_DIGEST]
        for key in keys:
            try:
                latest = runtime.timeseries.latest(key)
            except Exception as exc:
                warnings.append(f"latest() failed for series '{key}': {exc}")
                continue
            if latest is None:
                digest["series"][key] = {"latest_ts": None, "value": None, "age_hours": None}
                digest["stale_series"].append(key)
                continue
            ts, value = latest
            age_hours = round((now - _as_utc(ts)).total_seconds() / 3600.0, 2)
            digest["series"][key] = {
                "latest_ts": _as_utc(ts).isoformat(),
                "value": value,
                "age_hours": age_hours,
            }
            if age_hours > staleness_hours:
                digest["stale_series"].append(key)

        try:
            digest["graph"] = dict(runtime.graph.stats() or {})
        except Exception as exc:
            warnings.append(f"graph store unavailable: {exc}")
        try:
            digest["documents"] = {"count": int(runtime.documents.count())}
        except Exception as exc:
            warnings.append(f"document store unavailable: {exc}")
        try:
            digest["episodes"] = {"count": len(runtime.replay.episodes())}
        except Exception as exc:
            warnings.append(f"replay engine unavailable: {exc}")

    @staticmethod
    def _confidence(ctx: AgentContext, digest: dict[str, Any], warnings: list[str]) -> float:
        """Deterministic digest confidence: full twin, fresh data scores highest."""
        if ctx.runtime is None:
            return 0.1
        score = 0.9
        if digest["stale_series"]:
            score -= 0.15
        score -= 0.05 * min(len(warnings), 4)
        return round(max(0.1, score), 3)
