"""Sensing agent tests: freshness digest, staleness flags, degraded modes."""

from __future__ import annotations

from fintwinos.agents.base import AgentContext, Blackboard, TaskSpec
from fintwinos.agents.sensing import SensingAgent

from .conftest import build_fake_registry, build_fake_runtime


def _task(**inputs) -> TaskSpec:
    return TaskSpec(step_id="sensing", owner="sensing", objective="sense the twin", inputs=inputs)


async def test_digest_posted_with_series_graph_and_documents(ctx):
    output = await SensingAgent().run(_task(), ctx)
    digest = output.data
    assert set(digest["series"]) == {
        "market.eq_index", "treasury.deposits_total", "ops.queue_depth"
    }
    assert digest["graph"] == {"entities": 2, "relationships": 1}
    assert digest["documents"] == {"count": 3}
    assert digest["episodes"] == {"count": 1}
    # the digest is shared via the blackboard for every downstream agent
    assert ctx.blackboard.latest("sensing.digest") == digest


async def test_stale_series_flagged_with_warning(ctx):
    output = await SensingAgent().run(_task(), ctx)
    assert output.data["stale_series"] == ["ops.queue_depth"]
    assert any("stale" in w for w in output.warnings)
    assert output.confidence < 0.9  # staleness costs confidence


async def test_fresh_twin_has_no_stale_series(settings, offline_llm):
    runtime = build_fake_runtime(with_stale_series=False)
    registry = build_fake_registry(runtime, settings)
    ctx = AgentContext(
        runtime=runtime, registry=registry, llm=offline_llm,
        blackboard=Blackboard(), settings=settings, audit=runtime.audit,
    )
    output = await SensingAgent().run(_task(), ctx)
    assert output.data["stale_series"] == []
    assert not any("stale" in w for w in output.warnings)


async def test_staleness_threshold_is_configurable(ctx):
    # with a one-week threshold even the 120h-old series counts as fresh
    output = await SensingAgent().run(_task(staleness_hours=24 * 7), ctx)
    assert output.data["stale_series"] == []


async def test_no_runtime_degrades_with_warning(settings, offline_llm):
    registry = build_fake_registry(None, settings, with_tools=False)
    ctx = AgentContext(
        runtime=None, registry=registry, llm=offline_llm,
        blackboard=Blackboard(), settings=settings, audit=None,
    )
    output = await SensingAgent().run(_task(), ctx)
    assert output.data["series"] == {}
    assert any("no twin runtime" in w for w in output.warnings)
    assert output.confidence <= 0.1


async def test_sensing_appends_audit_record(ctx):
    before = len(ctx.audit)
    await SensingAgent().run(_task(), ctx)
    records = ctx.audit.records(action="sensing.digest_posted")
    assert len(records) == 1
    assert len(ctx.audit) > before
