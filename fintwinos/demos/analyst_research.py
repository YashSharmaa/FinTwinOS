"""Demo: filings research with citations and an adversarial critic pass.

Flow: load the sample filing corpus into the twin's document store → search
for risk factors through ``observe_filing_search`` → build a research brief in
which **every claim carries citations** (document ids plus retrieval scores) →
run a critic agent that challenges the brief: uncited claims, dangling
citations and weak lexical grounding are flagged before anything leaves the
desk.

Run with ``fintwinos demo analyst_research``. Fully offline-capable: without
an OpenAI key the brief is built extractively and the critic runs its
deterministic rule-based checks.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fintwinos.agents.base import AgentContext, BaseAgent, TaskSpec
from fintwinos.core.config import Settings
from fintwinos.demos._corpus import first_sentences, lexical_support, resolve_filing_corpus
from fintwinos.demos.stack import (
    DemoStack,
    as_plain,
    audit_summary,
    build_stack,
    call_tool,
    find_tool,
    render_audit_excerpt,
    render_header,
    schema_args,
)
from fintwinos.models.llm_routing.router import TaskClass

HEADLINE_QUERY = "principal risk factors liquidity and funding stress"

#: Research questions answered by the brief: (topic, retrieval query).
RESEARCH_TOPICS: list[tuple[str, str]] = [
    ("liquidity and funding", "liquidity funding deposit outflow contingency coverage ratio"),
    ("credit concentration", "credit concentration commercial real estate office loans allowance"),
    ("cyber and operational resilience", "cybersecurity ransomware vendor outage data centre recovery"),
    ("financial crime remediation", "anti-money laundering consent order remediation monitoring"),
]

_SEARCH_TOOLS = [
    "observe_filing_search",
    "observe_*filing*",
    "observe_document_search",
    "observe_*document*",
    "observe_*search*",
]


class BriefCritic(BaseAgent):
    """Critic agent that challenges a cited brief before it ships.

    Deterministic checks (always run): every claim must carry at least one
    citation, every cited document must exist in the store, and the claim's
    content tokens must be lexically supported by the cited documents. When a
    live LLM is available it adds qualitative commentary on top; offline, the
    rule-based verdict stands alone.
    """

    name = "demo.brief_critic"
    role = "critic"
    task_class = TaskClass.critique

    async def run(self, task: TaskSpec, ctx: AgentContext) -> Any:
        brief: list[dict[str, Any]] = task.inputs["brief"]
        doc_texts: dict[str, str] = task.inputs["doc_texts"]
        challenges: list[dict[str, Any]] = []
        supports: list[float] = []

        for claim in brief:
            topic = claim["topic"]
            citations = claim.get("citations") or []
            if not citations:
                challenges.append(
                    {"topic": topic, "severity": "high", "issue": "claim carries no citations"}
                )
                supports.append(0.0)
                continue
            cited_texts: list[str] = []
            for citation in citations:
                doc_id = citation.get("doc_id")
                text = doc_texts.get(str(doc_id))
                if text is None:
                    challenges.append(
                        {
                            "topic": topic,
                            "severity": "high",
                            "issue": f"dangling citation: '{doc_id}' is not in the document store",
                        }
                    )
                else:
                    cited_texts.append(text)
            support = lexical_support(claim["claim"], cited_texts)
            supports.append(support)
            if cited_texts and support < 0.4:
                challenges.append(
                    {
                        "topic": topic,
                        "severity": "medium",
                        "issue": f"weak grounding: only {support:.0%} of claim tokens "
                        "appear in the cited documents",
                    }
                )

        commentary = (
            "Rule-based critic: every claim was checked for citation presence, citation "
            "resolution and lexical grounding against the cited filings."
        )
        try:
            response = await self.ask_llm(
                ctx,
                "Critique this cited research brief in two sentences. Flag any claim that "
                f"overreaches its citations:\n{brief}",
            )
        except Exception:  # the deterministic critique above already stands
            response = None
        if response is not None and not response.offline and response.text.strip():
            commentary = response.text.strip()

        passed = not any(c["severity"] == "high" for c in challenges)
        coverage = round(sum(supports) / len(supports), 4) if supports else 0.0
        return self.output(
            step_id=task.step_id,
            summary=f"{len(challenges)} challenge(s); grounding coverage {coverage:.0%}",
            data={
                "challenges": challenges,
                "passed": passed,
                "coverage": coverage,
                "commentary": commentary,
            },
            confidence=coverage if supports else 0.0,
        )


async def arun(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
    stack: DemoStack | None = None,
) -> dict[str, Any]:
    """Async core of the analyst research demo."""
    stack = stack or build_stack(seed=seed, console=console, offline_ok=offline_ok, settings=settings)
    console = console or stack.console
    warnings: list[str] = []

    render_header(
        console,
        "Filings research",
        "Load the filing corpus, retrieve risk factors, and ship a brief where every "
        "claim is pinned to document ids, then let the critic try to break it.",
        stack.settings,
    )

    corpus = _load_corpus(stack)
    console.print(
        Panel(
            f"corpus source: {corpus['source']}\n"
            f"documents loaded this run: {corpus['docs_loaded']} · "
            f"documents in store: {corpus['docs_in_store']}",
            title="[bold]Document store[/bold]",
            border_style="green",
        )
    )

    search_tool = find_tool(stack.registry, _SEARCH_TOOLS)
    if search_tool is None:
        warnings.append("no observe_* filing-search tool registered; querying the store directly")
    headline_hits = await _search(stack, search_tool, HEADLINE_QUERY, k=5)
    _render_hits(console, headline_hits, stack)

    brief = await _build_brief(stack, search_tool, warnings)
    _render_brief(console, brief)

    critic = await _critic_pass(stack, brief)
    _render_critic(console, critic)

    audit = audit_summary(stack.registry.audit)
    render_audit_excerpt(console, audit["excerpt"])

    return {
        "demo": "analyst_research",
        "offline": stack.llm.offline,
        "corpus": corpus,
        "search": {
            "source": search_tool or "documents.search",
            "query": HEADLINE_QUERY,
            "hits": headline_hits,
        },
        "brief": brief,
        "critic": critic,
        "audit": audit,
        "warnings": warnings,
    }


def _load_corpus(stack: DemoStack) -> dict[str, Any]:
    """Load the sample filing corpus into the twin's document store (idempotent)."""
    docs, source = resolve_filing_corpus()
    loaded = 0
    for doc in docs:
        if stack.runtime.documents.get(doc["doc_id"]) is None:
            stack.runtime.documents.add(doc["doc_id"], doc["text"], doc["metadata"])
            loaded += 1
    stack.registry.audit.append(
        "demo.analyst_research",
        "corpus.loaded",
        {"source": source, "docs_loaded": loaded, "docs_in_store": stack.runtime.documents.count()},
    )
    return {
        "source": source,
        "docs_loaded": loaded,
        "docs_in_store": stack.runtime.documents.count(),
    }


async def _search(
    stack: DemoStack, tool: str | None, query: str, k: int = 5
) -> list[dict[str, Any]]:
    """Search filings through the registry tool, or the store as a fallback."""
    if tool is not None:
        spec = stack.registry.spec(tool)
        result = await call_tool(
            stack,
            tool,
            schema_args(spec, {"query": query, "k": k, "limit": k}),
            caller="demo.analyst_research",
        )
        if result.ok:
            hits = _coerce_hits(result.data)
            if hits:
                return hits[:k]
    raw = stack.runtime.documents.search(query, k=k)
    return _coerce_hits(raw)[:k]


def _coerce_hits(data: Any) -> list[dict[str, Any]]:
    """Normalise search payloads into ``[{"doc_id", "score"}, ...]``."""
    plain = as_plain(data)
    items: Any = plain
    if isinstance(plain, dict):
        for key in ("hits", "results", "matches", "documents"):
            if isinstance(plain.get(key), list):
                items = plain[key]
                break
    hits: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return hits
    for item in items:
        if isinstance(item, dict):
            doc_id = item.get("doc_id", item.get("id"))
            score = item.get("score", item.get("relevance", 0.0))
        elif isinstance(item, list | tuple) and len(item) >= 2:
            doc_id, score = item[0], item[1]
        else:
            continue
        if isinstance(doc_id, str) and isinstance(score, int | float):
            hits.append({"doc_id": doc_id, "score": round(float(score), 4)})
    return hits


def _render_hits(console: Console, hits: list[dict[str, Any]], stack: DemoStack) -> None:
    table = Table(
        title=f"Risk-factor retrieval · “{HEADLINE_QUERY}”",
        title_justify="left",
        border_style="green",
    )
    table.add_column("doc id", style="bold")
    table.add_column("score", justify="right")
    table.add_column("issuer / section", style="dim")
    for hit in hits:
        doc = stack.runtime.documents.get(hit["doc_id"]) or {}
        meta = doc.get("metadata") or {}
        table.add_row(
            hit["doc_id"],
            f"{hit['score']:.3f}",
            f"{meta.get('issuer', '-')} · {meta.get('section', '-')}",
        )
    console.print(table)


async def _build_brief(
    stack: DemoStack, search_tool: str | None, warnings: list[str]
) -> list[dict[str, Any]]:
    """Build the cited brief: one claim per research topic, citations attached."""
    brief: list[dict[str, Any]] = []
    for topic, query in RESEARCH_TOPICS:
        hits = await _search(stack, search_tool, query, k=3)
        citations = hits[:2]
        excerpts: list[str] = []
        for citation in citations:
            doc = stack.runtime.documents.get(citation["doc_id"])
            if doc and isinstance(doc.get("text"), str):
                excerpts.append(doc["text"])
        if excerpts:
            claim = first_sentences(excerpts[0], n=2)
            source = "extractive"
        else:
            claim = f"No filings in the corpus address {topic}; flagging a coverage gap."
            source = "coverage_gap"
            warnings.append(f"no retrieval hits for topic '{topic}'")

        if excerpts and not stack.llm.offline:
            try:
                response = await stack.llm.complete(
                    [
                        {
                            "role": "system",
                            "content": "You are an equity research analyst. Synthesise ONE factual "
                            "sentence strictly from the excerpts provided. No outside knowledge.",
                        },
                        {"role": "user", "content": f"Topic: {topic}\n\nExcerpts:\n" + "\n---\n".join(excerpts)},
                    ],
                    task=TaskClass.analysis,
                )
            except Exception as exc:  # extractive claim stands; LLM is enrichment only
                warnings.append(f"LLM synthesis unavailable for '{topic}': {type(exc).__name__}: {exc}")
            else:
                if not response.offline and response.text.strip():
                    claim = response.text.strip()
                    source = "llm"

        brief.append({"topic": topic, "claim": claim, "citations": citations, "source": source})
    return brief


def _render_brief(console: Console, brief: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for i, claim in enumerate(brief, start=1):
        refs = ", ".join(
            f"{c['doc_id']} ({c['score']:.2f})" for c in claim["citations"]
        ) or "no citations"
        lines.append(f"[bold]{i}. {claim['topic'].title()}[/bold]\n{claim['claim']}\n[dim]↳ cites: {refs}[/dim]")
    console.print(
        Panel(
            "\n\n".join(lines),
            title="[bold]Research brief, every claim cited[/bold]",
            border_style="green",
        )
    )


async def _critic_pass(stack: DemoStack, brief: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the critic agent over the brief and return its verdict."""
    doc_texts: dict[str, str] = {}
    for claim in brief:
        for citation in claim.get("citations", []):
            doc_id = str(citation.get("doc_id"))
            doc = stack.runtime.documents.get(doc_id)
            if doc and isinstance(doc.get("text"), str):
                doc_texts[doc_id] = doc["text"]

    critic = BriefCritic()
    ctx = AgentContext(
        runtime=stack.runtime,
        registry=stack.registry,
        llm=stack.llm,
        settings=stack.settings,
        audit=stack.registry.audit,
    )
    task = TaskSpec(
        step_id="critic_pass",
        owner=critic.name,
        objective="challenge the cited research brief",
        inputs={"brief": brief, "doc_texts": doc_texts},
    )
    output = await critic.run(task, ctx)
    stack.registry.audit.append(
        critic.name,
        "brief.critiqued",
        {"passed": output.data["passed"], "challenges": len(output.data["challenges"])},
    )
    return output.data


def _render_critic(console: Console, critic: dict[str, Any]) -> None:
    verdict = "[green]PASSED[/green]" if critic["passed"] else "[red]CHALLENGED[/red]"
    table = Table(
        title=f"Critic pass, {verdict} · grounding coverage {critic['coverage']:.0%}",
        title_justify="left",
        border_style="magenta",
    )
    table.add_column("topic", style="bold")
    table.add_column("severity")
    table.add_column("issue")
    if critic["challenges"]:
        for challenge in critic["challenges"]:
            style = "red" if challenge["severity"] == "high" else "yellow"
            table.add_row(
                challenge["topic"], f"[{style}]{challenge['severity']}[/{style}]", challenge["issue"]
            )
    else:
        table.add_row("-", "[green]none[/green]", "no challenges raised")
    console.print(table)
    console.print(Panel(critic["commentary"], title="[dim]critic commentary[/dim]", border_style="dim"))


def run(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the analyst research demo end-to-end and return structured results.

    Returns a dict with keys ``demo``, ``corpus``, ``search`` (headline query
    hits), ``brief`` (each claim with ``citations`` mapping to doc ids and
    scores), ``critic`` (``passed``/``challenges``/``coverage``), ``audit``
    and ``warnings``.
    """
    return asyncio.run(arun(offline_ok=offline_ok, console=console, seed=seed, settings=settings))


def main() -> None:
    """Contracted CLI entry point: ``fintwinos demo analyst_research``."""
    run()
