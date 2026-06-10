"""Tests for the benchmark-style adapters and their companion suites."""

from __future__ import annotations

from fintwinos.evals.adapters import (
    ExtractiveQASubject,
    FilingsQASuite,
    MultiTurnToolSuite,
    SequentialRouterSubject,
    load_bfcl_style,
    load_finmcp_style,
    load_secque_style,
)
from fintwinos.evals.function_calls import RuleBasedSubject, ToolCallExactnessSuite
from fintwinos.evals.harness import EvalCase, summarise

# --- loaders ----------------------------------------------------------------


def test_bfcl_loader_parses_fixture():
    cases = load_bfcl_style()
    assert 8 <= len(cases) <= 12
    for case in cases:
        assert case.suite == "bfcl_style"
        assert case.metadata["source_style"] == "BFCL"
        assert case.input["request"]
        catalog = case.input["catalog"]
        assert 2 <= len(catalog) <= 4
        names = {tool["name"] for tool in catalog}
        assert case.expected["tool"] in names  # ground truth always selectable
        for tool in catalog:
            assert tool["input_schema"].get("type") == "object"


def test_finmcp_loader_parses_fixture():
    cases = load_finmcp_style()
    assert 8 <= len(cases) <= 12
    for case in cases:
        assert case.suite == "finmcp_style"
        user_turns = [t for t in case.input["turns"] if t["role"] == "user"]
        assert len(user_turns) == len(case.expected["tool_sequence"]) >= 2


def test_secque_loader_parses_fixture():
    cases = load_secque_style()
    assert 8 <= len(cases) <= 12
    for case in cases:
        assert case.suite == "secque_style"
        assert case.input["context"] and case.input["question"]
        # every expected fact is genuinely present in the filing excerpt
        for fact in case.expected["key_facts"]:
            assert fact.lower() in case.input["context"].lower()


# --- suites over the adapter cases ------------------------------------------------


async def test_bfcl_cases_run_through_exactness_suite():
    suite = ToolCallExactnessSuite(cases=load_bfcl_style(), name="bfcl_style")
    results = await suite.run(RuleBasedSubject())
    summary = summarise(suite, results)
    assert summary.n_cases == len(suite.cases)
    # the deterministic baseline must at least pick the right function most of the time
    assert summary.metrics["tool_match_rate"] >= 0.8
    assert summary.metrics["hallucinated_tool_rate"] == 0.0


async def test_finmcp_sequences_score_positionally():
    suite = MultiTurnToolSuite()
    results = await suite.run(SequentialRouterSubject())
    summary = summarise(suite, results)
    assert summary.mean_score >= 0.8, [
        (r.case_id, r.details) for r in results if r.score < 1.0
    ]


async def test_finmcp_partial_sequence_credit():
    case = EvalCase(
        id="seq-partial",
        suite="finmcp_style",
        input={"turns": []},
        expected={"tool_sequence": ["a", "b", "c"]},
    )
    suite = MultiTurnToolSuite(cases=[case])

    async def subject(_case):
        return {"tool_sequence": ["a", "x", "c"]}

    results = await suite.run(subject)
    assert results[0].score == round(2 / 3, 6)
    assert results[0].passed is False


async def test_secque_extractive_baseline_finds_key_facts():
    suite = FilingsQASuite()
    results = await suite.run(ExtractiveQASubject())
    summary = summarise(suite, results)
    assert summary.mean_score >= 0.8, [
        (r.case_id, r.details["missing_facts"]) for r in results if r.score < 1.0
    ]


async def test_secque_scores_missing_facts():
    case = EvalCase(
        id="qa-miss",
        suite="secque_style",
        input={"context": "Revenue was $5 billion. Costs were $4 billion.",
               "question": "What was revenue?"},
        expected={"key_facts": ["$5 billion", "$9 billion"]},
    )
    suite = FilingsQASuite(cases=[case])
    results = await suite.run(ExtractiveQASubject())
    assert results[0].score == 0.5
    assert results[0].details["missing_facts"] == ["$9 billion"]
