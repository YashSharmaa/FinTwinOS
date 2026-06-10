"""Tests for the tool-call exactness suite: scoring math, subjects, fixtures."""

from __future__ import annotations

from fintwinos.core.config import Settings
from fintwinos.evals.function_calls import (
    LlmSubject,
    RuleBasedSubject,
    ToolCallExactnessSuite,
    default_catalog,
    extract_arguments,
    load_function_call_cases,
    route_request,
    score_arguments,
    score_prediction,
    value_matches,
)
from fintwinos.evals.harness import EvalCase, summarise
from fintwinos.models.llm_routing.client import LLMClient

OFFLINE_SETTINGS = Settings(offline=True, openai_api_key=None)


def _case(expected_tool: str = "observe_positions", arguments: dict | None = None) -> EvalCase:
    return EvalCase(
        id="t-1",
        suite="function_calls",
        input={"request": "irrelevant"},
        expected={"tool": expected_tool, "arguments": arguments or {}},
    )


# --- value/argument scoring math ------------------------------------------------


def test_value_matches_type_checks():
    assert value_matches(5, 5.0)            # JSON number semantics
    assert not value_matches(True, 1)       # bool is not a number here
    assert not value_matches(1, True)
    assert value_matches("x", "x")
    assert not value_matches("5", 5)
    assert value_matches([1, "a"], [1, "a"])
    assert not value_matches([1, "a"], [1, "b"])
    assert value_matches({"k": 1}, {"k": 1.0})
    assert not value_matches({"k": 1}, {"k": 1, "extra": 2})


def test_score_arguments_exact_partial_and_hallucinated_extras():
    assert score_arguments({}, {}) == 1.0
    assert score_arguments({"a": 1}, {"a": 1}) == 1.0
    assert score_arguments({"a": 1, "b": "x"}, {"a": 1}) == 0.5          # omission
    assert score_arguments({"a": 1}, {"a": 1, "b": 2}) == 0.5            # fabrication
    assert score_arguments({"a": 1}, {"a": 2}) == 0.0                     # wrong value
    assert score_arguments({"a": 1}, {"a": "1"}) == 0.0                   # wrong type


def test_score_prediction_components():
    catalog = default_catalog()
    case = _case("observe_positions", {"account_id": "ACC-1"})

    perfect = score_prediction(
        case, {"tool": "observe_positions", "arguments": {"account_id": "ACC-1"}}, catalog
    )
    assert perfect.passed and perfect.score == 1.0

    wrong_args = score_prediction(
        case, {"tool": "observe_positions", "arguments": {"account_id": "ACC-2"}}, catalog
    )
    assert not wrong_args.passed and wrong_args.score == 0.5

    wrong_tool = score_prediction(
        case, {"tool": "observe_account_balance", "arguments": {"account_id": "ACC-1"}}, catalog
    )
    assert not wrong_tool.passed and wrong_tool.score == 0.0

    hallucinated = score_prediction(case, {"tool": "observe_magic", "arguments": {}}, catalog)
    assert hallucinated.details["hallucinated_tool"] is True
    assert hallucinated.score == 0.0

    no_call = score_prediction(case, {"tool": None, "arguments": {}}, catalog)
    assert no_call.details["hallucinated_tool"] is False
    assert no_call.score == 0.0


# --- routing and extraction --------------------------------------------------------


def test_route_request_picks_obvious_tools():
    catalog = default_catalog()
    assert route_request("What is the balance on account ACC-1?", catalog) == (
        "observe_account_balance"
    )
    assert route_request(
        "Run a market stress for rates with a 200 bps shock over 10 days.", catalog
    ) == "simulate_market_stress"
    assert route_request("please make me a sandwich", catalog) is None


def test_extract_arguments_typed():
    schema = {
        "type": "object",
        "properties": {
            "account_id": {"type": "string"},
            "horizon_days": {"type": "integer"},
            "runoff_pct": {"type": "number"},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        },
        "required": [],
    }
    args = extract_arguments(
        "Stress account ACC-0042 with a 15 percent runoff over a 30 day horizon, "
        "treating severity as high.",
        schema,
    )
    assert args["account_id"] == "ACC-0042"
    assert args["horizon_days"] == 30 and isinstance(args["horizon_days"], int)
    assert args["runoff_pct"] == 15
    assert args["severity"] == "high"


def test_extract_arguments_ignores_digits_inside_identifiers():
    schema = {
        "type": "object",
        "properties": {"limit_id": {"type": "string"}, "new_limit": {"type": "number"}},
        "required": [],
    }
    args = extract_arguments(
        "Draft a limit change for LIM-FX-01 to a new limit of 2500000.", schema
    )
    assert args["limit_id"] == "LIM-FX-01"
    assert args["new_limit"] == 2500000  # not the "01" embedded in the identifier


# --- fixture + suite ------------------------------------------------------------


def test_fixture_has_around_25_well_formed_cases():
    cases = load_function_call_cases()
    assert 24 <= len(cases) <= 30
    catalog_names = {tool["name"] for tool in default_catalog()}
    assert len({c.id for c in cases}) == len(cases)
    for case in cases:
        assert case.input["request"]
        assert case.expected["tool"] in catalog_names


async def test_rule_based_subject_scores_at_least_090_with_zero_hallucinations():
    suite = ToolCallExactnessSuite()
    results = await suite.run(RuleBasedSubject())
    summary = summarise(suite, results)
    assert summary.mean_score >= 0.90, [
        (r.case_id, r.score, r.details) for r in results if r.score < 1.0
    ]
    assert summary.metrics["hallucinated_tool_rate"] == 0.0


async def test_rule_based_subject_is_deterministic():
    suite = ToolCallExactnessSuite()
    first = await suite.run(RuleBasedSubject())
    second = await suite.run(RuleBasedSubject())
    assert [(r.case_id, r.score) for r in first] == [(r.case_id, r.score) for r in second]


async def test_llm_subject_falls_back_to_rule_based_when_offline():
    llm = LLMClient(settings=OFFLINE_SETTINGS)
    assert llm.offline is True
    cases = load_function_call_cases()[:5]
    llm_subject = LlmSubject(llm)
    rule_subject = RuleBasedSubject()
    for case in cases:
        assert await llm_subject(case) == await rule_subject(case)


async def test_per_case_catalog_overrides_bound_catalog():
    tiny_catalog = [
        {
            "name": "get_weather",
            "description": "Get the weather forecast for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    case = EvalCase(
        id="override-1",
        suite="function_calls",
        input={"request": "Get the weather forecast for the city of 'Leeds'.",
               "catalog": tiny_catalog},
        expected={"tool": "get_weather", "arguments": {"city": "Leeds"}},
    )
    suite = ToolCallExactnessSuite(cases=[case])
    results = await suite.run(RuleBasedSubject())
    assert results[0].details["predicted_tool"] == "get_weather"
