"""Tool-call exactness benchmark: did the subject pick the right tool with the right args?

The suite ships its own frozen catalog (built from real :class:`ToolSpec` instances, so
band invariants are enforced) and ~25 handwritten natural-language cases in
``fixtures/function_calls.jsonl``. Freezing the catalog with the cases is deliberate —
exactness benchmarks are only meaningful when the answer key and the tool universe are
versioned together (the same convention BFCL uses).

Scoring per case (see :func:`score_prediction`):

- exact tool match ......... 0.5
- argument exactness ....... 0.5 (scaled by :func:`score_arguments`, type-checked;
  awarded only when the tool matches — arguments to the wrong tool are not comparable)
- ``hallucinated_tool`` is tracked per case (a predicted tool absent from the catalog)
  and rolled up as ``hallucinated_tool_rate`` in the suite summary.

Two subjects are provided:

- :class:`RuleBasedSubject` — a deterministic keyword router + typed argument
  extractor over the public catalog. No LLM, no network, no randomness: the floor
  baseline every model-based subject must beat, and the subject used offline.
- :class:`LlmSubject` — OpenAI function calling through the shared
  :class:`~fintwinos.models.llm_routing.client.LLMClient`; automatically delegates to
  :class:`RuleBasedSubject` when the client is offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fintwinos.core.types import RiskTier, SideEffectClass, ToolBand
from fintwinos.evals.harness import EvalCase, EvalResult, Suite
from fintwinos.models.llm_routing.client import LLMClient
from fintwinos.models.llm_routing.router import TaskClass
from fintwinos.tools.registry import ToolSpec

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FUNCTION_CALLS_FIXTURE = FIXTURES_DIR / "function_calls.jsonl"

# ---------------------------------------------------------------------------
# The frozen benchmark catalog
# ---------------------------------------------------------------------------


def _spec(name: str, description: str, schema: dict[str, Any], band: ToolBand,
          **kwargs: Any) -> ToolSpec:
    return ToolSpec(name=name, description=description, input_schema=schema, band=band, **kwargs)


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


DEFAULT_TOOLSPECS: list[ToolSpec] = [
    _spec(
        "observe_account_balance",
        "Read the current balance and currency of a deposit account in the twin.",
        _obj({"account_id": {"type": "string"}}, ["account_id"]),
        ToolBand.observe,
    ),
    _spec(
        "observe_positions",
        "List open positions and market values for an account.",
        _obj({"account_id": {"type": "string"}}, ["account_id"]),
        ToolBand.observe,
    ),
    _spec(
        "observe_customer_profile",
        "Fetch a customer profile with segment and risk rating.",
        _obj({"customer_id": {"type": "string"}}, ["customer_id"]),
        ToolBand.observe,
    ),
    _spec(
        "observe_case_file",
        "Retrieve a case record with its narrative and history.",
        _obj({"case_id": {"type": "string"}}, ["case_id"]),
        ToolBand.observe,
    ),
    _spec(
        "observe_market_series",
        "Read a market or operational time series window from the twin.",
        _obj(
            {"series_key": {"type": "string"},
             "window_days": {"type": "integer", "minimum": 1, "default": 30}},
            ["series_key"],
        ),
        ToolBand.observe,
    ),
    _spec(
        "observe_alert_queue",
        "List alerts filtered by status and severity.",
        _obj(
            {"status": {"type": "string",
                        "enum": ["new", "triaged", "investigating", "dismissed", "confirmed"]},
             "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]}},
            [],
        ),
        ToolBand.observe,
    ),
    _spec(
        "observe_liquidity_ladder",
        "Read the funding ladder and cash-flow buckets out to a horizon.",
        _obj({"horizon_days": {"type": "integer", "minimum": 1}}, ["horizon_days"]),
        ToolBand.observe,
    ),
    _spec(
        "simulate_market_stress",
        "Run a calibrated market stress scenario over the portfolio.",
        _obj(
            {"asset_class": {"type": "string", "enum": ["rates", "equity", "credit", "fx"]},
             "shock_bps": {"type": "integer"},
             "horizon_days": {"type": "integer", "minimum": 1}},
            ["asset_class", "shock_bps", "horizon_days"],
        ),
        ToolBand.simulate,
        risk_tier=RiskTier.medium,
    ),
    _spec(
        "simulate_liquidity_runoff",
        "Simulate deposit runoff and report survival days.",
        _obj(
            {"runoff_pct": {"type": "number", "minimum": 0},
             "horizon_days": {"type": "integer", "minimum": 1}},
            ["runoff_pct", "horizon_days"],
        ),
        ToolBand.simulate,
        risk_tier=RiskTier.medium,
    ),
    _spec(
        "simulate_aml_typology",
        "Generate synthetic transactions for an AML typology and score detection.",
        _obj(
            {"typology": {"type": "string",
                          "enum": ["structuring", "layering", "mule_network", "round_tripping"]},
             "n_transactions": {"type": "integer", "minimum": 1}},
            ["typology", "n_transactions"],
        ),
        ToolBand.simulate,
        risk_tier=RiskTier.medium,
    ),
    _spec(
        "simulate_ops_backlog",
        "Simulate customer operations backlog and SLA breaches under a staffing level.",
        _obj(
            {"staff_count": {"type": "integer", "minimum": 0},
             "horizon_days": {"type": "integer", "minimum": 1}},
            ["staff_count", "horizon_days"],
        ),
        ToolBand.simulate,
    ),
    _spec(
        "propose_case_action",
        "Draft a recommended next case action for human review.",
        _obj(
            {"case_id": {"type": "string"},
             "action": {"type": "string",
                        "enum": ["escalate", "close", "request_information", "file_sar"]}},
            ["case_id", "action"],
        ),
        ToolBand.propose,
        risk_tier=RiskTier.medium,
    ),
    _spec(
        "propose_limit_change",
        "Draft a trading limit change for human review.",
        _obj(
            {"limit_id": {"type": "string"}, "new_limit": {"type": "number", "minimum": 0}},
            ["limit_id", "new_limit"],
        ),
        ToolBand.propose,
        risk_tier=RiskTier.high,
    ),
    _spec(
        "propose_hedge",
        "Draft a hedge proposal on an instrument for a portfolio exposure.",
        _obj(
            {"instrument_id": {"type": "string"},
             "notional": {"type": "number", "minimum": 0},
             "direction": {"type": "string", "enum": ["long", "short"]}},
            ["instrument_id", "notional", "direction"],
        ),
        ToolBand.propose,
        risk_tier=RiskTier.high,
    ),
    _spec(
        "execute_send_customer_letter",
        "Send an approved letter to a customer via the local outbox.",
        _obj(
            {"customer_id": {"type": "string"},
             "template": {"type": "string",
                          "enum": ["complaint_ack", "kyc_refresh", "rate_change_notice"]}},
            ["customer_id", "template"],
        ),
        ToolBand.execute,
        risk_tier=RiskTier.high,
        side_effect=SideEffectClass.reversible,
        requires_human_approval=True,
    ),
    _spec(
        "execute_update_trading_limit",
        "Apply an approved trading limit change to the limits outbox.",
        _obj(
            {"limit_id": {"type": "string"}, "new_limit": {"type": "number", "minimum": 0}},
            ["limit_id", "new_limit"],
        ),
        ToolBand.execute,
        risk_tier=RiskTier.critical,
        side_effect=SideEffectClass.reversible,
        requires_human_approval=True,
    ),
]


def default_catalog() -> list[dict[str, Any]]:
    """The frozen benchmark catalog as MCP-style public tool dicts."""
    return [spec.to_public_dict() for spec in DEFAULT_TOOLSPECS]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "and", "or", "for", "of", "to", "in", "on", "with", "from", "by",
     "via", "is", "are", "be", "this", "that", "it", "as", "at", "into", "under", "out",
     "its", "their", "s", "so", "can", "i", "me", "all", "over"}
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalise(token: str) -> str:
    """Cheap symmetric singularisation: strip a trailing 's' from long tokens."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, lightly singularised."""
    return [_normalise(t) for t in _TOKEN_RE.findall(text.lower())]


# ---------------------------------------------------------------------------
# Routing: pick the best tool for a request
# ---------------------------------------------------------------------------

_BAND_PREFIXES = ("observe_", "simulate_", "propose_", "execute_")

BAND_CUES: dict[str, frozenset[str]] = {
    "observe": frozenset({"what", "show", "list", "read", "fetch", "check", "pull",
                          "plot", "open", "view", "get", "display", "has", "were"}),
    "simulate": frozenset({"simulate", "simulation", "stress", "run", "scenario",
                           "generate", "model"}),
    "propose": frozenset({"propose", "draft", "recommend", "recommendation", "suggest"}),
    "execute": frozenset({"send", "apply", "execute", "submit", "dispatch", "update"}),
}

_NAME_WEIGHT = 3.0
_DESC_WEIGHT = 1.0
_BAND_BONUS = 2.0
_MIN_ROUTE_SCORE = 3.0  # at least one tool-name token hit (or desc hit + band cue)


def _tool_band(tool: dict[str, Any]) -> str | None:
    band = tool.get("x_band")
    if band:
        return str(band)
    for prefix in _BAND_PREFIXES:
        if tool["name"].startswith(prefix):
            return prefix.rstrip("_")
    return None


def _name_tokens(tool: dict[str, Any]) -> set[str]:
    name = tool["name"]
    for prefix in _BAND_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return set(tokenize(name.replace("_", " ")))


def route_request(request: str, catalog: list[dict[str, Any]]) -> str | None:
    """Deterministic keyword routing over the catalog; ``None`` when nothing fits.

    Score per tool = 3 x (request tokens hitting the tool-name tokens)
    + 1 x (request tokens hitting description tokens not already in the name)
    + 2 if the request contains a cue word for the tool's band. Ties keep the first
    tool in name-sorted order, so routing is fully deterministic.
    """
    req_tokens = set(tokenize(request))
    best_name: str | None = None
    best_score = 0.0
    for tool in sorted(catalog, key=lambda t: t["name"]):
        name_tokens = _name_tokens(tool)
        desc_tokens = set(tokenize(tool.get("description", ""))) - STOPWORDS - name_tokens
        score = (
            _NAME_WEIGHT * len(name_tokens & req_tokens)
            + _DESC_WEIGHT * len(desc_tokens & req_tokens)
        )
        band = _tool_band(tool)
        if band in BAND_CUES and BAND_CUES[band] & req_tokens:
            score += _BAND_BONUS
        if score > best_score:
            best_score = score
            best_name = tool["name"]
    if best_score < _MIN_ROUTE_SCORE:
        return None
    return best_name


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

_ID_TOKEN_RE = re.compile(r"\b[A-Z]{2,6}(?:-[A-Z0-9]+)+\b")
_SERIES_RE = re.compile(r"\b[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\b")
_NUMBER_RE = re.compile(r"(?<![\w.\-])(\d+(?:\.\d+)?)(?![\w\-])")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
_CIK_RE = re.compile(r"\b\d{7,10}\b")
_ALLCAPS_RE = re.compile(r"\b[A-Z]{1,5}\b")
_FX_PAIR_RE = re.compile(r"\b[A-Z]{6}\b")
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")

CURRENCIES: frozenset[str] = frozenset(
    {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "CNY", "SEK", "NZD"}
)

_ID_PREFIXES: dict[str, str] = {
    "account_id": "ACC", "customer_id": "CUS", "case_id": "CASE", "instrument_id": "INS",
    "limit_id": "LIM", "trade_id": "TRD", "scenario_id": "SCN", "alert_id": "ALR",
}

_NUMERIC_SYNONYMS: dict[str, set[str]] = {
    "bps": {"basis"},
    "pct": {"percent", "%", "pc"},
    "days": {"horizon"},
    "window": {"last"},
}

_GENERIC_NUMERIC_PARTS = {"n", "num", "value"}
_MAX_KEYWORD_GAP = 40  # characters between a number and a supporting keyword


def _numeric_keywords(prop_name: str) -> set[str]:
    keywords: set[str] = set()
    for part in prop_name.lower().split("_"):
        if not part or part in _GENERIC_NUMERIC_PARTS:
            continue
        keywords.add(part)
        keywords.add(part[:-1] if part.endswith("s") and len(part) > 3 else part + "s")
        keywords |= _NUMERIC_SYNONYMS.get(part, set())
    return keywords


def _keyword_spans(lowered: str, keywords: set[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for kw in keywords:
        start = 0
        while True:
            idx = lowered.find(kw, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(kw)))
            start = idx + 1
    return spans


def _gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[1] <= b[0]:
        return b[0] - a[1]
    if b[1] <= a[0]:
        return a[0] - b[1]
    return 0


def _cast_number(raw: str, schema_type: str) -> int | float:
    value = float(raw)
    if schema_type == "integer" or value.is_integer():
        return int(value)
    return value


def _extract_number(request: str, prop_name: str, schema: dict[str, Any]) -> int | float | None:
    """Pick the number whose nearest property keyword is closest (within a window)."""
    numbers = [(m.group(1), m.span(1)) for m in _NUMBER_RE.finditer(request)]
    if not numbers:
        return None
    keywords = _numeric_keywords(prop_name)
    spans = _keyword_spans(request.lower(), keywords)
    best_raw: str | None = None
    best_gap = _MAX_KEYWORD_GAP + 1
    for raw, num_span in numbers:
        for kw_span in spans:
            gap = _gap(num_span, kw_span)
            if gap < best_gap:
                best_gap = gap
                best_raw = raw
    if best_raw is not None and best_gap <= _MAX_KEYWORD_GAP:
        return _cast_number(best_raw, schema.get("type", "number"))
    if len(numbers) == 1:  # a lone number with one numeric slot is unambiguous
        return _cast_number(numbers[0][0], schema.get("type", "number"))
    return None


def _part_matches(part: str, token: str) -> bool:
    """A token carries an enum part when it starts with it (light e-drop stemming:
    ``close`` also matches ``closing``, ``escalate`` matches ``escalating``)."""
    if token.startswith(part):
        return True
    return len(part) > 3 and part.endswith("e") and token.startswith(part[:-1])


def _match_enum(request: str, enum_values: list[Any]) -> Any | None:
    """Match an enum value when every underscore/hyphen part prefixes a request token."""
    tokens = _TOKEN_RE.findall(request.lower())
    for value in enum_values:
        parts = [p for p in re.split(r"[^a-z0-9]+", str(value).lower()) if p]
        if parts and all(any(_part_matches(part, tok) for tok in tokens) for part in parts):
            return value
    return None


def _extract_string(request: str, prop_name: str) -> str | None:
    name = prop_name.lower()
    if name in _ID_PREFIXES or name.endswith("_id"):
        ids = _ID_TOKEN_RE.findall(request)
        prefix = _ID_PREFIXES.get(name)
        if prefix is not None:
            for candidate in ids:
                if candidate.startswith(prefix + "-"):
                    return candidate
        return ids[0] if ids else None
    if name in {"series_key", "series"}:
        match = _SERIES_RE.search(request)
        return match.group(0) if match else None
    if name in {"currency"}:
        for token in re.findall(r"\b[A-Z]{3}\b", request):
            if token in CURRENCIES:
                return token
        return None
    if name in {"from_currency", "to_currency", "base", "quote", "base_currency",
                "quote_currency"}:
        found = [t for t in re.findall(r"\b[A-Z]{3}\b", request) if t in CURRENCIES]
        first = name in {"from_currency", "base", "base_currency"}
        if first:
            return found[0] if found else None
        return found[1] if len(found) > 1 else None
    if name in {"isin"}:
        match = _ISIN_RE.search(request)
        return match.group(0) if match else None
    if name in {"cik"}:
        match = _CIK_RE.search(request)
        return match.group(0) if match else None
    if name in {"pair"}:
        match = _FX_PAIR_RE.search(request)
        return match.group(0) if match else None
    if name in {"date", "start_date", "end_date", "value_date"}:
        match = _ISO_DATE_RE.search(request)
        return match.group(0) if match else None
    if name in {"symbol", "ticker"}:
        for token in _ALLCAPS_RE.findall(request):
            if len(token) >= 2 and token not in CURRENCIES:
                return token
        return None
    if name in {"query", "q", "search_term"}:
        match = re.search(r"\babout\s+(.+?)[.?!]?$", request, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    quoted = _QUOTED_RE.search(request)
    return quoted.group(1) if quoted else None


def extract_arguments(request: str, input_schema: dict[str, Any]) -> dict[str, Any]:
    """Deterministically extract typed arguments for one tool from a natural request.

    Only emits arguments it actually found — never guesses or fills defaults, so a
    miss shows up as a missing key in the exactness score rather than a fabrication.
    """
    arguments: dict[str, Any] = {}
    properties: dict[str, Any] = input_schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        if "enum" in prop_schema:
            value = _match_enum(request, list(prop_schema["enum"]))
            if value is not None:
                arguments[prop_name] = value
            continue
        prop_type = prop_schema.get("type", "string")
        if prop_type in {"integer", "number"}:
            number = _extract_number(request, prop_name, prop_schema)
            if number is not None:
                arguments[prop_name] = number
        elif prop_type == "string":
            text = _extract_string(request, prop_name)
            if text is not None:
                arguments[prop_name] = text
        elif prop_type == "boolean":
            keywords = _numeric_keywords(prop_name)
            tokens = set(tokenize(request))
            if keywords & tokens and ({"include", "with"} & set(tokenize(request))):
                arguments[prop_name] = True
    return arguments


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def value_matches(expected: Any, predicted: Any) -> bool:
    """Type-checked exact value comparison.

    - booleans only match booleans (``True != 1`` here, deliberately);
    - ints and floats cross-match when numerically equal (JSON number semantics);
    - lists match pairwise, dicts match on identical keys and matching values;
    - everything else requires identical type and equality.
    """
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return isinstance(expected, bool) and isinstance(predicted, bool) and expected == predicted
    if isinstance(expected, int | float) and isinstance(predicted, int | float):
        return float(expected) == float(predicted)
    if type(expected) is not type(predicted):
        return False
    if isinstance(expected, list):
        return len(expected) == len(predicted) and all(
            value_matches(e, p) for e, p in zip(expected, predicted, strict=True)
        )
    if isinstance(expected, dict):
        return set(expected) == set(predicted) and all(
            value_matches(expected[k], predicted[k]) for k in expected
        )
    return bool(expected == predicted)


def score_arguments(expected: dict[str, Any], predicted: dict[str, Any]) -> float:
    """Fraction of the union of argument keys that match exactly (with type checks).

    Using the key *union* penalises both omissions and hallucinated extra arguments
    symmetrically. Two empty dicts score 1.0.
    """
    keys = set(expected) | set(predicted)
    if not keys:
        return 1.0
    correct = sum(
        1 for k in keys
        if k in expected and k in predicted and value_matches(expected[k], predicted[k])
    )
    return correct / len(keys)


def score_prediction(
    case: EvalCase,
    prediction: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> EvalResult:
    """Score one tool-call prediction: 0.5 for the exact tool, 0.5 for exact arguments."""
    expected_tool = case.expected.get("tool")
    expected_args: dict[str, Any] = case.expected.get("arguments", {}) or {}
    predicted_tool = prediction.get("tool")
    predicted_args: dict[str, Any] = prediction.get("arguments") or {}

    tool_match = predicted_tool == expected_tool
    arg_score = score_arguments(expected_args, predicted_args) if tool_match else 0.0
    score = (0.5 if tool_match else 0.0) + 0.5 * arg_score
    catalog_names = {tool["name"] for tool in catalog}
    hallucinated = predicted_tool is not None and predicted_tool not in catalog_names
    return EvalResult(
        case_id=case.id,
        passed=score >= 0.999,
        score=round(score, 6),
        details={
            "expected_tool": expected_tool,
            "predicted_tool": predicted_tool,
            "tool_match": tool_match,
            "argument_score": round(arg_score, 6),
            "predicted_arguments": predicted_args,
            "hallucinated_tool": hallucinated,
        },
    )


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------


class RuleBasedSubject:
    """Deterministic keyword router + argument extractor over the public catalog.

    The offline floor baseline: no LLM, no network, no randomness. A per-case catalog
    in ``case.input["catalog"]`` (as in BFCL-style cases) overrides the bound one.
    """

    name = "rule_based"

    def __init__(self, catalog: list[dict[str, Any]] | None = None):
        self.catalog = catalog if catalog is not None else default_catalog()

    async def __call__(self, case: EvalCase) -> dict[str, Any]:
        catalog = case.input.get("catalog") or self.catalog
        request = str(case.input.get("request", ""))
        tool_name = route_request(request, catalog)
        if tool_name is None:
            return {"tool": None, "arguments": {}}
        schema = next(
            (t.get("input_schema", {}) for t in catalog if t["name"] == tool_name), {}
        )
        return {"tool": tool_name, "arguments": extract_arguments(request, schema)}


_LLM_SYSTEM_PROMPT = (
    "You are the tool-selection layer of FinTwinOS, an auditable digital twin of a "
    "financial institution. Given a user request, call exactly one of the available "
    "tools with precisely the arguments stated in the request. Never invent tool "
    "names or argument values that are not in the request."
)


class LlmSubject:
    """OpenAI function-calling subject; delegates to the rule-based baseline offline.

    The delegation (rather than skipping) keeps offline eval runs meaningful and
    deterministic: the report then measures the floor baseline instead of nothing.
    """

    name = "llm"

    def __init__(self, llm: LLMClient, catalog: list[dict[str, Any]] | None = None):
        self.llm = llm
        self.catalog = catalog if catalog is not None else default_catalog()
        self.fallback = RuleBasedSubject(self.catalog)

    @staticmethod
    def _openai_tools(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
            for tool in catalog
        ]

    async def __call__(self, case: EvalCase) -> dict[str, Any]:
        if self.llm.offline:
            return await self.fallback(case)
        catalog = case.input.get("catalog") or self.catalog
        request = str(case.input.get("request", ""))
        response = await self.llm.complete(
            [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
            task=TaskClass.classification,
            tools=self._openai_tools(catalog),
        )
        if response.tool_calls:
            call = response.tool_calls[0]
            return {"tool": call.get("name"), "arguments": call.get("arguments") or {}}
        return {"tool": None, "arguments": {}}


# ---------------------------------------------------------------------------
# Fixture loading and the suite
# ---------------------------------------------------------------------------


def load_function_call_cases(path: Path | None = None) -> list[EvalCase]:
    """Load the handwritten function-call cases from the bundled JSONL fixture."""
    path = path or FUNCTION_CALLS_FIXTURE
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(
                EvalCase(
                    id=row["id"],
                    suite="function_calls",
                    input={"request": row["request"]},
                    expected={
                        "tool": row["expected_tool"],
                        "arguments": row.get("expected_arguments", {}),
                    },
                    metadata={"fixture_line": line_no, **row.get("metadata", {})},
                )
            )
    return cases


class ToolCallExactnessSuite(Suite):
    """Did the subject pick the right tool with exactly the right arguments?

    Defaults to the bundled fixture cases, the frozen catalog and the deterministic
    :class:`RuleBasedSubject`; pass ``subject=LlmSubject(llm)`` to benchmark a model.
    """

    def __init__(
        self,
        cases: list[EvalCase] | None = None,
        catalog: list[dict[str, Any]] | None = None,
        subject: Any | None = None,
        name: str = "function_calls",
    ):
        super().__init__(name, cases if cases is not None else load_function_call_cases())
        self.catalog = catalog if catalog is not None else default_catalog()
        self._subject = subject

    def default_subject(self) -> Any:
        return self._subject if self._subject is not None else RuleBasedSubject(self.catalog)

    async def evaluate_case(self, case: EvalCase, subject: Any) -> EvalResult:
        prediction = await subject(case)
        catalog = case.input.get("catalog") or self.catalog
        return score_prediction(case, prediction, catalog)

    def extra_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        n = len(results)
        if n == 0:
            return {}
        hallucinated = sum(1 for r in results if r.details.get("hallucinated_tool"))
        tool_matches = sum(1 for r in results if r.details.get("tool_match"))
        arg_scores = [float(r.details.get("argument_score", 0.0)) for r in results]
        return {
            "hallucinated_tool_rate": hallucinated / n,
            "tool_match_rate": tool_matches / n,
            "argument_score_mean": sum(arg_scores) / n,
        }
