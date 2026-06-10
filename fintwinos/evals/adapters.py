"""Local-fixture adapters in the style of public benchmarks — no downloads, ever.

FinTwinOS ships small handwritten fixture files whose *case shape* mirrors three
public benchmark families, so the same harness, scoring and reporting code paths are
exercised that a full benchmark integration would use. Mapping to the originals:

- **BFCL-style** (``load_bfcl_style``) — mirrors the Berkeley Function-Calling
  Leaderboard's single-turn ``simple`` category: each row carries a natural-language
  ``question``, a per-case list of candidate ``functions`` (name / description /
  JSON-Schema ``parameters``, exactly OpenAI function-calling shape) and one
  ``ground_truth`` call. Cases run through the same
  :class:`~fintwinos.evals.function_calls.ToolCallExactnessSuite` scoring (0.5 tool +
  0.5 arguments) as BFCL's AST-match category. Our fixture is finance-flavoured and
  hand-written; swap the path for a converted BFCL export to run the real thing.
- **FinMCP-style** (``load_finmcp_style``) — mirrors multi-turn financial MCP
  tool-use benchmarks: a conversation of user turns over the FinTwinOS public tool
  catalog with an expected *tool sequence*. Scored by positional sequence accuracy in
  :class:`MultiTurnToolSuite` (exact sequence required to pass a case).
- **SECQUE-style** (``load_secque_style``) — mirrors SEC-filings question-answering
  benchmarks for financial analysts: a filing excerpt, an analyst question, and the
  key facts a correct answer must contain. Scored by key-fact recall in
  :class:`FilingsQASuite` against a deterministic extractive baseline subject.

All three loaders parse JSONL into :class:`~fintwinos.evals.harness.EvalCase` lists,
so any future real-benchmark converter only needs to emit the same row shape.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fintwinos.evals.function_calls import (
    STOPWORDS,
    default_catalog,
    route_request,
    tokenize,
)
from fintwinos.evals.harness import EvalCase, EvalResult, Suite

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BFCL_FIXTURE = FIXTURES_DIR / "bfcl_style.jsonl"
FINMCP_FIXTURE = FIXTURES_DIR / "finmcp_style.jsonl"
SECQUE_FIXTURE = FIXTURES_DIR / "secque_style.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_bfcl_style(path: Path | None = None) -> list[EvalCase]:
    """Parse a BFCL-style JSONL fixture into tool-call exactness cases.

    Row shape: ``{"id", "question", "functions": [{"name", "description",
    "parameters"}], "ground_truth": {"name", "arguments"}}``.
    """
    cases: list[EvalCase] = []
    for row in _read_jsonl(path or BFCL_FIXTURE):
        catalog = [
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object"}),
            }
            for fn in row["functions"]
        ]
        cases.append(
            EvalCase(
                id=row["id"],
                suite="bfcl_style",
                input={"request": row["question"], "catalog": catalog},
                expected={
                    "tool": row["ground_truth"]["name"],
                    "arguments": row["ground_truth"].get("arguments", {}),
                },
                metadata={"source_style": "BFCL", **row.get("metadata", {})},
            )
        )
    return cases


def load_finmcp_style(path: Path | None = None) -> list[EvalCase]:
    """Parse a FinMCP-style JSONL fixture into multi-turn tool-sequence cases.

    Row shape: ``{"id", "task", "turns": [{"role", "content"}],
    "expected_tool_sequence": [tool names]}``. Tools resolve against the FinTwinOS
    public catalog unless a row embeds its own ``catalog``.
    """
    cases: list[EvalCase] = []
    for row in _read_jsonl(path or FINMCP_FIXTURE):
        cases.append(
            EvalCase(
                id=row["id"],
                suite="finmcp_style",
                input={"turns": row["turns"], "catalog": row.get("catalog")},
                expected={"tool_sequence": row["expected_tool_sequence"]},
                metadata={"source_style": "FinMCP", "task": row.get("task", ""),
                          **row.get("metadata", {})},
            )
        )
    return cases


def load_secque_style(path: Path | None = None) -> list[EvalCase]:
    """Parse a SECQUE-style JSONL fixture into filings-QA key-fact cases.

    Row shape: ``{"id", "filing", "question", "expected_facts": [strings]}`` where
    each expected fact is a literal substring a correct answer must contain.
    """
    cases: list[EvalCase] = []
    for row in _read_jsonl(path or SECQUE_FIXTURE):
        cases.append(
            EvalCase(
                id=row["id"],
                suite="secque_style",
                input={"context": row["filing"], "question": row["question"]},
                expected={"key_facts": row["expected_facts"]},
                metadata={"source_style": "SECQUE", **row.get("metadata", {})},
            )
        )
    return cases


# ---------------------------------------------------------------------------
# FinMCP-style suite: multi-turn tool sequences
# ---------------------------------------------------------------------------


class SequentialRouterSubject:
    """Deterministic baseline: route each user turn independently over the catalog."""

    name = "sequential_router"

    def __init__(self, catalog: list[dict[str, Any]] | None = None):
        self.catalog = catalog if catalog is not None else default_catalog()

    async def __call__(self, case: EvalCase) -> dict[str, Any]:
        catalog = case.input.get("catalog") or self.catalog
        sequence = [
            route_request(str(turn.get("content", "")), catalog)
            for turn in case.input.get("turns", [])
            if turn.get("role") == "user"
        ]
        return {"tool_sequence": sequence}


class MultiTurnToolSuite(Suite):
    """Score predicted tool sequences against expected ones, position by position."""

    def __init__(
        self,
        cases: list[EvalCase] | None = None,
        catalog: list[dict[str, Any]] | None = None,
        subject: Any | None = None,
        name: str = "finmcp_style",
    ):
        super().__init__(name, cases if cases is not None else load_finmcp_style())
        self.catalog = catalog if catalog is not None else default_catalog()
        self._subject = subject

    def default_subject(self) -> Any:
        return self._subject if self._subject is not None else SequentialRouterSubject(self.catalog)

    async def evaluate_case(self, case: EvalCase, subject: Any) -> EvalResult:
        prediction = await subject(case)
        expected: list[str] = list(case.expected.get("tool_sequence", []))
        predicted: list[Any] = list(prediction.get("tool_sequence", []))
        denominator = max(len(expected), len(predicted), 1)
        matches = sum(
            1 for exp, pred in zip(expected, predicted, strict=False) if exp == pred
        )
        score = matches / denominator
        return EvalResult(
            case_id=case.id,
            passed=expected == predicted,
            score=round(score, 6),
            details={
                "expected_sequence": expected,
                "predicted_sequence": predicted,
                "positional_matches": matches,
            },
        )

    def extra_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}
        return {"mean_turn_accuracy": sum(r.score for r in results) / len(results)}


# ---------------------------------------------------------------------------
# SECQUE-style suite: filings QA scored by key-fact recall
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class ExtractiveQASubject:
    """Deterministic extractive baseline: top-k sentences by question-token overlap.

    Sentences are ranked by the count of non-stopword question tokens they contain
    (ties keep document order) and the top ``k`` are returned in document order — a
    classic extractive-QA floor that any LLM answerer must beat.
    """

    name = "extractive_qa"

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    async def __call__(self, case: EvalCase) -> dict[str, Any]:
        context = str(case.input.get("context", ""))
        question = str(case.input.get("question", ""))
        sentences = _split_sentences(context)
        question_tokens = set(tokenize(question)) - STOPWORDS
        ranked = sorted(
            range(len(sentences)),
            key=lambda i: (-len(set(tokenize(sentences[i])) & question_tokens), i),
        )
        chosen = sorted(ranked[: self.top_k])
        return {"answer": " ".join(sentences[i] for i in chosen)}


class FilingsQASuite(Suite):
    """Key-fact recall: every expected fact must appear verbatim in the answer to pass."""

    def __init__(
        self,
        cases: list[EvalCase] | None = None,
        subject: Any | None = None,
        name: str = "secque_style",
    ):
        super().__init__(name, cases if cases is not None else load_secque_style())
        self._subject = subject

    def default_subject(self) -> Any:
        return self._subject if self._subject is not None else ExtractiveQASubject()

    async def evaluate_case(self, case: EvalCase, subject: Any) -> EvalResult:
        prediction = await subject(case)
        answer = _normalise_text(str(prediction.get("answer", "")))
        facts: list[str] = list(case.expected.get("key_facts", []))
        found = [fact for fact in facts if _normalise_text(fact) in answer]
        missing = [fact for fact in facts if fact not in found]
        score = len(found) / len(facts) if facts else 1.0
        return EvalResult(
            case_id=case.id,
            passed=not missing,
            score=round(score, 6),
            details={"found_facts": found, "missing_facts": missing,
                     "answer_chars": len(answer)},
        )

    def extra_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}
        return {"fact_recall_mean": sum(r.score for r in results) / len(results)}
