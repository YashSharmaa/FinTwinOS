"""Task-class to model routing, with indicative cost estimation.

The institution should not pay frontier prices for classification, nor run planning on
a nano model. Routing is configuration-driven (see ``Settings``), so adopters can remap
tiers to any OpenAI models approved by their model-risk function.
"""

from __future__ import annotations

from enum import StrEnum

from fintwinos.core.config import Settings, get_settings


class TaskClass(StrEnum):
    planning = "planning"
    analysis = "analysis"
    critique = "critique"
    drafting = "drafting"
    extraction = "extraction"
    classification = "classification"
    cheap = "cheap"


# Indicative USD per 1M tokens (input, output). Estimates for budgeting/observability
# only — never for billing. Override via ModelRouter(price_table=...).
DEFAULT_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o4-mini": (1.10, 4.40),
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    price_table: dict[str, tuple[float, float]] | None = None,
) -> float:
    table = price_table or DEFAULT_PRICE_TABLE
    if model not in table:
        return 0.0
    in_price, out_price = table[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


class ModelRouter:
    PRIMARY_TASKS = {TaskClass.planning, TaskClass.analysis, TaskClass.critique}
    FAST_TASKS = {TaskClass.drafting, TaskClass.extraction}

    def __init__(self, settings: Settings | None = None,
                 price_table: dict[str, tuple[float, float]] | None = None):
        self.settings = settings or get_settings()
        self.price_table = price_table or DEFAULT_PRICE_TABLE

    def model_for(self, task: TaskClass) -> str:
        if task in self.PRIMARY_TASKS:
            return self.settings.llm_model_primary
        if task in self.FAST_TASKS:
            return self.settings.llm_model_fast
        return self.settings.llm_model_cheap

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return estimate_cost_usd(model, input_tokens, output_tokens, self.price_table)
