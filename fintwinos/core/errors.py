"""Exception hierarchy for FinTwinOS."""

from __future__ import annotations


class FinTwinError(Exception):
    """Base class for all FinTwinOS errors."""


class ToolNotFound(FinTwinError):
    """Raised when a tool name is not present in the registry."""


class SchemaValidationError(FinTwinError):
    """Raised when tool arguments fail JSON Schema validation."""

    def __init__(self, tool: str, message: str):
        self.tool = tool
        super().__init__(f"arguments for tool '{tool}' failed validation: {message}")


class PolicyViolation(FinTwinError):
    """Raised when a policy gate denies an action outright."""

    def __init__(self, message: str, rule: str | None = None):
        self.rule = rule
        super().__init__(message)


class ApprovalRequired(FinTwinError):
    """Raised when an execute-band tool is invoked without a valid approval token."""

    def __init__(self, tool: str):
        self.tool = tool
        super().__init__(
            f"tool '{tool}' has side effects and requires a human approval token; "
            "attach an ApprovalToken to the call context or route the decision to the "
            "human approval gateway"
        )


class CalibrationError(FinTwinError):
    """Raised when a simulator's calibration diagnostics fall outside tolerances."""


class IngestionError(FinTwinError):
    """Raised when an event envelope cannot be ingested into the twin."""
