"""YAML policy-pack loading for the FinTwinOS governance plane.

A *policy pack* is a YAML document that declares an ordered list of policy rules.
Packs let risk and compliance teams version-control policy as data, review it like
code, and load it into the foundation :class:`~fintwinos.policy.gates.PolicyGate`
without touching Python.

Pack format::

    name: treasury
    version: "1.0"
    description: Treasury transfer limits.
    rules:
      - name: treasury-deny-large-transfers
        description: Transfers above the desk limit are denied outright.
        effect: deny                      # allow | deny | require_approval
        bands: [execute]                  # optional; omit for any band
        tools: ["execute_post_transfer"]  # optional glob patterns
        min_risk_tier: high               # optional; rule applies at-or-above
        conditions:                       # optional operator conditions on arguments
          amount: {gt: 250000}

Conditions extend the base :class:`~fintwinos.policy.gates.PolicyRule` equality
semantics with operators, evaluated against the tool-call arguments:

- ``{"field": {"eq": v}}`` / bare scalar ``{"field": v}`` — equality
- ``{"field": {"ne": v}}`` — inequality
- ``{"field": {"in": [...]}}`` — membership
- ``{"field": {"lt"|"lte"|"gt"|"gte": x}}`` — numeric comparison

Field names may use dotted paths (``"details.amount"``) to reach into nested
argument dicts. A condition on a missing field never matches, and numeric
operators never match non-numeric values, so malformed arguments can never
*accidentally* satisfy an ``allow`` rule.

This module is part of the ``policy_gov`` plane; import directly from the
submodule::

    from fintwinos.policy.rules import load_pack, load_default_packs, gate_from_packs
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, field_validator
from pydantic import ValidationError as PydanticValidationError

from fintwinos.core.errors import FinTwinError
from fintwinos.core.types import RISK_ORDER, RiskTier, SideEffectClass, ToolBand
from fintwinos.policy.gates import PolicyGate, PolicyRule

#: Directory holding the policy packs shipped with FinTwinOS.
PACKS_DIR = Path(__file__).resolve().parent / "packs"

#: The condition operators a :class:`YamlRule` understands.
SUPPORTED_OPERATORS = frozenset({"eq", "ne", "in", "lt", "lte", "gt", "gte"})

_MISSING = object()

_RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "effect"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "effect": {"enum": ["allow", "deny", "require_approval"]},
        "bands": {
            "type": "array",
            "minItems": 1,
            "items": {"enum": [band.value for band in ToolBand]},
        },
        "tools": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "min_risk_tier": {"enum": [tier.value for tier in RiskTier]},
        "side_effects": {
            "type": "array",
            "minItems": 1,
            "items": {"enum": [se.value for se in SideEffectClass]},
        },
        "conditions": {"type": "object"},
    },
}

#: JSON Schema every policy-pack document must satisfy before rules are built.
PACK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "rules"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "version": {"type": ["string", "number"]},
        "description": {"type": "string"},
        "rules": {"type": "array", "minItems": 1, "items": _RULE_SCHEMA},
    },
}


class PolicyPackError(FinTwinError):
    """Raised when a policy pack cannot be parsed, validated or resolved."""


def _lookup_argument(arguments: dict[str, Any], path: str) -> Any:
    """Resolve a (possibly dotted) condition field path against call arguments.

    Returns the module-private ``_MISSING`` sentinel when the path cannot be
    resolved, which callers treat as "condition not satisfied".
    """
    if path in arguments:
        return arguments[path]
    current: Any = arguments
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _compare(operator: str, actual: Any, expected: Any) -> bool:
    """Evaluate one condition operator. Unknown / incomparable values never match."""
    if operator == "eq":
        return bool(actual == expected)
    if operator == "ne":
        return bool(actual != expected)
    if operator == "in":
        try:
            return actual in expected
        except TypeError:
            return False
    try:
        lhs, rhs = float(actual), float(expected)
    except (TypeError, ValueError):
        return False
    if operator == "lt":
        return lhs < rhs
    if operator == "lte":
        return lhs <= rhs
    if operator == "gt":
        return lhs > rhs
    if operator == "gte":
        return lhs >= rhs
    return False


class YamlRule(PolicyRule):
    """A :class:`PolicyRule` whose conditions support comparison operators.

    The base rule treats ``conditions`` as exact-equality constraints. This
    subclass normalises every condition into an operator mapping at validation
    time (bare scalars become ``{"eq": value}``) and overrides :meth:`matches`
    to evaluate the full operator set documented at module level. Because it
    *is* a ``PolicyRule``, lists of ``YamlRule`` plug straight into
    :class:`~fintwinos.policy.gates.PolicyGate`.
    """

    @field_validator("conditions", mode="before")
    @classmethod
    def _normalise_conditions(cls, value: Any) -> dict[str, dict[str, Any]]:
        """Normalise conditions to ``{field: {operator: operand}}`` and validate operators."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("conditions must be a mapping of field -> operator mapping")
        normalised: dict[str, dict[str, Any]] = {}
        for field_name, raw in value.items():
            if isinstance(raw, dict):
                if not raw:
                    raise ValueError(f"condition '{field_name}' declares no operators")
                unknown = sorted(set(raw) - SUPPORTED_OPERATORS)
                if unknown:
                    raise ValueError(
                        f"condition '{field_name}' uses unsupported operator(s) {unknown}; "
                        f"supported operators: {sorted(SUPPORTED_OPERATORS)}"
                    )
                if "in" in raw and not isinstance(raw["in"], (list, tuple)):
                    raise ValueError(
                        f"condition '{field_name}': the 'in' operator requires a list operand"
                    )
                normalised[field_name] = {op: raw[op] for op in raw}
            else:
                normalised[field_name] = {"eq": raw}
        return normalised

    def matches(self, spec: Any, arguments: dict[str, Any]) -> bool:
        """Mirror the base band/tool/tier matching, then evaluate operator conditions."""
        if self.bands is not None and spec.band not in self.bands:
            return False
        if self.tools is not None and not any(
            fnmatch.fnmatch(spec.name, pattern) for pattern in self.tools
        ):
            return False
        if self.min_risk_tier is not None and (
            RISK_ORDER[spec.risk_tier] < RISK_ORDER[self.min_risk_tier]
        ):
            return False
        if self.side_effects is not None and spec.side_effect not in self.side_effects:
            return False
        for field_path, operators in self.conditions.items():
            actual = _lookup_argument(arguments, field_path)
            if actual is _MISSING:
                return False
            for operator, expected in operators.items():
                if not _compare(operator, actual, expected):
                    return False
        return True


class PolicyPack(BaseModel):
    """A named, versioned bundle of :class:`YamlRule` definitions."""

    name: str
    version: str = "1.0"
    description: str = ""
    rules: list[YamlRule]

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, value: Any) -> str:
        return str(value)

    def to_yaml(self) -> str:
        """Serialise the pack back to canonical YAML (round-trips with :func:`load_pack`)."""
        body: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "rules": [rule.model_dump(mode="json", exclude_none=True) for rule in self.rules],
        }
        return yaml.safe_dump(body, sort_keys=False, default_flow_style=False)


def _validate_structure(data: dict[str, Any], source: str) -> None:
    """Validate the raw pack document against :data:`PACK_SCHEMA`, reporting every error."""
    validator = Draft202012Validator(PACK_SCHEMA)
    errors = sorted(
        validator.iter_errors(data),
        key=lambda err: (list(map(str, err.absolute_path)), err.message),
    )
    if errors:
        details = "; ".join(f"{err.json_path}: {err.message}" for err in errors)
        raise PolicyPackError(f"policy pack '{source}' failed schema validation: {details}")


def load_pack_model(path: Path | str) -> PolicyPack:
    """Load and validate a policy pack, returning the full :class:`PolicyPack` model.

    Raises :class:`PolicyPackError` with a precise message on any problem:
    missing file, malformed YAML, schema violations (with JSON paths),
    unsupported condition operators, or duplicate rule names.
    """
    path = Path(path)
    if not path.exists():
        raise PolicyPackError(f"policy pack not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyPackError(f"policy pack '{path}' is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyPackError(
            f"policy pack '{path}' must be a YAML mapping with 'name' and 'rules' keys"
        )
    _validate_structure(data, str(path))

    rules: list[YamlRule] = []
    seen: set[str] = set()
    for index, raw in enumerate(data["rules"]):
        label = raw.get("name") or f"#{index}"
        try:
            rule = YamlRule(**raw)
        except PydanticValidationError as exc:
            messages = "; ".join(err["msg"] for err in exc.errors())
            raise PolicyPackError(
                f"policy pack '{path}' rule '{label}' is invalid: {messages}"
            ) from exc
        if rule.name in seen:
            raise PolicyPackError(f"policy pack '{path}' contains duplicate rule name '{rule.name}'")
        seen.add(rule.name)
        rules.append(rule)

    return PolicyPack(
        name=data["name"],
        version=data.get("version", "1.0"),
        description=data.get("description", ""),
        rules=rules,
    )


def load_pack(path: Path | str) -> list[PolicyRule]:
    """Load a policy pack from ``path`` and return its rules in declared order.

    Every returned rule is a :class:`YamlRule` (a ``PolicyRule`` subclass), so the
    list can be handed directly to ``PolicyGate(rules=...)``.
    """
    return list(load_pack_model(path).rules)


def load_default_packs(packs_dir: Path | str | None = None) -> dict[str, list[PolicyRule]]:
    """Load every ``*.yaml`` pack shipped under ``fintwinos/policy/packs/``.

    Returns a mapping of pack name -> rules. Pack names must be unique across
    the directory; collisions raise :class:`PolicyPackError`.
    """
    directory = Path(packs_dir) if packs_dir is not None else PACKS_DIR
    if not directory.is_dir():
        raise PolicyPackError(f"policy pack directory not found: {directory}")
    packs: dict[str, list[PolicyRule]] = {}
    for pack_path in sorted(directory.glob("*.yaml")):
        pack = load_pack_model(pack_path)
        if pack.name in packs:
            raise PolicyPackError(
                f"duplicate policy pack name '{pack.name}' (second definition in {pack_path})"
            )
        packs[pack.name] = list(pack.rules)
    if not packs:
        raise PolicyPackError(f"no policy packs found in {directory}")
    return packs


def gate_from_packs(*names: str, packs_dir: Path | str | None = None) -> PolicyGate:
    """Build a :class:`PolicyGate` from one or more shipped packs, in the order given.

    With no names, every shipped pack is loaded in alphabetical order. Unknown
    pack names raise :class:`PolicyPackError` listing what is available.
    """
    available = load_default_packs(packs_dir=packs_dir)
    selected = list(names) if names else sorted(available)
    rules: list[PolicyRule] = []
    for name in selected:
        if name not in available:
            raise PolicyPackError(
                f"unknown policy pack '{name}'; available packs: {sorted(available)}"
            )
        rules.extend(available[name])
    return PolicyGate(rules=rules)
