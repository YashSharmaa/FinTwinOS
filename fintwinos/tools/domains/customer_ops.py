"""Customer operations tool pack.

Tools registered here:

- ``observe_queue_state`` — work-queue depth, load and SLA risk per queue.
- ``simulate_staffing_change`` — runs the ``customer_ops`` simulator for a staffing
  delta and returns wait-time / SLA metrics with confidence intervals.
- ``propose_response_draft`` — deterministic, compliance-safe customer response draft
  built from templates (no promises of outcomes, mandatory regulatory footer).
- ``execute_send_response`` — queues an outbound response as a JSON document in the
  local outbox; reversible while queued, human approval always required.
"""

from __future__ import annotations

from typing import Any

from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import RiskTier, Scenario, SideEffectClass, ToolBand, new_id
from fintwinos.tools.domains.common import (
    derive_seed,
    entities_of_type,
    get_entity_attributes,
    now_iso,
    object_schema,
    provenance_entry,
    run_simulator,
    simulation_payload,
    write_outbox_json,
)
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

OWNER = "customer_ops"

_CATEGORY_SUBJECTS = {
    "complaint": "Update on your complaint",
    "query": "Response to your recent enquiry",
    "dispute": "Update on your transaction dispute",
    "kyc_refresh": "Action required: information needed to keep your account up to date",
}

_CATEGORY_OPENINGS = {
    "complaint": (
        "Thank you for bringing your concerns to our attention. We take every complaint "
        "seriously and have logged your case for investigation under our complaints "
        "handling procedure."
    ),
    "query": (
        "Thank you for contacting us. Please find below the information relating to "
        "your enquiry."
    ),
    "dispute": (
        "Thank you for reporting this transaction. We have registered a formal dispute "
        "and temporarily flagged the transaction while our investigation proceeds."
    ),
    "kyc_refresh": (
        "As part of our regulatory obligations we periodically refresh the information "
        "we hold about our customers. We need some up-to-date details from you."
    ),
}

_CATEGORY_NEXT_STEPS = {
    "complaint": (
        "We aim to resolve complaints promptly. You will receive a written update or our "
        "final response within the regulatory timeframe, and we will contact you if we "
        "need further information."
    ),
    "query": (
        "If this does not fully answer your question, simply reply to this message and a "
        "member of the team will follow up."
    ),
    "dispute": (
        "We will contact you with the outcome of the investigation. No further action is "
        "needed from you at this stage unless we request additional evidence."
    ),
    "kyc_refresh": (
        "Please provide the requested information within 30 days using your secure "
        "portal. If we do not hear from you, we may be required to restrict the account "
        "until the refresh is complete."
    ),
}

_COMPLIANCE_FOOTER = (
    "This message is for information only and does not constitute financial advice. "
    "Nothing in this message is a commitment to a particular outcome. If you remain "
    "dissatisfied with our response you may be entitled to refer the matter to the "
    "relevant financial ombudsman or dispute-resolution scheme free of charge. "
    "Please never share passwords, one-time codes or other security credentials in "
    "any reply."
)


def _load_queues(runtime: TwinRuntime) -> list[dict[str, Any]]:
    """Load queue entities with deterministic derived load and SLA statistics."""
    rows: list[dict[str, Any]] = []
    for ref, attrs in entities_of_type(runtime.graph, "queue"):
        depth = float(attrs.get("depth", 0.0))
        arrival = float(attrs.get("arrival_rate_per_hour", 0.0))
        handle = float(attrs.get("avg_handle_minutes", 0.0))
        agents = float(attrs.get("agents_on_shift", 0.0))
        sla = float(attrs.get("sla_minutes", 0.0))
        load_erlangs = round(arrival * handle / 60.0, 4)
        utilisation = round(load_erlangs / agents, 4) if agents > 0 else None
        est_wait = round(depth * handle / agents, 2) if agents > 0 else None
        rows.append(
            {
                "queue_id": ref.entity_id,
                "name": attrs.get("name", ref.entity_id),
                "depth": int(depth),
                "arrival_rate_per_hour": arrival,
                "avg_handle_minutes": handle,
                "agents_on_shift": int(agents),
                "sla_minutes": sla,
                "offered_load_erlangs": load_erlangs,
                "utilisation": utilisation,
                "estimated_wait_minutes": est_wait,
                "sla_at_risk": bool(est_wait is not None and sla > 0 and est_wait > sla),
            }
        )
    rows.sort(key=lambda r: r["queue_id"])
    return rows


def register(registry: ToolRegistry, runtime: TwinRuntime) -> None:
    """Register the customer-operations tool pack on the registry against the runtime."""

    # -- observe_queue_state --------------------------------------------------------

    def observe_queue_state(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        queues = _load_queues(runtime)
        queue_id = arguments.get("queue_id")
        if queue_id is not None:
            queues = [q for q in queues if q["queue_id"] == queue_id]
            if not queues:
                raise ValueError(f"queue '{queue_id}' not found in the twin")
        return {
            "as_of": now_iso(),
            "count": len(queues),
            "queues": queues,
            "queues_at_risk": [q["queue_id"] for q in queues if q["sla_at_risk"]],
            "provenance": [provenance_entry("twin_core.graph", "entity_type:queue")],
        }

    registry.register(
        ToolSpec(
            name="observe_queue_state",
            description=(
                "Current state of customer-operations work queues: depth, arrival rate, "
                "handle time, staffing, derived utilisation, estimated wait and SLA risk."
            ),
            input_schema=object_schema(
                {
                    "queue_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Restrict to one queue.",
                    }
                }
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_queue_state,
    )

    # -- simulate_staffing_change ------------------------------------------------------

    def simulate_staffing_change(
        arguments: dict[str, Any], context: CallContext
    ) -> dict[str, Any]:
        queue_id = arguments["queue_id"]
        delta_agents = int(arguments["delta_agents"])
        horizon_hours = int(arguments.get("horizon_hours", 8))
        queues = {q["queue_id"]: q for q in _load_queues(runtime)}
        if queues and queue_id not in queues:
            raise ValueError(
                f"queue '{queue_id}' not found in the twin (known: {', '.join(sorted(queues))})"
            )
        baseline = queues.get(queue_id, {})
        # The simulator's staffing knob is ``n_agents`` (see twin_sim.customer_ops);
        # translate the requested delta against the queue's observed baseline staffing.
        baseline_agents = int(baseline.get("agents_on_shift") or 0) or 8
        params: dict[str, Any] = {
            "queue_id": queue_id,
            "delta_agents": delta_agents,
            "baseline_agents": baseline_agents,
            "n_agents": max(baseline_agents + delta_agents, 1),
            "horizon_hours": horizon_hours,
        }
        arrival = float(baseline.get("arrival_rate_per_hour") or 0.0)
        if arrival > 0:
            params["arrival_rate_per_hr"] = arrival
        sla_minutes = float(baseline.get("sla_minutes") or 0.0)
        if sla_minutes > 0:
            params["sla_minutes"] = sla_minutes
        scenario = Scenario(
            name=f"staffing_{queue_id}_{delta_agents:+d}",
            kind="what_if",
            params=params,
        )
        seed = arguments.get("seed")
        if seed is None:
            seed = derive_seed(
                registry.settings.seed, "customer_ops", queue_id, delta_agents, horizon_hours
            )
        result = run_simulator(runtime, "customer_ops", scenario, int(seed))
        return simulation_payload(result, scenario, "twin_sim.customer_ops")

    registry.register(
        ToolSpec(
            name="simulate_staffing_change",
            description=(
                "Run the customer-operations simulator for a staffing change on one queue "
                "and return wait-time / SLA metrics with confidence intervals and "
                "calibration. Deterministic for identical inputs."
            ),
            input_schema=object_schema(
                {
                    "queue_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Queue the staffing change applies to.",
                    },
                    "delta_agents": {
                        "type": "integer",
                        "minimum": -50,
                        "maximum": 50,
                        "description": "Agents added (positive) or removed (negative).",
                    },
                    "horizon_hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 72,
                        "description": "Simulation horizon in hours (default 8).",
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Override the deterministic derived seed.",
                    },
                },
                ["queue_id", "delta_agents"],
            ),
            band=ToolBand.simulate,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.none,
            provenance_required=True,
            owner=OWNER,
        ),
        simulate_staffing_change,
    )

    # -- propose_response_draft ----------------------------------------------------------

    def propose_response_draft(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        customer_id = arguments["customer_id"]
        category = arguments["category"]
        case_id = arguments.get("case_id")
        key_points = [str(point) for point in arguments.get("key_points", [])]

        customer = get_entity_attributes(runtime.graph, "customer", customer_id) or {}
        display_name = str(customer.get("name", "Customer"))

        subject = _CATEGORY_SUBJECTS[category]
        if case_id:
            subject = f"{subject} (case {case_id})"

        lines: list[str] = [f"Dear {display_name},", "", _CATEGORY_OPENINGS[category]]
        if case_id:
            lines += ["", f"Your reference for this matter is case {case_id}."]
        if key_points:
            lines += ["", "In particular:"]
            lines += [f"- {point}" for point in key_points]
        lines += ["", _CATEGORY_NEXT_STEPS[category]]
        lines += ["", _COMPLIANCE_FOOTER, "", "Kind regards,", "Customer Operations Team"]
        body = "\n".join(lines)

        return {
            "customer_id": customer_id,
            "case_id": case_id,
            "category": category,
            "subject": subject,
            "body": body,
            "requires_human_review": True,
            "generated_by": "rule_based_template",
            "provenance": [
                provenance_entry(
                    "fintwinos.tools.customer_ops",
                    f"deterministic response template:{category}",
                )
            ],
        }

    registry.register(
        ToolSpec(
            name="propose_response_draft",
            description=(
                "Draft a compliant customer response from deterministic templates: "
                "category-appropriate opening and next steps, caller-supplied key points, "
                "and a mandatory regulatory footer. Never sends anything; pair with "
                "execute_send_response after human review."
            ),
            input_schema=object_schema(
                {
                    "customer_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Customer the response is addressed to.",
                    },
                    "case_id": {
                        "type": "string",
                        "description": "Related case reference, if any.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["complaint", "query", "dispute", "kyc_refresh"],
                        "description": "Interaction category driving the template.",
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 10,
                        "description": "Specific points to include as bullets.",
                    },
                },
                ["customer_id", "category"],
            ),
            band=ToolBand.propose,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.none,
            owner=OWNER,
        ),
        propose_response_draft,
    )

    # -- execute_send_response ----------------------------------------------------------

    def execute_send_response(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        warnings: list[str] = []
        customer_id = arguments["customer_id"]
        if get_entity_attributes(runtime.graph, "customer", customer_id) is None:
            warnings.append(f"customer '{customer_id}' is not a known customer in the twin")

        approval = context.approval
        response_id = new_id("rsp")
        record = {
            "response_id": response_id,
            "record_type": "customer_response",
            "customer_id": customer_id,
            "case_id": arguments.get("case_id"),
            "channel": arguments["channel"],
            "subject": arguments["subject"],
            "body": arguments["body"],
            "ticket_id": arguments["ticket_id"],
            "requested_by": context.caller,
            "approved_by": approval.granted_by if approval else None,
            "approval_token_id": approval.token_id if approval else None,
            "status": "queued",
            "reversible": True,
            "created_at": now_iso(),
        }
        path = write_outbox_json(registry.settings, f"response_{response_id}.json", record)
        runtime.audit.append(
            context.caller,
            "outbox.response_queued",
            {
                "response_id": response_id,
                "customer_id": customer_id,
                "channel": arguments["channel"],
                "ticket_id": arguments["ticket_id"],
                "path": str(path),
            },
        )
        return {
            "response_id": response_id,
            "status": "queued",
            "outbox_path": str(path),
            "warnings": warnings,
            "provenance": [provenance_entry("fintwinos.outbox", f"response:{response_id}")],
        }

    registry.register(
        ToolSpec(
            name="execute_send_response",
            description=(
                "Queue an outbound customer response by writing a JSON document to the "
                "local outbox (never a real messaging system). Reversible while queued; "
                "always requires a human approval token, a policy allow rule and the "
                "execute-band enable flag."
            ),
            input_schema=object_schema(
                {
                    "customer_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Customer the response is addressed to.",
                    },
                    "case_id": {
                        "type": "string",
                        "description": "Related case reference, if any.",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["email", "secure_message", "letter"],
                        "description": "Delivery channel for the response.",
                    },
                    "subject": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                        "description": "Subject line of the response.",
                    },
                    "body": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Full response body, post human review.",
                    },
                    "ticket_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Workflow ticket authorising the send.",
                    },
                },
                ["customer_id", "channel", "subject", "body", "ticket_id"],
            ),
            band=ToolBand.execute,
            risk_tier=RiskTier.medium,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
            provenance_required=True,
            idempotent=False,
            owner=OWNER,
        ),
        execute_send_response,
    )
