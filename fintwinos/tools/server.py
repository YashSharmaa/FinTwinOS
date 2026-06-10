"""MCP-style JSON-RPC server exposing the FinTwinOS typed tool catalog over HTTP.

Endpoints:

- ``POST /rpc`` — JSON-RPC 2.0 with the MCP-flavoured methods ``initialize``,
  ``tools/list`` and ``tools/call`` (plus ``ping``). Tool failures come back as
  JSON-RPC error envelopes; calls blocked pending human approval use the dedicated
  error code :data:`APPROVAL_REQUIRED` (``-32001``) with
  ``data.requires_approval = true`` so clients can route them to an approval UI.
- ``GET /healthz`` — liveness plus audit-chain integrity.

The server adds **no** authority of its own: every call goes through
:meth:`fintwinos.tools.registry.ToolRegistry.call`, so schema validation, the hard
execute-band gates, the policy gate and the audit trail all apply identically over
HTTP and in-process.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import fintwinos
from fintwinos.core.types import ApprovalToken
from fintwinos.tools.envelope import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    jsonrpc_error,
    jsonrpc_result,
    parse_jsonrpc_request,
)
from fintwinos.tools.registry import CallContext, ToolRegistry

#: JSON-RPC error code for a generic tool failure (schema-valid call that errored).
TOOL_FAILED = -32000
#: JSON-RPC error code for a call blocked pending human approval / execute enablement.
APPROVAL_REQUIRED = -32001

#: MCP protocol revision this server speaks.
PROTOCOL_VERSION = "2024-11-05"

_INSTRUCTIONS = (
    "FinTwinOS typed tool catalog. Tools are banded observe_*/simulate_*/propose_*/"
    "execute_*; observe, simulate and propose are side-effect free, while execute "
    "requires a human approval token, an explicit policy allow rule and "
    "FINTWIN_EXECUTE_TOOLS_ENABLED=1, and only ever writes to a local outbox. "
    "Pass caller/ticket/approval details in params.context of tools/call."
)


def _default_registry() -> ToolRegistry:
    """Assemble the demo twin, register all simulators and build the catalog."""
    from fintwinos.tools.catalog import build_default_registry
    from fintwinos.twin_core.runtime import build_runtime
    from fintwinos.twin_sim import register_all

    runtime = build_runtime()
    register_all(runtime)
    return build_default_registry(runtime)


def _build_call_context(raw: dict[str, Any]) -> CallContext:
    """Construct a :class:`CallContext` from the ``params.context`` object.

    Raises:
        pydantic.ValidationError: when the embedded approval token is malformed.
    """
    approval_raw = raw.get("approval")
    approval: ApprovalToken | None = None
    if isinstance(approval_raw, dict):
        approval = ApprovalToken.model_validate(approval_raw)
    ticket_id = raw.get("ticket_id")
    idempotency_key = raw.get("idempotency_key")
    return CallContext(
        caller=str(raw.get("caller", "rpc-client")),
        ticket_id=str(ticket_id) if ticket_id is not None else None,
        approval=approval,
        dry_run=bool(raw.get("dry_run", False)),
        idempotency_key=str(idempotency_key) if idempotency_key is not None else None,
    )


def _classify_failure(request_id: Any, name: str, error: str, requires_approval: bool) -> dict:
    """Map a failed ToolResult onto the appropriate JSON-RPC error envelope."""
    if requires_approval:
        return jsonrpc_error(
            request_id,
            APPROVAL_REQUIRED,
            error,
            data={"requires_approval": True, "tool": name},
        )
    if "unknown tool" in error or "failed validation" in error:
        return jsonrpc_error(request_id, INVALID_PARAMS, error, data={"tool": name})
    return jsonrpc_error(request_id, TOOL_FAILED, error, data={"tool": name})


def create_app(registry: ToolRegistry | None = None) -> FastAPI:
    """Create the FastAPI application serving the typed tool catalog.

    Args:
        registry: An assembled :class:`ToolRegistry`. When ``None``, a default
            registry is built from ``fintwinos.twin_core.runtime.build_runtime()``
            plus ``fintwinos.twin_sim.register_all`` — the full demo twin.

    Returns:
        A FastAPI app with ``POST /rpc`` (JSON-RPC) and ``GET /healthz``.
    """
    if registry is None:
        registry = _default_registry()

    app = FastAPI(
        title="FinTwinOS tool server",
        version=fintwinos.__version__,
        description=(
            "MCP-style JSON-RPC surface over the FinTwinOS typed tool catalog. "
            + fintwinos.CREDIT
        ),
    )
    app.state.registry = registry

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness probe: tool count and audit-chain integrity."""
        return {
            "status": "ok",
            "version": fintwinos.__version__,
            "tools": len(registry),
            "audit_intact": registry.audit.verify(),
        }

    async def _handle_tools_call(request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            return jsonrpc_error(
                request_id, INVALID_PARAMS, "params.name must be a non-empty string"
            )
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return jsonrpc_error(request_id, INVALID_PARAMS, "params.arguments must be an object")
        raw_context = params.get("context") or {}
        if not isinstance(raw_context, dict):
            return jsonrpc_error(request_id, INVALID_PARAMS, "params.context must be an object")
        try:
            context = _build_call_context(raw_context)
        except ValidationError as exc:
            return jsonrpc_error(
                request_id, INVALID_PARAMS, f"invalid call context: {exc.error_count()} error(s)",
                data={"errors": json.loads(exc.json())},
            )

        result = await registry.call(name, arguments, context)
        if result.ok:
            return jsonrpc_result(request_id, result.model_dump(mode="json"))
        return _classify_failure(
            request_id, name, result.error or "tool call failed", result.requires_approval
        )

    @app.post("/rpc")
    async def rpc(request: Request) -> JSONResponse:
        """JSON-RPC 2.0 endpoint: initialize, tools/list, tools/call, ping."""
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(
                jsonrpc_error(None, PARSE_ERROR, "request body is not valid JSON")
            )
        if not isinstance(body, dict):
            return JSONResponse(
                jsonrpc_error(None, INVALID_REQUEST, "request must be a single JSON object")
            )
        parsed = parse_jsonrpc_request(body)
        if isinstance(parsed, dict):
            return JSONResponse(parsed)
        request_id, method, params = parsed

        if method == "initialize":
            envelope = jsonrpc_result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": {
                        "name": "fintwinos-tools",
                        "version": fintwinos.__version__,
                        "vendor": fintwinos.__author__,
                        "attribution": fintwinos.CREDIT,
                    },
                    "capabilities": {"tools": {"listChanged": False}},
                    "instructions": _INSTRUCTIONS,
                },
            )
        elif method == "tools/list":
            envelope = jsonrpc_result(request_id, {"tools": registry.public_catalog()})
        elif method == "tools/call":
            envelope = await _handle_tools_call(request_id, params)
        elif method == "ping":
            envelope = jsonrpc_result(request_id, {})
        else:
            envelope = jsonrpc_error(
                request_id, METHOD_NOT_FOUND, f"unknown method '{method}'"
            )
        return JSONResponse(envelope)

    return app
