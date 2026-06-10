"""JSON-RPC 2.0 envelopes in the MCP style (``tools/list``, ``tools/call``)."""

from __future__ import annotations

from typing import Any

JSONRPC_VERSION = "2.0"

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def mcp_call(tool_name: str, arguments: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Build a transport-neutral ``tools/call`` request envelope."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }


def mcp_list(request_id: str) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "method": "tools/list", "params": {}}


def jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": err}


def parse_jsonrpc_request(body: dict[str, Any]) -> tuple[Any, str, dict[str, Any]] | dict[str, Any]:
    """Validate a JSON-RPC request body.

    Returns ``(id, method, params)`` on success, or a ready-to-send error envelope.
    """
    request_id = body.get("id")
    if body.get("jsonrpc") != JSONRPC_VERSION:
        return jsonrpc_error(request_id, INVALID_REQUEST, "jsonrpc must be '2.0'")
    method = body.get("method")
    if not isinstance(method, str):
        return jsonrpc_error(request_id, INVALID_REQUEST, "method must be a string")
    params = body.get("params") or {}
    if not isinstance(params, dict):
        return jsonrpc_error(request_id, INVALID_PARAMS, "params must be an object")
    return request_id, method, params
