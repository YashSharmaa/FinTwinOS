"""Typed tool layer: registry, JSON-RPC envelope, and the four tool bands.

Domain tool catalogs live in ``fintwinos.tools.catalog`` and ``fintwinos.tools.domains``;
the MCP-compatible HTTP server lives in ``fintwinos.tools.server``.
"""

from fintwinos.tools.envelope import jsonrpc_error, jsonrpc_result, mcp_call, parse_jsonrpc_request
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

__all__ = [
    "CallContext",
    "ToolRegistry",
    "ToolSpec",
    "jsonrpc_error",
    "jsonrpc_result",
    "mcp_call",
    "parse_jsonrpc_request",
]
