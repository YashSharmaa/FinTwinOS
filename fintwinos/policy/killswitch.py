"""Kill switch: global and per-band halt flags, persisted and registry-enforced.

The kill switch is the platform's hard stop. State is a small JSON document at
``settings.data_dir / "killswitch.json"`` so it survives process restarts and
is shared by every process pointed at the same data directory:

- a **global** flag halts every tool band;
- **per-band** flags halt one band (most commonly ``execute``).

Enforcement happens at the registry boundary: :meth:`KillSwitch.guard` wraps a
:class:`~fintwinos.tools.registry.ToolRegistry`'s ``call`` so any halted call
is refused with ``ToolResult(ok=False, error="kill switch engaged: <reason>")``
before schema validation, policy or handlers ever run. The state file is
re-read on every check, so an engage/release in one process takes effect in
every guarded registry immediately — and an unreadable state file fails
*closed* (treated as a global halt).

Import directly from the submodule::

    from fintwinos.policy.killswitch import KillSwitch
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings, get_settings
from fintwinos.core.errors import ToolNotFound
from fintwinos.core.types import ToolBand, ToolResult, utcnow
from fintwinos.tools.registry import ToolRegistry

#: File name of the persisted state inside ``settings.data_dir``.
KILLSWITCH_FILENAME = "killswitch.json"

_GUARD_ATTR = "_killswitch_guard"


def _normalise_band(band: ToolBand | str | None) -> str | None:
    """Coerce a band argument to its canonical string value (or None for global)."""
    if band is None:
        return None
    if isinstance(band, ToolBand):
        return band.value
    try:
        return ToolBand(str(band).lower()).value
    except ValueError as exc:
        valid = ", ".join(b.value for b in ToolBand)
        raise ValueError(f"unknown tool band '{band}'; expected one of: {valid}") from exc


class KillSwitch:
    """Persisted global / per-band halt with registry-level enforcement.

    Parameters
    ----------
    settings:
        Supplies ``data_dir`` for the persisted state file. Defaults to the
        process settings.
    audit:
        Shared audit trail; every engage/release/blocked event is appended.
    path:
        Explicit state-file path, overriding ``settings.data_dir / "killswitch.json"``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        audit: AuditTrail | None = None,
        path: Path | str | None = None,
    ):
        self.settings = settings or get_settings()
        # NB: AuditTrail defines __len__, so an empty trail is falsy — compare to None.
        self.audit = audit if audit is not None else AuditTrail()
        self.path = Path(path) if path is not None else self.settings.data_dir / KILLSWITCH_FILENAME
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------------------

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {"global": {"engaged": False}, "bands": {}, "history": []}

    def _load(self) -> dict[str, Any]:
        """Read the persisted state, failing *closed* if the file is unreadable."""
        if not self.path.exists():
            return self._default_state()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state root must be a JSON object")
        except (OSError, ValueError):
            return {
                "global": {
                    "engaged": True,
                    "actor": "system",
                    "reason": f"kill switch state file unreadable ({self.path}); failing closed",
                    "at": utcnow().isoformat(),
                },
                "bands": {},
                "history": [],
            }
        state = self._default_state()
        if isinstance(data.get("global"), dict):
            state["global"] = data["global"]
        if isinstance(data.get("bands"), dict):
            state["bands"] = data["bands"]
        if isinstance(data.get("history"), list):
            state["history"] = data["history"]
        return state

    def _save(self, state: dict[str, Any]) -> None:
        """Atomically persist the state file (write temp, then replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- engage / release ----------------------------------------------------------------

    def engage(
        self, actor: str, reason: str, band: ToolBand | str | None = None
    ) -> dict[str, Any]:
        """Engage the kill switch globally (``band=None``) or for one band.

        ``reason`` is mandatory: it is persisted, audited, and echoed verbatim
        in every refused tool call so operators see *why* the platform halted.
        """
        if not reason or not reason.strip():
            raise ValueError("a reason is required to engage the kill switch")
        scope = _normalise_band(band)
        entry = {
            "engaged": True,
            "actor": actor,
            "reason": reason.strip(),
            "at": utcnow().isoformat(),
        }
        with self._lock:
            state = self._load()
            if scope is None:
                state["global"] = entry
            else:
                state["bands"][scope] = entry
            state["history"].append({"event": "engage", "scope": scope or "global", **entry})
            self._save(state)
        self.audit.append(
            actor,
            "killswitch.engaged",
            {"scope": scope or "global", "reason": reason.strip()},
        )
        return self.status()

    def release(
        self, actor: str, reason: str, band: ToolBand | str | None = None
    ) -> dict[str, Any]:
        """Release the global switch (``band=None``) or one band's switch."""
        if not reason or not reason.strip():
            raise ValueError("a reason is required to release the kill switch")
        scope = _normalise_band(band)
        entry = {
            "engaged": False,
            "actor": actor,
            "reason": reason.strip(),
            "at": utcnow().isoformat(),
        }
        with self._lock:
            state = self._load()
            if scope is None:
                state["global"] = entry
            else:
                state["bands"][scope] = entry
            state["history"].append({"event": "release", "scope": scope or "global", **entry})
            self._save(state)
        self.audit.append(
            actor,
            "killswitch.released",
            {"scope": scope or "global", "reason": reason.strip()},
        )
        return self.status()

    # -- queries -----------------------------------------------------------------------

    def is_engaged(self, band: ToolBand | str | None = None) -> bool:
        """True when halted: globally (``band=None``), or globally-or-for-``band``."""
        state = self._load()
        if state["global"].get("engaged"):
            return True
        scope = _normalise_band(band)
        if scope is None:
            return False
        return bool(state["bands"].get(scope, {}).get("engaged"))

    def blocking_reason(self, band: ToolBand | str) -> str | None:
        """The reason a call in ``band`` would be refused right now, or None."""
        state = self._load()
        if state["global"].get("engaged"):
            return state["global"].get("reason") or "global halt"
        scope = _normalise_band(band)
        flag = state["bands"].get(scope, {})
        if flag.get("engaged"):
            return flag.get("reason") or f"{scope} band halted"
        return None

    def status(self) -> dict[str, Any]:
        """A deep copy of the full persisted state (global flag, bands, history)."""
        return copy.deepcopy(self._load())

    # -- registry enforcement ----------------------------------------------------------------

    def guard(self, registry: ToolRegistry) -> ToolRegistry:
        """Wrap ``registry.call`` so halted calls are refused before dispatch.

        The wrapper resolves the tool's band from its spec and consults the
        persisted state on *every* call, so engage/release takes effect
        immediately, even across processes. Unknown tool names are delegated
        to the original ``call`` (which audits and refuses them itself).
        Idempotent: guarding an already-guarded registry is a no-op.
        """
        if getattr(registry, _GUARD_ATTR, None) is not None:
            return registry
        original = registry.call
        killswitch = self

        async def guarded_call(
            name: str, arguments: dict[str, Any], context: Any = None
        ) -> ToolResult:
            try:
                band = registry.spec(name).band
            except ToolNotFound:
                return await original(name, arguments, context)
            reason = killswitch.blocking_reason(band)
            if reason is not None:
                caller = getattr(context, "caller", None) or "anonymous"
                registry.audit.append(
                    caller,
                    "killswitch.blocked",
                    {"tool": name, "band": band.value, "reason": reason},
                )
                return ToolResult(ok=False, tool=name, error=f"kill switch engaged: {reason}")
            return await original(name, arguments, context)

        registry.call = guarded_call  # type: ignore[method-assign]
        setattr(registry, _GUARD_ATTR, (self, original))
        self.audit.append(
            "killswitch", "killswitch.guard_attached", {"tools": len(registry)}
        )
        return registry

    def unguard(self, registry: ToolRegistry) -> ToolRegistry:
        """Remove the guard installed by :meth:`guard`, restoring the original ``call``."""
        if getattr(registry, _GUARD_ATTR, None) is None:
            return registry
        del registry.call  # removes the instance attribute, revealing the class method
        delattr(registry, _GUARD_ATTR)
        self.audit.append("killswitch", "killswitch.guard_detached", {})
        return registry
