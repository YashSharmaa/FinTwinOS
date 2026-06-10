"""Hard USD ceilings on cumulative LLM spend.

``LLMClient`` meters indicative cost per call (``LLMResponse.cost_usd``);
``BudgetGuard`` turns that meter into an enforced limit: every recorded response
adds to a cumulative total, and the moment the total reaches the configured ceiling
a :class:`BudgetExceeded` is raised. Spend that already happened is still recorded
— the guard stops the *next* dollar, it does not pretend the last one back.

Three usage styles::

    guard = BudgetGuard(ceiling_usd=5.00, audit=runtime.audit)

    # 1. Explicit recording
    response = await llm.complete(messages)
    guard.record(response)

    # 2. Decorator (sync or async callables returning an LLMResponse)
    @guard.wrap
    async def ask(prompt: str) -> LLMResponse: ...

    # 3. Context manager (verifies headroom on entry)
    with guard:
        guard.record(await llm.complete(messages))

Every charge (and the breach itself) is appended to the shared ``AuditTrail`` when
one is supplied, so spend is part of the same tamper-evident record as everything
else in FinTwinOS.
"""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from fintwinos.core.audit import AuditTrail
from fintwinos.core.errors import FinTwinError

_EPS = 1e-12

F = TypeVar("F", bound=Callable[..., Any])


class BudgetExceeded(FinTwinError):
    """Raised when cumulative LLM spend reaches or exceeds the configured ceiling.

    Attributes:
        spent_usd: Total recorded spend at the moment of the breach.
        ceiling_usd: The configured ceiling.
        payload: Optional object associated with the breaching charge (e.g. the
            ``LLMResponse`` whose cost tripped the guard), so callers that catch
            the exception do not lose the response they already paid for.
    """

    def __init__(self, spent_usd: float, ceiling_usd: float, payload: Any = None) -> None:
        self.spent_usd = spent_usd
        self.ceiling_usd = ceiling_usd
        self.payload = payload
        super().__init__(
            f"LLM budget exceeded: spent ${spent_usd:.6f} of ${ceiling_usd:.6f} ceiling"
        )


class BudgetGuard:
    """Tracks cumulative LLM cost against a hard USD ceiling.

    Semantics:

    - :meth:`charge` / :meth:`record` add to the total *first*, then raise
      :class:`BudgetExceeded` if the total has reached the ceiling. Reaching the
      ceiling exactly trips the guard — it is a hard stop, not a soft target.
    - :meth:`ensure_available` raises immediately when the guard is already
      exhausted; the decorator and context-manager forms call it before doing any
      work so no new spend is initiated against a blown budget.
    - All methods are thread-safe.
    """

    def __init__(
        self,
        ceiling_usd: float,
        *,
        audit: AuditTrail | None = None,
        name: str = "llm-budget",
    ) -> None:
        """Create a guard.

        Args:
            ceiling_usd: Hard cumulative spend limit in USD (must be positive).
            audit: Optional shared audit trail; charges append ``budget.charged``
                records and a breach appends ``budget.exceeded``.
            name: Actor name used in audit records (useful with several guards).
        """
        if ceiling_usd <= 0:
            raise ValueError("ceiling_usd must be positive")
        self.ceiling_usd = float(ceiling_usd)
        self.audit = audit
        self.name = name
        self._spent = 0.0
        self._lock = threading.Lock()

    # -- state ----------------------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        """Cumulative recorded spend in USD."""
        with self._lock:
            return self._spent

    @property
    def remaining_usd(self) -> float:
        """Headroom left before the ceiling (never negative)."""
        with self._lock:
            return max(0.0, self.ceiling_usd - self._spent)

    @property
    def exhausted(self) -> bool:
        """True once cumulative spend has reached or exceeded the ceiling."""
        with self._lock:
            return self._spent >= self.ceiling_usd - _EPS

    def summary(self) -> dict[str, float | bool]:
        """Snapshot for dashboards and run reports."""
        with self._lock:
            spent = self._spent
        return {
            "ceiling_usd": self.ceiling_usd,
            "spent_usd": spent,
            "remaining_usd": max(0.0, self.ceiling_usd - spent),
            "exhausted": spent >= self.ceiling_usd - _EPS,
        }

    def reset(self) -> None:
        """Zero the meter (e.g. at the start of a new evaluation run)."""
        with self._lock:
            self._spent = 0.0

    # -- charging --------------------------------------------------------------------

    def charge(self, cost_usd: float, *, payload: Any = None) -> float:
        """Record a spend amount; raise :class:`BudgetExceeded` on breach.

        Args:
            cost_usd: Non-negative USD amount to add to the meter.
            payload: Optional object attached to the exception on breach.

        Returns:
            Remaining headroom in USD after the charge.

        Raises:
            BudgetExceeded: When the cumulative total reaches the ceiling. The
                charge is still recorded before raising.
        """
        cost = float(cost_usd)
        if cost < 0:
            raise ValueError("cost_usd must be non-negative")
        with self._lock:
            self._spent += cost
            spent = self._spent
        if self.audit is not None:
            self.audit.append(
                self.name,
                "budget.charged",
                {
                    "cost_usd": cost,
                    "spent_usd": spent,
                    "ceiling_usd": self.ceiling_usd,
                    "remaining_usd": max(0.0, self.ceiling_usd - spent),
                },
            )
        if spent >= self.ceiling_usd - _EPS:
            if self.audit is not None:
                self.audit.append(
                    self.name,
                    "budget.exceeded",
                    {"spent_usd": spent, "ceiling_usd": self.ceiling_usd},
                )
            raise BudgetExceeded(spent, self.ceiling_usd, payload=payload)
        return self.ceiling_usd - spent

    def record(self, response: Any) -> float:
        """Charge the cost of an ``LLMResponse`` (or anything with ``cost_usd``).

        Objects without a numeric ``cost_usd`` attribute are recorded as zero
        spend, so offline stub responses pass through for free.

        Returns:
            Remaining headroom in USD.

        Raises:
            BudgetExceeded: When this response's cost breaches the ceiling; the
                response is attached as ``payload`` on the exception.
        """
        cost = getattr(response, "cost_usd", 0.0)
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            cost = 0.0
        return self.charge(cost, payload=response)

    def ensure_available(self) -> None:
        """Raise :class:`BudgetExceeded` if the guard is already exhausted."""
        with self._lock:
            spent = self._spent
        if spent >= self.ceiling_usd - _EPS:
            raise BudgetExceeded(spent, self.ceiling_usd)

    # -- helpers -----------------------------------------------------------------------

    def wrap(self, fn: F) -> F:
        """Decorate a callable so its returned response is budget-metered.

        Works with both sync and async callables. Before invocation the guard
        verifies headroom (:meth:`ensure_available`); after invocation the return
        value's ``cost_usd`` is charged via :meth:`record`. If the charge breaches
        the ceiling, the raised :class:`BudgetExceeded` carries the response as
        ``payload``.
        """
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self.ensure_available()
                result = await fn(*args, **kwargs)
                self.record(result)
                return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            self.ensure_available()
            result = fn(*args, **kwargs)
            self.record(result)
            return result

        return sync_wrapper  # type: ignore[return-value]

    def __enter__(self) -> BudgetGuard:
        """Context-manager entry: verify there is headroom before any work starts."""
        self.ensure_available()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Context-manager exit: never suppresses exceptions."""
        return False
