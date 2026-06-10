"""A full day in the life: all four desks on one twin, plus governance roll-up.

Run: ``python examples/04_full_day.py`` (works fully offline).
"""

from fintwinos.demos import day_in_the_life

result = day_in_the_life.run(seed=7)

gov = result["governance"]
print()
print(f"sections completed: {', '.join(result['sections'])}")
print(f"tool calls by band: {gov['tool_calls_by_band']}")
print(f"audit chain: {gov['audit_records']} records, verified={gov['audit_verified']}")
print(f"llm usage: {gov['llm_usage']}")
