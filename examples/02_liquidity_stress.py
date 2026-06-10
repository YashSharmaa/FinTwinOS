"""USD liquidity squeeze rehearsal: stress presets, survival days, decision.

Run: ``python examples/02_liquidity_stress.py`` (works fully offline).
"""

from fintwinos.demos import liquidity

result = liquidity.run(seed=7)

print()
for preset, row in result["stress"].items():
    ci = row["ci"] or ["—", "—"]
    print(
        f"{preset:>18}: survival {row['survival_days']} days "
        f"(90% CI [{ci[0]}, {ci[1]}]), shock {row['shock_bps']:.0f} bps"
    )
print(f"case '{liquidity.CASE_ID}' status: {result['case']['status']}")
print(f"audit verified: {result['audit']['verified']} ({result['audit']['records']} records)")
