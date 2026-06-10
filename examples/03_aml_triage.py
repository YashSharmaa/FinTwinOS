"""AML ring surge: scored triage queue and a dual-controlled case closure.

Run: ``python examples/03_aml_triage.py`` (works fully offline).
"""

from fintwinos.demos import aml_triage

result = aml_triage.run(seed=7)

print()
print(f"queue of {result['queue_size']} alerts scored — AUC {result['scorer']['auc']}")
for row in result["triage"]:
    print(
        f"  top-{int(row['k']):>2}: precision {row['precision']:.0%}, recall {row['recall']:.0%}"
    )
blocked, approved = result["close_case"]["blocked"], result["close_case"]["approved"]
print(f"close without approval → ok={blocked['ok']} (requires_approval={blocked['requires_approval']})")
print(f"close with ApprovalToken → ok={approved['ok']}, outbox: {result['close_case']['outbox_dir']}")
