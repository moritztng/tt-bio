"""Render analyze.py's JSON as the counted model x input table (markdown on stdout).

Cell codes: ok = folded and the input's feature is in the structure; DROP = folded without the
feature and nothing said so; warn = folded without it after a printed Note; clash = the feature
is there but overlaps (< 1.0 A); ref = refused before any fold, with a named alternative; ERR =
failed at run time; fly = still running; - = not run.

Usage: table.py <analyze.json>
"""
import json
import sys
from collections import Counter

d = json.load(open(sys.argv[1]))


def code(v):
    o = v["outcome"]
    if o == "folded":
        c = v.get("check") or {}
        if c.get("ok", True):
            return "ok"
        if v.get("warned"):
            return "warn"
        return "clash" if c.get("present") and c.get("clash") else "DROP"
    return {"refused": "ref", "failed": "ERR", "in_flight": "fly", "not_run": "-"}[o]


models = list(d)
inputs = sorted({k for m in models for k in d[m]})
print("| input | " + " | ".join(models) + " |")
print("|---|" + "---|" * len(models))
tally = Counter()
for i in inputs:
    row = [code(d[m][i]) if i in d[m] else "-" for m in models]
    tally.update(row)
    print(f"| {i.rsplit('.', 1)[0]} | " + " | ".join(row) + " |")
done = sum(n for c, n in tally.items() if c not in ("fly", "-"))
print(f"\n{done}/{len(models) * len(inputs)} cells measured; " +
      ", ".join(f"{c} {n}" for c, n in tally.most_common()))
