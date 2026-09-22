#!/usr/bin/env python3
"""Stamp the three row-count sites in of3t-orchestrator.md from DISK, not from prose.

The audit requires the counts in words and the ledger moves while the pass is being written --
four hand-restamps in one pass, each one stale before the compose finished. Compute, write,
compose, in that order and with no thinking in between.
"""
import pathlib, re, sys
D = pathlib.Path("/home/moritz/.coworker")
W = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",
     11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",16:"sixteen",
     17:"seventeen",18:"eighteen",19:"nineteen"}
def word(n):
    if n < 20: return W[n]
    if n < 100: return {2:"twenty",3:"thirty",4:"forty",5:"fifty",6:"sixty",7:"seventy",8:"eighty",9:"ninety"}[n//10] + ("-"+W[n%10] if n%10 else "")
    return word(n//100) + " hundred" + (" " + word(n%100) if n%100 else "")
briefs  = len(list((D/"workstreams").glob("of3t-*.txt")))
markers = [m.name for m in (D/"state"/"concluded").glob("of3t-*")]
own     = [m for m in markers if m.startswith("of3t-orchestrator")]
rows    = len(markers) - len(own)
# --- defects: recompute from the UNION and reconcile the triage, same reason as the rows.
sys.path.insert(0, str(D/"wt"/"of3t-orchestrator"/"perf"/"of3t_orchestrator"))
from defects_union import defects_text
from status_vocab import statuses_by_defect
import json
st = statuses_by_defect(defects_text())
unf = sorted((d for d, v in st.items() if v == "UNFIXED"), key=lambda s: int(s[1:]))
tp = D/"state"/"of3t"/"UNFIXED_TRIAGE.json"; j = json.loads(tp.read_text())
cls = {d: c for c, ds in j["classes"].items() for d in ds}
for d in unf:                      # a newly-visible UNFIXED defect defaults to campaign-internal
    if d not in cls:
        j["classes"]["CAMPAIGN-INTERNAL"].append(d)
        j["reasons"].setdefault(d, "classified by stamp_row_counts.py; no user-facing claim made")
for c in j["classes"]:             # and one that stopped being UNFIXED leaves the split
    j["classes"][c] = [d for d in j["classes"][c] if d in set(unf)]
j["counts"] = {k: len(v) for k, v in j["classes"].items()}
j["unfixed_total"] = sum(j["counts"].values())
tp.write_text(json.dumps(j, indent=1) + "\n")
n_def = len(st)

p = D/"state"/"of3t-orchestrator.md"; t = p.read_text()
def ids(c): return ", ".join(sorted(j["classes"][c], key=lambda s: int(s[1:])))
t = re.sub(r"\*\*[A-Za-z ]+ defects filed\*\*, \*\*\d+ UNFIXED\*\* \(\d+\n?scope-excluded, \d+ USER-FACING, \d+ campaign-internal\)",
           f"**{word(n_def).capitalize()} defects filed**, **{j['unfixed_total']} UNFIXED** "
           f"({j['counts']['SCOPE-EXCLUDED']}\nscope-excluded, {j['counts']['USER-FACING']} USER-FACING, "
           f"{j['counts']['CAMPAIGN-INTERNAL']} campaign-internal)", t, count=1)
t = re.sub(r"\*\*The \d+ UNFIXED, named", f"**The {j['unfixed_total']} UNFIXED, named", t, count=1)
for c in ("SCOPE-EXCLUDED", "USER-FACING", "CAMPAIGN-INTERNAL"):
    t = re.sub(rf"    {c}( +)\d+  [^\n]*", lambda m, c=c: f"    {c}{m.group(1)}{j['counts'][c]}  {ids(c)}", t, count=1)
t = re.sub(r"ROWS: \*\*[a-z\- ]+ dispatched, [a-z\- ]+ concluded\*\*",
           f"ROWS: **{word(briefs)} dispatched, {word(rows)} concluded**", t, count=1)
t = re.sub(r"\(\d+ `of3t-\*` briefs including this row's own; \d+ markers",
           f"({briefs} `of3t-*` briefs including this row's own; {len(markers)} markers", t, count=1)
t = re.sub(r"One hundred [a-z\- ]+ rows dispatched, one\nhundred [a-z\- ]+ concluded",
           f"{word(briefs).capitalize()} rows dispatched, {word(rows)}\nconcluded", t, count=1)
t = re.sub(r"`state/concluded` holds \*\*[a-z\- ]+\*\* of3t files",
           f"`state/concluded` holds **{word(len(markers))}** of3t files", t, count=1)
t = re.sub(r"so \*\*[a-z\- ]+ rows have concluded\*\*", f"so **{word(rows)} rows have concluded**", t, count=1)
p.write_text(t)
print(f"stamped: {briefs} briefs, {len(markers)} markers, {rows} rows concluded, "
      f"{n_def} defects, {j['unfixed_total']} UNFIXED {j['counts']}")
