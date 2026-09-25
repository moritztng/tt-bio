#!/usr/bin/env python3
"""of3t-infab: classify every line where the AFTER2 trace differs from BEFORE (and from AFTER).

    trace_after2.py INFAB2.json OUT.json

Finding 5 is normalised first: `compute_kernel_config:None` is ttnn.softmax's own default, so the
explicit kwarg is dropped from both sides. What is left is split into the named findings
(1: D1 `multiply_` scalar 1.0 -> sqrt 24; 3: `deallocate`; 6: host uploads that moved, i.e. the
lines removed and the lines added are the same multiset of `from_torch`, which is 96d9d7767
staging `cl0_d`/`plm0_d` at the top of `build_dm_device_aux`) and anything else, which is a failure.
"""
import collections
import difflib
import json
import re
import sys
from pathlib import Path

W1 = Path("/tmp/of3t/of3t-infab/work")
W2 = Path("/tmp/of3t/of3t-infab/work2")


def lines(run):
    out = []
    for p in run["trace"]["processes"]:
        out += Path(p["path"]).read_text().splitlines()
    return [re.sub(r",?compute_kernel_config:None", "", l).replace("{,", "{") for l in out]


def classify(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    kinds = collections.Counter()
    other = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        la, lb = a[i1:i2], b[j1:j2]
        if tag == "replace" and len(la) == len(lb) and all(
                x.split("\t")[1] == "ttnn.multiply_" and x.replace(",1.0)", ",4.898979485566356)") == y
                for x, y in zip(la, lb)):
            kinds["finding1_D1_scalar"] += len(la)
            continue
        for x in la:
            op = x.split("\t")[1]
            (kinds.__setitem__("finding3_deallocate_removed", kinds["finding3_deallocate_removed"] + 1)
             if op == "ttnn.deallocate" else other.append("- " + x[:300]))
        for y in lb:
            other.append("+ " + y[:300])
    rem = collections.Counter(x[2:] for x in other if x[0] == "-")
    add = collections.Counter(y[2:] for y in other if y[0] == "+")
    if other and rem == add and all(x.split("\t")[1] == "ttnn.from_torch" for x in rem):
        kinds["finding6_upload_reorder"] = sum(rem.values())
        other = []
    return dict(kinds), other


r2 = json.loads(Path(sys.argv[1]).read_text())
r1 = json.loads(Path(sys.argv[1]).with_name("INFAB.json").read_text())
out = {}
for m, v in r2["models"].items():
    if not v["all_folds_succeeded"]:
        out[m] = {"folds_failed": True}
        continue
    runs = v["runs"]
    b, a2 = lines(runs[0]), lines(runs[1])
    k, other = classify(b, a2)
    rec = {"n_ops": [len(b), len(a2)], "B_vs_A2_identical_after_finding5": b == a2,
           "B_vs_A2_named": k, "B_vs_A2_unnamed": other[:40], "B_vs_A2_unnamed_n": len(other)}
    prev = r1["models"].get(m)
    if prev and prev["all_folds_succeeded"]:
        a1 = lines(prev["runs"][1])
        d = collections.Counter(l.split("\t")[1] for l in a1)
        d.subtract(collections.Counter(l.split("\t")[1] for l in a2))
        rec["A_vs_A2_identical"] = a1 == a2
        rec["A_minus_A2_op_counts"] = {x: n for x, n in d.items() if n}
    out[m] = rec
    print(m, json.dumps({x: y for x, y in rec.items() if x != "B_vs_A2_unnamed"}), flush=True)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
