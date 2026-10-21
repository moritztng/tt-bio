"""Compare the tests.sh legs: counts per leg, and tests that fail on this branch but pass on main
under the same interpreter. Usage: python junit_diff.py <scratch_root>/tests"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

D = Path(sys.argv[1])


def leg(name):
    p = D / f"{name}.xml"
    if not p.exists():
        return None
    out = {}
    for tc in ET.parse(p).iter("testcase"):
        k = f"{tc.get('classname')}::{tc.get('name')}"
        tags = {c.tag for c in tc}
        out[k] = "failed" if tags & {"failure", "error"} else "skipped" if "skipped" in tags else "passed"
    return out


res = {}
for py in ("310", "312"):
    land, main = leg(f"land{py}"), leg(f"main{py}")
    for name, r in (("land" + py, land), ("main" + py, main)):
        if r is not None:
            res[name] = {s: sum(v == s for v in r.values()) for s in ("passed", "failed", "skipped")}
    if land and main:
        res[f"regressions_py{py}"] = sorted(k for k, v in land.items() if v == "failed" and main.get(k) == "passed")
        res[f"fixed_py{py}"] = sorted(k for k, v in main.items() if v == "failed" and land.get(k) == "passed")
        res[f"failing_both_py{py}"] = sorted(k for k, v in land.items() if v == "failed" and main.get(k) == "failed")
        res[f"only_on_land_py{py}"] = sorted(k for k in land if k not in main)
print(json.dumps(res, indent=1))
