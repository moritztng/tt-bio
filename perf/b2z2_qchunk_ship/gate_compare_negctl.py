"""Negative control for digits_reproduced: it must FAIL when a digit actually moves."""
import json, sys
sys.path.insert(0, "perf/b2z2_qchunk_ship")
from gate_compare import digits_reproduced, control

ctl, _ = control()
LEG = "openfold3-7xi5-notmpl"
detail = ctl[LEG]["detail"]
raw = json.loads(open(f"perf/b2z2_gate/qchunkship_merged_work/{LEG}.json").read())

ok, why = digits_reproduced(detail, raw)
print(f"UNTOUCHED : ok={ok}  {why}")
assert ok, "positive control failed: real report should reproduce"

# perturb the one number the control prints as X=5.390 (cross.mean) by 0.01 A
bad = json.loads(json.dumps(raw))
t = bad["targets"]["7xi5_notmpl"]["kabsch_rmsd"]
t["cross"]["mean"] += 0.01
ok2, why2 = digits_reproduced(detail, bad)
print(f"PERTURBED : ok={ok2}  {why2}")
assert not ok2, "NEGATIVE CONTROL FAILED: check passed on moved digits"
print("\ncontrol detail:", detail)
print("NEGATIVE CONTROL OK -- the check breaks on a 0.01 A move")
