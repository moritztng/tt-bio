"""Campaign status, with `generate.py` as the single source of truth for what counts as done.

This script used to carry its own provenance rule ("a record is CURRENT only if mps == 3").
That duplicated `done_pairs()` and then drifted from it: after qb1 was relaunched at mps=5 the
rule inverted, so the live records were reported as stale and five superseded failures were
reported as current. A second copy of an acceptance predicate is a second thing to get wrong,
so there is only one now -- `done_pairs()` -- and everything here is derived from it.

Acceptance for Tier A: OUTSTANDING is empty on both hosts, i.e. every (target, model) pair
counts as done.

Usage: python3 scripts/abag_xm_status.py
"""
import collections
import importlib.util
import json
import pathlib
import statistics

_spec = importlib.util.spec_from_file_location(
    "abag_xm_generate", pathlib.Path(__file__).resolve().parent / "abag_xm_generate.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

records = []
if gen.PROGRESS.exists():
    for line in open(gen.PROGRESS):
        try:
            records.append(json.loads(line))
        except Exception:
            continue

done = gen.done_pairs()
targets = gen.all_targets()
expected = {(t, m) for t in targets for m in gen.MODELS}
outstanding = sorted(expected - done)

# The record that made a pair done -- superseded attempts on the same pair are not failures.
accepted = {}
for r in records:
    key = (r["target"], r["model"])
    if key in done and r.get("status") == "ok":
        accepted[key] = r

print(f"targets {len(targets)} x models {len(gen.MODELS)} = {len(expected)} pairs "
      f"| done {len(done)} | OUTSTANDING {len(outstanding)}")
print(f"records on disk: {len(records)} (superseded attempts are kept, not counted)")

for m in gen.MODELS:
    w = [r["wall_s"] for k, r in accepted.items() if k[1] == m and r.get("wall_s")]
    n_out = sum(1 for t, mm in outstanding if mm == m)
    if w:
        print(f"   {m:14s} done={len(w):3d} outstanding={n_out:3d}  "
              f"wall_s median {statistics.median(w):.0f}  min {min(w):.0f}  max {max(w):.0f}")
    else:
        print(f"   {m:14s} done=  0 outstanding={n_out:3d}")

# Why each outstanding pair is outstanding: its most recent record, or never attempted.
last = {}
for r in records:
    last[(r["target"], r["model"])] = r
reasons = collections.Counter()
for key in outstanding:
    r = last.get(key)
    if r is None:
        reasons["never attempted"] += 1
    elif r.get("status") == "ok":
        reasons[f"ok but mps={r.get('mps')} (regenerating at mps={gen.MPS})"] += 1
    else:
        reasons[r.get("status", "?")] += 1
print("   OUTSTANDING by reason:", dict(reasons) or "NONE")
if 0 < len(outstanding) <= 12:
    print("   OUTSTANDING pairs:", outstanding)

odd = [(k[0], k[1], r.get("n_cifs"), r.get("n_paes")) for k, r in sorted(accepted.items())
       if r.get("n_cifs") != gen.N_SAMPLES or r.get("n_paes") != gen.N_SAMPLES]
print("   done but wrong artifact count:", odd or "NONE")

# D12 (per-model config fairness contract): the resolved config must be constant WITHIN a
# generator. mps is exempt for the generators that ignore it (supports_multiplicity=False) --
# the same model-aware rule done_pairs() uses, so the two cannot drift apart.
for field in ("host_threads", "mps", "paired_msa"):
    by_model = {m: {r.get(field) for k, r in accepted.items() if k[1] == m} for m in gen.MODELS}
    checked = {m: v for m, v in by_model.items()
               if field != "mps" or m in gen.MPS_SENSITIVE_MODELS}
    flag = "" if all(len(v) <= 1 for v in checked.values()) else "   <-- NOT CONSTANT (D12)"
    shown = "  ".join(f"{m}={sorted(v, key=str)}"
                      + ("" if m in checked else " (ignored by this model)")
                      for m, v in by_model.items() if v)
    print(f"   {field:13s} per model: {shown}{flag}")
