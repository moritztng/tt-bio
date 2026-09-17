#!/usr/bin/env python3
"""Tests for the call-site join and the two sibling-set guards, with a negative control each.

Run: python3 perf/c12_linear_sites/test_sites.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sites as S                                                             # noqa: E402

FAIL = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ((" -- " + extra) if extra else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    R = S.TF.setup(S.PERF, S.Args())
    rows, _reuse = S.collect(R)
    cen = {k["key"]: k for k in
           json.loads((HERE / "census_budget_sweep2.json").read_text())["keys"]
           if k["arm"] == "linear"}
    got = {}
    for r in rows:
        got[r["key"]] = got.get(r["key"], 0.0) + r["calls"]
    check("27 keys reproduced", len(got) == 27 and set(got) == set(cen),
          "%d keys" % len(got))
    check("108608 calls reproduced", abs(sum(got.values()) - 108608) < 0.5,
          "%.0f" % sum(got.values()))
    bad = [k for k in cen if abs(cen[k]["calls"] - got.get(k, 0)) >= 0.5]
    check("every key's call count matches the census", not bad, str(bad))

    # negative control for the join: a census row with the wrong call count must be caught.
    cen2 = {k: dict(v) for k, v in cen.items()}
    cen2["linear|out=1x16x512x512|K=128"]["calls"] /= 2
    bad2 = [k for k in cen2 if abs(cen2[k]["calls"] - got.get(k, 0)) >= 0.5]
    check("negative control: a perturbed census call count IS flagged", len(bad2) == 1)

    # guard 1, reallocation: two members either side of a fresh allocation are two sets.
    S.ALLOCS["fake"] = {}
    base = [{"sig": "fake", "i": 1, "calls": 1, "owner": "U", "inst": "U#0", "key": "k",
             "act_addr": 9, "act_shape": [1, 4, 4], "K": 4, "w_addr": 1, "w_tid": 11, "N": 4,
             "act_B": 32, "B": 0.0},
            {"sig": "fake", "i": 2, "calls": 1, "owner": "U", "inst": "U#0", "key": "k",
             "act_addr": 9, "act_shape": [1, 4, 4], "K": 4, "w_addr": 2, "w_tid": 12, "N": 4,
             "act_B": 32, "B": 0.0}]
    check("two reads of one live buffer are ONE set", len(S.sibling_sets(base, 4)) == 1)
    S.ALLOCS["fake"] = {9: [2]}          # the buffer is refilled between op 1 and op 2
    check("negative control: a reallocation between them splits the set",
          S.sibling_sets(base, 4) == [])

    # guard 2, weight identity: the same weight tensor twice is a new iteration.
    S.ALLOCS["fake"] = {}
    same = [dict(base[0]), dict(base[1])]
    same[1]["w_addr"], same[1]["w_tid"] = 1, 11
    check("negative control: a repeated weight tensor splits the set",
          S.sibling_sets(same, 4) == [])
    # and the PairWeightedAveraging case: same recycled ADDRESS, different tensor, stays one set
    recycled = [dict(base[0]), dict(base[1])]
    recycled[1]["w_addr"] = 1
    check("a recycled weight ADDRESS with a new tensor id stays one set",
          len(S.sibling_sets(recycled, 4)) == 1)

    print("\n%d failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
