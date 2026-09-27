#!/usr/bin/env python3
"""The bcx-p10-arm sitting: harness-armed (a*) against product-armed (b*).

    report.py out/a1 out/b1 out/b2 out/a2 [--json out/report.json]

Per arm, `perf/bcx_p10_tritraj/report.py::one` (pro-rata amortised round, gate held/idle, AICLK
and loadavg1 DURING the window). GO: b within 2 % of a and at most 6.958 s. Equality: every
(trajectory, round) digest of a b arm against the same key in the a arms, `torch.equal` bytes.
"""
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.bcx_p10_tritraj.report import digests, one  # noqa: E402

BAR = 6.958


def main():
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [a for a in sys.argv[1:] if not a.startswith("--") and a != js]
    rows, raw = [], {}
    for p in paths:
        r, d = one(p)
        rows.append(r)
        raw[r["arm"]] = d
        c = r.get("clock", {})
        print(f"{r['arm']:4s}  round {r.get('round_s', float('nan')):.3f} s  "
              f"held {r.get('held_per_round', float('nan')):.3f}  "
              f"idle {r.get('idle_per_round', float('nan')):.3f}  "
              f"AICLK med {c.get('aiclk_med')} min {c.get('aiclk_min')} "
              f"load1 {c.get('load1')}  "
              f"hwm {r.get('host_hwm_gb')} GB")
    s = {}
    for k in "ab":
        rs = [r for r in rows if r["arm"].startswith(k) and "round_s" in r]
        if rs:
            s[k] = {"arms": [r["arm"] for r in rs],
                    "round_s": statistics.mean(r["round_s"] for r in rs),
                    "spread": [min(r["round_s"] for r in rs), max(r["round_s"] for r in rs)]}
    if "a" in s and "b" in s:
        s["b_over_a"] = s["b"]["round_s"] / s["a"]["round_s"]
        s["go"] = s["b_over_a"] <= 1.02 and s["b"]["round_s"] <= BAR
        print(f"a {s['a']['round_s']:.3f}  b {s['b']['round_s']:.3f}  b/a {s['b_over_a']:.4f}  "
              f"(GO: b/a <= 1.02 and b <= {BAR})  -> {'GO' if s['go'] else 'NO-GO'}")
    ref = {}
    for arm in s.get("a", {}).get("arms", []):
        for key, h in digests(raw[arm]).items():
            ref.setdefault(key, set()).add(h)
    same = diff = 0
    for arm in s.get("b", {}).get("arms", []):
        for key, h in digests(raw[arm]).items():
            if key in ref:
                same, diff = (same + 1, diff) if ref[key] == {h} else (same, diff + 1)
    s["equal"] = {"equal": same, "differ": diff, "a_keys": len(ref),
                  "a_self_consistent": all(len(v) == 1 for v in ref.values())}
    print(f"equality: {same} product-armed (trajectory, round) outputs equal the harness's, "
          f"{diff} differ, over {len(ref)} keys (a arms agree: "
          f"{s['equal']['a_self_consistent']})")
    for arm in s.get("b", {}).get("arms", []):
        st = raw[arm]["stamp"]
        s.setdefault("fast", {})[arm] = st.get("fast")
        s.setdefault("lever_stats", {})[arm] = st.get("lever_stats")
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": s}, indent=1,
                                               default=str))


if __name__ == "__main__":
    main()
