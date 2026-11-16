"""Served RFD4 per-design chip time, read from JapanFold task usage records (jfops JSON slices).

    python3 perf/bmk/rfd4_served_split.py /tmp/rfd4s/s*.json

Each usage target carries the sampled lease interval of one design on its chip. A chip that loads the
engine cold inside a design's lease shows that design ~load seconds longer than its siblings, so a fixed
start shows up as a spread inside a job; a job whose designs all run alike paid none. Prints, per residue
bucket: designs, median design interval, the in-job spread (max - min), the share of designs more than
20 s above their job's median, and how much of the agent-metered time the API's lease cap left unbilled.
"""
import json
import statistics as st
import sys
from collections import defaultdict


def rows(paths):
    seen = set()
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        cols = d["columns"]
        for r in d["rows"]:
            r = dict(zip(cols, r))
            if (r["job"], r["started_at"]) in seen:
                continue
            seen.add((r["job"], r["started_at"]))
            u = r["usage"]
            r["usage"] = json.loads(u) if isinstance(u, str) else u
            yield r


def main(paths):
    b = defaultdict(lambda: {"n": 0, "dur": [], "spread": [], "slow": 0, "billed": 0.0, "cut": 0.0,
                             "jobs": 0, "narrow": 0})
    for r in rows(paths):
        u = r["usage"] or {}
        durs = [t["intervals"][-1][1] - t["intervals"][0][0] for t in u.get("targets", [])
                if t.get("intervals")]
        durs = [x for x in durs if x > 1]
        if not durs:
            continue
        k = (r["residues"] or 0) // 64 * 64
        s = b[k]
        med = st.median(durs)
        s["jobs"] += 1
        s["n"] += len(durs)
        s["dur"] += durs
        s["spread"].append(max(durs) - min(durs))
        s["slow"] += sum(x > med + 20 for x in durs)
        s["billed"] += float(u.get("billable_s") or 0)
        s["cut"] += float(u.get("over_lease_s") or 0)
        s["narrow"] += int((r["chips"] or 1) < len(durs))
    print(f"{'res':>5} {'jobs':>5} {'designs':>7} {'p10 s':>6} {'p50 s':>6} {'p90 s':>6} "
          f"{'spread p50':>10} {'spread p90':>10} {'>med+20s':>8} {'chips<designs':>13} {'unbilled':>8}")
    tot = defaultdict(float)
    for k in sorted(b):
        s = b[k]
        d = sorted(s["dur"])
        q = lambda a, f: a[min(len(a) - 1, int(f * len(a)))]
        sp = sorted(s["spread"])
        un = s["cut"] / (s["billed"] + s["cut"]) if s["billed"] + s["cut"] else 0
        print(f"{k:>5} {s['jobs']:>5} {s['n']:>7} {q(d, .1):>6.1f} {q(d, .5):>6.1f} {q(d, .9):>6.1f} "
              f"{q(sp, .5):>10.1f} {q(sp, .9):>10.1f} {s['slow'] / s['n']:>8.1%} {s['narrow']:>13} {un:>8.1%}")
        for f in ("jobs", "n", "slow", "billed", "cut", "narrow"):
            tot[f] += s[f]
        tot["chip_s"] += sum(d)
    print(f"all: {int(tot['jobs'])} jobs, {int(tot['n'])} designs, {tot['chip_s'] / 3600:.1f} chip-h of design "
          f"leases; designs > job median + 20 s: {tot['slow'] / tot['n']:.1%}; jobs leased fewer chips than "
          f"designs: {int(tot['narrow'])}; billed {tot['billed'] / 3600:.1f} h, cut by the lease cap "
          f"{tot['cut'] / 3600:.1f} h ({tot['cut'] / (tot['billed'] + tot['cut']):.1%})")


if __name__ == "__main__":
    main(sys.argv[1:])
