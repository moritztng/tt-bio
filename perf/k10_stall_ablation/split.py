#!/usr/bin/env python3
"""Pair a tracy ops CSV against `ablate.py`'s recorded dispatch order and print the response table.

The two shipped sum counters are per-op sums over cores, so each is divided by THAT op's own
`CORE COUNT` -- its own, not the block's, because core counts differ between ops in one sweep.
The denominator for a stall fraction is the thread the counter belongs to:

    CB WAIT FRONT   -> DEVICE TRISC0 KERNEL DURATION   (unpack, blocked on input)
    CB RESERVE BACK -> DEVICE TRISC2 KERNEL DURATION   (pack, blocked on output room)

`cb_split.py` divides both by TRISC1 and calls the remainder compute; that is the wrong-thread
division and is not repeated here. Nothing below reports a "useful math" figure.

    python3 split.py run.json ops_perf_results_*.csv[.gz]
"""
from __future__ import annotations

import csv
import gzip
import json
import statistics as st
import sys
from pathlib import Path

MEASURED = ("GenericOpDeviceOperation", "MatmulDeviceOperation", "BinaryNgDeviceOperation")


def num(r, k):
    v = (r.get(k) or "").strip()
    try:
        return float(v)
    except ValueError:
        return 0.0


def load_rows(p: Path):
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    run = json.loads(Path(sys.argv[1]).read_text())
    rows = load_rows(Path(sys.argv[2]))
    order = run["dispatch_order"]
    by_name = {a["name"]: a for a in run["arms"]}

    # The measured region is what sits between the two ttnn.exp fences the rig lays down.
    u = [i for i, r in enumerate(rows) if r["OP CODE"] == "UnaryDeviceOperation"]
    if len(u) < 6:
        print(f"FAILED: found {len(u)} fence ops, expected 6", file=sys.stderr)
        return 1
    reg = rows[u[2] + 1:u[-3]]
    got = [r for r in reg if r["OP CODE"] in MEASURED]
    if len(got) != len(order):
        print(f"FAILED: {len(got)} measured device ops in the fenced region against "
              f"{len(order)} dispatched -- the pairing is not sound", file=sys.stderr)
        return 1

    per: dict[str, list[dict]] = {}
    for name, r in zip(order, got):
        if by_name[name]["opcode"] != r["OP CODE"]:
            print(f"FAILED: {name} expected {by_name[name]['opcode']}, CSV has {r['OP CODE']}",
                  file=sys.stderr)
            return 1
        cc = num(r, "CORE COUNT") or 1.0
        t0 = num(r, "DEVICE TRISC0 KERNEL DURATION [ns]")
        t2 = num(r, "DEVICE TRISC2 KERNEL DURATION [ns]")
        wf = num(r, "DEVICE COMPUTE CB WAIT FRONT [ns]") / cc
        rb = num(r, "DEVICE COMPUTE CB RESERVE BACK [ns]") / cc
        per.setdefault(name, []).append({
            "cores": int(cc),
            "wall_us": num(r, "DEVICE KERNEL DURATION [ns]") / 1e3,
            "in_us": wf / 1e3, "out_us": rb / 1e3,
            "in_frac": wf / t0 if t0 else 0.0,
            "out_frac": rb / t2 if t2 else 0.0,
            "anomaly": 1 if (t0 and wf > t0) or (t2 and rb > t2) else 0,
        })

    stat = {}
    for name, xs in per.items():
        stat[name] = {k: st.median([x[k] for x in xs])
                      for k in ("wall_us", "in_us", "out_us", "in_frac", "out_frac")}
        stat[name]["cores"] = xs[0]["cores"]
        stat[name]["n"] = len(xs)
        w = sorted(x["wall_us"] for x in xs)
        stat[name]["wall_spread"] = w[-1] / w[0] if w[0] else 0.0
    anom = sum(x["anomaly"] for xs in per.values() for x in xs)
    tot = sum(len(xs) for xs in per.values())

    print(f"# {tot} measured ops, {anom} normalise inconsistently "
          f"(per-core stall longer than its own thread's residency) = "
          f"{100.0 * anom / max(tot, 1):.1f} % residual error")
    print(f"# grid {run['env'].get('grid')}  arch {run['env'].get('arch')}  "
          f"host {run['env'].get('host')}  cap {run['env'].get('host_thread_cap')}")
    print()
    hdr = (f"{'arm':24s} {'cores':>5s} {'wall us':>9s} {'in us':>9s} {'in/T0':>7s} "
           f"{'out us':>8s} {'out/T2':>7s} {'spread':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for name in run["arms"]:
        n = name["name"]
        if n not in stat:
            continue
        s = stat[n]
        print(f"{n:24s} {s['cores']:5d} {s['wall_us']:9.2f} {s['in_us']:9.2f} "
              f"{100 * s['in_frac']:6.1f}% {s['out_us']:8.2f} {100 * s['out_frac']:6.1f}% "
              f"{s['wall_spread']:6.3f}x")

    # ---- the response table. Every knob against its OWN control, and the derivative that
    # decides whether the stall is the constraint at all.
    ctrl = {}
    for a in run["arms"]:
        k = a["knob"]
        if k in ("AA", "control"):
            continue
        base = {"kblock": "gen.AA.a", "cbdepth": "gen.AA.a", "fidelity": None,
                "quantum": None, "grid": None}.get(k)
        ctrl[k] = base

    def pick_control(a):
        k, n = a["knob"], a["name"]
        if n.startswith("gen.smallM.in0L1"):
            return "gen.smallM.dram"
        if n.startswith("gen.smallM.outL1"):
            return "gen.smallM.dram2"
        if n.startswith("mm.smallM.in0L1"):
            return "mm.smallM.dram"
        if n.startswith("gen.kb2.cb4x"):
            return "gen.kb2.cb2x"
        return "gen.AA.a" if n.startswith("gen.") else "mm.AA.a"

    floors = {}
    for pre in ("gen", "mm"):
        aa, bb = f"{pre}.AA.a", f"{pre}.AA.b"
        if aa in stat and bb in stat:
            floors[pre] = {
                "wall": max(stat[aa]["wall_us"], stat[bb]["wall_us"]) /
                        min(stat[aa]["wall_us"], stat[bb]["wall_us"]),
                "in": (abs(stat[aa]["in_frac"] - stat[bb]["in_frac"])),
            }
    print()
    for pre, f in floors.items():
        print(f"# A/A floor {pre}: wall {f['wall']:.4f}x, in/T0 {100 * f['in']:.2f} pp")
    print()
    hdr2 = (f"{'knob':16s} {'level':12s} {'d(in/T0) pp':>12s} {'d(wall)':>9s} "
            f"{'d(w)/d(s)':>10s} {'verdict':>12s}")
    print(hdr2)
    print("-" * len(hdr2))
    resp = []
    for a in run["arms"]:
        n, k = a["name"], a["knob"]
        if k in ("AA", "control") or n not in stat:
            continue
        c = pick_control(a)
        if c not in stat or c == n:
            continue
        pre = "gen" if n.startswith("gen.") else "mm"
        fl = floors.get(pre, {"wall": 1.0, "in": 0.0})
        d_in = stat[n]["in_frac"] - stat[c]["in_frac"]
        # d(stall) and d(wall) in SECONDS on the same op, so the derivative is dimensionless:
        # how many microseconds of op wall one microsecond of removed input stall buys.
        d_in_us = stat[n]["in_us"] - stat[c]["in_us"]
        d_wall_us = stat[n]["wall_us"] - stat[c]["wall_us"]
        deriv = d_wall_us / d_in_us if abs(d_in_us) > 1e-9 else float("nan")
        ratio = stat[c]["wall_us"] / stat[n]["wall_us"] if stat[n]["wall_us"] else 0.0
        moved = abs(d_in) > max(fl["in"], 0.005)
        wall_moved = max(ratio, 1 / ratio if ratio else 0) > fl["wall"]
        v = ("both" if moved and wall_moved else "stall only" if moved
             else "wall only" if wall_moved else "NULL")
        print(f"{k:16s} {a['level']:12s} {100 * d_in:11.2f} {ratio:8.4f}x {deriv:10.3f} {v:>12s}")
        resp.append({"knob": k, "level": a["level"], "arm": n, "control": c,
                     "d_in_pp": round(100 * d_in, 3), "wall_ratio": round(ratio, 5),
                     "d_wall_us": round(d_wall_us, 3), "d_in_us": round(d_in_us, 3),
                     "dwall_dstall": None if deriv != deriv else round(deriv, 4),
                     "verdict": v})

    outp = Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem + "_response.json")
    outp.write_text(json.dumps({"env": run["env"], "per_arm": stat, "floors": floors,
                                "anomaly_frac": anom / max(tot, 1), "response": resp}, indent=1))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
