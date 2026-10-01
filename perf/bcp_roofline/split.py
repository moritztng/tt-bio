#!/usr/bin/env python3
"""Host against device, per warm round, for serial (N=1) and interleaved (N>1) arms.

    split.py out/a1 out/b1 ...

N=1: a round is two consecutive `sequence_gradients` entries. The device column is the wall inside
the three on-card seams (evoformer/extra_msa/template x taped/backward/primal); host is the round
less that. Rounds 1 (jit compile) and 2 (lazy trunk load) are dropped.
N>1: the round is pro rata over the window every slot was running (perf/bcx_default/report.py's
rule); the device column is the gate's held time, since the seam events carry no slot.
AICLK and loadavg are the card-node samples inside the window.
"""
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bcx_default"))
import report as R  # noqa: E402


def serial(d):
    ev, clk = d["events"], d["aiclk"]
    starts = sorted(e["t0"] for e in ev if e["kind"] == "round_start")
    stop = [e["t0"] for e in ev if e["kind"] == "round_stop"]
    b = starts + stop[:1]
    rows = []
    for i in range(2, len(b) - 1):                          # drop rounds 1 and 2
        t0, t1 = b[i], b[i + 1]
        dev = sum(e["dt"] for e in ev if e["kind"] == "device"
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6)
        c = sorted(x[1] for x in clk if t0 <= x[0] <= t1)
        rows.append((t1 - t0, dev, c[len(c) // 2] if c else None, c[0] if c else None))
    return rows


for p in sys.argv[1:]:
    d = json.loads((pathlib.Path(p) / "round_events.json").read_text())
    n = d["stamp"].get("trajectories", 1)
    if not d["stamp"].get("interleave"):
        rows = serial(d)
        w = st.median(r[0] for r in rows)
        dv = st.median(r[1] for r in rows)
        h = st.median(r[0] - r[1] for r in rows)
        print(f"{pathlib.Path(p).name} N=1 warm rounds {len(rows)}: wall {w:.3f} s, device {dv:.3f} s, "
              f"host {h:.3f} s ({h / w * 100:.1f} %), AICLK med {st.median(r[2] for r in rows):.0f} "
              f"min {min(r[3] for r in rows)}; walls {[round(r[0], 3) for r in rows]}")
    else:
        o, _ = R.one(p)
        print(f"{o['arm']} N={n} round {o['round_s']:.3f} s, held {o['held_per_round']:.3f}, "
              f"idle {o['idle_per_round']:.3f}, AICLK {o.get('clock')}, hwm {o['host_hwm_gb']} GB")
