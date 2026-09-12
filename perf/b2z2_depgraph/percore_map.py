#!/usr/bin/env python3
"""Per-core map of one Pairformer block, straight out of the device profiler log.

The ops CSV only ever gives you a number summed over the cores that ran an op. That is enough to
say *which op* waits and useless for saying *which core* waits, which is the question this row
exists to answer. The device log has one row per zone marker per core on a common device clock,
so it can answer both, plus the one the ops CSV structurally cannot: who was everybody waiting for.

Emits, per replayed block:
  * per-core input-tile wait (CB-COMPUTE-WAIT-FRONT, ZONE_TOTAL accumulator, per RISC)
  * per-op per-core kernel begin/end, so a straggler surplus can be computed
  * per-op core coordinates, so wait can be regressed on position along a grid axis

Usage: percore_map.py <profile_log_device.csv> --ops-per-block N --reps R --out out.json
"""
import argparse
import json
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--ops-per-block", type=int, required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # op index is run host ID / 1024; the dispatcher bumps it by 1024 per program.
    kern = defaultdict(dict)        # (op, core) -> {risc: [start, end]}
    wait = defaultdict(float)       # (op, core, risc) -> accumulated cycles
    resv = defaultdict(float)
    ops_seen = set()
    unclosed = 0

    with open(a.log) as fh:
        fh.readline(), fh.readline()          # arch header, column header
        for line in fh:
            f = line.split(",", 12)
            if len(f) < 12:
                continue
            try:
                op = int(f[7]) // 1024
            except ValueError:
                continue
            t = int(f[5])
            if t == 0:
                continue                       # flush artefact, no timestamp
            zone, phase = f[10], f[11]
            risc = f[3]
            core = (int(f[1]), int(f[2]))
            if zone.endswith("-KERNEL"):
                d = kern[(op, core)].setdefault(risc, [None, None])
                if phase == "ZONE_START":
                    d[0] = t
                elif phase == "ZONE_END":
                    d[1] = t
                ops_seen.add(op)
            elif zone == "CB-COMPUTE-WAIT-FRONT":
                wait[(op, core, risc)] += float(f[6])
            elif zone == "CB-COMPUTE-RESERVE-BACK":
                resv[(op, core, risc)] += float(f[6])

    # the census replays the block `reps` times at the end of the run: take the last block.
    all_ops = sorted(ops_seen)
    need = a.ops_per_block * a.reps
    if len(all_ops) < need:
        print(f"only {len(all_ops)} ops in log, need {need}", file=sys.stderr)
        return 1
    window = all_ops[-need:]
    by_op_kern = defaultdict(dict)
    for (o, core), d in kern.items():
        by_op_kern[o][core] = d
    by_op_wait = defaultdict(lambda: defaultdict(float))
    for (o, core, _risc), v in wait.items():
        by_op_wait[o][core] += v
    by_op_resv = defaultdict(lambda: defaultdict(float))
    for (o, core, _risc), v in resv.items():
        by_op_resv[o][core] += v
    blocks = [window[i * a.ops_per_block:(i + 1) * a.ops_per_block] for i in range(a.reps)]

    out = {"log": a.log, "ops_per_block": a.ops_per_block, "reps": a.reps,
           "ops_in_log": len(all_ops), "window": [window[0], window[-1]], "blocks": []}

    for bi, block in enumerate(blocks):
        per_core_wait = defaultdict(float)
        per_core_resv = defaultdict(float)
        per_core_busy = defaultdict(float)
        op_rows = []
        for oi, op in enumerate(block):
            cores = {}
            for core, d in by_op_kern[op].items():
                st = [v[0] for v in d.values() if v[0] is not None]
                en = [v[1] for v in d.values() if v[1] is not None]
                if not st or not en:
                    continue
                cores[core] = (min(st), max(en))
            if not cores:
                continue
            first = min(v[0] for v in cores.values())
            last = max(v[1] for v in cores.values())
            surplus = sum(last - v[1] for v in cores.values())
            # who is last: the core that gates every other core on this op
            gate = max(cores.items(), key=lambda kv: kv[1][1])[0]
            for core, (s, e) in cores.items():
                per_core_busy[core] += e - s
            w = by_op_wait[op]
            r = by_op_resv[op]
            for core, v in w.items():
                per_core_wait[core] += v
            for core, v in r.items():
                per_core_resv[core] += v
            op_rows.append({
                "op": op, "i": oi, "cores": len(cores),
                "span": last - first, "first_start": first, "last_end": last,
                "first_to_last_start": max(v[0] for v in cores.values()) - first,
                "straggler_surplus": surplus,
                "gate_core": list(gate),
                "wait_total": sum(w.values()),
                "percore": {f"{c[0]},{c[1]}": [cores[c][0] - first, cores[c][1] - first,
                                               w.get(c, 0.0), r.get(c, 0.0)]
                            for c in cores},
            })
        out["blocks"].append({
            "block": bi,
            "span": max(o["last_end"] for o in op_rows) - min(o["first_start"] for o in op_rows),
            "per_core_wait": {f"{c[0]},{c[1]}": v for c, v in per_core_wait.items()},
            "per_core_resv": {f"{c[0]},{c[1]}": v for c, v in per_core_resv.items()},
            "per_core_busy": {f"{c[0]},{c[1]}": v for c, v in per_core_busy.items()},
            "ops": op_rows,
        })

    with open(a.out, "w") as fh:
        json.dump(out, fh)
    b = out["blocks"][len(out["blocks"]) // 2]
    print(f"ops in log {len(all_ops)}  window {window[0]}..{window[-1]}  blocks {len(blocks)}")
    print(f"median block: span {b['span'] / 1e6:.4f} ms  ops {len(b['ops'])}  "
          f"cores {len(b['per_core_wait'])}  wait {sum(b['per_core_wait'].values()) / 1e6:.3f} ms-cores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
