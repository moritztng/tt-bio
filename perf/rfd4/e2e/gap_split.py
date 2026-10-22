"""Where a profiled trunk call's time goes: kernels, device gaps between them, and host enqueue.

    python perf/rfd4/e2e/gap_split.py <ops.csv> [--aiclk 1350] [--begin elt_prof] [--end elt_end]

Reads the ops_perf_results csv an `elt_prof.sh` run writes and takes the ops between the two
signposts (the measured call). If the host enqueue span is well under the device span, the host
stayed ahead and the card never waited on it, so a trace replay of that call has nothing to remove.
The profiler slows the host, so a host-bound reading here is not proof the unprofiled run is; the
eager vs traced call times from pass_bench.py are the check for that.
"""
import argparse
import csv
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--aiclk", type=float, default=1350.0, help="MHz, converts device cycles to time")
    ap.add_argument("--begin", default="elt_prof")
    ap.add_argument("--end", default="elt_end")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    i0 = next(i for i, r in enumerate(rows) if r["OP CODE"] == a.begin)
    i1 = next(i for i, r in enumerate(rows) if r["OP CODE"] == a.end)
    ops = [r for r in rows[i0 + 1:i1] if r["OP TYPE"] != "signpost"]
    f = lambda r, k: float(r[k] or 0)  # noqa: E731
    kernel = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in ops) / 1e6
    gaps = [f(r, "OP TO OP LATENCY [ns]") for r in ops[1:]]
    dev = [r for r in ops if r["DEVICE FW START CYCLE"]]
    span = (int(dev[-1]["DEVICE FW END CYCLE"]) - int(dev[0]["DEVICE FW START CYCLE"])) / a.aiclk / 1e3
    host = (int(ops[-1]["HOST END TS"]) - int(ops[0]["HOST START TS"])) / 1e6
    print(f"ops {len(ops)}  kernel {kernel:.1f} ms  op-to-op {sum(gaps) / 1e6:.1f} ms  "
          f"device span {span:.1f} ms  host enqueue span {host:.1f} ms")
    buckets = Counter()
    for g in gaps:
        us = g / 1e3
        buckets["<10us" if us < 10 else "10-50us" if us < 50 else "50-500us" if us < 500 else ">500us"] += g / 1e6
    print("op-to-op ms by gap size:", {k: round(v, 1) for k, v in sorted(buckets.items())})


if __name__ == "__main__":
    main()
