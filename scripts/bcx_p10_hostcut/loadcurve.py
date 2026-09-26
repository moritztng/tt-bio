"""What co-tenant load does to the structure module, measured rather than argued.

`bcx-p10-hostfloor` found the composed round's host column reads 3.600 s on qb1 at loadavg 16.9
and 1.948 s on qb2 at loadavg 7.4, and ruled that a host number on this workload is not portable
across boxes. That is right, and it leaves the campaign without a rule: it can say a loaded
number is wrong but not by how much.

This measures the transfer function on ONE box, so nothing is subtracted across hosts. The
module is timed against a ladder of spinners this process starts and reaps itself, all in one
process so the program is compiled once and every arm runs the identical executable. Each arm
reports the spinner count, the loadavg1 that count actually produced, and the module's wall.

Loadavg1 is a one-minute EWMA, so every arm holds its spinners for `--settle` seconds before it
times anything; without that the early arms report the previous arm's load.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time


def _spin(stop):
    x = 0.0
    while not stop.is_set():
        for _ in range(100000):
            x += 1.0
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spinners", default="0,2,4,6,8,12,16,24",
                    help="comma-separated co-tenant spinner counts, in the order run")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--settle", type=float, default=75.0,
                    help="seconds to hold the spinners before timing, so loadavg1 catches up")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import smbench
    fwd, both, s, z, n = smbench.build()
    smbench.timed(lambda: fwd(s, z), args.warm)
    smbench.timed(lambda: both(s, z), args.warm)

    ctx = mp.get_context("fork")
    arms = []
    for k in [int(v) for v in args.spinners.split(",")]:
        stop = ctx.Event()
        procs = [ctx.Process(target=_spin, args=(stop,), daemon=True) for _ in range(k)]
        for p in procs:
            p.start()
        t0 = time.time()
        while time.time() - t0 < args.settle:
            time.sleep(1.0)
        rows = smbench.timed(lambda: both(s, z), args.reps)
        f_rows = smbench.timed(lambda: fwd(s, z), args.reps)
        stop.set()
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        arm = {"spinners": k, "forward": smbench.summary(f_rows),
               "forward_and_backward": smbench.summary(rows)}
        arms.append(arm)
        b = arm["forward_and_backward"]
        print("spinners %2d  load1 %5.2f  fwd+bwd %6.3f s  %4.2f cores  fwd %6.3f s"
              % (k, b["median_load1"], b["median_wall_s"], b["median_cores"],
                 arm["forward"]["median_wall_s"]), flush=True)

    report = {"n": n, "host": os.uname().nodename, "nproc": os.cpu_count(),
              "affinity_cores": len(os.sched_getaffinity(0)), "reps": args.reps,
              "settle_s": args.settle, "xla_flags": os.environ.get("XLA_FLAGS", ""),
              "utc": time.strftime("%FT%TZ", time.gmtime()), "arms": arms}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print("-> " + args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
