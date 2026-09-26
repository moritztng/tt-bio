"""How many cores AF2's structure module actually uses, and what more of them buy.

`bcx-p10-hostfloor` put the module at 1.307 s of the composed round's 1.948 s host floor and
measured 4.51 cores of 16 over the whole XLA host compute. That average cannot say whether the
module is thread-starved or thread-saturated, because its window also holds six gaps that are
serial by construction.

This runs the module alone, jitted, at the captured n=288, and reports wall AND this process's
CPU per rep. `cores = cpu/wall` is the module's own occupancy; sweeping the affinity mask gives
its scaling curve. No knob RAISES the XLA:CPU pool above the visible core count -- jaxlib builds
it from `tsl::port::MaxParallelism()`, which reads the affinity mask -- so a ladder down from
all cores is the only A/B available, and a curve already flat at the top is the finding.

`scripts/bcx_p10_hostcut/flagsweep.sh` drives this over the XLA:CPU configuration ladder and
over the affinity ladder. See `smbench.py` for what the cotangents are.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smbench                                                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=smbench.REF)
    ap.add_argument("--bc2", default=smbench.BC2)
    ap.add_argument("--params", default=smbench.PARAMS)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fwd, both, s, z, n = smbench.build(args.ref, args.bc2, args.params)
    import jax

    smbench.timed(lambda: fwd(s, z), args.warm)
    smbench.timed(lambda: both(s, z), args.warm)
    f_rows = smbench.timed(lambda: fwd(s, z), args.reps)
    b_rows = smbench.timed(lambda: both(s, z), args.reps)

    report = {"tag": args.tag, "n": n, "host": os.uname().nodename, "dtype": "float32",
              "reps": args.reps, "warm": args.warm,
              "affinity_cores": len(os.sched_getaffinity(0)), "nproc": os.cpu_count(),
              "xla_flags": os.environ.get("XLA_FLAGS", ""), "jax": jax.__version__,
              "utc": time.strftime("%FT%TZ", time.gmtime()),
              "forward": smbench.summary(f_rows),
              "forward_and_backward": smbench.summary(b_rows)}
    report["backward_implied_s"] = round(
        report["forward_and_backward"]["median_wall_s"] - report["forward"]["median_wall_s"], 4)
    print(json.dumps(report, indent=1, sort_keys=True), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
