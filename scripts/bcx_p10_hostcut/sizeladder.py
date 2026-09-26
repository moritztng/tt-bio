"""Is the structure module's host cost arithmetic, or is it the length of a serial chain?

The configuration ladder and the affinity ladder both came back flat, which leaves two
explanations for 1.16 s: the module is doing that much arithmetic, or it is a long chain of
small ops whose cost is latency rather than FLOPs. A size ladder separates them. The module's
work is O(n) in the single track and O(n^2) in the pair track, so a cost that is arithmetic
falls roughly with n^2 when n is cut, and a cost that is chain length barely moves.

One process, one arm per n, each jitted separately. The capture is sliced from the front, so
every arm runs real trunk statistics.
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
    ap.add_argument("--sizes", default="32,64,128,192,288")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = []
    for n in [int(v) for v in args.sizes.split(",")]:
        fwd, both, s, z, got = smbench.build(n_slice=n)
        smbench.timed(lambda: fwd(s, z), args.warm)
        smbench.timed(lambda: both(s, z), args.warm)
        f = smbench.summary(smbench.timed(lambda: fwd(s, z), args.reps))
        b = smbench.summary(smbench.timed(lambda: both(s, z), args.reps))
        arms.append({"n": got, "forward": f, "forward_and_backward": b})
        print("n %3d  fwd %6.4f s  fwd+bwd %6.4f s  %4.2f cores  load1 %5.2f"
              % (got, f["median_wall_s"], b["median_wall_s"], b["median_cores"],
                 b["median_load1"]), flush=True)

    base = arms[-1]
    for a in arms:
        a["wall_share_of_largest"] = round(
            a["forward_and_backward"]["median_wall_s"]
            / base["forward_and_backward"]["median_wall_s"], 4)
        a["n_share_of_largest"] = round(a["n"] / base["n"], 4)
    report = {"host": os.uname().nodename, "nproc": os.cpu_count(),
              "affinity_cores": len(os.sched_getaffinity(0)), "reps": args.reps,
              "xla_flags": os.environ.get("XLA_FLAGS", ""),
              "utc": time.strftime("%FT%TZ", time.gmtime()), "arms": arms}
    print(json.dumps({a["n"]: a["wall_share_of_largest"] for a in arms}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print("-> " + args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
