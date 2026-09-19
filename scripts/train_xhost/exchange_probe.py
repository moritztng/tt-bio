#!/usr/bin/env python3
"""What one step's gradient exchange costs, with the real payload and the real transport.

    # on qb1                                    # on qb2
    python3 exchange_probe.py --rank 0 \        python3 exchange_probe.py --rank 1 \
      --rendezvous /dev/shm/xp+ttuser@qb2         --rendezvous /dev/shm/xp+ttuser@qb1

No card is opened: the exchange happens after the step has pulled every gradient down to a
float32 host mirror, so it can be measured on its own and the number belongs to the link and the
transport rather than to the model. 7,111,515 float32 is ABodyBuilder3's parameter count, and the
array is built from integer arithmetic so both hosts hold bit-identical inputs with no file
crossing the wire to make them so.

Each round reports the full ``allreduce``: write, mirror, wait for the peer, sum in rank order.
That is the quantity the step pays. The digest of the sum is compared across ranks by the reduce
itself, so a fast exchange that lost the last mantissa bit fails here rather than in a run.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np

from tt_bio.train.hostreduce import HostReduce, master_hash

PARAMS = 7_111_515


def payload(n: int, rank: int) -> np.ndarray:
    """Integer arithmetic, so numpy 1.26 and numpy 2.5 build the same bits on both hosts."""
    return (np.arange(n, dtype=np.int64) % 1000 + rank).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--rendezvous", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--params", type=int, default=PARAMS)
    args = ap.parse_args()

    comm = HostReduce(args.rendezvous, args.rank, args.world, timeout=300.0)
    vec = payload(args.params, args.rank)
    times, digests = [], []
    try:
        for i in range(1, args.rounds + 1):
            t0 = time.perf_counter()
            total = comm.allreduce(vec, step=i)
            d = master_hash([total])
            comm.check_equal(d, step=i)
            times.append(time.perf_counter() - t0)
            digests.append(d.hex())
    finally:
        out = {"host": platform.node(), "rank": args.rank, "world": args.world,
               "python": platform.python_version(), "numpy": np.__version__,
               "bytes_per_rank": int(vec.nbytes), "peers": comm.peers,
               "rounds": [round(t, 3) for t in times],
               "median_s": round(statistics.median(times), 3) if times else None,
               "digests": sorted(set(digests)),
               "sent_mb": round(comm.transport.bytes_sent / 1e6, 1) if comm.transport else 0.0}
        print("PROBE " + json.dumps(out), flush=True)
        comm.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
