#!/usr/bin/env python3
"""Measure the host roof that actually binds the ESM-C embed job: zlib compression of float32.

The embed wall is not on the chip, so quoting a DRAM roof against it would be the wrong roof.
The binding term is np.savez_compressed, i.e. zlib at level 6 over float32 embedding bytes.
Measure that rate on real-shaped data, single-threaded (the roof one core can reach) and
threaded (the roof write_npz_many can reach), so the measured wall can be placed against a roof
rather than against an assertion.

Reports MB/s of *input* bytes, which is the rate the byte model needs: the job's input to the
writer is n_seqs * L * d_model * 4 bytes.
"""
import argparse
import json
import os
import statistics
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--residues", type=int, default=76)
    ap.add_argument("--d-model", type=int, default=960)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Realistic content: embedding rows are correlated and roughly gaussian, not white noise.
    # Compression rate depends on content, so this is generated to land near the ratio the real
    # output shows (1.14 MB out of 2.33 MB in for 8x76x960 float32, i.e. ~0.49).
    rng = np.random.default_rng(0)
    base = rng.standard_normal((args.residues, args.d_model), dtype=np.float32)
    blocks = [np.ascontiguousarray(base + rng.standard_normal((1, args.d_model), dtype=np.float32))
              for _ in range(args.n_seqs)]
    raw = [b.tobytes() for b in blocks]
    in_bytes = sum(len(r) for r in raw)

    def serial():
        return [zlib.compress(r, 6) for r in raw]

    workers = max(1, min(32, os.cpu_count() or 8))

    def threaded():
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda r: zlib.compress(r, 6), raw))

    def timed(fn):
        fn()  # warm
        ts = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            out = fn()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts), sum(len(o) for o in out)

    t_ser, out_ser = timed(serial)
    t_thr, out_thr = timed(threaded)

    res = dict(
        host=os.uname().nodename, cpus=os.cpu_count(), workers=workers,
        loadavg=round(os.getloadavg()[0], 2),
        n_seqs=args.n_seqs, residues=args.residues, d_model=args.d_model,
        input_bytes=in_bytes, output_bytes=out_ser,
        compress_ratio=round(out_ser / in_bytes, 4),
        serial_s=round(t_ser, 4),
        threaded_s=round(t_thr, 4),
        serial_MBps=round(in_bytes / t_ser / 1e6, 2),
        threaded_MBps=round(in_bytes / t_thr / 1e6, 2),
        thread_speedup=round(t_ser / t_thr, 3),
    )
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
