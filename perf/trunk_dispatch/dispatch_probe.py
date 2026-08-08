#!/usr/bin/env python3
"""Separate host dispatch from device compute for one real Protenix-v2 PairformerLayer.

The 298-aa campaign closed with "the trunk is ~91.5% host-dispatch-bound". This probe
measures that claim directly. Three timings on the same warm layer:

  serial   sync; t0; block(); sync  -> t_total   (what pf_layer.py --mode bench reports)
  issue    sync; t0; block(); t1    -> t_issue   (host returns; device still working)
  pipe     sync; t0; K x block(); sync -> per-block wall with dispatch overlapped,
                                          the in-model analogue

t_issue is an UPPER bound on pure host dispatch cost: if the device back-pressures the
command queue mid-block the host blocks inside the call. So t_issue << t_total is a sound
proof of device-boundedness; the reverse is ambiguous, which is why --n 64 is also run
(device time per op is tiny there, so t_issue is the pure host cost of ~the same op count).

    TT_VISIBLE_DEVICES=3 python3 perf/trunk_dispatch/dispatch_probe.py --n 320 --n 128 --n 64
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_split_298"))
from pf_layer import build_layer, TRI_HEAD_DIM  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def med(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, action="append", required=True)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--pipe", type=int, default=8, help="blocks chained per pipelined leg")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = build_layer(ckc)
    results = []

    for N in args.n:
        torch.manual_seed(0)
        s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for _ in range(args.warm):
            s, z = layer(s, z)
        ttnn.synchronize_device(dev)

        serial, issue = [], []
        for _ in range(args.iters):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            s, z = layer(s, z)
            t1 = time.perf_counter()
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            issue.append((t1 - t0) * 1e3)
            serial.append((t2 - t0) * 1e3)

        pipe = []
        for _ in range(3):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.pipe):
                s, z = layer(s, z)
            ttnn.synchronize_device(dev)
            pipe.append((time.perf_counter() - t0) * 1e3 / args.pipe)

        r = dict(n=N, c_z=c_z,
                 serial_ms=round(med(serial), 2), issue_ms=round(med(issue), 2),
                 pipe_ms=round(med(pipe), 2),
                 serial_series=[round(x, 2) for x in serial],
                 issue_series=[round(x, 2) for x in issue],
                 pipe_series=[round(x, 2) for x in pipe])
        r["issue_frac_of_serial"] = round(r["issue_ms"] / r["serial_ms"], 3)
        results.append(r)
        print(f"N={N:4d} c_z={c_z}  serial={r['serial_ms']:8.2f} ms  "
              f"issue={r['issue_ms']:8.2f} ms ({100*r['issue_frac_of_serial']:.1f}%)  "
              f"pipe={r['pipe_ms']:8.2f} ms", flush=True)
        ttnn.deallocate(z)
        ttnn.deallocate(s)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
