#!/usr/bin/env python3
"""The instrument's float64 arithmetic per Evoformer block, across the token counts BC2 runs.

Card-free. `blockcount.py` measures the traced n=288 census shape by shape. This reduces that
census to the three tensors that carry it and sweeps n, so the floor can be put beside a round
measured at a different length.

The reduction is checked, not assumed. At n=288 it gives

    softmax     2 x (n, 4, n, n)      191,102,976 elements   against 192,439,296 traced  (-0.7 %)
    layer_norm  8 x (n, n, 128)        84,934,656 elements   against  85,524,480 traded  (-0.7 %)

so the two triangle attentions and the eight pair layer norms ARE the block, and everything
else in the census is below a percent.

Same exclusions as `blockcount.py`: no `ttnn.to_torch` off the card, no `ttnn.from_torch` back,
no backward. This is a LOWER bound on the forward half of the instrument, per block.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics as st
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import torch                                                           # noqa: E402

from blockcount import time_layer_norm, time_softmax                   # noqa: E402


def block(n, reps):
    sm = time_softmax((n, 4, n, n), reps)
    ln = time_layer_norm((n, n, 128), reps)
    return {"n": n, "softmax_calls": 2, "layer_norm_calls": 8,
            "softmax_elements": 2 * n * 4 * n * n, "layer_norm_elements": 8 * n * n * 128,
            "softmax_s_median": round(st.median(sm), 6),
            "layer_norm_s_median": round(st.median(ln), 6),
            "softmax_s_per_block": round(2 * st.median(sm), 4),
            "layer_norm_s_per_block": round(8 * st.median(ln), 4),
            "s_per_block": round(2 * st.median(sm) + 8 * st.median(ln), 4),
            "s_per_48_block_forward": round(48 * (2 * st.median(sm) + 8 * st.median(ln)), 2),
            "loadavg1": round(os.getloadavg()[0], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="224,256,288")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "LADDER.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    # One untimed pass so the allocator's first-touch cost does not land on the first n.
    time_softmax((64, 4, 64, 64), 1)
    rows = [block(int(x), args.reps) for x in args.ns.split(",")]
    blob = {"host": os.uname().nodename, "when_utc": time.strftime("%FT%TZ", time.gmtime()),
            "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
            "opened_a_device": False, "threads": args.threads, "reps": args.reps,
            "nproc": os.cpu_count(), "loadavg": os.getloadavg(), "rows": rows}
    pathlib.Path(args.out).write_text(json.dumps(blob, indent=1))
    for r in rows:
        print(json.dumps(r))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
