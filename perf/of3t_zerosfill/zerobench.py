#!/usr/bin/env python3
"""What the 36.4 ms of `zeros [384,1,384,128]` actually is, priced against the roof that binds.

`of3t-bwattrib` (noexact arm, pc card 0, 1350 MHz) put ONE verb at 35.1 % of the real 33.63 s
backward: `zeros [384,1,384,128] BFLOAT16 TILE`, 324 calls, 11.796 s self, 36.408 ms a call at
1.04 GB/s. The shape is 37.75 MB. A write-only fill at the 440 GB/s bandwidth roof that binds
this DRAM-bound workload is 0.086 ms, so the production path is ~424x off its own roof.

424x off a WRITE-ONLY roof is not a slow kernel, it is the wrong operation, and this repo
already names it: `tt_bio/autograd.py:34` -- "`ttnn.zeros(..., device=)` builds its zeros on
the host and uploads them". That is a PCIe transfer, and the backward's own measured PCIe rate
is ~2.2 GB/s. This script is the break control at op level: same shape, same dtype, same
layout, same device, five ways of getting a zero tensor, each timed with the queue drained so
the number is the op's and not the dispatch queue's.

    A  zeros_host    ttnn.zeros(shape, device=dev)        the production path
    B  zeros_like    ttnn.zeros_like(rows)                what ag.DEVICE_ZEROS already selects
    C  full          ttnn.full(shape, 0.0, device=dev)    a device fill by another name
    D  cache+clone   one cached zero, ttnn.clone per call  read+write, 2x the roof traffic
    E  cached        one cached zero handed out directly   no traffic at all

and then the whole slot-backward of `_v_create_qkv_heads` end to end, A against B, because a
verb-level win that the concat swallows is not a win.

    python3 perf/of3t_zerosfill/zerobench.py --out perf/of3t_zerosfill/out/zerobench.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                      # noqa: E402

import ttnn                                                              # noqa: E402

NS = time.perf_counter_ns

# The pair track at crop 384, exactly as the backward issues it.
SHAPE = [384, 1, 384, 128]
NBYTES = 384 * 1 * 384 * 128 * 2          # BFLOAT16
ROOF_GB_S = 440.0                          # of3t/BACKWARD.md, measured machine balance 247-338


def _sync(dev):
    ttnn.synchronize_device(dev)


def timeit(fn, dev, reps, warmup=2):
    """Median and spread of `fn`, queue drained each rep so the number is the op's."""
    for _ in range(warmup):
        t = fn()
        _sync(dev)
        del t
    xs = []
    for _ in range(reps):
        t0 = NS()
        t = fn()
        _sync(dev)
        xs.append((NS() - t0) / 1e6)
        del t
    return {"ms_median": round(statistics.median(xs), 4),
            "ms_min": round(min(xs), 4), "ms_max": round(max(xs), 4), "n": reps}


def rate(ms, nbytes=NBYTES):
    gb_s = (nbytes / 1e9) / (ms / 1e3)
    return {"gb_s": round(gb_s, 2), "x_off_roof": round(ROOF_GB_S / gb_s, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default="perf/of3t_zerosfill/out/zerobench.json")
    a = ap.parse_args()

    res = {"doc": __doc__.splitlines()[0], "argv": sys.argv, "shape": SHAPE,
           "bytes": NBYTES, "roof_gb_s": ROOF_GB_S,
           "env": {"host": socket.gethostname(),
                   "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                            capture_output=True, text=True).stdout.strip(),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "loadavg": os.getloadavg(),
                   "board": "pc card 0 -- Blackhole p150a, custom 130-core firmware"},
           "ops": {}, "slot_backward": {}}

    dev = ttnn.open_device(device_id=0)
    try:
        with during() as clk:
            dt, lay = ttnn.bfloat16, ttnn.TILE_LAYOUT

            rows = ttnn.zeros(SHAPE, dtype=dt, layout=lay, device=dev)
            _sync(dev)
            cached = ttnn.zeros(SHAPE, dtype=dt, layout=lay, device=dev)
            _sync(dev)

            plans = [
                ("A_zeros_host", lambda: ttnn.zeros(SHAPE, dtype=dt, layout=lay, device=dev)),
                ("B_zeros_like", lambda: ttnn.zeros_like(rows)),
                ("C_full", lambda: ttnn.full(SHAPE, 0.0, dtype=dt, layout=lay, device=dev)),
                ("D_cache_clone", lambda: ttnn.clone(cached)),
                ("E_cached", lambda: cached),
            ]
            for name, fn in plans:
                try:
                    r = timeit(fn, dev, a.reps)
                    r.update(rate(r["ms_median"]))
                    res["ops"][name] = r
                except Exception as e:                                   # noqa: BLE001
                    res["ops"][name] = {"error": f"{type(e).__name__}: {e}"}
                print(name, res["ops"][name], flush=True)

            # --- the whole slot backward, A against B ------------------------------------
            # `_v_create_qkv_heads`' bw for one slot at B=384, L=384, H*dh=128: the zero, the
            # three-part concat into the packed width, and nothing else. `rows` here stands in
            # for merge_heads_value(g), which is a separate verb and not this row's subject.
            B, L, W = 384, 384, 128

            def slot_bw(zero_fn, s=0):
                def go():
                    zero = zero_fn()
                    parts = [rows if i == s else zero for i in range(3)]
                    return ttnn.concat(parts, dim=-1)
                return go

            for name, zf in (("A_zeros_host", lambda: ttnn.zeros(SHAPE, dtype=dt, layout=lay,
                                                                 device=dev)),
                             ("B_zeros_like", lambda: ttnn.zeros_like(rows))):
                try:
                    r = timeit(slot_bw(zf), dev, max(4, a.reps // 3))
                    # concat writes 3x the slot width and reads 3x, so its own roof traffic
                    # is 6 * NBYTES; the zero adds 1x write on top.
                    r["gb_s_concat_traffic"] = round((6 * NBYTES / 1e9) /
                                                     (r["ms_median"] / 1e3), 2)
                    res["slot_backward"][name] = r
                except Exception as e:                                   # noqa: BLE001
                    res["slot_backward"][name] = {"error": f"{type(e).__name__}: {e}"}
                print("slot", name, res["slot_backward"][name], flush=True)

        res["env"]["aiclk_during"] = clk.summary()
        res["env"]["aiclk_line"] = clk.line()
    finally:
        ttnn.close_device(dev)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(res["env"].get("aiclk_line"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
