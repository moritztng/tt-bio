"""Can the DiT's 18 per-block pair-bias projections be hoisted into one matmul, and does
`ttnn.slice` eat the saving?

`model.py:1492` is the largest single op in RFD3: `_tuned_linear([1,685,704,128], [128,32])`,
36 calls/step (18 blocks x 2 recycles), 26.742 ms/step, 42.6 % of this card's measured 390.0 GB/s
read roof (perf/p56/linear_census.json). Every one of the 18 blocks in a recycle reads the SAME
`z` tensor object -- `LocalTokenTransformer.run_device` loops `a = block(a, s, z, additive_mask)`
with one `z` -- so the input is read 18 times to produce 18 disjoint 16-wide slabs of output.

Fused, that is one `[128, 288]` projection reading `z` once. P3.21c priced it at ~20 ms/step and
then blocked it on `ttnn.slice` being a copy rather than a view
(tt-bio-ttnn-slice-not-a-view-and-allocation-order-sensitivity) -- but the slice's cost was never
measured, only assumed to exceed the saving. This measures it.

Packing matters. Each block needs 16 columns (N_HEAD=16). Packed at 16 the block-i slab starts at
column 16*i, so odd blocks start mid-tile, and p52 measured that a piece narrower than a 32-wide
tile puts the whole op 15-20x below its bandwidth floor. So the shipping packing pads each block's
slot to 32 (fused N = 576, half padding) and every slice starts tile-aligned. Arm E is the naive
16-wide packing, kept to confirm that cliff rather than assert it.

Write volume is unchanged by the padding: today's 18 outputs are logically 16 wide and already
tile-padded to 32, so 18x32 = 576 is exactly what ships today.

Traffic model, bf16, one recycle (685 x 704 = 482,240 tiled elements per channel):

    shipped  18 x (123.4 MB in + 30.9 MB out) = 2.221 GB read + 0.556 GB write
    fused     1 x (123.4 MB in + 277.7 MB out) = 0.123 GB read + 0.278 GB write
    slices   18 x (30.9 MB in + 30.9 MB out)   = 0.556 GB read + 0.556 GB write
    fused route                                 0.679 GB read + 0.834 GB write

PREDICTION, written before the run: A ~13.4 ms per recycle (half the census 26.742), D 6.5-7.0 ms,
so 12-14 ms/step saved. KILL GATE: if D >= A - 4.0 ms per recycle (8 ms/step), NO-GO -- ttnn.slice
has priced the fusion out for the second time, and it is written down as such.

    ~/.coworker/scripts/benchlock.sh rfd3-matched-batch-denominator-reopen -- env \
      TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:rfd3-matched-batch-denominator-reopen \
      RFD3_TUNE_MATMUL=1 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p60_pairbias_fusion.py
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                     # noqa: E402
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p60/pairbias_fusion.json")
I = 685                 # tokens at the page fixture (585 target + 100 binder)
C_PAIR = 128
N_HEAD = 16
N_BLOCK = 18
SLOT = 32               # tile-aligned slot per block; 16 real columns + 16 of padding
NWARM, NREP = 2, 6
RD_ROOF, WR_ROOF = 390.0e9, 269.6e9


def timeit(fn, dev):
    for _ in range(NWARM):
        out = fn()
        _free(out)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(NREP):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        _free(out)
    return statistics.median(ts) * 1e3, min(ts) * 1e3, max(ts) * 1e3


def _free(out):
    if out is None:
        return
    for t in (out if isinstance(out, (list, tuple)) else [out]):
        ttnn.deallocate(t)


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    dt = ttnn.bfloat16

    def lin(x, w):
        return M._tuned_linear(x, w, ckc=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)

    g = torch.Generator().manual_seed(0)
    p = ttnn.from_torch(torch.randn(1, I, I, C_PAIR, generator=g), dtype=dt,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    w16 = [ttnn.from_torch(torch.randn(C_PAIR, N_HEAD, generator=g), dtype=dt,
                           layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(N_BLOCK)]
    # block i occupies columns [SLOT*i, SLOT*i + N_HEAD); the rest is zero padding.
    wide = torch.zeros(C_PAIR, SLOT * N_BLOCK)
    for i in range(N_BLOCK):
        wide[:, SLOT * i:SLOT * i + N_HEAD] = torch.randn(C_PAIR, N_HEAD, generator=g)
    w576 = ttnn.from_torch(wide, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    w288 = ttnn.from_torch(torch.randn(C_PAIR, N_HEAD * N_BLOCK, generator=g), dtype=dt,
                           layout=ttnn.TILE_LAYOUT, device=dev)

    rows = []

    def record(arm, ms, lo, hi, note=""):
        rows.append({"arm": arm, "ms_per_recycle": round(ms, 4), "min": round(lo, 4),
                     "max": round(hi, 4), "ms_per_step": round(2 * ms, 4), "note": note})
        print("%-46s %8.4f ms/recycle  (%7.4f - %7.4f)  %8.4f ms/step  %s"
              % (arm, ms, lo, hi, 2 * ms, note), flush=True)

    # A -- what ships: 18 separate [128,16] projections of the same p.
    record("A shipped 18 x linear[128,16]",
           *timeit(lambda: [lin(p, w) for w in w16], dev))

    # B -- the fused matmul alone.
    record("B fused 1 x linear[128,576]",
           *timeit(lambda: lin(p, w576), dev))

    # C -- the 18 tile-aligned 16-wide slices out of the fused result.
    y576 = lin(p, w576)
    ttnn.synchronize_device(dev)
    record("C 18 x slice 16-wide @ tile-aligned 32*i",
           *timeit(lambda: [ttnn.slice(y576, [0, 0, 0, SLOT * i], [1, I, I, SLOT * i + N_HEAD])
                            for i in range(N_BLOCK)], dev))

    # D -- the buildable route, one arm, not A+C added.
    def route():
        y = lin(p, w576)
        return [ttnn.slice(y, [0, 0, 0, SLOT * i], [1, I, I, SLOT * i + N_HEAD])
                for i in range(N_BLOCK)] + [y]

    record("D fused + 18 slices (the route)", *timeit(route, dev))
    ttnn.deallocate(y576)

    # E -- the naive 16-wide packing, to confirm the p52 sub-tile cliff rather than assert it.
    y288 = lin(p, w288)
    ttnn.synchronize_device(dev)
    record("E 18 x slice 16-wide @ mid-tile 16*i",
           *timeit(lambda: [ttnn.slice(y288, [0, 0, 0, N_HEAD * i], [1, I, I, N_HEAD * (i + 1)])
                            for i in range(N_BLOCK)], dev),
           "naive packing")
    ttnn.deallocate(y288)

    a = next(r for r in rows if r["arm"].startswith("A"))["ms_per_recycle"]
    d = next(r for r in rows if r["arm"].startswith("D"))["ms_per_recycle"]
    saved_step = 2 * (a - d)
    gate = "GO" if saved_step >= 8.0 else "NO-GO"
    print("\nA %.4f  D %.4f  saved %.4f ms/step = %.3f s/design over 200 steps  -> %s"
          % (a, d, saved_step, saved_step * 200 / 1e3, gate), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"rows": rows, "tokens": I, "c_pair": C_PAIR, "n_head": N_HEAD, "n_block": N_BLOCK,
         "slot": SLOT, "n_warm": NWARM, "n_rep": NREP, "host": "qb2", "card": 2,
         "ttnn": "0.68.0", "tune_matmul": os.environ.get("RFD3_TUNE_MATMUL"),
         "saved_ms_per_step": round(saved_step, 4),
         "saved_s_per_design": round(saved_step * 200 / 1e3, 4),
         "kill_gate_ms_per_step": 8.0, "verdict": gate,
         "read_roof_GB_s_measured": RD_ROOF / 1e9,
         "write_roof_GB_s_measured": WR_ROOF / 1e9}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
