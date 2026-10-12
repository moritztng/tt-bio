#!/usr/bin/env python3
"""The AF2-IG trimul channel move at the census shape, against its own roofline.

The WH per-op census at 195 tokens (tree 8e053986e, .114 chip 7 @1000 MHz,
out/af2igwh_census_8e053986e/s0-opsall/census.json) puts ONE op at the top of a 5.152 s fold:

    0.4739 s synced / 1664 calls = 284.8 us, generic_op,
    af2.py:<lambda> > reblock_permute.py:reblock_permute_gated,
    ([1, 195, 195, 256]:BFLOAT16:L1, [1, 64, 195, 195]:BFLOAT16:L1)

That is `reblock_permute_gated`: read the p and g slices of the wide trimul projection, apply
`p * sigmoid(g)`, and permute (0,3,1,2) into [1, C, N, N]. 284.8 us for ~19 MB of tensor traffic
that never leaves L1 is one to two orders off any byte roof, so the question this answers is WHICH
resource binds. Four things could, and they are separable here:

  attribution  the census number is a synced one-call timing. `steady` enqueues `--steady` calls
               behind ONE sync, so `synced - steady` is what the census charges to dispatch.
  the reader   32 SERIALISED L1 round trips a group (one row = the p page and the g page, then a
               barrier). `--batches` issues 2, 4, 8 rows behind one barrier
               (`TT_BIO_GATED_READ_BATCH`, default-off, bit-exact).
  the writer   2048 local 32-byte gather transactions a group, which is the kernel's own recorded
               critical path. The `plain` arm is `reblock_permute` on [1,N,N,64] -> [1,64,N,N]:
               the SAME number of gather transactions and the SAME number of reader round trips
               on the busiest core, with half the bytes read and no gate arithmetic. If `gated`
               and `plain` land together, the gate and the bytes are free and the gather binds.
  the grid     98 groups over 72 cores is 2 waves at 68 % occupancy. `--sweep` runs N where the
               group count lands differently (192 -> 72 groups = exactly 1 full wave) at nearly
               the same byte count, so the wave term is read off directly.

Every arm that produces the gated result is checked `torch.equal` against the batch-1 arm, which
is the shipped kernel, so no number here is bought with a changed value.

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=. python perf/spd/af2_gated_move.py --out out/gmove.json

Device time only: no AICLK is sampled here, the pool job's `aiclk` helper does that, and a number
from this script is only valid against a 1000 MHz WH reading.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch                                  # noqa: E402
import ttnn                                   # noqa: E402

from tt_bio import reblock_permute as R       # noqa: E402
from tt_bio import tenstorrent as T           # noqa: E402

TILE = 32


def work_split(dev, num_groups):
    """``(cores, deepest core's group count, groups/core, waves)`` for this many groups."""
    plan = R._split_plan(dev, num_groups)
    gx, gy, (num_cores, _cg, _1, _2, work1, work2) = plan
    return {"grid": [gx, gy], "grid_cores": gx * gy, "cores": num_cores,
            "groups": num_groups, "deepest": max(work1, work2),
            "groups_per_core": num_groups / (gx * gy)}


def traffic(n, cw, c):
    """Exact bytes and transactions one gated call moves at this shape.

    Read off the two kernels, not estimated. A group is (it, jt, ct) and owns 32 output tiles;
    `rows_valid` is how many of a row-tile's 32 rows are real, the rest being tile padding that
    the reader still reads (from row 0, to keep the group a fixed 32 pushes) and the writer
    zeroes once a group.
    """
    nt, ct = -(-n // TILE), c // TILE
    groups = nt * nt * ct
    tile_b = TILE * TILE * 2
    rows = [min(TILE, n - it * TILE) for it in range(nt)]          # per row-tile
    # reader: 32 rows x (p page + g page) per group, padding rows included
    read_pages = groups * TILE * 2
    read_b = read_pages * tile_b
    useful_read_b = sum(rows) * nt * ct * 2 * tile_b               # the non-padding part
    # writer: per group, 32 channels x rows_valid x 2 faces local 32-byte reads, then 32 tile writes
    gather_tx = sum(TILE * r * 2 for r in rows) * nt * ct
    gather_b = gather_tx * 16 * 2        # one face row, FACE_WIDTH=16 at bf16, is 32 B
    write_tx = groups * TILE
    write_b = write_tx * tile_b
    return {"n": n, "cw": cw, "c": c, "nt": nt, "ct": ct, "groups": groups,
            "read_pages": read_pages, "read_B": read_b, "useful_read_B": useful_read_b,
            "gather_tx": gather_tx, "gather_B": gather_b,
            "write_tx": write_tx, "write_B": write_b,
            "tensor_B": useful_read_b + write_b,                   # p + g in, out out
            "noc_B": read_b + gather_b + write_b,
            "logical_B": (n * n * c * 2) * 3}                      # unpadded p, g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=195, help="token count; the census shape is 195")
    ap.add_argument("--cw", type=int, default=256, help="wide projection channels (4 x slice)")
    ap.add_argument("--slice", dest="sl", type=int, default=64)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--steady", type=int, default=20,
                    help="calls behind ONE sync, for the dispatch-free per-call time")
    ap.add_argument("--batches", default="1,2,4,8",
                    help="TT_BIO_GATED_READ_BATCH values to A/B; 1 is the shipped kernel")
    ap.add_argument("--sweep", default="192,195,224,256",
                    help="token counts for the wave-occupancy arm; empty to skip")
    ap.add_argument("--out", default="out/af2_gated_move.json")
    a = ap.parse_args()
    assert a.cw == 4 * a.sl, "the gated reader slices a four-way projection"

    dev = T.get_device()
    mc_l1 = ttnn.L1_MEMORY_CONFIG
    torch.manual_seed(0)

    def up(t, mc=mc_l1):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=mc)

    N, C = a.n, a.sl
    xw_t = (torch.randn(1, N, N, a.cw) * 0.7).to(torch.bfloat16)
    xw = up(xw_t)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, N, N]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, dev, mc_l1)
    p_off, g_off = 2 * C, 0                   # chunk(x, 4)[2] * sigmoid(chunk(x, 4)[0])
    # The ungated control: same writer gather, same reader round trips, half the bytes, no gate.
    x64 = up(xw_t[..., p_off:p_off + C].contiguous())
    p_only, g_only = x64, up(xw_t[..., g_off:g_off + C].contiguous())

    def gated():
        return R.reblock_permute_gated(xw, p_off, g_off, C, out=out)

    def plain():
        o = R.reblock_permute(x64, memory_config=mc_l1)
        ttnn.deallocate(o)

    def stock_permute():
        o = ttnn.permute(x64, (0, 3, 1, 2), memory_config=mc_l1)
        ttnn.deallocate(o)

    def unfused():
        """What the fused kernel replaced: the gate, then the move."""
        m = ttnn.multiply(p_only, g_only, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                          memory_config=mc_l1)
        o = ttnn.permute(m, (0, 3, 1, 2), memory_config=mc_l1)
        ttnn.deallocate(m)
        ttnn.deallocate(o)

    def runs(fn):
        """Whether an arm's ops exist on this wheel; a missing one is dropped, not fatal."""
        try:
            fn()
            ttnn.synchronize_device(dev)
            return True
        except Exception as e:                                     # noqa: BLE001
            print(f"arm dropped: {type(e).__name__}: {e}", flush=True)
            return False

    def synced(fn):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) * 1e3

    def steady(fn, m):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(m):
            fn()
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) * 1e3 / m

    arms = {}
    for b in [int(s) for s in a.batches.split(",") if s]:
        arms[f"gated_b{b}"] = (gated, b)
    arms["plain"] = (plain, None)
    arms["stock_permute"] = (stock_permute, None)
    if runs(unfused):
        arms["unfused"] = (unfused, None)

    def arm(fn, b):
        if b is not None:
            R.set_gated_read_batch(b)
        return fn

    # Warm every arm first (JIT per program), then interleave the timed reps so a clock or
    # occupancy drift hits every arm the same way.
    digests = {}
    for name, (fn, b) in arms.items():
        arm(fn, b)
        for _ in range(a.warm):
            fn()
        ttnn.synchronize_device(dev)
        if b is not None:
            digests[name] = ttnn.to_torch(out).clone()

    ref = digests.get("gated_b1")
    equal = {k: (ref is not None and torch.equal(ref, v)) for k, v in digests.items()}

    t_sync = {k: [] for k in arms}
    t_fast = {k: [] for k in arms}
    for _ in range(a.reps):
        for name, (fn, b) in arms.items():
            arm(fn, b)
            t_sync[name].append(synced(fn))
            t_fast[name].append(steady(fn, a.steady))
    R.set_gated_read_batch(1)

    tr = traffic(N, a.cw, C)
    ws = work_split(dev, tr["groups"])
    med_s = {k: st.median(v) for k, v in t_sync.items()}
    med_f = {k: st.median(v) for k, v in t_fast.items()}

    def rate(ms, b):
        return b / (ms * 1e-3) / 1e9

    base = med_f["gated_b1"]
    res = {
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "arch": str(dev.arch()), "loadavg": os.getloadavg(),
        "shape": {"n": N, "cw": a.cw, "slice": C},
        "reps": a.reps, "warm": a.warm, "steady_calls": a.steady,
        "traffic": tr, "work_split": ws,
        "census_us_per_call": 284.8,
        "synced_ms": med_s, "steady_ms": med_f,
        "spread_frac": {k: (max(v) - min(v)) / st.median(v) for k, v in t_fast.items()},
        "bit_equal_to_shipped": equal,
        "speedup_vs_b1": {k: base / v for k, v in med_f.items()},
        "dispatch_ms": {k: med_s[k] - med_f[k] for k in arms},
        "gated_b1_GBs": {
            "tensor": rate(base, tr["tensor_B"]),       # p + g read, out written
            "noc": rate(base, tr["noc_B"]),             # + the writer's local gather
            "logical": rate(base, tr["logical_B"]),     # unpadded, what a roofline usually quotes
        },
        # The transaction view: what the busiest core issues inside one call.
        "critical_core": {
            "groups": ws["deepest"],
            "gather_tx": tr["gather_tx"] // tr["groups"] * ws["deepest"],
            "reader_round_trips": ws["deepest"] * TILE,
            "ns_per_gather_tx": base * 1e6 / (tr["gather_tx"] // tr["groups"] * ws["deepest"]),
        },
        "all_steady_ms": t_fast, "all_synced_ms": t_sync,
    }

    if a.sweep:
        # The sweep's own operands are up to 42 MB at N=256 and the chip's L1 is ~108 MB, so give
        # the main arms' tensors back first.
        for t in (xw, out, x64, g_only):
            ttnn.deallocate(t)
        sweep = {}
        for n in [int(s) for s in a.sweep.split(",") if s]:
            xs = up((torch.randn(1, n, n, a.cw) * 0.7).to(torch.bfloat16))
            os_ = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, n, n]), ttnn.bfloat16,
                                                 ttnn.TILE_LAYOUT, dev, mc_l1)
            f = lambda: R.reblock_permute_gated(xs, p_off, g_off, C, out=os_)  # noqa: E731
            for _ in range(a.warm):
                f()
            ms = st.median([steady(f, a.steady) for _ in range(max(3, a.reps // 3))])
            t = traffic(n, a.cw, C)
            sweep[str(n)] = {"steady_ms": ms, "traffic": t, "work_split": work_split(dev, t["groups"]),
                             "tensor_GBs": rate(ms, t["tensor_B"]),
                             "ms_per_group_wave": ms / work_split(dev, t["groups"])["deepest"]}
            ttnn.deallocate(xs)
            ttnn.deallocate(os_)
        res["sweep"] = sweep

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("shape", "work_split", "steady_ms", "synced_ms",
                                          "speedup_vs_b1", "bit_equal_to_shipped",
                                          "gated_b1_GBs", "critical_core", "dispatch_ms")},
                     indent=1))
    if "sweep" in res:
        for n, s in res["sweep"].items():
            print(f"sweep n={n:>4} groups={s['traffic']['groups']:>4} "
                  f"deepest={s['work_split']['deepest']} "
                  f"occ={s['work_split']['groups_per_core']:.2f} "
                  f"{s['steady_ms']:.3f} ms  {s['tensor_GBs']:.1f} GB/s "
                  f"{s['ms_per_group_wave']:.4f} ms/wave")
    print("wrote", p)


if __name__ == "__main__":
    main()
