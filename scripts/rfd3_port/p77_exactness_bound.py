#!/usr/bin/env python3
"""p77 -- what bit-exactness costs, priced as deliberately-wrong-output timing arms.

§8 of the state doc: the two largest remaining prizes are blocked by the bit-exactness
requirement, not by physics, and the honest way to put that in front of Moritz is a number
rather than an argument. Same move `p50_bf16_softmax_bound.py` made for the bf16 softmax
question: run the shapes, report the ledger delta, write no kernel.

The atom decoder materialises `[1, 4, 6051, 6080]` scores where only K = 128 columns per row
carry a non-mask value -- 47.5x redundant. A gathered formulation is NOT bit-exact: `exp(-9984 -
max)` underflows to exactly 0.0 so the SET of contributing terms is identical, but summing 128
contiguous terms is a different reduction order from summing them scattered through 6080, and
fp32 addition is not associative. `ttnn.transformer.scaled_dot_product_attention` is not
bit-exact either, by construction -- flash-style online softmax has a different reduction order.

So these arms are bounds, not levers. Their outputs are wrong on purpose. Nothing here is ever
wired into the model.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p77_exactness_bound.py \
          perf/p77/exactness_bound.json
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
from tt_bio import softmax_generic                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p77/exactness_bound.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# The decoder + atom encoder call site, at the page fixture.
H, L, NK, DH = 4, 6051, 6080, 32
K_SPARSE = 128                       # neighbours per row in the gathered formulation
CALLS = 9                            # per diffusion step
STEPS = 200
SCALE = DH ** -0.5


def timeit(fn, dev, n=N, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out)


def main():
    dev = get_device()
    torch.manual_seed(42)
    rows = []

    q = ttnn.from_torch(torch.randn(1, H, L, DH), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    kt = ttnn.from_torch(torch.randn(1, H, DH, NK), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(torch.randn(1, H, NK, DH), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    bias32 = ttnn.from_torch(torch.randn(1, H, L, NK), dtype=ttnn.float32,
                             layout=ttnn.TILE_LAYOUT, device=dev)

    # --- arm A: the shipped dense chain, with the bf16 softmax already landed ------------------
    def dense():
        sc = ttnn.matmul(q, kt)
        sc32 = ttnn.typecast(sc, ttnn.float32)
        ttnn.deallocate(sc)
        sc32 = ttnn.multiply(sc32, SCALE)
        sc32 = ttnn.add(sc32, bias32)
        a = softmax_generic.softmax_bf16(sc32, ttnn.bfloat16)
        ttnn.deallocate(sc32)
        o = ttnn.matmul(a, v)
        ttnn.deallocate(a)
        ttnn.deallocate(o)

    t_dense = timeit(dense, dev)
    rows.append({"arm": "dense_shipped", "ms_per_call": round(t_dense, 4),
                 "exact": True, "note": "the chain the model runs today"})

    # --- arm B: the same chain on gathered [1, H, L, 128] scores -------------------------------
    kg = ttnn.from_torch(torch.randn(1, H, DH, K_SPARSE), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    vg = ttnn.from_torch(torch.randn(1, H, K_SPARSE, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    biasg = ttnn.from_torch(torch.randn(1, H, L, K_SPARSE), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=dev)

    def sparse():
        sc = ttnn.matmul(q, kg)
        sc32 = ttnn.typecast(sc, ttnn.float32)
        ttnn.deallocate(sc)
        sc32 = ttnn.multiply(sc32, SCALE)
        sc32 = ttnn.add(sc32, biasg)
        a = softmax_generic.softmax_bf16(sc32, ttnn.bfloat16)
        ttnn.deallocate(sc32)
        o = ttnn.matmul(a, vg)
        ttnn.deallocate(a)
        ttnn.deallocate(o)

    t_sparse = timeit(sparse, dev)
    rows.append({"arm": "sparse_K128", "ms_per_call": round(t_sparse, 4), "exact": False,
                 "note": "gathered scores; the softmax row sum is over 128 terms, not 6080"})

    # --- arm C: ttnn's flash SDPA on the same q/k/v --------------------------------------------
    t_sdpa = None
    sdpa_err = None
    kk = ttnn.from_torch(torch.randn(1, H, NK, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    try:
        def sdpa():
            o = ttnn.transformer.scaled_dot_product_attention(q, kk, v, is_causal=False,
                                                              scale=SCALE)
            ttnn.deallocate(o)
        t_sdpa = timeit(sdpa, dev)
        rows.append({"arm": "sdpa_flash", "ms_per_call": round(t_sdpa, 4), "exact": False,
                     "note": "no per-row pair bias; online softmax, different reduction order"})
    except Exception as e:                                              # noqa: BLE001
        sdpa_err = str(e)[:300]
        rows.append({"arm": "sdpa_flash", "ms_per_call": None, "exact": False,
                     "note": "rejected: " + sdpa_err})

    def prize(t):
        if t is None:
            return None, None
        d = (t_dense - t) * CALLS
        return round(d, 3), round(d * STEPS / 1000.0, 3)

    sp_step, sp_design = prize(t_sparse)
    sd_step, sd_design = prize(t_sdpa)

    for r in rows:
        print("[p77] %-14s %9s ms/call  exact=%s  %s"
              % (r["arm"], r["ms_per_call"], r["exact"], r["note"]), flush=True)
    print("[p77] sparse K=128 would save %s ms/step = %s s/design at %d steps"
          % (sp_step, sp_design, STEPS))
    print("[p77] flash SDPA would save  %s ms/step = %s s/design at %d steps"
          % (sd_step, sd_design, STEPS))
    print("[p77] both arms produce WRONG OUTPUT. They are bounds, not levers.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "H": H, "L": L, "n_key": NK, "head_dim": DH, "k_sparse": K_SPARSE,
        "calls_per_step": CALLS, "num_timesteps": STEPS,
        "sparse_saved_ms_per_step": sp_step, "sparse_saved_s_per_design": sp_design,
        "sdpa_saved_ms_per_step": sd_step, "sdpa_saved_s_per_design": sd_design,
        "sdpa_error": sdpa_err,
        "host": "qb2", "card": int(os.environ.get("TT_VISIBLE_DEVICES", "1")),
    }, indent=2))


if __name__ == "__main__":
    main()
