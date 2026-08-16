#!/usr/bin/env python3
"""S-F: fold the pair transition's RESIDUAL ADD into the row block, and let fc2 land in L1.

`tt_bio/esmfold2.py:1129` is `pair = ttnn.add(pair, self.pair_transition(pair))`. The add is a
full-tensor op sitting OUTSIDE `SwiGLUFFN.__call__`'s row-block loop, so at 512 aa it reads two
134.2 MB operands and writes a third, and it reads one of them (the FFN output) straight back out
of the DRAM the concat has just written it to. Per call, with C-in already assumed:

  today   slice 134 read | fc2 134 write | concat 134 read + 134 write
          | add 134 (pair) + 134 (r3) read + 134 write            = 938 MB
  with F  slice 134 read | fc2 -> L1, 0 | add(p_L1, o_L1) -> 134 write
          | concat 134 read + 134 write                           = 536 MB

402 MB/call removed = 0.95 ms at a 422 GB/s roof = 0.51 s/fold. The same traffic model predicted
C-in at 0.34 s and C-in measured 0.295, i.e. it runs ~15 % high, so expect 0.43-0.51 s.

Nothing here is a kernel. `_ffn` gains an output memory_config and the residual add moves inside
the block loop; `SwiGLUFFN.__call__` returns `x + ffn(x)` for the 4-D row-blocked path and the
call site drops its own add.

The add is elementwise and row-independent, so per-block and full-tensor must agree bit-for-bit.
That is the gate, not an expectation: `torch.equal` on the full output, nothing weaker.

Why fc2's output can go to L1 here when holding all 16 block outputs in L1 cannot: this arm
consumes each block's output IMMEDIATELY (the add reads it and frees it), so L1 holds one 8.39 MB
block, not 134 MB. `p3_s_split_512_c0.json`'s `fc2_l1` and `cat_l1` arms both TT_THROW at
program.cpp:1052 doing exactly the thing this arm avoids.

ARMS (batched: 4 calls per synchronize, median of 5, never per-op-synced)
  base    slice->L1, body (fc2 -> DRAM), concat, then the full-tensor residual add. This is the
          shipped chain with lever E and lever C-in, plus the residual the call site owns.
  F       slice->L1, body (fc2 -> L1), per-block residual add -> DRAM, concat. No outer add.
  F_chunk F but fed by the shipped eager `ttnn.chunk` instead of C-in's lazy L1 slice, so F's
          value is known both with and without C-in landing first.
  base_chunk  the matching baseline for F_chunk.

KILL GATES, PRE-COMMITTED
  J1  (base - F) * 538 < 0.25 s      -> F is NO-GO by size.
  J2  F not torch.equal to base      -> F is DEAD. Do not scope it by size.
  J3  F raises on L1                 -> record the size; F needs the same refusal cache as E.
"""
import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import esmc as EC

CALLS_PER_FOLD = 538


def timed(fn, dev, reps=4, batches=5, warm=2):
    import time
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / reps)
    return st.median(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L, C_Z, D_FF, R = a.size, 256, 1024, EC._PAIR_FFN_ROW_BLOCK

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
          else ttnn.types.BlackholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
    to = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "size": L, "rows": R, "calls_per_fold": CALLS_PER_FOLD,
           "ms": {}, "raw": {}, "roofs": {}}

    N = 4096
    ra = to(torch.randn(1, 1, N * 4, N)); rb = to(torch.randn(1, 1, N * 4, N))
    m_add, _ = timed(lambda: ttnn.deallocate(ttnn.add(ra, rb)), dev, reps=2, batches=5)
    res["roofs"]["dram_add_GBps"] = round(3 * N * 4 * N * 2 / (m_add / 1e3) / 1e9, 1)
    ttnn.deallocate(ra); ttnn.deallocate(rb)
    print("roofs", res["roofs"], flush=True)

    x = to(torch.randn(1, L, L, C_Z))
    nw = to(torch.ones(C_Z)); nb = to(torch.zeros(C_Z))
    w1a = to(torch.randn(C_Z, D_FF) * 0.02)
    w1b = to(torch.randn(C_Z, D_FF) * 0.02)
    w2 = to(torch.randn(D_FF, C_Z) * 0.02)
    l1cfg = dict(l1_out=True, l1_bw=T._PAIR_FFN_FC1_BW, l1_block_w=T._PAIR_FFN_FC1_BLOCK_W)
    nblk = -(-L // R)
    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

    sl = lambda i: ttnn.slice(x, [0, i * R, 0, 0], [1, min((i + 1) * R, L), L, C_Z],
                              memory_config=L1)
    ln = lambda t: ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5,
                                   compute_kernel_config=ck, memory_config=L1)

    def ffn_block(p, out_mc):
        xn = ln(p)
        h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
        h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
        ttnn.deallocate(xn)
        gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                              memory_config=L1)
        ttnn.deallocate(h1); ttnn.deallocate(h2)
        o = ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                        core_grid=T.CORE_GRID_MAIN, memory_config=out_mc)
        ttnn.deallocate(gated)
        return o

    def run_base(lazy=True, keep=False):
        parts = None if lazy else ttnn.chunk(x, nblk, dim=1)
        outs = []
        for i in range(nblk):
            p = sl(i) if lazy else parts[i]
            o = ffn_block(p, DR)
            ttnn.deallocate(p)
            outs.append(o)
        r = ttnn.concat(outs, dim=1)
        for o in outs:
            ttnn.deallocate(o)
        s = ttnn.add(x, r)
        ttnn.deallocate(r)
        if keep:
            return s
        ttnn.deallocate(s)
        return None

    def run_F(lazy=True, keep=False):
        parts = None if lazy else ttnn.chunk(x, nblk, dim=1)
        outs = []
        for i in range(nblk):
            p = sl(i) if lazy else parts[i]
            o = ffn_block(p, L1)
            outs.append(ttnn.add(p, o, memory_config=DR))
            ttnn.deallocate(o); ttnn.deallocate(p)
        s = ttnn.concat(outs, dim=1)
        for o in outs:
            ttnn.deallocate(o)
        if keep:
            return s
        ttnn.deallocate(s)
        return None

    arms = [("base", lambda: run_base(True)), ("F", lambda: run_F(True)),
            ("base_chunk", lambda: run_base(False)), ("F_chunk", lambda: run_F(False))]
    for name, fn in arms:
        try:
            m, raw = timed(fn, dev)
        except Exception as exc:
            res["ms"][name] = None
            res["raw"][name] = "EXC: %s" % (str(exc).splitlines()[0][:220],)
            print("%-11s EXC %s" % (name, res["raw"][name]), flush=True)
            continue
        res["ms"][name], res["raw"][name] = round(m, 4), [round(v, 4) for v in raw]
        print("%-11s %8.4f ms  %s" % (name, m, res["raw"][name]), flush=True)

    # J2, the gate.
    try:
        rb_, rf = run_base(True, keep=True), run_F(True, keep=True)
        tb, tf = ttnn.to_torch(rb_), ttnn.to_torch(rf)
        res["F_torch_equal"] = bool(torch.equal(tb, tf))
        res["F_max_abs_diff"] = float((tb.float() - tf.float()).abs().max())
        ttnn.deallocate(rb_); ttnn.deallocate(rf)
    except Exception as exc:
        res["F_torch_equal"] = None
        res["F_parity_exc"] = str(exc).splitlines()[0][:220]

    ms = res["ms"]
    pf = lambda d: round(d * CALLS_PER_FOLD / 1e3, 3)
    if ms.get("F") is not None:
        res["F_delta_ms"] = round(ms["base"] - ms["F"], 4)
        res["F_s_per_fold"] = pf(res["F_delta_ms"])
    if ms.get("F_chunk") is not None:
        res["F_chunk_delta_ms"] = round(ms["base_chunk"] - ms["F_chunk"], 4)
        res["F_chunk_s_per_fold"] = pf(res["F_chunk_delta_ms"])
    if ms.get("F") is not None and ms.get("base_chunk") is not None:
        res["cin_plus_F_s_per_fold"] = pf(ms["base_chunk"] - ms["F"])
    res["s_per_fold"] = {k: (pf(v) if v is not None else None) for k, v in ms.items()}
    res["J1_pass"] = bool(res.get("F_s_per_fold", 0) >= 0.25)
    res["J2_pass"] = bool(res.get("F_torch_equal"))
    res["verdict"] = ("GO" if res["J1_pass"] and res["J2_pass"]
                      else "DEAD-PARITY" if res.get("F_torch_equal") is False else "NO-GO")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
