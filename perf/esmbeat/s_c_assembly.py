#!/usr/bin/env python3
"""S-C: price the pair FFN's row-block assembly IN CHAIN, before any kernel is written.

C is the fused blocked pair FFN. What it would delete is not arithmetic, it is the two copies
that wrap the row block in `SwiGLUFFN.__call__` (tt_bio/esmc.py:463-467):

    parts = ttnn.chunk(x, L // 32, dim=1)          # copies all of x
    return ttnn.concat([self._ffn(p, ...) ...], 1) # copies all of the output back

`ttnn.slice` does not alias its parent (state/esmfold2-to-3p4x.md 11.18), so neither copy can be
removed by a smaller kernel; the only route is for the matmuls themselves to read and write at a
page offset. That is a real kernel change to `tt_bio/mm_generic.py`, and this screen decides
whether it is worth writing -- the E10 discipline: price what the fusion deletes against the
shipped chain FIRST, on its own, before any parity work.

The 1.4651 ms/call that state/esmfold2-to-3p4x.md 11.18 attributes to chunk+concat is an
ISOLATED number and that document says so in the same section: isolated per-op timings there sum
to 17.93 ms against a MEASURED 14.657 ms for the chain they belong to, an inflation of
1.078-1.223x (memory tt-bio-isolated-op-timing-oversync-inflates-cost). This screen measures the
same thing in chain, which is the number C is actually worth.

538 pair transitions per fold at 512 aa. Not carried in: it is L1_FC1_STATS[0] / 2 / (512/32) =
17216 / 2 / 16 from perf/esmbeat/ab_d_512_c0_quiet.json, an executed counter.

ARMS (all batched: 4 chain calls per synchronize, median of 5 batches, never per-op-synced)

  ship       the shipped chain: chunk -> 16 x _ffn -> concat.
  noasm      the same 16 _ffn bodies, with the chunk hoisted OUT of the timed region (the parts
             are cut once, before timing, and reused) and the concat deleted. Its output is a
             list, not a tensor, so this arm is a stopwatch and not a correctness arm.
             ship - noasm IS the in-chain cost of the assembly, i.e. C's ceiling.
  lnfull     one ttnn.layer_norm over the whole [1,L,L,256], versus the 16 block layer_norms the
             shipped chain runs. C's input half needs the layer_norm hoisted out of the block
             loop (the matmul reads a row block of x_norm at a page offset), so this arm prices
             that restructure and, more importantly, CHECKS IT: layer_norm reduces over the last
             dim only, so it is row-independent in exact arithmetic, but the ttnn kernel derives
             its blocking from the shape and a different blocking can round differently.

PREDICTIONS, WRITTEN BEFORE THE RUN
  ship - noasm lands 1.20-1.36 ms/call, i.e. 0.64-0.73 s of fold, if the 1.4651 ms isolated
  number deflates by the measured 1.078-1.223x band. Below that band the isolated number was
  wrong about the chain.
  lnfull is within +/-5 % of 16 block layer_norms (same bytes, same reduction).
  lnfull is torch.equal to the concatenation of the 16 block layer_norms: 50/50, and it is the
  single fact that decides whether C's input half is buildable at all.

KILL GATES, PRE-COMMITTED
  G1  (ship - noasm) * 538 < 0.45 s  ->  C is NO-GO. Record the number and stop. C cannot reach
      the bar on its own and a custom offset kernel with a parity risk is not worth less than
      twice what D returned.
  G2  lnfull not torch.equal to the block layer_norms -> C's input half is BARRED. Re-price C at
      the concat half only (measure `noconcat`, below) and re-apply G1 to that number.
  G3  lnfull slower than the 16 block layer_norms by more than 20 % of the G1 delta -> the
      restructure eats its own win on the input side; fall back to concat-only, same as G2.

  noasm is split into its two halves so G2/G3 have a number to fall back to:
  nochunk (parts pre-cut, concat kept) and noconcat (chunk kept, concat dropped).
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

assert Path(T.__file__).resolve().is_relative_to(REPO), "tt_bio from %s" % T.__file__

CALLS_PER_FOLD = 538  # pair transitions per 512 aa fold; see the docstring for its provenance


def timed(fn, dev, reps=4, batches=5, warm=2):
    """Batched wall per call. One synchronize per batch, never one per call."""
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
    L, C_Z, D_FF, ROWS = a.size, 256, 1024, EC._PAIR_FFN_ROW_BLOCK

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    # The same config every TorchWrapper builds (tt_bio/tenstorrent.py:5410-5420), spelled out
    # here rather than reached for through a module that only holds it per instance.
    ck = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
          else ttnn.types.BlackholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
    to = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = to(torch.randn(1, L, L, C_Z))
    nw = to(torch.ones(C_Z))
    nb = to(torch.zeros(C_Z))
    w1a = to(torch.randn(C_Z, D_FF) * 0.02)
    w1b = to(torch.randn(C_Z, D_FF) * 0.02)
    w2 = to(torch.randn(D_FF, C_Z) * 0.02)

    def ln(t):
        return ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ck)

    def body(p, x_norm=None):
        """One block of the shipped `_ffn(split=True, l1_gated=True)`."""
        xn = ln(p) if x_norm is None else x_norm
        l1 = dict(l1_out=True, l1_bw=T._PAIR_FFN_FC1_BW, l1_block_w=T._PAIR_FFN_FC1_BLOCK_W)
        h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1)
        h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1)
        if x_norm is None:
            ttnn.deallocate(xn)
        gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                              memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h1)
        ttnn.deallocate(h2)
        out = ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                          core_grid=T.CORE_GRID_MAIN)
        ttnn.deallocate(gated)
        return out

    nblk = -(-L // ROWS)
    pre = ttnn.chunk(x, nblk, dim=1)  # cut once, for the arms that hoist the chunk out

    def arm_ship():
        parts = ttnn.chunk(x, nblk, dim=1)
        outs = [body(p) for p in parts]
        for p in parts:
            ttnn.deallocate(p)
        r = ttnn.concat(outs, dim=1)
        for o in outs:
            ttnn.deallocate(o)
        ttnn.deallocate(r)

    def arm_nochunk():
        outs = [body(p) for p in pre]
        r = ttnn.concat(outs, dim=1)
        for o in outs:
            ttnn.deallocate(o)
        ttnn.deallocate(r)

    def arm_noconcat():
        parts = ttnn.chunk(x, nblk, dim=1)
        outs = [body(p) for p in parts]
        for p in parts:
            ttnn.deallocate(p)
        for o in outs:
            ttnn.deallocate(o)

    def arm_noasm():
        outs = [body(p) for p in pre]
        for o in outs:
            ttnn.deallocate(o)

    def arm_lnblocks():
        for p in pre:
            ttnn.deallocate(ln(p))

    def arm_lnfull():
        ttnn.deallocate(ln(x))

    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "size": L, "rows": ROWS, "blocks": nblk,
           "calls_per_fold": CALLS_PER_FOLD, "ms": {}, "raw": {}}
    for name, fn in (("ship", arm_ship), ("nochunk", arm_nochunk), ("noconcat", arm_noconcat),
                     ("noasm", arm_noasm), ("lnblocks", arm_lnblocks), ("lnfull", arm_lnfull)):
        m, raw = timed(fn, dev)
        res["ms"][name], res["raw"][name] = round(m, 4), [round(v, 4) for v in raw]
        print("%-9s %8.4f ms  %s" % (name, m, res["raw"][name]), flush=True)

    # G2: the fact that decides whether C's input half exists at all.
    full = ln(x)
    blocks = ttnn.concat([ln(p) for p in pre], dim=1)
    res["ln_torch_equal"] = bool(torch.equal(ttnn.to_torch(full), ttnn.to_torch(blocks)))
    ttnn.deallocate(full)
    ttnn.deallocate(blocks)

    d = res["ms"]["ship"] - res["ms"]["noasm"]
    res["assembly_ms_per_call"] = round(d, 4)
    res["assembly_s_per_fold"] = round(d * CALLS_PER_FOLD / 1e3, 3)
    res["concat_only_s_per_fold"] = round(
        (res["ms"]["ship"] - res["ms"]["noconcat"]) * CALLS_PER_FOLD / 1e3, 3)
    res["ln_full_vs_blocks_ms"] = round(res["ms"]["lnfull"] - res["ms"]["lnblocks"], 4)
    res["G1_pass"] = res["assembly_s_per_fold"] >= 0.45
    res["G2_pass"] = res["ln_torch_equal"]
    res["G3_pass"] = res["ln_full_vs_blocks_ms"] <= 0.20 * d
    res["verdict"] = ("GO" if res["G1_pass"] and res["G2_pass"] and res["G3_pass"]
                      else "GO-CONCAT-ONLY" if res["G1_pass"] else "NO-GO")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
