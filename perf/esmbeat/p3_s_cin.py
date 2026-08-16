#!/usr/bin/env python3
"""S-Cin: can the pair FFN's row-block CHUNK be deleted with no kernel, now that E has landed?

Lever C is the pair FFN's row-block assembly: `ttnn.chunk` on the way in and `ttnn.concat` on the
way out of `SwiGLUFFN.__call__` (tt_bio/esmc.py:550-552). p2 screened it at 0.63 s/fold on qb2 and
p3's plan proposed landing the OUTPUT half with `ttnn.experimental.slice_write` and no kernel.

That route is dead on inspection, before any device time: `slice_write`'s own docstring on the
installed wheel says "Supports only Row Major Tensors" and the pair tensor is TILE_LAYOUT. It also
refuses to slice the last dim, which is fine, but row-major is not. Untilize/retilize around it
would cost more than the 0.32 s it would save (and can fall back to a single core).

What this screen prices instead is the INPUT half, which has a no-kernel route the plan did not
name. Today:

    parts = ttnn.chunk(x, nblk, dim=1)     # 16 blocks materialised in DRAM, eagerly
    xn    = layer_norm(part, -> L1)        # lever E: reads the block back from DRAM

Per block that is one 8.39 MB DRAM write (chunk) plus one 8.39 MB DRAM read (layer_norm's operand)
that a slice straight into L1 would not pay:

    part = ttnn.slice(x, [0,i*R,0,0], [1,(i+1)*R,L,C], memory_config=L1)   # DRAM read, L1 write
    xn   = layer_norm(part, -> L1)                                          # L1 -> L1

`ttnn.slice` takes a `memory_config` kwarg on this wheel (checked by introspection). It is still a
copy and not a view (tt-bio-ttnn-slice-not-a-view-and-allocation-order-sensitivity) -- the win is
not that the copy disappears, it is that its destination and the next op's source both move on
chip. Predicted 268 MB/call of removed DRAM traffic = 0.638 ms at a 420 GB/s roof = 0.34 s/fold,
which should reproduce p2's own `ship - nochunk` (0.6357 ms/call, 0.342 s/fold) because pre-cutting
the parts and slicing them into L1 leave the layer_norm reading the same number of DRAM bytes.

Slicing lazily also drops the peak DRAM footprint by the 134 MB of eagerly-materialised blocks.

ARMS (batched: 4 chain calls per synchronize, median of 5, never per-op-synced)
  chain      the shipped chain WITH E: chunk -> 16 x (ln->L1, fc1 pair, SiLU*mul, fc2) -> concat
  cin_l1     the same, with chunk replaced by a lazy per-block ttnn.slice into L1
  cin_dram   the same, sliced lazily into DRAM -- the control that separates "lazy slice instead
             of eager chunk" from "the slice lands in L1". If cin_dram already carries the win,
             the mechanism is the eager materialisation, not the memory config.
  noconcat   chunk kept, concat deleted (C's output half, priced with E on)
  noasm      parts pre-cut outside the timed region, concat deleted (C's ceiling, with E on)
  cin_noasm  lazy L1 slice, concat deleted (what C-in + a built C-out would reach)
  slice_l1 / slice_dram / chunk_only   the three copies alone, for attribution

PARITY, and it is the gate: the full concatenated output of `cin_l1` must be `torch.equal` to the
full concatenated output of `chain`. Anything weaker is not a parity check on this lineage.

KILL GATES, PRE-COMMITTED
  H1  (chain - cin_l1) * 538 < 0.20 s  ->  C-in is NO-GO by size. Record and stop.
  H2  cin_l1 not torch.equal to chain  ->  C-in is DEAD. Do not scope it by size, do not force it.
  H3  cin_l1 worse than cin_dram       ->  the L1 destination is not the mechanism; re-derive
      before building, because the shipped patch would then be the wrong one-liner.
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

CALLS_PER_FOLD = 538  # pair transitions per 512 aa fold (L1_FC1_STATS[0]/2/(512/32))


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
           "grid": [g.x, g.y], "size": L, "rows": R, "c_z": C_Z, "d_ff": D_FF,
           "calls_per_fold": CALLS_PER_FOLD, "ms": {}, "raw": {}, "roofs": {}}

    # DRAM roof, measured in this process on this card, not carried in.
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
    L1 = ttnn.L1_MEMORY_CONFIG
    DR = ttnn.DRAM_MEMORY_CONFIG

    def ln(t, mc=L1):
        return ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5,
                               compute_kernel_config=ck, memory_config=mc)

    def sl(i, mc):
        return ttnn.slice(x, [0, i * R, 0, 0], [1, min((i + 1) * R, L), L, C_Z],
                          memory_config=mc)

    def block(p, free_p):
        """One block of the shipped `_ffn(split=True, l1_gated=True)` with lever E on."""
        xn = ln(p)
        if free_p:
            ttnn.deallocate(p)
        h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
        h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
        ttnn.deallocate(xn)
        gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                              memory_config=L1)
        ttnn.deallocate(h1); ttnn.deallocate(h2)
        out = ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                          core_grid=T.CORE_GRID_MAIN)
        ttnn.deallocate(gated)
        return out

    pre = ttnn.chunk(x, nblk, dim=1)   # cut once, for the arms that hoist the chunk out

    def run_chain(mode, concat=True, keep=False):
        """mode: 'chunk' | 'slice_l1' | 'slice_dram' | 'pre'."""
        if mode == "chunk":
            parts = ttnn.chunk(x, nblk, dim=1)
            outs = [block(p, True) for p in parts]
        elif mode == "pre":
            outs = [block(p, False) for p in pre]
        else:
            mc = L1 if mode == "slice_l1" else DR
            outs = [block(sl(i, mc), True) for i in range(nblk)]
        if not concat:
            for o in outs:
                ttnn.deallocate(o)
            return None
        r = ttnn.concat(outs, dim=1)
        for o in outs:
            ttnn.deallocate(o)
        if keep:
            return r
        ttnn.deallocate(r)
        return None

    arms = (
        ("chain",     lambda: run_chain("chunk")),
        ("cin_l1",    lambda: run_chain("slice_l1")),
        ("cin_dram",  lambda: run_chain("slice_dram")),
        ("noconcat",  lambda: run_chain("chunk", concat=False)),
        ("noasm",     lambda: run_chain("pre", concat=False)),
        ("cin_noasm", lambda: run_chain("slice_l1", concat=False)),
        ("chunk_only", lambda: [ttnn.deallocate(p) for p in ttnn.chunk(x, nblk, dim=1)]),
        ("slice_l1_only", lambda: [ttnn.deallocate(sl(i, L1)) for i in range(nblk)]),
        ("slice_dram_only", lambda: [ttnn.deallocate(sl(i, DR)) for i in range(nblk)]),
    )
    for name, fn in arms:
        try:
            m, raw = timed(fn, dev)
        except Exception as exc:                       # an L1 refusal is a result, not a crash
            res["ms"][name] = None
            res["raw"][name] = "EXC: %s" % (str(exc).splitlines()[0][:200],)
            print("%-16s EXC %s" % (name, res["raw"][name]), flush=True)
            continue
        res["ms"][name], res["raw"][name] = round(m, 4), [round(v, 4) for v in raw]
        print("%-16s %8.4f ms  %s" % (name, m, res["raw"][name]), flush=True)

    # ---- H2, the gate: full-output torch.equal, chunk vs lazy L1 slice ----------------------
    ref = run_chain("chunk", keep=True)
    got = run_chain("slice_l1", keep=True)
    tr, tg = ttnn.to_torch(ref), ttnn.to_torch(got)
    res["cin_torch_equal"] = bool(torch.equal(tr, tg))
    res["cin_max_abs_diff"] = float((tr.float() - tg.float()).abs().max())
    ttnn.deallocate(ref); ttnn.deallocate(got)
    # and the sliced block itself against the chunked one, so a failure is attributable
    c0 = ttnn.to_torch(pre[0])
    s0 = ttnn.to_torch(sl(0, L1))
    res["slice_block_torch_equal"] = bool(torch.equal(c0, s0))

    ms = res["ms"]
    per_fold = lambda d: round(d * CALLS_PER_FOLD / 1e3, 3)
    if ms.get("cin_l1") is not None:
        res["cin_delta_ms"] = round(ms["chain"] - ms["cin_l1"], 4)
        res["cin_s_per_fold"] = per_fold(res["cin_delta_ms"])
    if ms.get("cin_dram") is not None:
        res["cin_dram_delta_ms"] = round(ms["chain"] - ms["cin_dram"], 4)
        res["cin_dram_s_per_fold"] = per_fold(res["cin_dram_delta_ms"])
    if ms.get("noconcat") is not None:
        res["cout_delta_ms"] = round(ms["chain"] - ms["noconcat"], 4)
        res["cout_s_per_fold"] = per_fold(res["cout_delta_ms"])
    if ms.get("noasm") is not None:
        res["asm_ceiling_ms"] = round(ms["chain"] - ms["noasm"], 4)
        res["asm_ceiling_s_per_fold"] = per_fold(res["asm_ceiling_ms"])
    if ms.get("cin_noasm") is not None:
        res["cin_plus_cout_ms"] = round(ms["chain"] - ms["cin_noasm"], 4)
        res["cin_plus_cout_s_per_fold"] = per_fold(res["cin_plus_cout_ms"])
    res["s_per_fold"] = {k: (per_fold(v) if v is not None else None) for k, v in ms.items()}

    res["H1_pass"] = bool(res.get("cin_s_per_fold", 0) >= 0.20)
    res["H2_pass"] = bool(res.get("cin_torch_equal"))
    res["H3_pass"] = bool(ms.get("cin_l1") is not None and ms.get("cin_dram") is not None
                          and ms["cin_l1"] <= ms["cin_dram"])
    res["verdict"] = ("GO" if res["H1_pass"] and res["H2_pass"] and res["H3_pass"]
                      else "DEAD-PARITY" if not res["H2_pass"] else "NO-GO")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
