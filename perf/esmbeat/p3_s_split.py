#!/usr/bin/env python3
"""S-split: re-split the pair FFN chain AFTER lever E, and re-price the fused kernel against it.

p3 §1.1 split this chain when the block layer_norm still wrote to DRAM. E moved that write on chip
and C-in (screened GO in p3_s_cin_512_c0.json) moves the row-block slice on chip too, so every
per-op number in §1.1 is now against a baseline nobody runs. The state doc's own closing lesson is
that a residual inherits its mechanism label from the measurement that named it and that label
expires -- so this re-splits rather than re-using.

Arms are cumulative prefixes of the post-E, post-C-in block, all reading a lazily sliced L1 block:

  s          slice(x, block i) -> L1                                (C-in's copy)
  s_ln       + layer_norm -> L1                                     (lever E)
  s_ln_fc1   + the two N=1024 fc1 halves, L1 out
  s_ln_mul   + SwiGLU multiply with SiLU folded in, L1 out
  s_full     + fc2, DRAM out                                        (= cin_noasm)

Differences give ln / fc1 / mul / fc2 with the memory configs that actually ship. Two extra arms
price the one remaining assembly copy:

  fc2_l1     s_full with fc2 writing to L1 instead of DRAM -- if the 16 block outputs fit in L1 the
             concat reads on chip and only writes 134 MB instead of moving 268. Not expected to be
             portable (16 x 8.39 MB = 134 MB against 153.5 MB of total L1 on qb2's 110-core grid);
             measured here to SIZE the prize, not to propose it.
  cat_l1     the concat itself, fed from L1 block outputs.

Also dumps every ttnn symbol that could write a TILE tensor into another tensor at an offset, so
C's output half is either given a no-kernel route or recorded as needing one, on evidence.
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
    ma = to(torch.randn(1, 1, N, N)); mb = to(torch.randn(N, N))
    m_mm, _ = timed(lambda: ttnn.deallocate(ttnn.linear(
        ma, mb, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)),
        dev, reps=2, batches=5)
    res["roofs"]["mm_hifi4_TFLOPs"] = round(2 * N ** 3 / (m_mm / 1e3) / 1e12, 1)
    ttnn.deallocate(ma); ttnn.deallocate(mb)
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

    def prefix(stop, fc2_mc=DR, keep=False):
        outs = []
        for i in range(nblk):
            p = sl(i)
            if stop == "s":
                ttnn.deallocate(p); continue
            xn = ln(p); ttnn.deallocate(p)
            if stop == "s_ln":
                ttnn.deallocate(xn); continue
            h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
            h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
            ttnn.deallocate(xn)
            if stop == "s_ln_fc1":
                ttnn.deallocate(h1); ttnn.deallocate(h2); continue
            gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                  memory_config=L1)
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            if stop == "s_ln_mul":
                ttnn.deallocate(gated); continue
            o = ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                            core_grid=T.CORE_GRID_MAIN, memory_config=fc2_mc)
            ttnn.deallocate(gated)
            outs.append(o)
        if keep:
            return outs
        for o in outs:
            ttnn.deallocate(o)
        return None

    def arm_cat_l1():
        outs = prefix("s_full", fc2_mc=L1, keep=True)
        r = ttnn.concat(outs, dim=1)
        for o in outs:
            ttnn.deallocate(o)
        ttnn.deallocate(r)

    arms = [("s", lambda: prefix("s")),
            ("s_ln", lambda: prefix("s_ln")),
            ("s_ln_fc1", lambda: prefix("s_ln_fc1")),
            ("s_ln_mul", lambda: prefix("s_ln_mul")),
            ("s_full", lambda: prefix("s_full")),
            ("fc2_l1", lambda: prefix("s_full", fc2_mc=L1)),
            ("cat_l1", arm_cat_l1)]
    for name, fn in arms:
        try:
            m, raw = timed(fn, dev)
        except Exception as exc:
            res["ms"][name] = None
            res["raw"][name] = "EXC: %s" % (str(exc).splitlines()[0][:220],)
            print("%-10s EXC %s" % (name, res["raw"][name]), flush=True)
            continue
        res["ms"][name], res["raw"][name] = round(m, 4), [round(v, 4) for v in raw]
        print("%-10s %8.4f ms  %s" % (name, m, res["raw"][name]), flush=True)

    ms = res["ms"]
    d = lambda hi, lo: (round(ms[hi] - ms[lo], 4)
                        if ms.get(hi) is not None and ms.get(lo) is not None else None)
    res["per_op_ms"] = {"slice": ms.get("s"), "ln": d("s_ln", "s"), "fc1": d("s_ln_fc1", "s_ln"),
                        "mul": d("s_ln_mul", "s_ln_fc1"), "fc2": d("s_full", "s_ln_mul")}
    res["per_op_s_per_fold"] = {k: (round(v * CALLS_PER_FOLD / 1e3, 3) if v is not None else None)
                                for k, v in res["per_op_ms"].items()}
    gf = (2 * (L * L) * C_Z * D_FF * 2 + 2 * (L * L) * D_FF * C_Z) / 1e9
    res["gflop_per_call"] = round(gf, 1)
    res["compute_floor_ms"] = round(gf / res["roofs"]["mm_hifi4_TFLOPs"] / 1e3 * 1e3, 4)
    res["s_per_fold"] = {k: (round(v * CALLS_PER_FOLD / 1e3, 3) if v is not None else None)
                         for k, v in ms.items()}
    res["fc2_l1_delta_ms"] = d("fc2_l1", "s_full")

    # C-out: is there ANY op on this wheel that writes a TILE tensor into another at an offset?
    cands = {}
    for mod, nm in [("ttnn", n) for n in dir(ttnn)] + \
                   [("ttnn.experimental", n) for n in dir(ttnn.experimental)]:
        if any(k in nm for k in ("slice_write", "scatter", "index_put", "assign", "copy",
                                 "fill_", "update_cache", "paged")):
            f = getattr(ttnn if mod == "ttnn" else ttnn.experimental, nm, None)
            doc = (getattr(f, "__doc__", "") or "")
            cands["%s.%s" % (mod, nm)] = " ".join(doc.split())[:260]
    res["write_into_candidates"] = cands

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("raw", "write_into_candidates")},
                     indent=1))
    for k, v in cands.items():
        print("CAND %-44s %s" % (k, v[:170]))


if __name__ == "__main__":
    main()
