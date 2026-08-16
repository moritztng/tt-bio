#!/usr/bin/env python3
"""S-F: name the binding limit of the shipped pair FFN, then price the fusion fork against it.

The p2 screen measured the ASSEMBLY around the row block (chunk+concat, 1.171 ms/call, 0.63 s of
fold). This one measures the block BODY, which is 11x larger, and puts each piece of it on a roof
measured on this same card in this same process -- never a roof carried in from another host
(memory roofline-roof-must-be-measured-not-asserted).

Question it answers: the shipped pair transition is 14.63 ms/call on qb2 against a 3.66 ms/call
compute floor (412 GFLOP at the measured HiFi4 roof) and ~0.63 ms/call of DRAM traffic. It
saturates NEITHER roof, so "DRAM-traffic bound" is no longer true of it and the fusion fork cannot
be priced by subtracting bytes. This screen splits the body per op, in chain, batched, and reports
each piece against both roofs.

All arms are batched (4 chain calls per synchronize, median of 5) and never per-op-synced
(memory tt-bio-isolated-op-timing-oversync-inflates-cost).
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio import esmc as EC
assert Path(T.__file__).resolve().is_relative_to(REPO), "tt_bio from %s" % T.__file__

CALLS_PER_FOLD = 538


def timed(fn, dev, reps=4, batches=5, warm=2):
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
    ck = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
          else ttnn.types.BlackholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "cores": g.x * g.y, "size": L, "rows": ROWS,
           "is_small_grid": bool(getattr(T, "_IS_SMALL_GRID", False)),
           "l1_fc1": bool(EC._PAIR_FFN_L1_FC1), "split": bool(EC._SPLIT_SWIGLU),
           "ttnn": getattr(ttnn, "__version__", "?"), "ms": {}, "raw": {}, "roofs": {}}

    torch.manual_seed(0)
    to = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ---- roofs, measured here, this process, this card -------------------------------------
    N = 4096
    ra = to(torch.randn(1, 1, N * 4, N)); rb = to(torch.randn(1, 1, N * 4, N))
    m_add, _ = timed(lambda: ttnn.deallocate(ttnn.add(ra, rb)), dev, reps=2, batches=5)
    add_bytes = 3 * N * 4 * N * 2
    res["roofs"]["dram_add_GBps"] = round(add_bytes / (m_add / 1e3) / 1e9, 1)
    ttnn.deallocate(ra); ttnn.deallocate(rb)
    ma = to(torch.randn(1, 1, N, N)); mb = to(torch.randn(N, N))
    m_mm, _ = timed(lambda: ttnn.deallocate(
        ttnn.linear(ma, mb, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                    core_grid=T.CORE_GRID_MAIN)), dev, reps=2, batches=5)
    res["roofs"]["mm_hifi4_TFLOPs"] = round(2 * N ** 3 / (m_mm / 1e3) / 1e12, 1)
    ttnn.deallocate(ma); ttnn.deallocate(mb)
    print("roofs", res["roofs"], flush=True)

    # ---- production tensors -----------------------------------------------------------------
    x = to(torch.randn(1, L, L, C_Z))
    nw = to(torch.ones(C_Z)); nb = to(torch.zeros(C_Z))
    w1a = to(torch.randn(C_Z, D_FF) * 0.02)
    w1b = to(torch.randn(C_Z, D_FF) * 0.02)
    w2 = to(torch.randn(D_FF, C_Z) * 0.02)
    l1cfg = dict(l1_out=True, l1_bw=T._PAIR_FFN_FC1_BW, l1_block_w=T._PAIR_FFN_FC1_BLOCK_W)

    def ln(t, mc=None):
        kw = {"memory_config": mc} if mc is not None else {}
        return ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5,
                               compute_kernel_config=ck, **kw)

    nblk = -(-L // ROWS)
    pre = ttnn.chunk(x, nblk, dim=1)              # row blocks, cut once
    prn = [ln(p) for p in pre]                    # row blocks, pre-normed (DRAM)

    def a_ln():
        for p in pre:
            ttnn.deallocate(ln(p))

    def a_fc1():
        for xn in prn:
            h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
            h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
            ttnn.deallocate(h1); ttnn.deallocate(h2)

    def a_fc1mul():
        for xn in prn:
            h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
            h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
            gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                  memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(h1); ttnn.deallocate(h2); ttnn.deallocate(gated)

    def a_body():                                  # fc1 + mul + fc2, ln hoisted, no assembly
        for xn in prn:
            h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
            h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
            gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                  memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            out = ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                              core_grid=T.CORE_GRID_MAIN)
            ttnn.deallocate(gated); ttnn.deallocate(out)

    def a_chain():                                 # the shipped call, end to end
        parts = ttnn.chunk(x, nblk, dim=1)
        outs = []
        for p in parts:
            xn = ln(p)
            h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
            h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
            ttnn.deallocate(xn)
            gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                  memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            outs.append(ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                    core_grid=T.CORE_GRID_MAIN))
            ttnn.deallocate(gated)
        for p in parts:
            ttnn.deallocate(p)
        r = ttnn.concat(outs, dim=1)
        for o in outs:
            ttnn.deallocate(o)
        ttnn.deallocate(r)

    # F1: does an L1 layer_norm output let fc1 read its operand on chip? no kernel needed.
    def a_body_l1ln():
        for p in pre:
            xn = ln(p, ttnn.L1_MEMORY_CONFIG)
            h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
            h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
            ttnn.deallocate(xn)
            gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                  memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            out = ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                              core_grid=T.CORE_GRID_MAIN)
            ttnn.deallocate(gated); ttnn.deallocate(out)

    def a_body_ln():                               # ln + body, no assembly (= chain - assembly)
        for p in pre:
            xn = ln(p)
            h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
            h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
            ttnn.deallocate(xn)
            gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                  memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            out = ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                              core_grid=T.CORE_GRID_MAIN)
            ttnn.deallocate(gated); ttnn.deallocate(out)

    for name, fn in (("chain", a_chain), ("body_ln", a_body_ln), ("body", a_body),
                     ("fc1", a_fc1), ("fc1mul", a_fc1mul), ("ln", a_ln),
                     ("body_l1ln", a_body_l1ln)):
        m, raw = timed(fn, dev)
        res["ms"][name], res["raw"][name] = round(m, 4), [round(v, 4) for v in raw]
        print("%-10s %8.4f ms  %s" % (name, m, res["raw"][name]), flush=True)

    # parity of the L1-layer_norm arm, torch.equal and nothing weaker
    lnd = ttnn.concat([ln(p) for p in pre], dim=1)
    lnl = ttnn.concat([ln(p, ttnn.L1_MEMORY_CONFIG) for p in pre], dim=1)
    res["l1_ln_torch_equal"] = bool(torch.equal(ttnn.to_torch(lnd), ttnn.to_torch(lnl)))
    ttnn.deallocate(lnd); ttnn.deallocate(lnl)

    ms = res["ms"]
    gf = (2 * (L * L) * C_Z * D_FF * 2 + 2 * (L * L) * D_FF * C_Z) / 1e9   # fc1 pair + fc2, GFLOP
    res["gflop_per_call"] = round(gf, 1)
    roof_tf = res["roofs"]["mm_hifi4_TFLOPs"]
    roof_gb = res["roofs"]["dram_add_GBps"]
    res["compute_floor_ms"] = round(gf / roof_tf / 1e3 * 1e3, 4)
    # DRAM the shipped body cannot avoid: read x_norm once, write out once.
    body_bytes = 2 * L * L * C_Z * 2
    res["body_dram_floor_ms"] = round(body_bytes / (roof_gb * 1e9) * 1e3, 4)
    res["pct_compute_roof"] = {k: round(100 * res["compute_floor_ms"] / ms[k], 1)
                               for k in ("chain", "body_ln", "body")}
    res["assembly_ms"] = round(ms["chain"] - ms["body_ln"], 4)
    res["ln_ms"] = round(ms["body_ln"] - ms["body"], 4)
    res["fc2_ms"] = round(ms["body"] - ms["fc1mul"], 4)
    res["mul_ms"] = round(ms["fc1mul"] - ms["fc1"], 4)
    res["fc1_ms"] = round(ms["fc1"], 4)
    res["l1ln_delta_ms"] = round(ms["body_l1ln"] - ms["body_ln"], 4)
    res["s_per_fold"] = {k: round(v * CALLS_PER_FOLD / 1e3, 3) for k, v in ms.items()}
    res["headroom_to_compute_floor_s"] = round(
        (ms["chain"] - res["compute_floor_ms"]) * CALLS_PER_FOLD / 1e3, 3)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
