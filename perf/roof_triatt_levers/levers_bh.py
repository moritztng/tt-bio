#!/usr/bin/env python3
"""The two mechanisms `roof-tri-close` named, measured against what the fold ACTUALLY calls.

`roof-tri-close` measured both on stock ttnn microbenchmarks: `ttnn.transformer.
scaled_dot_product_attention` with a batch-1 bias (1.88x without the mask) and a bare
`ttnn.matmul` with no program config (2.10x on 32 cores against 72). Neither is the call the
Boltz-2 512 aa fold issues:

  * triangle attention goes through `tenstorrent._tri_att_sdpa` -> `triatt_sdpa.sdpa`, a
    transcribed kernel that fills the whole mask once per head into a persistent CB
    (`PERSISTENT_MASK`), measured 2.431x over the native op at this size;
  * the triangle product goes through `ttnn.matmul(program_config=_triangle_mul_program_config(
    seq_len_tiles))`, a `MatmulMultiCoreReuseMultiCast` config whose `per_core_M/N` are derived
    from the grid, with `transpose_b` folded in.

So both prizes may already be spent. This asks the question on the production call, on Blackhole,
which is the part the number has to ship on.

Harness reused as-is: `perf/roof_tri_arith/tri_mech.time_arms` (interleaved arms, warm, min over
blocks, one synchronize per timed region). No new tracer code.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "roof_tri_arith"))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

import tt_bio.tenstorrent as T                                                # noqa: E402
from tt_bio import triatt_sdpa as TA                                          # noqa: E402
from tri_mech import time_arms                                                # noqa: E402

assert Path(T.__file__).is_relative_to(ROOT), (T.__file__, ROOT)

DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32


def tri_pc(slt: int, gx: int, gy: int, in0_bw: int, sub=(1, 1)):
    """`_triangle_mul_program_config` with the grid as an argument instead of the module default.

    Every field is derived exactly as production derives it; only `COMPUTE_GRID_MAIN` is replaced.
    `in0_block_w` is held at production's value on every arm, which is the one field that decides
    bit-exactness (`_trimul_in0_block_w`), so the ladder is an occupancy A/B and nothing else.
    """
    pcm, pcn = -(-slt // gy), -(-slt // gx)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=in0_bw,
        out_subblock_h=sub[0], out_subblock_w=sub[1], out_block_h=pcm, out_block_w=pcn,
        per_core_M=pcm, per_core_N=pcn, transpose_mcast=False, fused_activation=None,
        fuse_batch=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "levers_bh.json")
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--chan", type=int, default=128, help="trimul channel batch")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--skip-sdpa", action="store_true")
    ap.add_argument("--skip-product", action="store_true")
    a = ap.parse_args()

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    S, C, H, DH = a.seq, a.chan, a.heads, a.head_dim
    slt = -(-S // TILE)
    in0_bw = T._trimul_in0_block_w(slt)
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)

    def t(shape):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.float32).bfloat16(),
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=DRAM)

    arms, keep, notes = {}, [], {}
    fl_prod = 2 * C * S ** 3
    fl_sdpa = 2 * 2 * S * H * S * S * DH

    # ---- 1. the triangle product, production call, grid as the only variable -----------------
    grids = [(gx, gy)]
    for g in [(8, 8), (8, 5), (8, 4), (4, 8), (4, 4), (2, 4), (2, 2), (1, 1)]:
        if g[0] <= gx and g[1] <= gy and g not in grids:
            grids.append(g)
    if not a.skip_product:
        af, bf = t((1, C, S, S)), t((1, C, S, S))
        keep += [af, bf]
        for g in grids:
            pc = tri_pc(slt, g[0], g[1], in0_bw)
            used = (-(-slt // pc.per_core_N), -(-slt // pc.per_core_M))   # cores actually engaged
            notes["prod_g%dx%d" % g] = {
                "declared_cores": g[0] * g[1], "engaged_cores": used[0] * used[1],
                "per_core_M": pc.per_core_M, "per_core_N": pc.per_core_N, "in0_block_w": in0_bw}
            arms["prod_g%dx%d" % g] = (
                lambda pc=pc: ttnn.matmul(af, bf, compute_kernel_config=ckc, memory_config=DRAM,
                                          program_config=pc, dtype=ttnn.bfloat16,
                                          transpose_b=True), fl_prod, 4)
        # The output subblock, at the production grid. `out_subblock_h/w` is what the FPU
        # amortises one `matmul_tiles` sequence over; production pins 1x1, and the dest register
        # file holds 4 tiles under fp32_dest_acc, so 2x2 is legal and unmeasured. It does not
        # touch `in0_block_w`, so the K accumulation order -- the one thing that decides
        # bit-exactness here -- is unchanged.
        for g in ((gx, gy), (8, 8), (8, 4)):
            pcm, pcn = -(-slt // g[1]), -(-slt // g[0])
            for sub in ((2, 2), (1, 2), (2, 1), (1, 4), (4, 1)):
                if pcm % sub[0] or pcn % sub[1] or sub[0] * sub[1] > 4:
                    continue
                nm = "prod_g%dx%d_sub%dx%d" % (g[0], g[1], sub[0], sub[1])
                pc = tri_pc(slt, g[0], g[1], in0_bw, sub)
                arms[nm] = (
                    lambda pc=pc: ttnn.matmul(af, bf, compute_kernel_config=ckc,
                                              memory_config=DRAM, program_config=pc,
                                              dtype=ttnn.bfloat16, transpose_b=True), fl_prod, 4)
                notes[nm] = {"declared_cores": g[0] * g[1],
                             "engaged_cores": (-(-slt // pcn)) * (-(-slt // pcm)),
                             "per_core_M": pcm, "per_core_N": pcn, "in0_block_w": in0_bw,
                             "out_subblock": list(sub)}
        # roof-tri-close's own core ladder, reproduced arm for arm: `ttnn.matmul(core_grid=...)`.
        # That kwarg does NOT narrow the program config above -- it sends the call to a different
        # factory, which derives `in0_block_w = 1` and its own blocking (the same thing
        # `_pair_proj_program_config`'s docstring records for `ttnn.linear(core_grid=)`). So the
        # WH ladder compared two program FAMILIES, not one family on two grids, and that is the
        # candidate cause of the sign inversion. These arms say whether it reproduces here.
        for g in ((gx, gy), (8, 8), (8, 4), (4, 4)):
            if g[0] > gx or g[1] > gy:
                continue
            nm = "coregrid_%dx%d" % g
            arms[nm] = (
                lambda g=g: ttnn.matmul(af, bf, compute_kernel_config=ckc, memory_config=DRAM,
                                        dtype=ttnn.bfloat16,
                                        core_grid=ttnn.CoreGrid(y=g[1], x=g[0])), fl_prod, 4)
        # what roof-tri-close actually timed: no program config at all. Same operands and the
        # same FLOPs, without the transpose, exactly as that row called it.
        arms["prod_ttnn_default"] = (
            lambda: ttnn.matmul(af, bf, compute_kernel_config=ckc, memory_config=DRAM,
                                dtype=ttnn.bfloat16), fl_prod, 2)
        arms["prod_g%dx%d_AA" % (gx, gy)] = arms["prod_g%dx%d" % (gx, gy)]

    # ---- 2. triangle attention: the shipped kernel against the two stock arms ----------------
    if not a.skip_sdpa:
        q, k, v = t((S, H, S, DH)), t((S, H, S, DH)), t((S, H, S, DH))
        bias = t((1, H, S, S))
        keep += [q, k, v, bias]
        scale = DH ** -0.5
        qc, kc_ = T._sdpa_chunks_shipped(S, S)
        pc_sdpa = T._sdpa_program_config(q_chunk_size=qc, k_chunk_size=kc_)
        notes["sdpa"] = {"q_chunk": qc, "k_chunk": kc_,
                         "fused_pairs": list(TA.fused_pairs(S, H, DH, gx * gy))}

        def prod_sdpa(ablate=()):
            # `triatt_sdpa._ABLATE` is read at CALL time and lands in `defines_extra`, which is part
            # of the sdpa_generic program-cache key -- so flipping it per arm compiles a second
            # program and lets the ablation interleave with its control in ONE process instead of
            # being a process-level env switch. MASKADD removes the bias add from the compute
            # kernel: the arm returns WRONG VALUES and only its time means anything.
            was, TA._ABLATE = TA._ABLATE, tuple(ablate)
            try:
                o = T._tri_att_sdpa(q, k, v, bias, scale)
            finally:
                TA._ABLATE = was
            assert o is not None
            return o

        arms["sdpa_production"] = (prod_sdpa, fl_sdpa, 4)
        arms["sdpa_prod_nomaskadd"] = (lambda: prod_sdpa(("MASKADD",)), fl_sdpa, 4)
        arms["sdpa_stock_mask"] = (
            lambda: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=scale, memory_config=DRAM,
                program_config=pc_sdpa, compute_kernel_config=ckc), fl_sdpa, 4)
        arms["sdpa_stock_nomask"] = (
            lambda: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=scale, memory_config=DRAM,
                program_config=pc_sdpa, compute_kernel_config=ckc), fl_sdpa, 4)
        arms["sdpa_production_AA"] = arms["sdpa_production"]
        arms["sdpa_prod_nomaskadd_AA"] = arms["sdpa_prod_nomaskadd"]

    best, err = time_arms(arms, list(arms), a.blocks, dev)

    # ---- 3. bit-exactness of every grid arm against production -------------------------------
    exact = {}
    if not a.skip_product:
        ref_arm = "prod_g%dx%d" % (gx, gy)
        ro = arms[ref_arm][0]()
        ref = ttnn.to_torch(ro)
        ttnn.deallocate(ro)
        for n in list(arms):
            if not n.startswith("prod_g") or n.endswith("_AA") or n not in best \
                    or n == ref_arm:
                continue
            o = arms[n][0]()
            got = ttnn.to_torch(o)
            ttnn.deallocate(o)
            exact[n] = {"equal": bool(torch.equal(ref, got)),
                        "max_abs": float((ref.float() - got.float()).abs().max())}

    rows = []
    for n, dt in best.items():
        rows.append({"arm": n, "ms": dt * 1e3, "TFLOPs": arms[n][1] / dt / 1e12})
    rows.sort(key=lambda r: r["arm"])
    out = {"host": platform.node(), "arch": str(dev.arch()), "grid": [gx, gy],
           "seq": S, "chan": C, "heads": H, "head_dim": DH, "blocks": a.blocks,
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "notes": notes, "refused": err, "bitexact": exact, "rows": rows,
           "triatt_stats": {"served": TA.STATS[0], "declined": TA.STATS[1],
                            "rejects": {str(k): v for k, v in TA.REJECTS.items()}}}
    a.out.write_text(json.dumps(out, indent=1))
    w = max(len(r["arm"]) for r in rows)
    for r in rows:
        e = exact.get(r["arm"])
        tag = "" if e is None else ("  bit-exact" if e["equal"] else "  max_abs %.4g" % e["max_abs"])
        print("%-*s  %9.4f ms  %8.2f TFLOP/s%s" % (w, r["arm"], r["ms"], r["TFLOPs"], tag),
              flush=True)
    print("triatt served/declined:", TA.STATS, flush=True)
    for x in keep:
        ttnn.deallocate(x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
