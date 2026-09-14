#!/usr/bin/env python3
"""The shape-honest arithmetic roof for every matmul class that carries the 512 aa fold's FLOPs.

`perf/roof_budget/` priced the arithmetic term at the dense-cube rate and got 2.092 s, then
dismissed it. `perf/roof_pair_transition/` measured one unit's own shapes and found 28 % of the
cube. If that fraction is typical the arithmetic floor is 7.47 s and it binds instead of the
6.934 s traffic floor. This file measures the fraction for the classes carrying 97 % of the FLOPs.

Construction, the same one `roof_pair_transition`'s `mm3only` used and for the same reason: real
shapes, the operand residency the fold keeps, the fold's own row blocking, and NOTHING that is not
a matmul. Where more than one residency or program config is plausible the arm set carries both
and the table takes the FASTER one, so the fraction is an upper bound on the rate and the implied
arithmetic floor is a LOWER bound on the arithmetic floor. A floor that binds under that rule
binds for real.

Method, unchanged from shape_roof.py: one process, one device, one session; `reps` enqueues per
sync so host dispatch is amortised; the arm set runs whole once per block in a fixed order so a
JIT warm-up or a clock ramp cannot bias one arm against another; minimum over blocks. The dense
cube runs in this same session, so every `pct_of_cube` is internally consistent.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tt_bio.device_lease import CardSetLease                                   # noqa: E402

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG

S = 512          # tokens at 512 aa
CZ = 128         # pair channel
DEPTH = 1024     # MSA_PAD_MULTIPLE
HD = 32          # triangle-attention head dim
NH = 4           # triangle-attention heads
TOK = 768        # diffusion token channel
ATOM_N, ATOM_W, ATOM_C = 140, 128, 128


def _f(*d):
    """2*prod(d) -- the FLOP count of a matmul whose (batch..., M, K, N) are d."""
    n = 2
    for x in d:
        n *= x
    return n


def build(dev):
    arch = dev.arch()
    kcls = (ttnn.types.WormholeComputeKernelConfig if arch == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    cc = dev.compute_with_storage_grid_size()
    gx, gy = cc.x, cc.y
    cg = ttnn.CoreGrid(y=min(10, gy), x=min(11, gx))     # the fold's CORE_GRID_MAIN, clamped

    def t(shape, mc=DRAM):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    def lin(x, w, out_mc, grid=None):
        kw = {"core_grid": grid} if grid is not None else {}
        return ttnn.linear(x, w, compute_kernel_config=kc, memory_config=out_mc,
                           dtype=ttnn.bfloat16, **kw)

    # ---- operands, allocated once -------------------------------------------------------
    z = t((1, S, S, CZ))                       # the pair tensor, 67.1 MB, DRAM as the fold keeps it
    zflat = t((S * S, CZ))                     # the same tensor as a 2-D matmul, same arithmetic
    w640 = t((CZ, 5 * CZ))                     # trimul [g_a|g_b|p_a|p_b|g_out]
    w544 = t((CZ, 3 * NH * HD + CZ + 32))      # tri-att [q|k|v|g|bias]
    w128 = t((CZ, CZ))                         # both out-projections
    ta = t((1, CZ, S, S)); tb = t((1, CZ, S, S))   # the trimul einsum's two operands
    q = t((S, NH, S, HD)); k = t((S, NH, S, HD)); v = t((S, NH, S, HD))
    bias = t((1, NH, S, S))
    tw1 = t((CZ, 4 * CZ)); tw3 = t((4 * CZ, CZ))   # pair Transition fc1/fc2 and fc3
    x768 = t((1, S, TOK))
    x1536 = t((1, S, 2 * TOK))
    d768 = t((TOK, TOK)); d1536 = t((TOK, 2 * TOK)); d3072 = t((TOK, 4 * TOK))
    d1536_768 = t((2 * TOK, TOK))
    x768_l1 = t((1, S, TOK), L1)
    x1536_l1 = t((1, S, 2 * TOK), L1)
    opa = t((S * HD, DEPTH)); opb = t((DEPTH, S * HD))
    pwa_a = t((DEPTH, HD, S)); pwa_b = t((1, S, S)); pwa_b2 = t((S, S))
    pwa_flat = t((DEPTH * HD, S))
    atom_x = t((1, ATOM_N, ATOM_W, ATOM_C)); atom_w = t((ATOM_C, 2 * ATOM_C))
    atom_flat = t((ATOM_N * ATOM_W, ATOM_C))
    cube4 = t((4096, 4096)); cube4b = t((4096, 4096))
    wide_l1 = t((1, 16, S, 4 * CZ), L1)         # one hidden-width Transition row block in L1
    blk = t((1, 16, S, CZ))                     # one shipped Transition row block, DRAM

    keep = [z, zflat, w640, w544, w128, ta, tb, q, k, v, bias, tw1, tw3, x768, x1536,
            d768, d1536, d3072, d1536_768, x768_l1, x1536_l1, opa, opb, pwa_a, pwa_b,
            atom_x, atom_w, cube4, cube4b, wide_l1, blk, pwa_b2, pwa_flat, atom_flat]

    def blocked(w, h, out_mc=DRAM, grid=None):
        """The pair tensor through one linear, row-blocked h rows at a time, as the fold blocks."""
        def run():
            cs = ttnn.chunk(z, -(-S // h), dim=1)
            outs = [lin(c, w, out_mc, grid) for c in cs]
            for c in cs:
                ttnn.deallocate(c)
            for o in outs[1:]:
                ttnn.deallocate(o)
            return outs[0]
        return run

    def sdpa(qc, kcz):
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=cc, exp_approx_mode=False,
                                    q_chunk_size=qc, k_chunk_size=kcz)
        return lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5, program_config=pc)

    NB_ = S // 16                                # 32 row blocks at the shipped h = 16

    def trans_block(kind):
        """One shipped row block's matmuls, repeated over the unit's 32 blocks.

        No ttnn.chunk here: chunk allocates and copies (it is not a view), and it is not a
        matmul, so it would sit inside a denominator that exists to price matmuls.
        """
        def run():
            out = None
            for _ in range(NB_):
                if kind == "fc12":
                    a = lin(blk, tw1, L1, cg)
                    b = lin(blk, tw1, L1, cg)
                    ttnn.deallocate(b)
                    o = a
                else:
                    o = lin(wide_l1, tw3, DRAM, cg)
                if out is not None:
                    ttnn.deallocate(out)
                out = o
            return out
        return run

    def trans_mm3only(h=16):
        """The roof-pair-transition arm verbatim: chunk, three matmuls per block, L1 between."""
        def run():
            cs = ttnn.chunk(z, -(-S // h), dim=1)
            parts = []
            for c in cs:
                a = lin(c, tw1, L1, cg)
                b = lin(c, tw1, L1, cg)
                ttnn.deallocate(b)
                parts.append(lin(a, tw3, DRAM, cg))
                ttnn.deallocate(a)
            for c in cs:
                ttnn.deallocate(c)
            for o in parts[1:]:
                ttnn.deallocate(o)
            return parts[0]
        return run

    NB = S // 16                                          # 32 row blocks at the shipped h=16
    FC1 = _f(1, 16 * S, CZ, 4 * CZ) * NB                  # one fc1-shaped matmul over the unit
    A = {}
    # name -> (callable, FLOP, reps)
    A["cube4096"] = (lambda: ttnn.matmul(cube4, cube4b, compute_kernel_config=kc,
                                         memory_config=DRAM), _f(4096, 4096, 4096), 3)

    # TriangleMultiplication -- in-projection [g_a|g_b|p_a|p_b|g_out], 10.98 % of the fold's FLOPs
    A["trimul_in_whole"] = (lambda: lin(z, w640, DRAM), _f(1, S * S, CZ, 5 * CZ), 2)
    A["trimul_in_h64"] = (blocked(w640, 64), _f(1, S * S, CZ, 5 * CZ), 2)
    A["trimul_in_h64_cg"] = (blocked(w640, 64, grid=cg), _f(1, S * S, CZ, 5 * CZ), 2)
    A["trimul_in_flat"] = (lambda: lin(zflat, w640, DRAM), _f(1, S * S, CZ, 5 * CZ), 2)
    # TriangleMultiplication -- the triangle product itself, 8.78 %
    A["trimul_einsum"] = (lambda: ttnn.matmul(ta, tb, compute_kernel_config=kc,
                                              memory_config=DRAM), _f(CZ, S, S, S), 3)
    # the two 128->128 output projections, 4.40 % together
    A["pair_out128_whole"] = (lambda: lin(z, w128, DRAM), _f(1, S * S, CZ, CZ), 3)
    A["pair_out128_h64"] = (blocked(w128, 64), _f(1, S * S, CZ, CZ), 3)
    A["pair_out128_flat"] = (lambda: lin(zflat, w128, DRAM), _f(1, S * S, CZ, CZ), 3)
    A["pair_out128_flat_cg"] = (lambda: lin(zflat, w128, DRAM, cg), _f(1, S * S, CZ, CZ), 3)

    # TriangleAttention -- fused qkv+gate+bias in-projection, 9.33 %
    A["triatt_in_whole"] = (lambda: lin(z, w544, DRAM), _f(1, S * S, CZ, 17 * 32), 2)
    A["triatt_in_h64"] = (blocked(w544, 64), _f(1, S * S, CZ, 17 * 32), 2)
    A["triatt_in_flat"] = (lambda: lin(zflat, w544, DRAM), _f(1, S * S, CZ, 17 * 32), 2)
    # TriangleAttention -- the fused SDPA, QK^T and AV together, 17.56 %
    SD = 2 * _f(S * NH, S, S, HD)
    for qc in (32, 64, 128, 256, 512):
        A["triatt_sdpa_q%d" % qc] = (sdpa(qc, 512), SD, 2)
    A["triatt_sdpa_q128_k128"] = (sdpa(128, 128), SD, 2)
    A["triatt_sdpa_q256_k256"] = (sdpa(256, 256), SD, 2)

    # the pair Transition, 13.17 % -- fc1/fc2 (8.78 %) and fc3 (4.39 %) separately, plus the
    # roof-pair-transition arm itself as the cross-session control
    A["trans_fc12"] = (trans_block("fc12"), 2 * FC1, 3)
    A["trans_fc3"] = (trans_block("fc3"), FC1, 3)
    A["trans_mm3only"] = (trans_mm3only(), 3 * FC1, 3)

    # the diffusion token layer, 27.18 % over four widths
    for nm, x, w, n in (("768x768", x768, d768, TOK), ("768x1536", x768, d1536, 2 * TOK),
                        ("768x3072", x768, d3072, 4 * TOK), ("1536x768", x1536, d1536_768, TOK)):
        kdim = x.shape[-1]
        A["dit_%s" % nm] = (lambda x=x, w=w: lin(x, w, DRAM), _f(1, S, kdim, n), 200)
    for nm, x, w, n in (("768x768", x768_l1, d768, TOK), ("768x1536", x768_l1, d1536, 2 * TOK),
                        ("768x3072", x768_l1, d3072, 4 * TOK),
                        ("1536x768", x1536_l1, d1536_768, TOK)):
        kdim = x.shape[-1]
        A["dit_%s_l1" % nm] = (lambda x=x, w=w: lin(x, w, L1), _f(1, S, kdim, n), 200)

    # OuterProductMean, 4.02 % -- the MSA depth contraction
    A["opm"] = (lambda: ttnn.matmul(opa, opb, compute_kernel_config=kc, memory_config=DRAM),
                _f(S * HD, DEPTH, S * HD), 1)
    # PairWeightedAveraging, 1.00 %
    A["pwa"] = (lambda: ttnn.matmul(pwa_a, pwa_b, compute_kernel_config=kc, memory_config=DRAM),
                _f(DEPTH, HD, S, S), 3)
    A["pwa_2d"] = (lambda: ttnn.matmul(pwa_a, pwa_b2, compute_kernel_config=kc,
                                       memory_config=DRAM), _f(DEPTH, HD, S, S), 3)
    A["pwa_flat"] = (lambda: lin(pwa_flat, pwa_b2, DRAM), _f(DEPTH, HD, S, S), 3)
    # the atom transformer, 0.64 %
    A["atom"] = (lambda: lin(atom_x, atom_w, DRAM), _f(ATOM_N, ATOM_W, ATOM_C, 2 * ATOM_C), 100)
    A["atom_l1"] = (lambda: lin(atom_x, atom_w, L1), _f(ATOM_N, ATOM_W, ATOM_C, 2 * ATOM_C), 100)
    A["atom_flat"] = (lambda: lin(atom_flat, atom_w, DRAM),
                      _f(ATOM_N, ATOM_W, ATOM_C, 2 * ATOM_C), 100)
    A["trimul_einsum_AA2"] = A["trimul_einsum"]
    A["cube4096_AA"] = A["cube4096"]
    return A, keep, (gx, gy), (cg.x, cg.y)


def time_arms(arms, order, blocks, dev, warm=2):
    err, live = {}, []
    for n in order:
        fn = arms[n][0]
        try:
            for _ in range(warm):
                ttnn.deallocate(fn())
            live.append(n)
        except Exception as e:                                                # noqa: BLE001
            err[n] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])
            print("SKIP %-24s %s" % (n, err[n]), flush=True)
    order = live
    ttnn.synchronize_device(dev)
    best = {n: None for n in order}
    for _ in range(blocks):
        for n in list(order):
            if n not in best:
                continue
            fn, _flop, reps = arms[n]
            outs = []
            try:
                t0 = time.perf_counter()
                for _ in range(reps):
                    outs.append(fn())
                    # keep the enqueue depth but bound the live footprint: an L1-resident output
                    # at reps=40 overruns L1 on its own and takes the whole session with it.
                    if len(outs) > 4:
                        ttnn.deallocate(outs.pop(0))
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / reps
            except Exception as e:                                            # noqa: BLE001
                err[n] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])
                print("DROP %-24s %s" % (n, err[n]), flush=True)
                best.pop(n, None)
                for o in outs:
                    ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                continue
            for o in outs:
                ttnn.deallocate(o)
            best[n] = dt if best[n] is None else min(best[n], dt)
    return {n: v for n, v in best.items() if v is not None}, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "shape_roofs.json")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    lease = CardSetLease().acquire()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    try:
        arms, keep, grid, cgu = build(dev)
        order = [n for n in arms if not a.only or n in a.only.split(",")]
        best, err = time_arms(arms, order, a.blocks, dev)
        order = [n for n in order if n in best]
        rows = [{"arm": n, "ms": best[n] * 1e3, "TFLOPs": arms[n][1] / best[n] / 1e12,
                 "reps": arms[n][2]} for n in order]
        cube = next((r["TFLOPs"] for r in rows if r["arm"] == "cube4096"), None)
        for r in rows:
            r["pct_of_cube"] = 100 * r["TFLOPs"] / cube if cube else None
        out = {"host": platform.node(), "arch": str(dev.arch()), "grid": list(grid),
               "core_grid_used": list(cgu), "blocks": a.blocks,
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "cube4096_TFLOPs": cube, "refused": err, "rows": rows}
        a.out.write_text(json.dumps(out, indent=1))
        w = max(len(r["arm"]) for r in rows)
        for r in rows:
            print("%-*s  %10.4f ms  %8.2f TFLOP/s  %6.1f %% of cube"
                  % (w, r["arm"], r["ms"], r["TFLOPs"], r["pct_of_cube"] or 0), flush=True)
        for x in keep:
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
