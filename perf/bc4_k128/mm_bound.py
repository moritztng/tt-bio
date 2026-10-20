#!/usr/bin/env python3
"""What bounds AF2's K=128 pair projections on Blackhole: the FPU, the output write, or ttnn's plan.

bc4-bet-block's mm_rate read ttnn.matmul at [N*N,128]@[128,640] at 19 TFLOP/s from DRAM, LoFi no faster than
HiFi2. This times the same products (plus the pair transition's 128->512 and 512->128) every way ttnn offers
without a new kernel, each against float64:

* mm      ttnn.matmul, DRAM in/out, ttnn's own plan (block's baseline)
* mmm     experimental.minimal_matmul, a block grid (the `pair_mm` op), HiFi2 and LoFi at the best block
* bfp8o   the best minimal_matmul block writing bfloat8_b (half the output bytes)
* bfp8io  bfloat8_b activations and weight in, bf16 out
* hs      height-sharded L1 in and out, 1D in1-mcast program config (what a fused kernel would hold)
* hso     DRAM in, height-sharded L1 out
* act     fused sigmoid epilogue on mm and on the best minimal_matmul block
* write / copy  controls: fill the output-sized tensor, and clone it (read + write), DRAM

    TT_VISIBLE_DEVICES=<card> python perf/bc4_k128/mm_bound.py --ns 224,288 --out x.jsonl
"""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="224,288")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--burst", type=int, default=20)
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
GX, GY = g.x, g.y
DRAM = ttnn.DRAM_MEMORY_CONFIG
FID = dict(hifi2=ttnn.MathFidelity.HiFi2, lofi=ttnn.MathFidelity.LoFi)
out = open(A.out, "w")


def ckc(fid="hifi2"):
    return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=FID[fid], math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(f):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(A.reps):
        t0 = time.perf_counter()
        for _ in range(A.burst): ttnn.deallocate(f())
        ttnn.synchronize_device(dev); ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
    return statistics.median(ms), max(ms) - min(ms)


def emit(r):
    r["aiclk"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{T_CARD}/tt_aiclk").read())
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n"); out.flush()


import os
T_CARD = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]


def run(base, arm, f, ref=None, flop=None, nbytes=None, act=None):
    r = dict(base, arm=arm)
    try:
        if ref is not None:
            o = f(); got = ttnn.to_torch(o).double().reshape(ref.shape); ttnn.deallocate(o)
            want = torch.sigmoid(ref) if act else ref
            r["rel_l2_vs_f64"] = round(float((got - want).norm() / want.norm()), 5)
        ms, sp = timed(f)
        r.update(ms=round(ms, 4), spread_ms=round(sp, 4))
        if flop: r["tflops"] = round(flop / ms / 1e9, 2)
        if nbytes: r["gbs"] = round(nbytes / ms / 1e6, 1)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:240]}"
    emit(r)
    return r


def divs(n, cap):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def sub(Mb, Nb):
    """Widest subblock (h*w <= 4, fp32 dest) dividing the block."""
    best = (1, 1)
    for h in (1, 2, 4):
        for w in (1, 2, 4):
            if h * w <= 4 and Mb % h == 0 and Nb % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def hs_cfg(Mt, nt, kt):
    """1D in1-mcast config over the most cores that split Mt evenly, or None."""
    for c in range(GX * GY, 0, -1):
        if Mt % c == 0:
            break
    pm = Mt // c
    sh, sw = sub(pm, nt)
    cores = ttnn.num_cores_to_corerangeset(c, ttnn.CoreCoord(GX, GY), row_wise=True)
    prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY), in0_block_w=kt, out_subblock_h=sh,
        out_subblock_w=sw, per_core_M=pm, per_core_N=nt, fuse_batch=True, fused_activation=None,
        mcast_in0=False)
    mk = lambda cols: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                        ttnn.ShardSpec(cores, [pm * 32, cols], ttnn.ShardOrientation.ROW_MAJOR))
    return c, pm, prog, mk


gen = torch.Generator().manual_seed(0)
emit(dict(grid=[GX, GY], arch=str(dev.arch()), load=open("/proc/loadavg").read().split()[:3]))
SHAPES = dict(in_proj=(128, 640), out_proj=(128, 128), trans1=(128, 512), trans2=(512, 128))
for N in map(int, A.ns.split(",")):
    M = N * N
    for name, (K, Nn) in SHAPES.items():
        base = dict(n=N, op=name, M=M, K=K, N=Nn)
        hx = torch.randn(1, 1, M, K, generator=gen).bfloat16()
        hw = (torch.randn(K, Nn, generator=gen) / K ** 0.5).bfloat16()
        ref = hx.double() @ hw.double()
        flop = 2 * M * K * Nn
        byt = 2 * (M * K + K * Nn + M * Nn)
        x = ttnn.from_torch(hx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        w = ttnn.from_torch(hw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        c2, c0 = ckc(), ckc("lofi")
        run(base, "mm", lambda: ttnn.matmul(x, w, memory_config=DRAM, compute_kernel_config=c2), ref, flop, byt)
        run(base, "mm_lofi", lambda: ttnn.matmul(x, w, memory_config=DRAM, compute_kernel_config=c0),
            ref, flop, byt)
        run(base, "mm_sigmoid", lambda: ttnn.matmul(x, w, memory_config=DRAM, compute_kernel_config=c2,
                                                    activation="sigmoid"), ref, flop, byt, act=True)
        # minimal_matmul block grid
        Mt, kt, nt = M // 32, K // 32, Nn // 32
        best = None
        for Mb in (4, 8, 16):
            for Nb in divs(nt, 8):
                for Kb in sorted({min(kt, 4), min(kt, 8)}):
                    if Mt % Mb or kt % Kb: continue
                    sh, sw = sub(Mb, Nb)
                    cfg = ttnn.MinimalMatmulConfig(M_block_size=Mb, K_block_size=Kb, N_block_size=Nb,
                                                   subblock_h=sh, subblock_w=sw,
                                                   compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY))
                    r = run(base, f"mmm{Mb}_{Kb}_{Nb}_{sh}x{sw}",
                            lambda cfg=cfg: ttnn.experimental.minimal_matmul(
                                input_tensor=x, weight_tensor=w, compute_kernel_config=c2, config=cfg),
                            ref, flop, byt)
                    if "ms" in r and (best is None or r["ms"] < best[0]):
                        best = (r["ms"], cfg, r["arm"])
        if best:
            cfg = best[1]
            mmm = lambda ck=c2, **kw: ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w,
                                                                       compute_kernel_config=ck, config=cfg, **kw)
            run(dict(base, block=best[2]), "mmm_best_lofi", lambda: mmm(c0), ref, flop, byt)
            run(dict(base, block=best[2]), "mmm_best_bfp8o", lambda: mmm(dtype=ttnn.bfloat8_b), ref, flop,
                2 * (M * K + K * Nn) + 1.0625 * M * Nn)
            run(dict(base, block=best[2]), "mmm_best_sigmoid", lambda: mmm(
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)), ref, flop, byt, act=True)
        # bfp8 activations (and weight) in, bf16 out
        x8 = ttnn.typecast(x, ttnn.bfloat8_b); w8 = ttnn.typecast(w, ttnn.bfloat8_b)
        run(base, "mm_bfp8io", lambda: ttnn.matmul(x8, w8, memory_config=DRAM, compute_kernel_config=c2,
                                                   dtype=ttnn.bfloat16), ref, flop,
            1.0625 * (M * K + K * Nn) + 2 * M * Nn)
        if best:
            run(dict(base, block=best[2]), "mmm_best_bfp8io", lambda: ttnn.experimental.minimal_matmul(
                input_tensor=x8, weight_tensor=w8, compute_kernel_config=c2, config=best[1],
                dtype=ttnn.bfloat16), ref, flop, 1.0625 * (M * K + K * Nn) + 2 * M * Nn)
        for t in (x8, w8): ttnn.deallocate(t)
        # height-sharded L1
        try:
            c, pm, prog, mk = hs_cfg(Mt, nt, kt)
            hb = dict(base, cores=c, per_core_M=pm)
            xs = ttnn.to_memory_config(x, mk(K))
            run(hb, "hs", lambda: ttnn.matmul(xs, w, program_config=prog, memory_config=mk(Nn),
                                              compute_kernel_config=c2), ref, flop, byt)
            run(hb, "hs_lofi", lambda: ttnn.matmul(xs, w, program_config=prog, memory_config=mk(Nn),
                                                   compute_kernel_config=c0), ref, flop, byt)
            run(hb, "hso", lambda: ttnn.matmul(x, w, program_config=prog, memory_config=mk(Nn),
                                               compute_kernel_config=c2), ref, flop, byt)
            run(hb, "hs_in_dram_out", lambda: ttnn.matmul(xs, w, program_config=prog, memory_config=DRAM,
                                                          compute_kernel_config=c2), ref, flop, byt)
            ttnn.deallocate(xs)
        except Exception as e:
            emit(dict(base, arm="hs", error=f"{type(e).__name__}: {str(e)[:240]}"))
        # controls: write and copy of the output-sized tensor
        o = ttnn.matmul(x, w, memory_config=DRAM, compute_kernel_config=c2)
        run(base, "write", lambda: ttnn.full(o.shape, 1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                             device=dev, memory_config=DRAM), nbytes=2 * M * Nn)
        run(base, "copy", lambda: ttnn.clone(o, memory_config=DRAM), nbytes=4 * M * Nn)
        run(base, "read_x_copy", lambda: ttnn.clone(x, memory_config=DRAM), nbytes=4 * M * K)
        for t in (o, x, w): ttnn.deallocate(t)
emit(dict(done=True, load=open("/proc/loadavg").read().split()[:3]))
