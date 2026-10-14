#!/usr/bin/env python3
"""ESMC's FFN half-layer, y = x + fc2(silu(fc1a(LN(x))) * fc1b(LN(x))), interleaved vs one block-sharded L1 chain.

ln_bench (qb1, p150a) found a block-sharded layer_norm 0.065 ms against the shipped interleaved 0.091, lost again
(0.108) once the tensor is moved into and out of the shard. So a sharded op only pays inside a chain whose producer
and consumer are sharded too. This times the whole FFN half both ways on one tensor:
  il   shipped form: interleaved DRAM, ttnn.linear(core_grid=main), gate as multiply with a SILU input activation
  mm   same, the three projections on minimal_matmul (esmc._ESMC_MM)
  sh   x block-sharded on gx x gy cores (gx splits d and d_ff, gy splits rows); sharded LN, 2D-mcast matmuls with
       sharded in0 and a sharded output, sharded gate and residual add; nothing leaves L1
  sh_rt  sh plus the move of x into the shard and of y back to DRAM (what one layer pays if the chain stops here)
Rows are padded to a multiple of 32 x gy (ESMC's page batch is 7 x 544 = 3808 rows; --rows 3840 by default).
Error of every arm against float64 of the same bf16 operands.

    python perf/spd_gap_esmf/ffn_chain_bench.py OUT [--d 960 --ff 2560]   # 300M; 600M: --d 1152 --ff 3072
"""
import argparse, json, math, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--rows", type=int, default=3840)
ap.add_argument("--d", type=int, default=960)
ap.add_argument("--ff", type=int, default=2560)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=16)
A = ap.parse_args()
OUT = Path(A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); s = json.dumps(kw, default=str)
    LOG.write(s + "\n"); LOG.flush(); print(s[:600], flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
g = torch.Generator().manual_seed(0)
R, D, F = A.rows, A.d, A.ff
xh = torch.randn(R, D, generator=g).to(torch.bfloat16)
lw = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
lb = (0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
w1a = (torch.randn(D, F, generator=g) / math.sqrt(D)).to(torch.bfloat16)
w1b = (torch.randn(D, F, generator=g) / math.sqrt(D)).to(torch.bfloat16)
w2 = (torch.randn(F, D, generator=g) / math.sqrt(F)).to(torch.bfloat16)
h = torch.nn.functional.layer_norm(xh.double(), (D,), lw.double(), lb.double(), eps=1e-5)
ref = xh.double() + (torch.nn.functional.silu(h @ w1a.double()) * (h @ w1b.double())) @ w2.double()

tt = lambda t, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                                          dtype=ttnn.bfloat16, memory_config=mc)
x = tt(xh.reshape(1, 1, R, D))
lwd, lbd = tt(lw.reshape(1, D)), tt(lb.reshape(1, D))
w1ad, w1bd, w2d = tt(w1a), tt(w1b), tt(w2)
ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
grid = dev.compute_with_storage_grid_size()
Mt, Dt, Ft = R // 32, D // 32, F // 32
log(ev="setup", rows=R, d=D, ff=F, arch=str(dev.arch()), grid=str(grid))
SILU = [ttnn.UnaryOpType.SILU]


def ffn_il(lin):
    hn = ttnn.layer_norm(x, weight=lwd, bias=lbd, epsilon=1e-5, compute_kernel_config=ck)
    a, b = lin(hn, w1ad), lin(hn, w1bd)
    ttnn.deallocate(hn)
    gt = ttnn.multiply(a, b, input_tensor_a_activations=SILU)
    ttnn.deallocate(a); ttnn.deallocate(b)
    o = lin(gt, w2d)
    ttnn.deallocate(gt)
    y = ttnn.add(x, o)
    ttnn.deallocate(o)
    return y


ARMS = {
    "il": lambda: ffn_il(lambda t, w: ttnn.linear(t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                                  core_grid=T.CORE_GRID_MAIN)),
    "mm": lambda: ffn_il(lambda t, w: ttnn.experimental.minimal_matmul(t, w, compute_kernel_config=ck,
                                                                      dtype=ttnn.bfloat16)),
}


def subblock(pm, pn, cap=4):
    w = max(s for s in range(1, min(cap, pn) + 1) if pn % s == 0)
    return max(s for s in range(1, cap // w + 1) if pm % s == 0), w


def plan(gx, gy):
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = lambda nt: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(
        cores, [Mt // gy * 32, nt // gx * 32], ttnn.ShardOrientation.ROW_MAJOR))
    mcd, mcf = mc(Dt), mc(Ft)
    bw = Dt // gx
    lnpc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(gx, gy),
                                                       subblock_w=max(s for s in (1, 2, 4) if bw % s == 0),
                                                       block_h=Mt // gy, block_w=bw, inplace=False)

    def mmpc(kt, nt):
        pm, pn = Mt // gy, nt // gx
        sh, sw = subblock(pm, pn)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=kt // gx, out_subblock_h=sh, out_subblock_w=sw,
            per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=True)
    pc1, pc2 = mmpc(Dt, Ft), mmpc(Ft, Dt)

    def chain(xs):
        hn = ttnn.layer_norm(xs, weight=lwd, bias=lbd, epsilon=1e-5, compute_kernel_config=ck,
                             memory_config=mcd, program_config=lnpc)
        a = ttnn.linear(hn, w1ad, program_config=pc1, memory_config=mcf, compute_kernel_config=ck, dtype=ttnn.bfloat16)
        b = ttnn.linear(hn, w1bd, program_config=pc1, memory_config=mcf, compute_kernel_config=ck, dtype=ttnn.bfloat16)
        ttnn.deallocate(hn)
        gt = ttnn.multiply(a, b, input_tensor_a_activations=SILU, memory_config=mcf)
        ttnn.deallocate(a); ttnn.deallocate(b)
        o = ttnn.linear(gt, w2d, program_config=pc2, memory_config=mcd, compute_kernel_config=ck, dtype=ttnn.bfloat16)
        ttnn.deallocate(gt)
        y = ttnn.add(xs, o, memory_config=mcd)
        ttnn.deallocate(o)
        return y
    return mcd, chain


for gx in range(grid.x, 0, -1):
    for gy in range(grid.y, 0, -1):
        if Dt % gx or Ft % gx or Mt % gy or gx * gy < 32:
            continue
        mcd, chain = plan(gx, gy)
        xs = ttnn.to_memory_config(x, mcd)
        ARMS[f"sh{gx}x{gy}"] = lambda chain=chain, xs=xs: chain(xs)

        def rt(chain=chain, mcd=mcd):
            s = ttnn.to_memory_config(x, mcd)
            y = chain(s)
            ttnn.deallocate(s)
            d = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(y)
            return d
        ARMS[f"sh{gx}x{gy}_rt"] = rt

live = {}
for name, fn in ARMS.items():
    try:
        o = fn(); ttnn.synchronize_device(dev)
        d = ttnn.to_torch(o).double().reshape(R, D)
        ttnn.deallocate(o)
        err = d - ref
        log(ev="arm_ok", arm=name, finite=bool(torch.isfinite(d).all()),
            rel_l2=float(err.norm() / ref.norm()), max_err_peak=float(err.abs().max() / ref.abs().max()))
        live[name] = fn
    except Exception as e:  # an L1 refusal or an illegal config is a result
        log(ev="arm_refused", arm=name, err=str(e).splitlines()[0][:300] + " | " + " ".join(str(e).split())[100:400])

times = {k: [] for k in live}
for rep in range(A.reps):
    for name, fn in live.items():
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(A.calls):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        times[name].append((time.perf_counter() - t0) / A.calls * 1e3)
    log(ev="rep", rep=rep, ms={k: round(v[-1], 4) for k, v in times.items()})
summ = {k: dict(median_ms=round(statistics.median(v), 4), spread_ms=round(max(v) - min(v), 4))
        for k, v in times.items()}
log(ev="summary", **summ)
(OUT / "result.json").write_text(json.dumps(summ, indent=1))
