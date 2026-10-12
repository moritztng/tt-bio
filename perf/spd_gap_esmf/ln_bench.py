#!/usr/bin/env python3
"""ESMC's block layer_norm at the page shape: interleaved (shipped) vs block-sharded, and what the conversion costs.

op census (qb1 p150a, 1350 MHz): layer_norm is 14.6 ms of the 300M's 85.8 ms per batch of 7 and 19.0 of the 600M's
128.5, ~0.12 ms per [7, 544, 960] call = ~120 GB/s on a DRAM-bound op whose roof is ~4x that. Each arm normalises the
same bf16 tensor; time = back-to-back slope of --calls calls between two synchronizes, arms round-robin per rep,
error vs a float64 layer_norm. `sh*` arms time the sharded op alone; `sh*_rt` include the to/from-shard moves the
interleaved model would pay.

    python perf/spd_gap_esmf/ln_bench.py OUT [--d 960]
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--b", type=int, default=7)
ap.add_argument("--l", type=int, default=544)
ap.add_argument("--d", type=int, default=960, help="960 = 300M, 1152 = 600M")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=32)
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
xh = (torch.randn(A.b, A.l, A.d, generator=g) * 3 + 0.5).to(torch.bfloat16)
wh = (1 + 0.1 * torch.randn(A.d, generator=g)).to(torch.bfloat16)
bh = (0.1 * torch.randn(A.d, generator=g)).to(torch.bfloat16)
ref = torch.nn.functional.layer_norm(xh.double(), (A.d,), wh.double(), bh.double(), eps=1e-5)

tt = lambda t, mc=ttnn.DRAM_MEMORY_CONFIG, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
    t, layout=lay, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
x = tt(xh)
w = tt(wh.reshape(1, A.d))  # TILE, as torch_to_tt stores it
b = tt(bh.reshape(1, A.d))
CK = ttnn.types.BlackholeComputeKernelConfig if "blackhole" in str(dev.arch()).lower() else \
    ttnn.WormholeComputeKernelConfig
ck4 = CK(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
ck2 = CK(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True)
grid = dev.compute_with_storage_grid_size()
log(ev="setup", shape=[A.b, A.l, A.d], arch=str(dev.arch()), grid=str(grid))

M, N = A.b * A.l // 32, A.d // 32   # tile rows, tile cols
x2 = ttnn.reshape(x, [1, 1, A.b * A.l, A.d])


def shard_plan(gx, gy):
    """Block shard: gy core rows split the M tile rows, gx core cols split the N tile cols (both must divide)."""
    if M % gy or N % gx or gx > grid.x or gy > grid.y:
        return None
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(cores, [M // gy * 32, N // gx * 32], ttnn.ShardOrientation.ROW_MAJOR))
    bw = N // gx
    sub = max(s for s in (1, 2, 4) if bw % s == 0)
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(gx, gy), subblock_w=sub,
                                                     block_h=M // gy, block_w=bw, inplace=False)
    return mc, pc


ARMS = {
    "il_hifi4": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck4),
    "il_hifi2": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck2),
    "il_hifi4_l1out": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck4,
                                              memory_config=ttnn.L1_MEMORY_CONFIG),
}
REF_SHAPE = {}
for gx in range(1, grid.x + 1):
    for gy in range(1, grid.y + 1):
        p = shard_plan(gx, gy)
        if p is None or gx * gy < 40:
            continue
        mc, pc = p
        xs = ttnn.to_memory_config(x2, mc)
        tag = f"sh{gx}x{gy}"
        ARMS[tag] = (lambda xs=xs, mc=mc, pc=pc: ttnn.layer_norm(
            xs, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck4, memory_config=mc, program_config=pc))

        def rt(mc=mc, pc=pc):
            s = ttnn.to_memory_config(x2, mc)
            o = ttnn.layer_norm(s, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck4, memory_config=mc,
                                program_config=pc)
            ttnn.deallocate(s)
            d = ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            return d
        ARMS[tag + "_rt"] = rt

live = {}
for name, fn in ARMS.items():
    try:
        o = fn(); ttnn.synchronize_device(dev)
        d = ttnn.to_torch(o).double().reshape(ref.shape)
        rec = dict(ev="arm_ok", arm=name, mc=str(o.memory_config().memory_layout),
                   finite=bool(torch.isfinite(d).all()),
                   rel_rms=float(((d - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()),
                   max_abs=float((d - ref).abs().max()))
        ttnn.deallocate(o); live[name] = fn; log(**rec)
    except Exception as e:  # noqa: BLE001
        log(ev="arm_refused", arm=name, err=str(e)[:300])

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
