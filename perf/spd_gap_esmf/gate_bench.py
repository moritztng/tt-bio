#!/usr/bin/env python3
"""The pair FFN's gate, silu(h1) * h2, on one row block of ESMFold2's trunk at 512 aa: what makes it cost.

ffn-split (qb2 p300c, 1350 MHz): the gate is ~4.4 of the FFN's 10.9 ms per call, 16 calls of
ttnn.multiply(h1, h2, input_tensor_a_activations=[SILU]) on [1, 32, 512, 1024] bf16 L1-interleaved halves,
~0.28 ms each where 96 MiB of L1 traffic is ~0.03 ms. Each arm is one way to produce the same product, timed as the
back-to-back slope of --calls calls between two synchronizes (each output freed as it is enqueued), arms round-robin per rep, error vs float64.

    python perf/spd_gap_esmf/gate_bench.py OUT [--rows 32]
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--rows", type=int, default=16, help="16 keeps h1, h2 and their sharded copies (64 MiB) in L1")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--hidden", type=int, default=1024)
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
L1 = ttnn.L1_MEMORY_CONFIG
SILU = ttnn.UnaryOpType.SILU
g = torch.Generator().manual_seed(0)
xh = torch.randn(1, A.rows, A.n, A.cz, generator=g).to(torch.bfloat16)
wah = (torch.randn(A.cz, A.hidden, generator=g) / A.cz ** 0.5).to(torch.bfloat16)
wbh = (torch.randn(A.cz, A.hidden, generator=g) / A.cz ** 0.5).to(torch.bfloat16)
ref = (torch.nn.functional.silu(xh.double() @ wah.double()) * (xh.double() @ wbh.double()))

tt = lambda t, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                                           dtype=ttnn.bfloat16, memory_config=mc)
x = tt(xh, L1)
wa, wb = tt(wah), tt(wbh)
ck = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                   packer_l1_acc=True)
l1 = dict(l1_out=True, l1_bw=T._PAIR_FFN_FC1_BW, l1_block_w=T._PAIR_FFN_FC1_BLOCK_W)
h1 = T._pair_proj_linear(x, wa, ck, ttnn.bfloat16, **l1)
h2 = T._pair_proj_linear(x, wb, ck, ttnn.bfloat16, **l1)
log(ev="setup", h1_mc=str(h1.memory_config()), shape=list(h1.shape), arch=str(dev.arch()),
    grid=str(dev.compute_with_storage_grid_size()))


def sharded(t, gx, gy):
    mt, nt = A.rows * A.n // 32, A.hidden // 32
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(cores, [mt // gy * 32, nt // gx * 32], ttnn.ShardOrientation.ROW_MAJOR))
    return ttnn.to_memory_config(t, mc), mc


s1, smc = sharded(h1, 8, 8)
s2, _ = sharded(h2, 8, 8)

def fc1a_silu():
    """fc1a on the shipped L1 config with the silu in its epilogue: the gate is then a plain FPU multiply."""
    cfg = T._pair_proj_config(x, wa, bw_cap=T._PAIR_FFN_FC1_BW, out_l1=True, block_w=T._PAIR_FFN_FC1_BLOCK_W,
                              silu=True)
    return ttnn.linear(x, wa, memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=ck, program_config=cfg)


def two_step():
    a = fc1a_silu()
    o = ttnn.multiply(a, h2, memory_config=L1)
    ttnn.deallocate(a)
    return o


# Each arm returns the product (a fresh tensor) or None; the caller frees what it returns.
ARMS = {
    "mul_silu": lambda: ttnn.multiply(h1, h2, input_tensor_a_activations=[SILU], memory_config=L1),
    "silu_in_fc1a+mul": two_step,
    "mul": lambda: ttnn.multiply(h1, h2, memory_config=L1),
    "silu": lambda: ttnn.silu(h1, memory_config=L1),
    "mul_silu_legacy": lambda: ttnn.multiply(h1, h2, input_tensor_a_activations=[SILU], memory_config=L1,
                                             use_legacy=True),
    "mul_legacy": lambda: ttnn.multiply(h1, h2, memory_config=L1, use_legacy=True),
    "mul_silu_sh8x8": lambda: ttnn.multiply(s1, s2, input_tensor_a_activations=[SILU], memory_config=smc),
    "mul_sh8x8": lambda: ttnn.multiply(s1, s2, memory_config=smc),
    "fc1a": lambda: T._pair_proj_linear(x, wa, ck, ttnn.bfloat16, **l1),
    "fc1a_cfg_silu": lambda: fc1a_silu(),
    "fc1a_lin": lambda: ttnn.linear(x, wa, compute_kernel_config=ck, memory_config=L1, dtype=ttnn.bfloat16,
                                    core_grid=T.CORE_GRID_MAIN),
    "fc1a_lin_silu": lambda: ttnn.linear(x, wa, compute_kernel_config=ck, memory_config=L1, dtype=ttnn.bfloat16,
                                         core_grid=T.CORE_GRID_MAIN, activation="silu"),
}
REFS = {"silu_in_fc1a+mul": ref, "mul_silu": ref, "mul_silu_legacy": ref, "mul_silu_sh8x8": ref}
live = {}
for name, fn in ARMS.items():
    try:
        o = fn(); ttnn.synchronize_device(dev)
        rec = dict(ev="arm_ok", arm=name, mc=str(o.memory_config().memory_layout))
        if name in REFS:
            d = ttnn.to_torch(o).double().reshape(ref.shape)
            rec.update(finite=bool(torch.isfinite(d).all()),
                       rel_rms=float(((d - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()))
        ttnn.deallocate(o); live[name] = fn; log(**rec)
    except Exception as e:  # noqa: BLE001
        log(ev="arm_refused", arm=name, err=str(e)[:300])

times = {k: [] for k in live}
for rep in range(A.reps):
    for name, fn in live.items():
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(A.calls):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev); dt = (time.perf_counter() - t0) / A.calls * 1e3
        times[name].append(dt)
    log(ev="rep", rep=rep, ms={k: round(v[-1], 4) for k, v in times.items()})
summ = {k: dict(median_ms=round(statistics.median(v), 4), spread_ms=round(max(v) - min(v), 4))
        for k, v in times.items()}
log(ev="summary", **summ)
(OUT / "result.json").write_text(json.dumps(summ, indent=1))
