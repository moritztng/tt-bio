"""Boltz-2's hoisted DiT conditioning linear, s_hat (1, 512, 768) @ W (768, 36864) + b, per diffusion step.
The WH census (b2f, 10-13) reads it at 2.0 ms and 0.25 of the DRAM roof, and HiFi2 did not move it, so the
question is the program config. Arms: ttnn's default, 1-D mcast_in0 over N at a few per-core widths, and 2-D
grids; weights bf16 and bfp8 (fast). One JSON line per arm: ms (median of 20 after 3 warm), rel_rms vs float64.

usage: TT_VISIBLE_DEVICES=<chip> timeout 600 python perf/spd_ditsdpa/cond_mm.py
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

dev = T.get_device()
grid = dev.compute_with_storage_grid_size()
gen = torch.Generator().manual_seed(0)
M, K, N = 512, 768, 36864
x = torch.randn(1, M, K, generator=gen).to(torch.bfloat16)
w = (torch.randn(K, N, generator=gen) / K ** 0.5).to(torch.bfloat16)
b = (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
xt = up(x, ttnn.bfloat16)


def run(wt, bt, ref, name, pc=None):
    f = lambda: ttnn.linear(xt, wt, bias=bt, compute_kernel_config=ckc, program_config=pc)
    try:
        for _ in range(3):
            ttnn.deallocate(f())
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); o = f(); ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0)
            ttnn.deallocate(o)
        o = ttnn.to_torch(f()).reshape(M, N).double()
        assert torch.isfinite(o).all()
        rel = float((o - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
        print(json.dumps({"arm": name, "ms": round(sorted(ts)[10] * 1e3, 3), "rel_rms": round(rel, 5)}), flush=True)
    except Exception as e:
        print(json.dumps({"arm": name, "err": str(e).splitlines()[0][:200]}), flush=True)


def run_chunks(wc, bc, refc, name, pc):
    f = lambda: [ttnn.linear(xt, a, bias=c, compute_kernel_config=ckc, program_config=pc) for a, c in zip(wc, bc)]
    try:
        for _ in range(3):
            [ttnn.deallocate(o) for o in f()]
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); os_ = f(); ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0)
            [ttnn.deallocate(o) for o in os_]
        rel = 0.0
        for o, r in zip(f(), refc):
            o = ttnn.to_torch(o).reshape(r.shape).double()
            assert torch.isfinite(o).all()
            rel = max(rel, float((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()))
        print(json.dumps({"arm": name, "ms": round(sorted(ts)[10] * 1e3, 3), "rel_rms": round(rel, 5)}), flush=True)
    except Exception as e:
        print(json.dumps({"arm": name, "err": str(e).splitlines()[0][:200]}), flush=True)


for wname, wdt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
    wt, bt = up(w, wdt), up(b.reshape(1, N), wdt)
    wq = ttnn.to_torch(ttnn.typecast(wt, ttnn.bfloat16)).reshape(K, N).double() if wdt != ttnn.bfloat16 else w.double()
    bq = ttnn.to_torch(ttnn.typecast(bt, ttnn.bfloat16)).reshape(-1)[:N].double() if wdt != ttnn.bfloat16 else b.double()
    ref = x.reshape(M, K).double() @ wq + bq
    run(wt, bt, ref, f"{wname}/default")
    cores = grid.x * grid.y
    # The full-N output is ~512 KB per core, so every explicit config clashes in L1 (cmm 10-13). The fold slices
    # the output per layer anyway: time the same product as C column chunks (no concat), each its own linear.
    for C in (2, 4, 8):
        nc = N // C
        wc = [up(w[:, i * nc:(i + 1) * nc], wdt) for i in range(C)]
        bc = [up(b[i * nc:(i + 1) * nc].reshape(1, nc), wdt) for i in range(C)]
        refc = [ref[:, i * nc:(i + 1) * nc] for i in range(C)]
        pn = -(-(nc // 32) // cores)
        cfgs = [("default", None)]
        for ib in (2, 4, 8):
            for sub in (1, 2, 4):
                if pn % sub == 0:
                    cfgs.append((f"1d ib{ib} pn{pn} sw{sub}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=grid, in0_block_w=ib, out_subblock_h=1, out_subblock_w=sub,
                        per_core_M=M // 32, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=True)))
        for name, pc in cfgs:
            run_chunks(wc, bc, refc, f"{wname}/C{C} {name}", pc)
