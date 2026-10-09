"""Which GEMM orientation the trimul tail should be built on: one `[N*N, c_z] @ [c_z, c_z]` pass.

The tail (`trimul_tail.fused_tail`) runs two such passes plus a gate on `minimal_matmul`'s
(M, K, N, sh, sw) = (4, 8, 1, 4, 1): N_block 1, so every core owns one output tile column, the
activation is multicast along the row of cores and each activation tile feeds one output tile.
This times the bare pass under that config, under wider N blocks (each activation tile feeds 2-8
output tiles) and under `ttnn.matmul`'s 1D program (the weight multicast, every core reading its
own activation rows, all 8 output columns per core), and checks each against the production
config with torch.equal (one K block everywhere, so the contraction order should not move).

usage: TT_VISIBLE_DEVICES=<chip> python perf/spd_trimul/tail_gemm.py [--n 736] [--calls 20] [--reps 3]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=736)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--fid", action="store_true", help="only production's block, at every fidelity / acc setting")
ap.add_argument("--b1d", action="store_true",
                help="only production's block and the 1D program over a batched view [B, N*N/B, c_z]")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
gx, gy = T.COMPUTE_GRID_MAIN
g = torch.Generator().manual_seed(0)
x = torch.randn(1, A.n * A.n, A.cz, generator=g).to(torch.bfloat16)
w = (torch.randn(A.cz, A.cz, generator=g) * A.cz ** -0.5).to(torch.bfloat16)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
xd, wd = up(x), up(w)
mt, kt, nt = A.n * A.n // 32, A.cz // 32, A.cz // 32
P = A.n * A.n * A.cz * 2


def mm(M, K, N, sh, sw, ckc=ckc):
    cfg = ttnn.MinimalMatmulConfig(M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh,
                                   subblock_w=sw, compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
    return lambda: ttnn.experimental.minimal_matmul(input_tensor=xd, weight_tensor=wd, bias_tensor=None,
                                                    compute_kernel_config=ckc, dtype=ttnn.bfloat16, config=cfg)


def mm1d(sh, sw):
    cores = gx * gy
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=kt, out_subblock_h=sh,
        out_subblock_w=sw, per_core_M=-(-mt // cores), per_core_N=nt, fuse_batch=True,
        fused_activation=None, mcast_in0=False)
    return lambda: ttnn.matmul(xd, wd, program_config=pc, compute_kernel_config=ckc,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


ARMS = {"prod_4_8_1_4_1": mm(4, 8, 1, 4, 1)}
if A.fid:
    F = ttnn.MathFidelity
    for fid in ("HiFi4", "HiFi3", "HiFi2", "LoFi"):
        for acc in (True, False):
            k = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(F, fid), math_approx_mode=False,
                                                 fp32_dest_acc_en=acc, packer_l1_acc=True)
            ARMS[f"{fid}_acc{int(acc)}"] = mm(4, 8, 1, 4, 1, ckc=k)
            ARMS[f"{fid}_acc{int(acc)}_m8"] = mm(8, 8, 1, 4, 1, ckc=k)
if A.b1d:
    # The unbatched 1D plan wants the whole per-core output in one CB (15.7 MB, tg1). A batched
    # view keeps the same rows and the same single K block, and per_core_M becomes per batch.
    for B in (23, 46, 92, 184):
        mb = mt // B
        xb = ttnn.reshape(xd, (B, mb * 32, A.cz))
        for sh, sw in ((1, 8), (2, 4), (1, 4)):
            pcm = -(-mb // (gx * gy))
            if pcm % sh:
                continue
            pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=kt,
                out_subblock_h=sh, out_subblock_w=sw, per_core_M=pcm, per_core_N=nt,
                fuse_batch=False, fused_activation=None, mcast_in0=False)
            ARMS[f"mm1d_b{B}_{sh}x{sw}"] = (lambda xb=xb, pc=pc: ttnn.reshape(ttnn.matmul(
                xb, wd, program_config=pc, compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16), (1, A.n * A.n, A.cz)))
for M in (() if A.fid or A.b1d else (1, 2, 4, 8)):
    for N in (2, 4, 8):
        for sh, sw in ((1, 4), (2, 2), (4, 1), (1, 2), (2, 1), (1, 1)):
            if M % sh or N % sw or mt % M:
                continue
            ARMS[f"mm_{M}_8_{N}_{sh}_{sw}"] = mm(M, 8, N, sh, sw)
for sh, sw in (() if A.fid or A.b1d else ((1, 4), (2, 2), (4, 1))):
    ARMS[f"mm1d_{sh}x{sw}"] = mm1d(sh, sw)

R64 = (x.double() @ w.double()).reshape(1, A.n * A.n, A.cz)
ref = None
for name, f in ARMS.items():
    rec = {"arm": name}
    try:
        y = f(); ttnn.synchronize_device(dev)
        yt = ttnn.to_torch(y)
        ttnn.deallocate(y)
        if ref is None:
            ref = yt
        rec["equal"] = bool(torch.equal(yt, ref))
        rec["rel_rms_f64"] = ((yt.double() - R64).pow(2).mean().sqrt() / R64.pow(2).mean().sqrt()).item()
        rec["max_abs"] = (yt.float() - ref.float()).abs().max().item()
        ms = []
        for _ in range(A.reps):
            t0 = time.perf_counter()
            outs = [f() for _ in range(A.calls)]
            ttnn.synchronize_device(dev)
            ms.append((time.perf_counter() - t0) * 1e3 / A.calls)
            for o in outs:
                ttnn.deallocate(o)
        rec["ms"] = min(ms)
        rec["spread_ms"] = max(ms) - min(ms)
        rec["GBps"] = 2 * P / rec["ms"] / 1e6
    except Exception as e:                                                    # noqa: BLE001
        rec["err"] = str(e).splitlines()[0][:200]
    print(json.dumps(rec), flush=True)
