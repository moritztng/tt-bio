"""Boltz-2 PWA head contraction at the served shape: plan sweep.

cen1 (748 tok x 8192-row MSA, normal, WH .107 at AICLK 1000) puts `_head_contract` at 13.5 ms x 192 calls = 2.6 s of
device time per fold, ~15-20 TFLOP/s against ~65 for the OPM contraction on the same chip. pwa3/pwa4 showed fidelity,
fp32 dest, transpose_b and bfp8 weights move it by under 4 %, and pwa6 showed ttnn's per-head batch-1 plan reads the
same 4.9 ms per 320 rows. What was never tried is the batched plan itself: per_core_N is 3 tiles on an 8-wide grid, so
each core runs 1x3 subblocks with little in0 reuse. This sweeps explicit plans for one row block:

  base     ttnn.matmul(vh, W, transpose_b=True, core_grid=CORE_GRID_MAIN) (what main runs)
  planner  the same without core_grid
  wT       base on pre-transposed weights (no transpose_b), plus the transpose itself timed apart
  r_M_N_K  MatmulMultiCoreReuseProgramConfig on the batched operands, per_core_M M, per_core_N N, in0_block_w K
  h1d_K    per head: slice vh[h], 1D in1-mcast plan over the grid with the head's [T, T] weight, concat back

Every arm is graded against float64 (rel_rms). usage: TT_VISIBLE_DEVICES=<chip> python headmm_sweep.py OUT [ROWS=672]
[TOKENS=748] [REPS=5]
"""
import json, math, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 672
T_ = int(sys.argv[3]) if len(sys.argv) > 3 else 748
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
LOG = open(OUT / "sweep.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


def aiclk():
    out = {}
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            out[p.name.split("!")[1]] = int((p / "tt_aiclk").read_text().split()[0])
        except Exception:
            pass
    return out


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

torch.manual_seed(0)
dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
H, HD = 8, 32
M = ROWS * HD
TL = ttnn.TILE_LAYOUT
GX, GY = T.COMPUTE_GRID_MAIN
lin = dict(compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
log(ev="start", rows=ROWS, tokens=T_, reps=REPS, grid=[GX, GY], arch=str(dev.arch()), aiclk=aiclk())

bf = lambda t: t.to(torch.bfloat16)
up = lambda t: ttnn.from_torch(bf(t), layout=TL, device=dev, dtype=ttnn.bfloat16)
vh_h = bf(torch.randn(H, M, T_))
w_h = bf(torch.softmax(torch.randn(H, T_, T_), -1))
ref = vh_h.double() @ w_h.double().transpose(-1, -2)                  # [H, M, T]
VH, W = up(vh_h), up(w_h)
WT = up(w_h.transpose(-1, -2).contiguous())
Mt, Nt, Kt = M // 32, math.ceil(T_ / 32), math.ceil(T_ / 32)


def err(x):
    x = ttnn.to_torch(x).double()[..., :T_]; d = x - ref
    return dict(rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_abs=float(d.abs().max()))


def timed(name, fn, **meta):
    try:
        o = fn(); ttnn.synchronize_device(dev); e = err(o); ttnn.deallocate(o)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter(); o = fn(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
        ms = statistics.median(ts)
        log(ev="arm", arm=name, ms=ms, tflops=2 * H * M * T_ * T_ / ms / 1e9, err=e, aiclk=aiclk(), **meta)
    except Exception as ex:
        log(ev="arm_fail", arm=name, err=str(ex)[:300], **meta)


def subblock(pm, pn, cap=4):
    w = max(d for d in range(1, cap + 1) if pn % d == 0)
    h = max(d for d in range(1, cap // w + 1) if pm % d == 0)
    return h, w


timed("base", lambda: ttnn.matmul(VH, W, transpose_b=True, **lin))
timed("planner", lambda: ttnn.matmul(VH, W, transpose_b=True, compute_kernel_config=ckc))
timed("wT", lambda: ttnn.matmul(VH, WT, **lin))
timed("transpose_w", lambda: ttnn.transpose(W, -2, -1))

for pn in (24, 12, 8, 6, 4, 3):
    for pm in (4, 8, 16, 21, 32):
        if Nt % pn or Mt % pm:
            continue
        blocks = H * (Mt // pm) * (Nt // pn)
        if blocks < GX * GY // 2:
            continue
        for k in (2, 4, 8):
            sh, sw = subblock(pm, pn)
            # bf16 in0/in1 double-buffered, bf16 out, fp32 intermediate (packer_l1_acc)
            l1 = (2 * pm * k + 2 * k * pn) * 2048 + pm * pn * (2048 + 4096)
            if l1 > 1_200_000:
                continue
            pc = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(GX, GY), in0_block_w=k,
                                                        out_subblock_h=sh, out_subblock_w=sw, per_core_M=pm, per_core_N=pn)
            timed(f"r_{pm}_{pn}_{k}", lambda: ttnn.matmul(VH, WT, program_config=pc, compute_kernel_config=ckc),
                  blocks=blocks, l1=l1, subblock=[sh, sw])

heads = [ttnn.slice(VH, [h, 0, 0], [h + 1, M, Nt * 32]) for h in range(H)]
whs = [ttnn.slice(WT, [h, 0, 0], [h + 1, Nt * 32, Nt * 32]) for h in range(H)]
for k in (2, 4, 8):
    pm = math.ceil(Mt / (GX * GY))
    sh, sw = subblock(pm, Nt)
    try:
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(GX, GY), in0_block_w=k, out_subblock_h=sh, out_subblock_w=sw,
            per_core_M=pm, per_core_N=Nt, fuse_batch=True, fused_activation=None, mcast_in0=False)
    except Exception as ex:
        log(ev="arm_fail", arm=f"h1d_{k}", err=str(ex)[:300]); continue
    timed(f"h1d_{k}", lambda: ttnn.concat([ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc)
                                           for a, b in zip(heads, whs)], dim=0), subblock=[sh, sw])
timed("h_planner", lambda: ttnn.concat([ttnn.matmul(a, b, compute_kernel_config=ckc) for a, b in zip(heads, whs)], dim=0))
timed("slice_cost", lambda: ttnn.concat([ttnn.slice(VH, [h, 0, 0], [h + 1, M, Nt * 32]) for h in range(H)], dim=0))
log(ev="end", aiclk=aiclk())
