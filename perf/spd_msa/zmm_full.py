"""spd-msa: wrong elements over the WHOLE OPM contraction output, per in0_block_w.

zmm_ibw.py checked ~6-9 % of the rows: in0_block_w 1-3 clean in 6 draws, 4 and up wrong in some. This
compares every element against an fp32 product of the same bf16 operands (its own error is ~1e-6
here, the defects are off by 1, 2 or 4) for ttnn's auto plan on the unpadded K (today's path) and the
opm_contract_config plan with in0_block_w forced to each of IBWS on K padded to a multiple of 96 rows,
ttnn's auto plan on that padded K (what any depth whose K tiles the grid width divides gets today),
and the plan at each width with packer_l1_acc OFF and an fp32 output (partials spill in fp32; `_nf`).
Kernel config as the model otherwise: HiFi3, fp32 dest acc, packer_l1_acc on.

usage: TT_VISIBLE_DEVICES=<chip> python zmm_full.py OUT TOKENS DEPTH [SEEDS=2] [IBWS=1,2,3,4]
(the _nf arms' ms include no typecast back to bf16)
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TOK, DEPTH = int(sys.argv[2]), int(sys.argv[3])
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
IBWS = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,3,4").split(",")]
LOG = open(OUT / "zmm_full.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

torch.set_num_threads(24)
dev = T.get_device()
grid = dev.compute_with_storage_grid_size()
ckcs = {acc: ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                                    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=acc)
        for acc in (False, True)}
N = TOK * 32
k = DEPTH + (-DEPTH % 96)
base = T.opm_contract_config(N // 32, N // 32, k // 32, grid)
assert base is not None
plan = lambda ibw: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=ibw, out_subblock_h=base.out_subblock_h,
    out_subblock_w=base.out_subblock_w, out_block_h=base.out_block_h, out_block_w=base.out_block_w,
    per_core_M=base.per_core_M, per_core_N=base.per_core_N, transpose_mcast=False, fused_activation=None,
    fuse_batch=False)
ws = [w for w in IBWS if (k // 32) % w == 0]
arms = ([("auto_unpad", DEPTH, None, True, None), ("auto_pad", k, None, True, None)]
        + [(f"ibw{w}", k, plan(w), True, None) for w in ws]
        + [(f"ibw{w}_nf", k, plan(w), False, ttnn.float32) for w in ws])
log(ev="start", tokens=TOK, depth=DEPTH, k=k, arms=[a[0] for a in arms], seeds=SEEDS, elements=N * N)
for seed in range(SEEDS):
    torch.manual_seed(2000 + seed)
    a_h = (torch.randn(N, k) / DEPTH ** 0.5).bfloat16(); a_h[:, DEPTH:] = 0
    b_h = torch.randn(N, k).bfloat16(); b_h[:, DEPTH:] = 0
    t0 = time.perf_counter()
    ref = a_h.float() @ b_h.float().T
    log(ev="ref", seed=seed, s=round(time.perf_counter() - t0, 1))
    for name, kk, pc, acc, dt in arms:
        ckc = ckcs[acc]
        a = ttnn.from_torch(a_h[:, :kk].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(b_h[:, :kk].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        try:
            z = ttnn.matmul(a, b, transpose_b=True, program_config=pc, compute_kernel_config=ckc, dtype=dt)
        except Exception as e:  # a plan the circular buffers refuse
            log(ev="arm", seed=seed, arm=name, err=str(e).splitlines()[0][:200])
            ttnn.deallocate(a); ttnn.deallocate(b); continue
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        ttnn.deallocate(ttnn.matmul(a, b, transpose_b=True, program_config=pc, compute_kernel_config=ckc, dtype=dt))
        ttnn.synchronize_device(dev); ms = (time.perf_counter() - t0) * 1e3
        ttnn.deallocate(a); ttnn.deallocate(b)
        e = (ttnn.to_torch(z).float() - ref).abs_()
        ttnn.deallocate(z)
        bad = (e > 0.25).nonzero()
        log(ev="arm", seed=seed, arm=name, ms=round(ms, 2), max_err=round(e.max().item(), 4), n_bad=len(bad),
            bad=[(i, j, round(e[i, j].item(), 3), round(ref[i, j].item(), 3)) for i, j in bad.tolist()[:10]])
        del e
log(ev="end")
