"""spd-msa: does the OPM contraction's wrong-element rate follow in0_block_w (K tiles DST sums per pack)?

zmm_sweep.py: the unpadded auto path (in0_block_w 1 at a prime K tile count) put no wrong element at 48
shapes; padded auto (in0_block_w = Kt / grid_x, 17-54) at 5 of 32; the OPM_CFG plan (in0_block_w <= 8) at
2 of 48. Every wrong element is off by exactly 1, 2 or 4 on outputs of order 1. This holds the plan fixed
(opm_contract_config's 2D multicast at TOKENS, K padded to a multiple of grid_x) and varies only in0_block_w,
plus the fidelity at one width, over SEEDS data draws, the model's kernel config otherwise (fp32 dest acc,
packer_l1_acc on). Rows checked against float64: the first ROWS and the first 64 of every per_core_M band.

usage: TT_VISIBLE_DEVICES=<chip> python zmm_ibw.py OUT [TOKENS=736] [DEPTH=9947] [SEEDS=3] [ROWS=2048]
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 736
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 9947
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NROWS = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
LOG = open(OUT / "zmm_ibw.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

torch.set_num_threads(16)
dev = T.get_device()
grid = dev.compute_with_storage_grid_size()
F = ttnn.MathFidelity
ckc = lambda fid: ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                         fp32_dest_acc_en=True, packer_l1_acc=True)
N = TOK * 32
k = DEPTH + T.opm_kpad_rows(DEPTH, grid.x)
kt = k // 32
base = T.opm_contract_config(N // 32, N // 32, kt, grid)
assert base is not None, "opm_contract_config refused this shape"
plan = lambda ibw: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=ibw, out_subblock_h=base.out_subblock_h,
    out_subblock_w=base.out_subblock_w, out_block_h=base.out_block_h, out_block_w=base.out_block_w,
    per_core_M=base.per_core_M, per_core_N=base.per_core_N, transpose_mcast=False, fused_activation=None,
    fuse_batch=False)
ibws = [d for d in (1, 2, 3, 4, 6, 8, 12, 13, 24, 26, 39) if kt % d == 0]
arms = [(f"ibw{d}", plan(d), F.HiFi3) for d in ibws]
arms += [("ibw8_hifi2", plan(8), F.HiFi2), ("ibw8_hifi4", plan(8), F.HiFi4), ("auto_pad", None, F.HiFi3)]
log(ev="start", tokens=TOK, depth=DEPTH, k=k, base_ibw=base.in0_block_w, pcm=base.per_core_M,
    obw=base.out_block_w, arms=[a[0] for a in arms], seeds=SEEDS, rows=NROWS)
pcm = base.per_core_M * 32
rows = torch.tensor(sorted(set(range(min(NROWS, N))) | {r for s in range(0, N, pcm) for r in range(s, min(s + 64, N))}))
for seed in range(SEEDS):
    torch.manual_seed(1000 + seed)
    a_h = (torch.randn(N, k) / DEPTH ** 0.5).bfloat16(); a_h[:, DEPTH:] = 0
    b_h = torch.randn(N, k).bfloat16(); b_h[:, DEPTH:] = 0
    ref = a_h[rows].double() @ b_h.double().T
    a = ttnn.from_torch(a_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(b_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    for name, pc, fid in arms:
        try:
            z = ttnn.matmul(a, b, transpose_b=True, program_config=pc, compute_kernel_config=ckc(fid))
        except Exception as e:  # a width the circular buffers refuse
            log(ev="arm", seed=seed, arm=name, err=str(e).splitlines()[0][:200]); continue
        zt = ttnn.to_torch(z)[rows].double()
        ttnn.deallocate(z)
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        ttnn.deallocate(ttnn.matmul(a, b, transpose_b=True, program_config=pc, compute_kernel_config=ckc(fid)))
        ttnn.synchronize_device(dev); ms = (time.perf_counter() - t0) * 1e3
        e = (zt - ref).abs()
        bad = (e > 0.25).nonzero()
        log(ev="arm", seed=seed, arm=name, ms=round(ms, 2), max_err=round(e.max().item(), 4),
            rms=float(e.pow(2).mean().sqrt()), n_bad=len(bad),
            bad=[(rows[i].item(), j, round(e[i, j].item(), 3), round(ref[i, j].item(), 3)) for i, j in bad.tolist()[:8]])
    ttnn.deallocate(a); ttnn.deallocate(b)
log(ev="end")
