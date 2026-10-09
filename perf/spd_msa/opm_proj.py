"""spd-msa: the OPM's per-chunk a/b projections, [512, 736, 128] x [128, 32], 40 per OPM call (86 ms of ~470).

Each reads a 96 MB normed chunk to write one 32-wide tile column: 1.7 ms in the fold, ~0.5 ms at the DRAM roof.
Arms, each against the same input: `grid` (the call as the model makes it, core_grid=CORE_GRID_MAIN), `auto`
(no core_grid), 1D in1-broadcast configs over per_core_M, and `ab` (one [128, 64] projection for both, which
the caller would then have to split). Reports median ms, spread, GB/s and max |diff| against `grid`.

usage: TT_VISIBLE_DEVICES=<chip> python opm_proj.py OUT [ROWS=512] [TOKENS=736] [REPS=10]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 736
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
LOG = open(OUT / "opm_proj.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                             math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
log(ev="start", rows=ROWS, tokens=TOK, arch=str(dev.arch()), grid=[g.x, g.y])
torch.manual_seed(0)
ft = lambda t: ttnn.from_torch(t.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
x = ft(torch.randn(ROWS, TOK, 128))
wa, wb = torch.randn(128, 32) / 11.3, torch.randn(128, 32) / 11.3
w, wab = ft(wa), ft(torch.cat([wa, wb], 1))
nbytes = ROWS * TOK * (128 + 32) * 2


def timed(fn):
    o = fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        ttnn.deallocate(o)
        t0 = time.perf_counter(); o = fn(); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return o, statistics.median(ts), max(ts) - min(ts)


ref, ms, sp = timed(lambda: ttnn.linear(x, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN))
ref_t = ttnn.to_torch(ref)
log(ev="arm", arm="grid", ms=ms, spread=sp, gbs=nbytes / ms / 1e6)


def arm(name, fn, cols=slice(None)):
    try:
        o, ms, sp = timed(fn)
        d = (ttnn.to_torch(o)[..., cols].double() - ref_t.double()).abs().max().item()
        ttnn.deallocate(o)
        log(ev="arm", arm=name, ms=ms, spread=sp, gbs=nbytes / ms / 1e6, max_abs=d)
    except Exception as e:
        log(ev="fail", arm=name, err=str(e)[:200])


arm("auto", lambda: ttnn.linear(x, w, compute_kernel_config=ckc))
arm("ab_grid", lambda: ttnn.linear(x, wab, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN), slice(0, 32))
arm("ab_auto", lambda: ttnn.linear(x, wab, compute_kernel_config=ckc), slice(0, 32))
Mt = ROWS * TOK // 32
ncores = g.x * g.y
for pcm in sorted({-(-Mt // ncores), -(-Mt // ncores // 4) * 4, -(-Mt // ncores // 8) * 8}):
    for ibw in (1, 2, 4):
        for sh in (1, 2, 4, 8):
            if pcm % sh:
                continue
            cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(g.x, g.y), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=1,
                out_block_h=max(d for d in range(1, 33) if pcm % d == 0 and d % sh == 0), out_block_w=1, per_core_M=pcm, per_core_N=1, fuse_batch=True,
                fused_activation=None, mcast_in0=False)
            arm(f"1d_m{pcm}_k{ibw}_s{sh}", lambda: ttnn.linear(x, w, compute_kernel_config=ckc, program_config=cfg))
log(ev="end")
