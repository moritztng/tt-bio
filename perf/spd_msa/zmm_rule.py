"""spd-msa: `opm_contract_config`'s plan against ttnn's auto config for the OPM contraction, across sizes.

The rule was read off zmm_pcm.py at 512/736/1024 tokens and depth 9947. This times both at every (tokens,
depth) given, the padded depth `opm_kpad_rows` produces and the unpadded one, and logs the rule's plan, both
medians and the max |diff|. A size where the rule loses to auto is a size the rule must not take.

usage: TT_VISIBLE_DEVICES=<chip> python zmm_rule.py OUT [TOKENS=256,384,512,640,736,896,1024] [DEPTHS=9947,4097,13602]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TOKS = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "256,384,512,640,736,896,1024").split(",")]
DEPTHS = [int(d) for d in (sys.argv[3] if len(sys.argv) > 3 else "9947,4097,13602").split(",")]
REPS = 3
LOG = open(OUT / "zmm_rule.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
grid = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                             math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
log(ev="start", tokens=TOKS, depths=DEPTHS, arch=str(dev.arch()), grid=[grid.x, grid.y])


def timed(a, b, cfg):
    z = ttnn.matmul(a, b, transpose_b=True, program_config=cfg, compute_kernel_config=ckc)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        ttnn.deallocate(z)
        t0 = time.perf_counter()
        z = ttnn.matmul(a, b, transpose_b=True, program_config=cfg, compute_kernel_config=ckc)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return z, statistics.median(ts), max(ts) - min(ts)


for depth in DEPTHS:
    for tok in TOKS:
        N = tok * 32
        for k in sorted({depth, depth + T.opm_kpad_rows(depth)}):
            torch.manual_seed(0)
            a = ttnn.from_torch((torch.randn(N, k) / k ** 0.5).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
            b = ttnn.from_torch(torch.randn(N, k).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
            Kt = a.padded_shape[1] // 32
            cfg = T.opm_contract_config(N // 32, N // 32, Kt, grid)
            z0, ms0, sp0 = timed(a, b, None)
            rec = dict(tokens=tok, depth=depth, k=k, kt=Kt, auto_ms=ms0, auto_spread=sp0)
            if cfg is not None:
                ends = lambda z: torch.cat([ttnn.to_torch(z[:256, :]), ttnn.to_torch(z[N - 512:, :])]).double()
                r0 = ends(z0)
                z1, ms1, sp1 = timed(a, b, cfg)
                rec.update(pcm=cfg.per_core_M, ibw=cfg.in0_block_w, obw=cfg.out_block_w, rule_ms=ms1,
                           rule_spread=sp1, x=ms0 / ms1, max_abs=(ends(z1) - r0).abs().max().item())
                ttnn.deallocate(z1)
            ttnn.deallocate(z0); ttnn.deallocate(a); ttnn.deallocate(b)
            log(ev="size", **rec)
log(ev="end")
