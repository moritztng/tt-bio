"""spd-msa: the whole-M OPM contraction z = a b^T with per_core_M rounded UP to a number with useful divisors.

ttnn's auto config sets per_core_M = ceil(Mt / grid_y). At 736 tokens that is 82 = 2 x 41, so the output block
height can only be 1, 41 or 82 and the auto plan is stuck with small blocks (237 ms at K 312 tiles on Wormhole,
where two 352-row calls with a 4 x 23 block take 2 x 84 ms). A larger per_core_M still covers M with the same
grid rows (the last row of cores gets fewer tiles), so the blocking the half-M calls get is available to the
whole call. This times, per token count: auto, then explicit 2D-multicast plans over per_core_M in
[ceil, ceil + 12] that keep ceil(Mt / per_core_M) <= grid_y, out_block_h in {2, 3, 4, 6, 8} dividing it, the
widest out_block_w dividing per_core_N that `_matmul_cb_bytes` fits, in0_block_w over divisors of Kt in 2..16.

usage: TT_VISIBLE_DEVICES=<chip> python zmm_pcm.py OUT [DEPTH=9947] [TOKENS=736,512,1024] [REPS=3]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 9947
TOKS = [int(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "736,512,1024").split(",")]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
LOG = open(OUT / "zmm_pcm.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
gx, gy = g.x, g.y
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                             math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
K = DEPTH + T.opm_kpad_rows(DEPTH)
Kt = K // 32
budget = T._matmul_cb_budget()
divs = lambda n: [d for d in range(1, n + 1) if n % d == 0]
log(ev="start", depth=DEPTH, k=K, tokens=TOKS, arch=str(dev.arch()), grid=[gx, gy], budget=budget)


def timed(a, b, **kw):
    z = ttnn.matmul(a, b, transpose_b=True, compute_kernel_config=ckc, **kw); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        ttnn.deallocate(z)
        t0 = time.perf_counter()
        z = ttnn.matmul(a, b, transpose_b=True, compute_kernel_config=ckc, **kw); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return z, statistics.median(ts), max(ts) - min(ts)


for tok in TOKS:
    N = tok * 32
    Mt, Nt = N // 32, N // 32
    torch.manual_seed(0)
    pad = lambda x: torch.nn.functional.pad(x, (0, K - DEPTH)).bfloat16()
    a = ttnn.from_torch(pad(torch.randn(N, DEPTH) / DEPTH ** 0.5), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    b = ttnn.from_torch(pad(torch.randn(N, DEPTH)), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    # First and last rows: the last row of cores is the one a padded per_core_M leaves partly empty.
    ends = lambda z: torch.cat([ttnn.to_torch(z[:256, :]), ttnn.to_torch(z[N - 512:, :])]).double()
    z0, ms0, sp0 = timed(a, b)
    ref = ends(z0); ttnn.deallocate(z0)
    log(ev="auto", tokens=tok, m=N, kt=Kt, ms=ms0, spread=sp0)
    pcn = -(-Nt // gx)
    pc0 = -(-Mt // gy)
    todo = []
    for pcm in range(pc0, pc0 + 13):
        if -(-Mt // pcm) > gy:
            continue
        for obh in (2, 3, 4, 6, 8):
            if pcm % obh:
                continue
            for ibw in [d for d in divs(Kt) if 2 <= d <= 16]:
                fit = [w for w in divs(pcn) if T._matmul_cb_bytes(ibw, obh, w, 2) <= budget]
                if fit:
                    todo.append((pcm, obh, max(fit), ibw))
    log(ev="plans", tokens=tok, pc0=pc0, pcn=pcn, n=len(todo))
    for pcm, obh, obw, ibw in todo:
        sw = max(w for w in divs(obw) if w <= 4)
        sh = max(h for h in divs(obh) if h * sw <= 4)
        if sh * sw < 4:  # a 4-tile subblock wants the other orientation
            sh = max(h for h in divs(obh) if h <= 4)
            sw = max(w for w in divs(obw) if w * sh <= 4)
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
            out_block_h=obh, out_block_w=obw, per_core_M=pcm, per_core_N=pcn, transpose_mcast=False,
            fused_activation=None, fuse_batch=False)
        try:
            z, ms, sp = timed(a, b, program_config=cfg)
            d = (ends(z) - ref).abs().max().item()
            ttnn.deallocate(z)
            log(ev="cfg", tokens=tok, pcm=pcm, obh=obh, obw=obw, ibw=ibw, sub=[sh, sw], ms=ms, spread=sp,
                x_auto=ms0 / ms, max_abs_vs_auto=d)
        except Exception as e:
            log(ev="cfg_fail", tokens=tok, pcm=pcm, obh=obh, obw=obw, ibw=ibw, err=str(e)[:200])
    ttnn.deallocate(a); ttnn.deallocate(b)
log(ev="end")
