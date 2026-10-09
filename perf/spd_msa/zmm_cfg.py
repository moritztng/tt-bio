"""spd-msa: the OPM contraction z = a b^T under explicit 2D-multicast program configs, against ttnn's auto config.

At a K padded to a grid-width multiple the auto config takes in0_block_w = Kt / grid_x and then shrinks the
output block until the circular buffers fit L1, so two half-M calls beat one whole-M call (2 x 101.6 vs 237.3
ms at 312 tiles, Wormhole HiFi3). This sweeps in0_block_w (divisors of Kt) against the output block (divisors
of per_core_M / per_core_N) for every plan `_matmul_cb_bytes` prices under the bank, and reports each one's
time and its max difference from the auto call.

usage: TT_VISIBLE_DEVICES=<chip> python zmm_cfg.py OUT [DEPTH=9947] [FID=hifi3] [REPS=3] [TOKENS=736] [ROWS=736,352]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 9947
FID = sys.argv[3] if len(sys.argv) > 3 else "hifi3"
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
TOK = int(sys.argv[5]) if len(sys.argv) > 5 else 736
ROWS = [int(r) for r in (sys.argv[6] if len(sys.argv) > 6 else f"{TOK},352").split(",")]
LOG = open(OUT / "zmm_cfg.jsonl", "a")


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
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity={"hifi4": ttnn.MathFidelity.HiFi4, "hifi3": ttnn.MathFidelity.HiFi3}[FID],
    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
N = TOK * 32
K = DEPTH + T.opm_kpad_rows(DEPTH, gx) if hasattr(T, "opm_kpad_rows") else -(-DEPTH // (32 * gx)) * 32 * gx
Kt, Nt = K // 32, N // 32
budget = T._matmul_cb_budget()
log(ev="start", tokens=TOK, rows=ROWS, depth=DEPTH, k=K, fid=FID, arch=str(dev.arch()), grid=[gx, gy], budget=budget)
torch.manual_seed(0)
a_h = torch.nn.functional.pad((torch.randn(N, DEPTH) / DEPTH ** 0.5), (0, K - DEPTH)).bfloat16()
b_h = torch.nn.functional.pad(torch.randn(N, DEPTH), (0, K - DEPTH)).bfloat16()
a = ttnn.from_torch(a_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
b = ttnn.from_torch(b_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
divs = lambda n: [d for d in range(1, n + 1) if n % d == 0]


def timed(am, **kw):
    z = ttnn.matmul(am, b, transpose_b=True, compute_kernel_config=ckc, **kw); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        ttnn.deallocate(z)
        t0 = time.perf_counter()
        z = ttnn.matmul(am, b, transpose_b=True, compute_kernel_config=ckc, **kw); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return z, statistics.median(ts), max(ts) - min(ts)


for m in [r * 32 for r in ROWS]:
    am = a if m == N else a[:m, :]
    Mt = m // 32
    z0, ms0, sp0 = timed(am)
    ref = ttnn.to_torch(z0[:256, :]); ttnn.deallocate(z0)
    log(ev="auto", m=m, kt=Kt, ms=ms0, spread=sp0)
    pcm, pcn = -(-Mt // gy), -(-Nt // gx)
    plans = []
    for ibw in divs(Kt):
        for obh in divs(pcm):
            for obw in divs(pcn):
                if T._matmul_cb_bytes(ibw, obh, obw, 2) <= budget:
                    plans.append((ibw, obh, obw))
    # Per in0_block_w keep the plans with the largest output blocks: the shrink is what the auto config pays.
    best = {}
    for ibw, obh, obw in plans:
        best.setdefault(ibw, []).append((obh * obw, obh, obw))
    todo = [(ibw, obh, obw) for ibw, l in best.items() for _, obh, obw in sorted(l, reverse=True)[:3]]
    log(ev="plans", m=m, pcm=pcm, pcn=pcn, n=len(plans), tried=len(todo))
    for ibw, obh, obw in todo:
        sw = max(w for w in divs(obw) if w <= 4)
        sh = max(h for h in divs(obh) if h * sw <= 4)
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
            out_block_h=obh, out_block_w=obw, per_core_M=pcm, per_core_N=pcn, transpose_mcast=False,
            fused_activation=None, fuse_batch=False)
        try:
            z, ms, sp = timed(am, program_config=cfg)
            d = (ttnn.to_torch(z[:256, :]).double() - ref.double()).abs().max().item()
            ttnn.deallocate(z)
            log(ev="cfg", m=m, ibw=ibw, obh=obh, obw=obw, sub=[sh, sw], ms=ms, spread=sp, x_auto=ms0 / ms,
                max_abs_vs_auto=d, cb=T._matmul_cb_bytes(ibw, obh, obw, 2))
        except Exception as e:
            log(ev="cfg_fail", m=m, ibw=ibw, obh=obh, obw=obw, err=str(e)[:200])
    if am is not a:
        ttnn.deallocate(am)
log(ev="end")
