"""spd-boltz2: PWA's batched head matmul, [H, rows*32, T] x [H, T, T]^T (tenstorrent.py `_heads_fused`).

Boltz-2 c730 runs it 288 times at 12.0 ms (16 TF, 0.25 of roof, census m11) on 704-row blocks, and concatenates
the eight [T, T] head weights again for every block. Arms against the same inputs: `tb` (the call as the model
makes it), `pre` (weights transposed once, no transpose_b), `pre_auto` (no core_grid), `head2d` (one 2D matmul
per head on pre-sliced operands), `cfg_*` (MatmulMultiCoreReuseProgramConfig over per_core_M / in0_block_w).
Reports median ms, TF and max |diff| against `tb`.

usage: TT_VISIBLE_DEVICES=<chip> python pwa_heads_mm.py OUT [ROWS=704] [TOKENS=736] [HEADS=8] [REPS=8]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 704
TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 736
H = int(sys.argv[4]) if len(sys.argv) > 4 else 8
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
LOG = open(OUT / "pwa_heads_mm.jsonl", "a")


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
log(ev="start", rows=ROWS, tokens=TOK, heads=H, arch=str(dev.arch()), grid=[g.x, g.y])
torch.manual_seed(0)
ft = lambda t: ttnn.from_torch(t.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
vh = ft(torch.randn(H, ROWS * 32, TOK))
ws_t = [torch.randn(1, TOK, TOK) / TOK ** 0.5 for _ in range(H)]
ws = [ft(w) for w in ws_t]
flops = 2 * H * ROWS * 32 * TOK * TOK
lin = dict(compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)


def timed(fn):
    o = fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        ttnn.deallocate(o)
        t0 = time.perf_counter(); o = fn(); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return o, statistics.median(ts), max(ts) - min(ts)


def tb():
    w = ttnn.concat(ws, dim=0)
    o = ttnn.matmul(vh, w, transpose_b=True, **lin)
    ttnn.deallocate(w)
    return o


ref, ms, sp = timed(tb)
ref_t = ttnn.to_torch(ref).double()
log(ev="arm", arm="tb", ms=ms, spread=sp, tf=flops / ms / 1e9)
w = ttnn.concat(ws, dim=0)
_, ms, sp = timed(lambda: ttnn.matmul(vh, w, transpose_b=True, **lin))
log(ev="arm", arm="tb_noconcat", ms=ms, spread=sp, tf=flops / ms / 1e9)
_, ms, sp = timed(lambda: ttnn.concat(ws, dim=0))
log(ev="arm", arm="concat_only", ms=ms, spread=sp)
wt = ttnn.transpose(w, -2, -1)


def arm(name, fn):
    try:
        o, ms, sp = timed(fn)
        d = (ttnn.to_torch(o).double() - ref_t).abs().max().item()
        ttnn.deallocate(o)
        log(ev="arm", arm=name, ms=ms, spread=sp, tf=flops / ms / 1e9, max_abs=d)
    except Exception as e:
        log(ev="fail", arm=name, err=str(e)[:300])


arm("pre", lambda: ttnn.matmul(vh, wt, **lin))
arm("pre_auto", lambda: ttnn.matmul(vh, wt, compute_kernel_config=ckc))
vhs = [ttnn.slice(vh, [h, 0, 0], [h + 1, ROWS * 32, TOK]) for h in range(H)]
wts = [ttnn.transpose(x, -2, -1) for x in ws]


def head2d():
    os_ = [ttnn.matmul(a, b, **lin) for a, b in zip(vhs, wts)]
    o = ttnn.concat(os_, dim=0)
    for x in os_:
        ttnn.deallocate(x)
    return o


arm("head2d", head2d)
arm("head2d_auto", lambda: ttnn.concat([ttnn.matmul(a, b, compute_kernel_config=ckc) for a, b in zip(vhs, wts)], dim=0))
# K padded to whole even tiles: 736 = 23 tiles is prime, so in0_block_w can only be 1 or 23. Zero K rows/cols add
# exact zeros. `kpad` takes vh already padded (the model would pad mc's tokens before the v projection, which has
# no bias); `kpad_n` pads N as well.
KP = int(sys.argv[6]) if len(sys.argv) > 6 else 768
vhp = ttnn.pad(vh, [(0, 0), (0, 0), (0, KP - TOK)], 0.0)
wtp = ttnn.pad(wt, [(0, 0), (0, KP - TOK), (0, 0)], 0.0)
arm("kpad", lambda: ttnn.matmul(vhp, wtp, **lin))
arm("kpad_auto", lambda: ttnn.matmul(vhp, wtp, compute_kernel_config=ckc))
wtpn = ttnn.pad(wt, [(0, 0), (0, KP - TOK), (0, KP - TOK)], 0.0)
_, ms, sp = timed(lambda: ttnn.matmul(vhp, wtpn, **lin))
log(ev="arm", arm="kpad_n", ms=ms, spread=sp, tf=flops / ms / 1e9)
_, ms, sp = timed(lambda: ttnn.pad(vh, [(0, 0), (0, 0), (0, KP - TOK)], 0.0))
log(ev="arm", arm="pad_vh_only", ms=ms, spread=sp)
log(ev="end")
sys.exit(0)
Mt, Nt, Kt = ROWS, -(-TOK // 32), -(-TOK // 32)
for pcn in (Nt,):
    for pcm in (2, 4, 8, 16):
        for ibw in (1, Kt):
            for sw in (1,):
                sh = max(d for d in (1, 2, 4, 8) if pcm % d == 0 and d * sw <= 4)
                cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(g.x, g.y), in0_block_w=ibw, out_subblock_h=sh,
                    out_subblock_w=sw, per_core_M=pcm, per_core_N=pcn)
                arm(f"cfg_m{pcm}_n{pcn}_k{ibw}_s{sh}x{sw}",
                    lambda: ttnn.matmul(vh, wt, compute_kernel_config=ckc, program_config=cfg))
log(ev="end")
