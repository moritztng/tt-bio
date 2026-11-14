"""spd-diffusion lever 10 probe: math fidelity of the fp32 diffusion linears. The census has the token DiT's
linears at the HiFi4 math roof; this measures what HiFi3 / HiFi2 cost in error against a float64 reference
and save in time, at the shapes the sampler runs (c730: 730 tokens, 5919 atoms, 5 samples).

usage: TT_VISIBLE_DEVICES=N python fid32.py OUT.jsonl
Per (shape, fidelity): device-synced time over REPS calls, max and mean |y - y64| and their ratios to HiFi4's,
run-to-run torch.equal. AICLK sampled around every timing.
"""
import json, sys, time
from pathlib import Path

import torch

OUT = Path(sys.argv[1]); REPS = 20
try:
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
except Exception:
    pass
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
LOG = open(OUT, "a")
# (name, M, rows, K, N): token DiT transition / attention projections, atom transformer linears
SHAPES = [("dit_ffn_up", 5, 730, 768, 3072), ("dit_ffn_down", 5, 730, 1536, 768), ("dit_qkv", 5, 730, 768, 768),
          ("atom_128", 5, 5919, 128, 128), ("atom_256", 5, 5919, 128, 256)]


def aiclk():
    vals = []
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            vals.append(int((p / "tt_aiclk").read_text().split()[0]))
        except Exception:
            pass
    return max(vals) if vals else None


def log(**kw):
    kw.update(t_unix=time.time())
    LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)


def up(t):
    return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)


g = torch.Generator().manual_seed(0)
for name, M, R, K, N in SHAPES:
    x_h = torch.randn(M, R, K, generator=g)
    w_h = torch.randn(K, N, generator=g) * K ** -0.5
    y64 = x_h.double() @ w_h.double()
    x, w = up(x_h), up(w_h)
    ref = None
    for fid in ("HiFi4", "HiFi3", "HiFi2"):
        ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                     fp32_dest_acc_en=True, packer_l1_acc=True)
        run = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
        a = run(); ttnn.synchronize_device(dev); ah = torch.Tensor(ttnn.to_torch(a)).double(); ttnn.deallocate(a)
        b = run(); ttnn.synchronize_device(dev); bh = torch.Tensor(ttnn.to_torch(b)).double(); ttnn.deallocate(b)
        c0 = aiclk(); t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(run())
        ttnn.synchronize_device(dev)
        us = round((time.perf_counter() - t0) / REPS * 1e6, 1)
        d = (ah - y64).abs()
        mx, mn = float(d.max()), float(d.mean())
        ref = ref or (mx, mn)
        log(shape=name, M=M, R=R, K=K, N=N, fid=fid, us=us, aiclk=[c0, aiclk()], max_abs_vs_f64=mx,
            mean_abs_vs_f64=mn, max_ratio_vs_hifi4=round(mx / ref[0], 3), mean_ratio_vs_hifi4=round(mn / ref[1], 3),
            run_to_run_equal=bool(torch.equal(ah, bh)))
log(shape="end")
