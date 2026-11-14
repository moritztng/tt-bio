"""spd-msa: the OPM contraction time at every K that is a multiple of the core grid's width, beside K - 1.

ttnn's auto config for all-DRAM-interleaved matmuls blocks K as Kt / grid_x when grid_x divides Kt and one
tile otherwise (matmul_program_config.cpp, `all_dram_interleaved`), then shrinks the output block until the
circular buffers fit L1. This maps where padding K up to a grid_x multiple pays and where the L1 shrink
eats it, at the whole-row M (23552) and the row-blocked M (11264).

usage: TT_VISIBLE_DEVICES=<chip> python zmm_kpad3.py OUT [FID=hifi3] [KT0=128] [KT1=480] [REPS=4]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
FID = sys.argv[2] if len(sys.argv) > 2 else "hifi3"
KT0, KT1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (128, 480)
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
LOG = open(OUT / "zmm_kpad3.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity={"hifi4": ttnn.MathFidelity.HiFi4, "hifi3": ttnn.MathFidelity.HiFi3}[FID],
    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
N = 736 * 32
log(ev="start", fid=FID, arch=str(dev.arch()), grid=[g.x, g.y])
torch.manual_seed(0)
a_h = (torch.randn(N, KT1 * 32) / 100).bfloat16()
b_h = torch.randn(N, KT1 * 32).bfloat16()
for kt in range(-(-KT0 // g.x) * g.x, KT1 + 1, g.x):
    for k in (kt - 1, kt):
        a = ttnn.from_torch(a_h[:, :k * 32], layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(b_h[:, :k * 32], layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for m in (N, 11264):
            am = a if m == N else a[:m, :]
            try:
                z = ttnn.matmul(am, b, transpose_b=True, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
                ts = []
                for _ in range(REPS):
                    ttnn.deallocate(z)
                    t0 = time.perf_counter()
                    z = ttnn.matmul(am, b, transpose_b=True, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(z)
                log(ev="arm", kt=k, m=m, ms_med=statistics.median(ts), spread=max(ts) - min(ts),
                    tflops=2 * m * N * k * 32 / statistics.median(ts) / 1e9)
            except Exception as e:
                log(ev="arm_fail", kt=k, m=m, err=str(e)[:300])
            if am is not a:
                ttnn.deallocate(am)
        ttnn.deallocate(a); ttnn.deallocate(b)
log(ev="end")
