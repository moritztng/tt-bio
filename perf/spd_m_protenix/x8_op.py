"""trimul_x8 at the op: the triangle contraction (bmm_res) writes x as bf16 or bfp8, then permute_back_pk moves it
into the [1, N, N, C] bf16 pair. Error of the moved pair vs float64, and ms per (contraction + move), interleaved,
AICLK sampled during each arm.

    python x8_op.py OUT [--n 512] [--c 128]
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=16)
A = ap.parse_args()
OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time()
    s = json.dumps(kw, default=str); LOG.write(s + "\n"); LOG.flush(); print(s[:400], flush=True)


import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import bmm_res as BR, permute_back_pk as PK

dev = T.get_device()
node = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
CLKF = OUT / "aiclk.tsv"; CLKF.touch()
_SAMPLER = subprocess.Popen([sys.executable, "-c", f"""
import os, time
f = open({str(CLKF)!r}, "a"); parent = os.getppid()
while os.getppid() == parent:
    try: v = open("/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read().split()[0]
    except Exception: v = "-1"
    f.write(f"{{time.monotonic()}}\\t{{v}}\\n"); f.flush(); time.sleep(0.25)
"""])
log(ev="start", arch=str(dev.arch()), args=vars(A))


def aiclk(t0, t1):
    v = [int(m) for ts, m in (l.split("\t") for l in CLKF.read_text().splitlines())
         if t0 - 0.3 <= float(ts) <= t1 + 0.3 and int(m) > 0]
    return (min(v), statistics.median(v), max(v)) if v else None


N, C = A.n, A.c
torch.manual_seed(0)
a = (torch.randn(1, C, N, N) * 0.5).bfloat16()
b = (torch.randn(1, C, N, N) * 0.5).bfloat16()
ref = torch.einsum("bcik,bckj->bijc", a.double(), b.double())
put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
ta, tb = put(a), put(b)
grid = T.COMPUTE_GRID_MAIN
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)


def arm(dt):
    x = BR.matmul(ta, tb, compute_kernel_config=ckc, out_dtype=dt)
    assert x is not None and x.dtype == dt, (x, dt)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, N, N, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    PK.permute_back_pk(x, out, grid)
    ttnn.deallocate(x)
    return out


ARMS = {"bf16": ttnn.bfloat16, "x8": ttnn.bfloat8_b}
for name, dt in ARMS.items():
    o = arm(dt); ttnn.synchronize_device(dev)
    got = ttnn.to_torch(o).double(); ttnn.deallocate(o)
    rel = float((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    log(ev="check", arm=name, rel_rms_f64=rel, max_abs=float((got - ref).abs().max()))

ms = {k: [] for k in ARMS}
for r in range(A.reps):
    for name, dt in ARMS.items():
        ttnn.synchronize_device(dev); t0 = time.monotonic()
        for _ in range(A.calls):
            ttnn.deallocate(arm(dt))
        ttnn.synchronize_device(dev); t1 = time.monotonic()
        ms[name].append((t1 - t0) * 1e3 / A.calls)
        log(ev="rep", arm=name, rep=r, ms_per_call=ms[name][-1], aiclk=aiclk(t0, t1))
log(ev="done", median_ms={k: statistics.median(v) for k, v in ms.items()}, served=BR.STATS)
_SAMPLER.terminate()
