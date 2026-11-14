"""The ending triangle attention's tail into z^T in one program, against today's two programs.

Today: gated_out_proj writes the update u [B, S, N] (P), then pair_transpose_add reads u and z and writes
z (3P). `gated_out_proj(..., resid_t=z)` keeps each 32 x 32 block of u in L1 and adds it into z's
orientation itself. Both add bf16 u to bf16 z in a 32-bit DEST, so z must come out torch.equal.
Times both, interleaved, with the AICLK sampled during each arm; g head-major at bf16 and at bfp8
(fast mode's triatt_ln_b8g).

    python tail_t.py OUT [--n 512] [--cz 256]
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("TT_BIO_TRIATT_TAIL", "1")
os.environ.setdefault("TT_BIO_TRIATT_TAIL_T", "1")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--heads", type=int, default=8)
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
from tt_bio import pair_transpose_add as PTA, triatt_qkv as TQ

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


N, C, H = A.n, A.cz, A.heads
torch.manual_seed(0)
o_h = (torch.randn(N, H, N, 32) * 0.5).bfloat16()
g_h = torch.randn(N, H, N, 32).bfloat16()
w_h = (torch.randn(H * 32, C) / (H * 32) ** 0.5).bfloat16()
z_h = torch.randn(1, N, N, C).bfloat16()
put = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
od, wd = put(o_h), put(w_h)


def two(g, z):
    u = TQ.gated_out_proj(od, g, wd, ckc)
    assert u is not None
    PTA.transpose_add(u, z)
    ttnn.deallocate(u)


def one(g, z):
    assert TQ.gated_out_proj(od, g, wd, ckc, resid_t=z) is z


arms = {}
for gname, gdt in (("g16", ttnn.bfloat16), ("g8", ttnn.bfloat8_b)):
    with T.levers(["triatt_ln_b8g"] if gdt == ttnn.bfloat8_b else []):
        gd = put(g_h, gdt)
        z1, z2 = put(z_h), put(z_h)
        two(gd, z1); one(gd, z2)
        a, b = ttnn.to_torch(z1), ttnn.to_torch(z2)
        log(ev="equal", g=gname, equal=bool(torch.equal(a, b)), max_abs=float((a.float() - b.float()).abs().max()))
        ttnn.deallocate(z1); ttnn.deallocate(z2)
    zt = put(z_h)
    arms[f"two_{gname}"] = (gdt, lambda gd=gd, zt=zt: two(gd, zt))
    arms[f"one_{gname}"] = (gdt, lambda gd=gd, zt=zt: one(gd, zt))

times = {a: [] for a in arms}; clocks = {a: [] for a in arms}
for r in range(A.reps + 1):
    for a, (gdt, f) in arms.items():
        with T.levers(["triatt_ln_b8g"] if gdt == ttnn.bfloat8_b else []):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter(); m0 = time.monotonic()
            for _ in range(A.calls):
                f()
            ttnn.synchronize_device(dev)
            if r:   # rep 0 compiles and warms
                times[a].append(1e3 * (time.perf_counter() - t0) / A.calls)
                clocks[a].append(aiclk(m0, time.monotonic()))
for a in arms:
    t = statistics.median(times[a])
    log(ev="arm", arm=a, ms_med=round(t, 4), ms_min=round(min(times[a]), 4), ms_max=round(max(times[a]), 4),
        aiclk=[c for c in clocks[a] if c])
log(ev="stats", gop=TQ.GOP_STATS, tadd=PTA.STATS)
log(ev="done")
_SAMPLER.terminate()
