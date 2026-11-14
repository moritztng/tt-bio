"""pair_transpose_add at Protenix-v2's shape against a same-bytes add_, per reader/writer shuffle split.

The ending triangle attention's update goes back into z through pair_transpose_add (3P of DRAM traffic). The
WH census reads it at ~2.75 ms a call against ~0.69 ms of DRAM time, so something other than DRAM sets it.
This times, interleaved: add_ of two [1, N, N, C] bf16 tensors (the same 3P, no shuffle: the measured roof),
pair_transpose alone (2P with the shuffle), and transpose_add at several QR (output tiles the reader shuffles,
the writer takes the rest and the z traffic). Every QR must give the same bytes (torch.equal against QR 8).

    python tadd_qr.py OUT [--n 512] [--cz 256] [--qr 0,4,8,12,16,24]
"""
import argparse, json, statistics, subprocess, sys, time, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--qr", default="0,4,8,12,16,24")
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
from tt_bio import pair_transpose as PT, pair_transpose_add as PTA

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


N, C = A.n, A.cz
torch.manual_seed(0)
u_h = (torch.randn(N, N, C) * 0.05).bfloat16()
z_h = torch.randn(1, N, N, C).bfloat16()
put = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
ud, zt, z2 = put(u_h), put(z_h), put(z_h)
qrs = [int(x) for x in A.qr.split(",")]

ref = None
for q in qrs:
    PTA.QR = q
    zq = put(z_h)
    assert PTA.eligible(ud, zq)
    PTA.transpose_add(ud, zq)
    got = ttnn.to_torch(zq); ttnn.deallocate(zq)
    if ref is None:
        exact = z_h.double() + u_h.double().permute(1, 0, 2).unsqueeze(0)
        ref = got
        log(ev="parity", qr=q, rel=float((got.double() - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()))
    log(ev="equal", qr=q, equal=bool(torch.equal(got, ref)))


def arm_fns():
    def fused(q):
        def f():
            PTA.QR = q
            PTA.transpose_add(ud, zt)
        return f
    arms = {"add_3P": lambda: ttnn.add_(zt, z2), "transpose_2P": lambda: ttnn.deallocate(PT.pair_transpose(ud))}
    arms.update({f"tadd_qr{q}": fused(q) for q in qrs})
    return arms


arms = arm_fns()
times = {a: [] for a in arms}; clocks = {a: [] for a in arms}
for f in arms.values():
    f(); ttnn.synchronize_device(dev)
for r in range(A.reps):
    for a, f in arms.items():
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter(); m0 = time.monotonic()
        for _ in range(A.calls):
            f()
        ttnn.synchronize_device(dev)
        times[a].append(1e3 * (time.perf_counter() - t0) / A.calls); clocks[a].append(aiclk(m0, time.monotonic()))
P = N * N * C * 2
for a in arms:
    t = statistics.median(times[a]); nbytes = {"add_3P": 3, "transpose_2P": 2}.get(a, 3) * P
    log(ev="arm", arm=a, ms_med=round(t, 4), ms_min=round(min(times[a]), 4), ms_max=round(max(times[a]), 4),
        GBps=round(nbytes / t / 1e6, 1), aiclk=[c for c in clocks[a] if c])
log(ev="done")
_SAMPLER.terminate()
