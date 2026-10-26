"""triatt_ln with `triatt_ln_b8g` alone (gate bfp8 via a gate-first weight) against the plain bf16 kernel.

q, k, v and the bias must come out torch.equal (same dot products, same packing); the gate is checked
against a float64 LayerNorm + projection reference, beside the bf16 gate's own error. Times both arms,
interleaved, start and ending (TR) variants, AICLK sampled during each.

    python tl_g8.py OUT [--n 512] [--cz 256]
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

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
from tt_bio import triatt_ln as TL

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
c = H * 32
torch.manual_seed(0)
x_h = torch.randn(N, N, C).bfloat16()
lw_h, lb_h = (1 + 0.1 * torch.randn(C)).bfloat16(), (0.1 * torch.randn(C)).bfloat16()
w_h = (torch.randn(C, 4 * c + 32) / C ** 0.5).bfloat16()
put = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
xd, wd = put(x_h), put(w_h)
lwd, lbd = put(lw_h), put(lb_h)
ckc = (ttnn.MathFidelity.HiFi2, False, True, False)
GRID = tuple(T.COMPUTE_GRID_MAIN)
ARMS = {"bf16": ["triatt_ln"], "g8": ["triatt_ln", "triatt_ln_b8g"]}


def run(arm, tr):
    with T.lever_scope(ARMS[arm]):
        r = TL.ln_qkvgb_heads(xd, lwd, lbd, wd, ckc, H, 32, 8, GRID, transposed=tr)
    assert r is not None, TL.REJECTS
    return r


def gate_ref(tr):
    x = x_h.double()
    x = x.transpose(0, 1) if tr else x
    y = torch.nn.functional.layer_norm(x, (C,), lw_h.double(), lb_h.double(), 1e-5) @ w_h.double()
    return y[..., 3 * c:4 * c].reshape(N, N, H, 32).permute(0, 2, 1, 3)   # [B, H, S, 32]


def rel(a, ref):
    return float(((a.double() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())


for tr in (False, True):
    (q0, k0, v0), g0, b0 = run("bf16", tr)
    s0 = TL.STATS[2]
    (q1, k1, v1), g1, b1 = run("g8", tr)
    t = lambda a: ttnn.to_torch(a).reshape(*[int(d) for d in a.shape])   # noqa: E731
    ref = gate_ref(tr)
    log(ev="check", tr=tr, served_b8=TL.STATS[2] - s0, g_dtype=str(g1.dtype), q_dtype=str(q1.dtype),
        qkv_equal=all(torch.equal(t(a), t(b)) for a, b in ((q0, q1), (k0, k1), (v0, v1))), bias_equal=torch.equal(t(b0), t(b1)),
        g_rel_bf16=rel(t(g0), ref), g_rel_g8=rel(t(g1), ref))
    for a in (q0, k0, v0, g0, b0, q1, k1, v1, g1, b1):
        ttnn.deallocate(a)

times = {}; clocks = {}
for r in range(A.reps + 1):
    for tr in (False, True):
        for arm in ARMS:
            key = f"{arm}_{'tr' if tr else 'start'}"
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter(); m0 = time.monotonic()
            for _ in range(A.calls):
                for a in (lambda o: list(o[0]) + list(o[1:]))(run(arm, tr)):
                    ttnn.deallocate(a)
            ttnn.synchronize_device(dev)
            if r:
                times.setdefault(key, []).append(1e3 * (time.perf_counter() - t0) / A.calls)
                clocks.setdefault(key, []).append(aiclk(m0, time.monotonic()))
for key in times:
    log(ev="arm", arm=key, ms_med=round(statistics.median(times[key]), 4), ms_min=round(min(times[key]), 4),
        aiclk=[c for c in clocks[key] if c])
log(ev="stats", tl=TL.STATS, rejects={str(k): v for k, v in TL.REJECTS.items()})
log(ev="done")
_SAMPLER.terminate()
