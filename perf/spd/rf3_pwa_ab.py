"""spd-m-rf3: PairWeightedAveraging at RF3's shape, fused head path vs unpadded, on one chip.

RF3's MSA module runs PWA 40 times a fold at depth 1024 x 512 tokens, c_m 64, 8 heads x 32. Head
dim 32 sends it down `_heads_fused` (`_PWA_FULL_HEADS_FUSED`, measured on Wormhole Boltz-2 only),
four general permutes of a [rows, T, 256] tensor. `_heads_unpadded` pays two, and on Blackhole
`permute_120` is a tile-transpose chain (2.9x the permute there). Wormhole, 10-27: fused 44.3 ms,
unpadded 59.5, bit-identical. The bmm / proj arms are `_PWA_BMM` and `_PWA_FUSED_PROJ` on the fused
path. Arms rotate per rep so clock drift hits all; each is graded against a float64 reference built
from the device's own token weights. AICLK of every node is sampled while the timed loops run, and
the loop runs on until the sampler has a few rows (a 32-chip Galaxy takes ~1 s a sweep).

usage: TT_VISIBLE_DEVICES=<chip> python rf3_pwa_ab.py OUT [TOKENS=512] [ROWS=1024] [REPS=7]
"""
import glob, json, os, statistics, sys, threading, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
T_ = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 7
LOG = open(OUT / "pwa_ab.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
SAMPLES = []


def _sampler():
    while True:
        row = {}
        for n in NODES:
            try:
                row[n] = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            except Exception:
                pass
        SAMPLES.append(row); time.sleep(0.25)


threading.Thread(target=_sampler, daemon=True).start()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

torch.manual_seed(0)
dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
C_M, C_Z, H, HD = 64, 128, 8, 32
bf = lambda t: t.to(torch.bfloat16)
lin_w = lambda o, i: bf(torch.randn(o, i) / i ** 0.5)
up = lambda t: ttnn.from_torch(bf(t), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
log(ev="start", tokens=T_, rows=ROWS, reps=REPS, visible=os.environ.get("TT_VISIBLE_DEVICES"), arch=str(dev.arch()))


def err(x, ref):
    x = x.double(); d = x - ref
    return dict(rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                max_abs=float(d.abs().max()), finite=bool(torch.isfinite(x).all()))


sd = {"norm_m.weight": bf(1 + 0.1 * torch.randn(C_M)), "norm_m.bias": bf(0.1 * torch.randn(C_M)),
      "norm_z.weight": bf(1 + 0.1 * torch.randn(C_Z)), "norm_z.bias": bf(0.1 * torch.randn(C_Z)),
      "proj_m.weight": lin_w(H * HD, C_M), "proj_g.weight": lin_w(H * HD, C_M),
      "proj_z.weight": lin_w(H, C_Z), "proj_o.weight": lin_w(C_M, H * HD)}
pwa = T.PairWeightedAveraging(HD, H, sd, ckc)
m = bf(torch.randn(1, ROWS, T_, C_M)); z = bf(torch.randn(1, T_, T_, C_Z))
m_tt, z_tt = up(m), up(z)
ws = pwa.head_weights(z_tt, None)
w_host = torch.cat([ttnn.to_torch(w).double().reshape(1, T_, T_) for w in ws])
x = m[0].double(); mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
mn = (x - mu) / torch.sqrt(var + 1e-5) * sd["norm_m.weight"].double() + sd["norm_m.bias"].double()
v = (mn @ sd["proj_m.weight"].double().t()).reshape(ROWS, T_, H, HD)
g = torch.sigmoid(mn @ sd["proj_g.weight"].double().t()).reshape(ROWS, T_, H, HD)
ref = (torch.einsum("hij,rjhc->rihc", w_host, v) * g).reshape(ROWS, T_, H * HD) @ sd["proj_o.weight"].double().t()
del x, mu, var, mn, v, g

ARMS = {"fused": (True, False, False), "unpadded": (False, False, False), "bmm": (True, True, False),
        "proj": (True, False, True), "bmm_proj": (True, True, True)}


def call(arm):
    T._PWA_FULL_HEADS_FUSED, T._PWA_BMM, T._PWA_FUSED_PROJ = ARMS[arm]
    t0 = time.perf_counter()
    out = pwa(m_tt, None, weights=ws)
    ttnn.synchronize_device(dev)
    return out, (time.perf_counter() - t0) * 1e3


outs = {}
for arm in ARMS:                                   # warm the program cache, keep one output to grade
    outs[arm], _ = call(arm)
    outs[arm] = ttnn.to_torch(outs[arm]).reshape(ROWS, T_, C_M)
ms = {a: [] for a in ARMS}
SAMPLES.clear()
r = 0
while r < REPS or (len(SAMPLES) < 4 and r < 60):
    names = list(ARMS)
    for arm in names[r % len(names):] + names[:r % len(names)]:
        o, t = call(arm); ms[arm].append(t); ttnn.deallocate(o)
    r += 1
clk = {n: (min(v), int(statistics.median(v)), max(v)) for n in NODES
       if (v := [s[n] for s in SAMPLES if s.get(n, 0) > 0])}
for arm in ARMS:
    log(ev="pwa", arm=arm, ms=ms[arm], ms_med=statistics.median(ms[arm]), err=err(outs[arm], ref),
        stats=list(T.PWA_FUSED_STATS) + list(T.PWA_UNPADDED_STATS))
base = statistics.median(ms["fused"])
log(ev="pwa_ab", aiclk=clk, reps=r, samples=len(SAMPLES),
    delta_ms={a: statistics.median(ms[a]) - base for a in ARMS if a != "fused"},
    bitident={a: bool(torch.equal(outs[a], outs["fused"])) for a in ARMS if a != "fused"},
    vs_fused={a: err(outs[a], outs["fused"].double()) for a in ARMS if a != "fused"})
