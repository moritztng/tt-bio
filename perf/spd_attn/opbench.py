"""spd-attn op bench: Protenix-v2's attention sites at the c730 shapes, today's call against this row's levers.

  ta:     triangle attention [736, 8, 736, 32] bf16 + [1, 8, 736, 736] bias through `_tri_att_sdpa_at`, with
          `_SDPA_FUSED_PADDED` off (today: stock op) and on (padded fused pair), plus every padded pair in
          `fused_pairs(padded=True)` order up to --pairs, so the ranking can be checked against the device.
  atom:   the diffusion atom-attention module (5 samples, 5,919 atoms, 4 heads), windows + core + re-layout:
          today's windowed path (fp32 = normal, bf16 = TT_BIO_LPX) against the superset window (fp32 explicit,
          bf16 explicit, bf16 fused SDPA).
Accuracy: rel_rms of each arm against a float64 torch evaluation of the same operands.
Timing: back-to-back slope (NCALL calls, one sync) x REPS, arms interleaved round-robin, AICLK sampled out of process.
usage: opbench.py OUT CHIP [ta|atom|all] [--pairs N]
"""
import argparse, json, os, statistics, subprocess, sys, time, types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("chip", type=int); ap.add_argument("which", nargs="?", default="all")
ap.add_argument("--pairs", type=int, default=6); ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); s = json.dumps(kw, default=str); LOG.write(s + "\n"); LOG.flush(); print(s[:400], flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [a.chip])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import triatt_sdpa as TS, protenix as PX
from tt_bio.protenix import AtomTransformer as AT

dev = T.get_device()
NODES = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                if os.path.exists(f"/proc/self/fd/{fd}")
                and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
CLKF = OUT / "aiclk.tsv"; CLKF.touch()
subprocess.Popen([sys.executable, "-c", f"""
import os, time
f = open({str(CLKF)!r}, "a"); parent = os.getppid()
while os.getppid() == parent:
    try: v = open("/sys/class/tenstorrent/tenstorrent!{NODES[0]}/tt_aiclk").read().split()[0]
    except Exception: v = "-1"
    f.write(str(time.monotonic()) + chr(9) + v + chr(10)); f.flush(); time.sleep(0.25)
"""])
log(ev="nodes_open", nodes=NODES, arch=str(dev.arch()), grid=list(T.COMPUTE_GRID_MAIN), git=os.popen(
    f"git -C {Path(__file__).resolve().parents[2]} rev-parse --short HEAD").read().strip())
up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
rel = lambda o, r: float(((o.double() - r).pow(2).mean() / r.pow(2).mean()).sqrt())
ARMS, REF, SEL = {}, {}, {}

# ---- triangle attention
if a.which in ("ta", "all"):
    S, H, D = 736, 8, 32
    torch.manual_seed(0)
    hq, hk, hv = (torch.randn(S, H, S, D) for _ in range(3))
    hb = torch.randn(1, H, S, S) * 2
    q, k, v, b = (up(t, ttnn.bfloat16) for t in (hq, hk, hv, hb))
    sc = D ** -0.5
    # The fused kernels add the bias before scaling: exp((qk + b - max) * scale).
    # Scored on every 23rd batch row: a float64 reference of the whole call is 4 GB of host memory.
    SEL["ta"] = slice(0, S, 23)
    q16, k16, v16 = (ttnn.to_torch(t)[SEL["ta"]].double() for t in (q, k, v))
    REF["ta"] = torch.softmax((q16 @ k16.transpose(-1, -2) + ttnn.to_torch(b).double()) * sc, -1) @ v16

    def ta_ladder(padded):
        def call():
            T._SDPA_FUSED_PADDED = padded
            return T._tri_att_sdpa_at(q, k, v, b, sc)
        return call
    ARMS["ta stock (padded off)"] = ("ta", ta_ladder(False))
    ARMS["ta ladder (padded on)"] = ("ta", ta_ladder(True))
    cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
    for qc, kc in TS.fused_pairs(S, H, D, cores, ttnn.bfloat16, padded=True)[:a.pairs]:
        ARMS[f"ta padded q{qc} k{kc}"] = ("ta", lambda qc=qc, kc=kc: TS.sdpa(
            q, k, v, b, sc, qc, kc, q_split_cap=0, padded_mask=True))
    log(ev="ta_pairs", cores=cores, pairs=TS.fused_pairs(S, H, D, cores, ttnn.bfloat16, padded=True)[:a.pairs])

# ---- atom attention module
if a.which in ("atom", "all"):
    M, N, H, dh, NQ, NK = 5, 5919, 4, 32, 32, 128
    NP = -(-N // NQ) * NQ; nb = NP // NQ
    torch.manual_seed(1)
    hx = {o: torch.randn(M, N, H * dh) * 0.5 for o in "qkv"}
    hz = torch.randn(nb, H, NQ, NK)
    key = torch.arange(nb)[:, None, None] * NQ - 48 + torch.arange(NK)[None, None, :]
    mask = ((key >= 0) & (key < N)).expand(nb, NQ, NK).float()

    def windowed_f64(q, k, v, z):
        # The 128-key window by definition: key j of block i is left-padded row i*32 + j.
        sp = lambda x: x.reshape(M, -1, H, dh).permute(0, 2, 1, 3)
        qb = sp(torch.nn.functional.pad(q, (0, 0, 0, NP - N))).reshape(M, H, nb, NQ, dh)
        kp, vp = (sp(torch.nn.functional.pad(x, (0, 0, 48, NP + NK - 48 - N))) for x in (k, v))
        idx = torch.arange(nb)[:, None] * NQ + torch.arange(NK)[None, :]
        bias = z.permute(1, 0, 2, 3) + torch.where(mask < 0.5, -1e9, 0.0).double()[None]
        o = torch.softmax(qb @ kp[:, :, idx].transpose(-1, -2) * dh ** -0.5 + bias, -1) @ vp[:, :, idx]
        return o.reshape(M, H, NP, dh).permute(0, 2, 1, 3).reshape(M, NP, H * dh)[:, :N]
    REF["atom"] = windowed_f64(*(hx[o].double() for o in "qkv"), hz.double())
    CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                           fp32_dest_acc_en=True, packer_l1_acc=False)

    def ns(dt, sdpa=False):
        t = {o: up(hx[o], dt) for o in "qkv"}
        n = types.SimpleNamespace(N_HEADS=H, HEAD_DIM=dh, N_QUERIES=NQ, N_KEYS=NK, PAD_LEFT=48, _kv_widx={},
                                  device=dev, dtype=dt, _sdpa=sdpa, compute_kernel_config=CKC,
                                  _softmax_ckc=T.softmax_ckc("protenix.atom_transformer"),
                                  _softmax_f64=T.host_f64_softmax_site("protenix.atom_transformer"))
        n._sdpa_ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                                       fp32_dest_acc_en=False, packer_l1_acc=False)
        for f in ("_kv_window_idx", "_windows_kv", "_windows_q_m", "_windows_kv_m", "_attention_m", "_superset",
                  "_superset_bias", "_attention_superset", "_make_pad_bias"):
            setattr(n, f, types.MethodType(getattr(AT, f), n))
        n._lin = lambda x, w, b_=None: t[w.split(".")[-2][-1]]   # linear_q/k/v -> the prepared operand
        return n

    def windowed(dt):
        n = ns(dt)
        z = up(hz, dt)
        pad = n._make_pad_bias(mask)
        pad = ttnn.to_layout(ttnn.concat([pad] * M, dim=0), ttnn.TILE_LAYOUT)
        x = up(torch.zeros(M, N, H * dh), dt)
        return lambda: n._attention_m(x, x, None, "attention.", N, NP, M, pad, z_pre=z)

    def superset(dt, sdpa):
        n = ns(dt, sdpa)
        zs = n._superset_bias(up(hz, dt), mask, nb)
        x = up(torch.zeros(M, N, H * dh), dt)
        return lambda: n._attention_superset(x, x, "attention.", N, NP, zs)
    ARMS["atom windowed fp32 (normal today)"] = ("atom", windowed(ttnn.float32))
    ARMS["atom superset fp32"] = ("atom", superset(ttnn.float32, False))
    ARMS["atom windowed bf16 (lpx today)"] = ("atom", windowed(ttnn.bfloat16))
    ARMS["atom superset bf16 explicit"] = ("atom", superset(ttnn.bfloat16, False))
    ARMS["atom superset bf16 sdpa"] = ("atom", superset(ttnn.bfloat16, True))

live = {}
for name, (site, call) in ARMS.items():
    try:
        o = call()
        if o is None:
            log(ev="declined", arm=name); continue
        o = ttnn.to_torch(o).double()
        o = (o[SEL[site]] if site in SEL else o).reshape(REF[site].shape)
        log(ev="check", arm=name, finite=bool(torch.isfinite(o).all()), rel_rms_vs_f64=rel(o, REF[site]),
            picks={f"{k}": v for k, v in T.SDPA_CHUNK_PICKS.items()} if site == "ta" else None)
        live[name] = call
    except Exception as e:
        log(ev="refused", arm=name, error=str(e).splitlines()[0][:300])

samples = {k: [] for k in live}; clk = {k: [] for k in live}
ncall = {k: (20 if ARMS[k][0] == "ta" else 10) for k in live}


def clk_window(t0, t1):
    return [int(v) for t, v in (l.split() for l in open(CLKF)) if t0 <= float(t) <= t1 and v.isdigit()]


for rep in range(a.reps):
    for k in (list(live) if rep % 2 == 0 else list(reversed(live))):
        ttnn.synchronize_device(dev); t0 = time.monotonic(); p0 = time.perf_counter()
        for _ in range(ncall[k]):
            live[k]()
        ttnn.synchronize_device(dev)
        samples[k].append((time.perf_counter() - p0) / ncall[k] * 1e3); clk[k] += clk_window(t0, time.monotonic())
for k, v in samples.items():
    c = sorted(clk[k]); med = statistics.median(v)
    log(ev="arm", arm=k, site=ARMS[k][0], ms=med, ms_min=min(v), ms_max=max(v),
        spread_pct=(max(v) - min(v)) / med * 100, reps=a.reps, ncall=ncall[k],
        aiclk=dict(n=len(c), med=c[len(c) // 2] if c else None, min=c[0] if c else None, max=c[-1] if c else None))
log(ev="end")
