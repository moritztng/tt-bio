"""spd-trikern: triangle attention's fused SDPA at Protenix-v2 shapes, the shipped kernel against kernel variants.

Arms are kernel configurations of `tt_bio.triatt_sdpa.sdpa` at one (q_chunk, k_chunk) pair, by default the pair the
shipped ladder picks (`T._tri_att_sdpa_at` is timed too, as the control that the pair is the fold's).
  ablate:X[+Y]   instrument: TT_BIO_TRIATT_ABLATE stages removed (EXP, ROWSUM, MAX, PRELOAD, PV); wrong on purpose
  lever:NAME     a candidate (rowsum_mm, mask_l1acc, exp_epi, exp_shift, ckc=LoFi/1/0), graded by rel_rms against float64 beside the shipped arm's
Timing: NCALL back-to-back calls per sample, one sync, arms interleaved round-robin with direction flipped every rep,
AICLK sampled out of process during every sample. Output: OUT/bench.jsonl.
usage: tabench.py OUT CHIP [--seq 736 ...] [--pair 256 768] [--arms ...] [--reps 10]
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("chip", type=int)
ap.add_argument("--seq", type=int, nargs="+", default=[736])
ap.add_argument("--pair", type=int, nargs=2, default=None, help="q_chunk k_chunk; default: the ladder's pick")
ap.add_argument("--arms", nargs="+", default=["ladder", "base", "ablate:ROWSUM", "ablate:MAX", "ablate:PRELOAD",
                                               "ablate:EXP", "ablate:PV", "ablate:ROWSUM+MAX", "base2"])
ap.add_argument("--reps", type=int, default=10); ap.add_argument("--ncall", type=int, default=20)
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
from tt_bio import triatt_sdpa as TS

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
ROOT = Path(__file__).resolve().parents[2]
REV = (ROOT / "REVISION").read_text().strip() if (ROOT / "REVISION").exists() else \
    os.popen(f"git -C {ROOT} rev-parse --short HEAD 2>/dev/null").read().strip()
log(ev="nodes_open", nodes=NODES, arch=str(dev.arch()), grid=list(T.COMPUTE_GRID_MAIN), git=REV, argv=sys.argv)
up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
rel = lambda o, r: float(((o.double() - r).pow(2).mean() / r.pow(2).mean()).sqrt())

ARMS, REF, SEL, OUTS = {}, {}, {}, {}
H, D = 8, 32
for S in a.seq:
    torch.manual_seed(S)
    hq, hk, hv = (torch.randn(S, H, S, D) for _ in range(3))
    hb = torch.randn(1, H, S, S) * 2
    q, k, v, b = (up(t, ttnn.bfloat16) for t in (hq, hk, hv, hb))
    sc = D ** -0.5
    SEL[S] = slice(0, S, 23)
    q16, k16, v16 = (ttnn.to_torch(t)[SEL[S]].double() for t in (q, k, v))
    REF[S] = torch.softmax((q16 @ k16.transpose(-1, -2) + ttnn.to_torch(b).double()) * sc, -1) @ v16
    T.SDPA_CHUNK_PICKS.clear()
    o = T._tri_att_sdpa_at(q, k, v, b, sc)
    picks = {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()}
    log(ev="ladder_pick", seq=S, picks=picks)
    qc, kc = a.pair or next(iter(T.SDPA_CHUNK_PICKS.values()))[:2]

    def arm(ablate=(), lever=None, q=q, k=k, v=v, b=b, sc=sc, qc=qc, kc=kc):
        def call():
            prev = TS._ABLATE, TS.ROWSUM_MM, TS.MASK_L1ACC, TS._CKC_OVERRIDE, TS.EXP_EPILOGUE, TS.EXP_SHIFT
            TS._ABLATE, TS.ROWSUM_MM, TS.MASK_L1ACC = tuple(ablate), lever == "rowsum_mm", lever == "mask_l1acc"
            TS.EXP_EPILOGUE, TS.EXP_SHIFT = lever == "exp_epi", lever == "exp_shift"
            if lever and lever.startswith("ckc="):
                TS._CKC_OVERRIDE = TS.ckc_from_env(lever[4:].replace("/", ","))
            try:
                return TS.sdpa(q, k, v, b, sc, qc, kc, q_split_cap=0, padded_mask=True)
            finally:
                TS._ABLATE, TS.ROWSUM_MM, TS.MASK_L1ACC, TS._CKC_OVERRIDE, TS.EXP_EPILOGUE, TS.EXP_SHIFT = prev
        return call
    for name in a.arms:
        if name == "ladder":
            ARMS[f"{S} ladder"] = (S, lambda q=q, k=k, v=v, b=b, sc=sc: T._tri_att_sdpa_at(q, k, v, b, sc))
        elif name.startswith("base"):
            ARMS[f"{S} {name} q{qc} k{kc}"] = (S, arm())
        elif name.startswith("ablate:"):
            ARMS[f"{S} {name} q{qc} k{kc}"] = (S, arm(ablate=name[7:].split("+")))
        elif name.startswith("lever:"):
            ARMS[f"{S} {name} q{qc} k{kc}"] = (S, arm(lever=name[6:]))

live = {}
for name, (S, call) in ARMS.items():
    try:
        o = call()
        if o is None:
            log(ev="declined", arm=name, rejects=dict(TS.REJECTS)); continue
        o = ttnn.to_torch(o).double()
        os_ = o[SEL[S]].reshape(REF[S].shape)
        base = next((n for n in OUTS if n.startswith(f"{S} base")), None)
        log(ev="check", arm=name, finite=bool(torch.isfinite(os_).all()), rel_rms_vs_f64=rel(os_, REF[S]),
            torch_equal_base=None if base is None else bool(torch.equal(o, OUTS[base])),
            max_abs_vs_base=None if base is None else float((o - OUTS[base]).abs().max()))
        if "base" in name:
            OUTS[name] = o
        live[name] = call
    except Exception as e:
        log(ev="refused", arm=name, error=str(e).splitlines()[0][:400])


def clk_window(t0, t1):
    return [int(v) for t, v in (l.split() for l in open(CLKF)) if t0 <= float(t) <= t1 and v.isdigit()]


samples = {n: [] for n in live}; clk = {n: [] for n in live}
for rep in range(a.reps):
    for arm_ in (list(live) if rep % 2 == 0 else list(reversed(live))):
        ttnn.synchronize_device(dev); t0 = time.monotonic(); p0 = time.perf_counter()
        for _ in range(a.ncall):
            live[arm_]()
        ttnn.synchronize_device(dev)
        samples[arm_].append((time.perf_counter() - p0) / a.ncall * 1e3); clk[arm_] += clk_window(t0, time.monotonic())
for arm_, ts in samples.items():
    c = sorted(clk[arm_]); med = statistics.median(ts); S = ARMS[arm_][0]
    log(ev="arm", arm=arm_, ms=med, ms_min=min(ts), ms_max=max(ts), spread_pct=(max(ts) - min(ts)) / med * 100,
        tflops=4 * S * H * S * S * D / med / 1e9, reps=a.reps, ncall=a.ncall,
        aiclk=dict(n=len(c), med=c[len(c) // 2] if c else None, min=c[0] if c else None, max=c[-1] if c else None))
log(ev="end")
