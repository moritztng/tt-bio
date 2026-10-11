"""spd-ditsdpa op bench: Protenix-v2 fast mode's token DiT SDPA at c730, today's stock call against the bias-once kernel.

Shape as the fold runs it: q/k/v [5, 16, 736, 64] bf16 (head dim 48 padded to 64 with zero lanes, 730 tokens
padded to 736 by `_sdpa_masked`, the 6 pad keys masked), bias [1, 16, 736, 736] bf16, scale 48**-0.5.
Arms: the shipped stock call, stock q/k chunk variants, `token_dit_sdpa_pm` (the lever) and every pair
`fused_pairs` offers plus fixed ones on `triatt_sdpa.sdpa`, with and without the mask preload.
Accuracy: rel_rms of each arm against float64 on samples 0 and 4. Timing: back-to-back slope (NCALL calls, one
sync) x REPS, arms interleaved, AICLK sampled out of process during the timed calls.
usage: opbench.py OUT CHIP [--reps N] [--seq 736] [--batch 5]
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("chip", type=int)
ap.add_argument("--reps", type=int, default=10); ap.add_argument("--seq", type=int, default=730)
ap.add_argument("--batch", type=int, default=5, help="DiT samples (Boltz-2 page cell: 1)")
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
REV = os.popen(f"git -C {ROOT} rev-parse --short HEAD 2>/dev/null").read().strip() or (
    (ROOT / "REVISION").read_text().strip() if (ROOT / "REVISION").exists() else "unknown")
log(ev="nodes_open", nodes=NODES, arch=str(dev.arch()), grid=list(T.COMPUTE_GRID_MAIN), git=REV)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
rel = lambda o, r: float(((o.double() - r).pow(2).mean() / r.pow(2).mean()).sqrt())

M, H, DH, DP, N = a.batch, 16, 48, 64, a.seq
S = -(-N // 32) * 32
sc = DH ** -0.5
torch.manual_seed(N)
hq, hk, hv = (torch.nn.functional.pad(torch.randn(M, H, N, DH), (0, DP - DH)) for _ in range(3))
hb = torch.randn(1, H, N, N) * 2
# Exactly what `_sdpa_masked` hands the kernel: both axes padded to the tile, the new keys masked.
q, k, v, b, _ = T._sdpa_pad_ragged(up(hq), up(hk), up(hv), up(hb))
log(ev="shapes", q=list(q.shape), b=list(b.shape))
SEL = [0, M - 1]
qd, kd, vd = (ttnn.to_torch(t)[SEL, :, :N].double() for t in (q, k, v))
bd = ttnn.to_torch(b)[:, :, :N, :N].double()
# The fused kernels add the bias before scaling: exp((qk + b - max) * scale).
REF = torch.softmax((qd @ kd[:, :, :N].transpose(-1, -2) + bd) * sc, -1) @ vd[:, :, :N]

ARMS = {}


def stock(qc, kc):
    pc = T._sdpa_program_config(q_chunk_size=qc, k_chunk_size=kc)
    return lambda: T.fused_sdpa(q, k, v, attn_mask=b, scale=sc, program_config=pc)


def preload(fn, on):
    def call():
        prev = TS.QK_MASK_PRELOAD; TS.QK_MASK_PRELOAD = on
        try:
            return fn()
        finally:
            TS.QK_MASK_PRELOAD = prev
    return call


ARMS["stock shipped"] = lambda: T.fused_sdpa(q, k, v, attn_mask=b, scale=sc, program_config=T._sdpa_program_config_for_lengths(
    q.shape[2], k.shape[2], q.shape[0] * q.shape[1], site="token_dit", d=q.shape[3]))
for qc, kc in ((128, 256), (64, 256), (256, 128), (128, 128), (64, 128), (256, 64), (32, 256), (128, 512), (64, 512)):
    ARMS[f"stock q{qc} k{kc}"] = stock(qc, kc)
for qc, kc in ((128, 256), (64, 256), (128, 512)):
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=T.COMPUTE_GRID_MAIN, exp_approx_mode=True,
                                q_chunk_size=qc, k_chunk_size=kc)
    ARMS[f"stock q{qc} k{kc} exp_approx"] = lambda pc=pc: T.fused_sdpa(q, k, v, attn_mask=b, scale=sc, program_config=pc)
ARMS["pm lever"] = lambda: T.token_dit_sdpa_pm(q, k, v, b, sc)
cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
pairs = [(qc, kc, pad) for pad in (False, True) for qc, kc in TS.fused_pairs(S, H, DP, cores, ttnn.bfloat16, padded=pad, batch=M)]
log(ev="pairs", cores=cores, pairs=pairs)
extra = [(qc, kc, True) for qc in (96, 128, 160, 192, 256) for kc in (256, 384, 768)]
for qc, kc, pad in pairs + [x for x in extra if x[:2] not in {p[:2] for p in pairs}]:
    fn = lambda qc=qc, kc=kc, pad=pad: TS.sdpa(q, k, v, b, sc, qc, kc, q_split_cap=0, padded_mask=pad)
    ARMS[f"pm q{qc} k{kc}"] = preload(fn, True)
    ARMS[f"pm q{qc} k{kc} no preload"] = preload(fn, False)
FLOPS = 4 * M * H * S * S * DP

live, outs = {}, {}
for name, call in ARMS.items():
    try:
        o = call()
        if o is None:
            log(ev="declined", arm=name, rejects={str(kk): vv for kk, vv in TS.REJECTS.items()}); continue
        o = ttnn.to_torch(o)[SEL, :, :N].double()
        log(ev="check", arm=name, finite=bool(torch.isfinite(o).all()), rel_rms_vs_f64=rel(o, REF),
            equal_shipped=bool(torch.equal(o, outs["stock shipped"])) if "stock shipped" in outs else None,
            pick=T._DIT_PM_PICK.get((M, S, H, DP, ttnn.bfloat16)) if name == "pm lever" else None)
        outs[name] = o
        live[name] = call
    except Exception as e:
        log(ev="refused", arm=name, error=str(e).splitlines()[0][:300])

samples = {n: [] for n in live}; clk = {n: [] for n in live}
NCALL = 20


def clk_window(t0, t1):
    return [int(v) for t, v in (l.split() for l in open(CLKF)) if t0 <= float(t) <= t1 and v.isdigit()]


for rep in range(a.reps):
    for arm in (list(live) if rep % 2 == 0 else list(reversed(live))):
        ttnn.synchronize_device(dev); t0 = time.monotonic(); p0 = time.perf_counter()
        os_ = [live[arm]() for _ in range(NCALL)]
        ttnn.synchronize_device(dev)
        samples[arm].append((time.perf_counter() - p0) / NCALL * 1e3); clk[arm] += clk_window(t0, time.monotonic())
        for o in os_:
            ttnn.deallocate(o)
for arm, ts in samples.items():
    c = sorted(clk[arm]); med = statistics.median(ts)
    log(ev="arm", arm=arm, ms=med, ms_min=min(ts), ms_max=max(ts), tflops=FLOPS / med / 1e9,
        spread_pct=(max(ts) - min(ts)) / med * 100, reps=a.reps, ncall=NCALL,
        aiclk=dict(n=len(c), med=c[len(c) // 2] if c else None, min=c[0] if c else None, max=c[-1] if c else None))
log(ev="end")
run = [int(l.split()[1]) for l in open(CLKF) if l.split()[1:] and l.split()[1].isdigit()]
log(ev="aiclk_run", n=len(run), hist={str(v): run.count(v) for v in sorted(set(run))})
