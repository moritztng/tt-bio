"""The diffusion atom-attention module as Protenix-v2 runs it at 730 tokens (5,919 atoms, 5 samples):
window gather + attention core + output re-layout, today's fp32 path against single-gather and SDPA variants.

  fp32_loop   today: _windows_q_m, then _windows_kv_m (per-sample pad + 4 slices + concat, x5, then concat) for
              K and V, explicit fp32 attention (matmul, scale, + z, + pad bias, softmax, matmul), re-layout.
  fp32_gather the same core, K/V windows from ONE ttnn.embedding over all five samples.
  bf16_sdpa   flat q/k/v typecast to bf16, one gather each, stock SDPA q32 k128 with the step-invariant bias
              (z + pad, pre-scaled) cast once per fold, output cast back to fp32.
  bfp8_sdpa   the same with bfp8 operands.
Timing: back-to-back slope (N calls, one sync), 10 reps interleaved round-robin, AICLK sampled out of process.
usage: atomwin.py OUT CHIP
"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

OUT = Path(sys.argv[1]).resolve(); CHIP = int(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time(); s = json.dumps(kw, default=str); LOG.write(s + "\n"); LOG.flush(); print(s[:400], flush=True)

from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [CHIP])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio.protenix import AtomTransformer as AT
from tt_bio.eltwise_fusion import scale_add

dev = T.get_device(trace="diffusion")
NODE = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
               if os.path.exists(f"/proc/self/fd/{fd}") and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})[0]
CLKF = OUT / "aiclk.tsv"; CLKF.touch()
subprocess.Popen([sys.executable, "-c", f"""
import os, time
f = open({str(CLKF)!r}, "a"); parent = os.getppid()
while os.getppid() == parent:
    try: v = open("/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read().split()[0]
    except Exception: v = "-1"
    f.write(str(time.monotonic()) + chr(9) + v + chr(10)); f.flush(); time.sleep(0.25)
"""])
log(ev="nodes_open", nodes=[NODE], arch=str(dev.arch()))

M, N, H, dh, NQ, NK = 5, 5919, 4, 32, 32, 128
NP = -(-N // NQ) * NQ; nb = NP // NQ
ns = SimpleNamespace(N_HEADS=H, HEAD_DIM=dh, N_QUERIES=NQ, N_KEYS=NK, PAD_LEFT=48, _kv_widx={}, device=dev)
ns._kv_window_idx = lambda *a: AT._kv_window_idx(ns, *a)
ns._windows_kv = lambda x, N_, NP_: AT._windows_kv(ns, x, N_, NP_)
CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)

torch.manual_seed(0)
host = {o: torch.randn(M, N, H * dh) * 0.5 for o in "qkv"}
z_h = torch.randn(nb, H, NQ, NK)
pad_h = torch.where(torch.rand(M * nb, 1, NQ, NK) < 0.05, -1e9, 0.0)
up = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
q32, k32, v32 = (up(host[o], ttnn.float32) for o in "qkv")
z32, pad32 = up(z_h, ttnn.float32), up(pad_h, ttnn.float32)
# SDPA scales its mask with the scores: softmax((qk^T + m) * s) == softmax(qk^T * s + z + pad) for m = (z + pad) / s
s = dh ** -0.5
mask_h = (z_h.repeat(M, 1, 1, 1) + pad_h).clamp(min=-1e4) / s

# one gather over all M samples: sample m's padded rows start at m * Lp
Lp = 48 + NP + NK
ii = (torch.arange(M).reshape(M, 1, 1) * Lp + torch.arange(nb).reshape(1, nb, 1) * NQ
      + torch.arange(NK).reshape(1, 1, NK)).reshape(1, M * nb * NK)
IDX = up(ii.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

def kv_gather(x, out_dt):
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.pad(x, [[0, 0], [48, Lp - 48 - N], [0, 0]], 0.0)
    x = ttnn.reshape(x, (M * Lp, H * dh))
    x = ttnn.embedding(IDX, x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.permute(ttnn.reshape(x, (M * nb, NK, H, dh)), (0, 2, 1, 3))
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=out_dt)

def relayout(o):
    o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (M, NP, H * dh))
    o = ttnn.slice(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0], [M, N, H * dh])
    return ttnn.to_layout(o, ttnn.TILE_LAYOUT)

def core_explicit(Qb, Kb, Vb):
    sc = T.batched_matmul(Qb, ttnn.permute(Kb, (0, 1, 3, 2)), compute_kernel_config=CKC)
    sc = ttnn.reshape(scale_add(ttnn.reshape(sc, (M, nb * H, NQ, NK)), s, ttnn.reshape(z32, (1, nb * H, NQ, NK))),
                      (M * nb, H, NQ, NK))
    sc = ttnn.add(sc, pad32)
    return T.batched_matmul(ttnn.softmax(sc, dim=-1), Vb, compute_kernel_config=CKC)

def arm_fp32(gather):
    def call():
        Qb = AT._windows_q_m(ns, q32, M, N, NP)
        if gather:
            Kb, Vb = kv_gather(k32, ttnn.float32), kv_gather(v32, ttnn.float32)
        else:
            Kb, Vb = AT._windows_kv_m(ns, k32, M, N, NP), AT._windows_kv_m(ns, v32, M, N, NP)
        return relayout(core_explicit(Qb, Kb, Vb))
    return call

def arm_sdpa(dt, fid):
    m = up(mask_h, dt)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=tuple(T.COMPUTE_GRID_MAIN), q_chunk_size=32, k_chunk_size=128)
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False)
    def call():
        qb = ttnn.typecast(q32, ttnn.bfloat16); kb = ttnn.typecast(k32, ttnn.bfloat16); vb = ttnn.typecast(v32, ttnn.bfloat16)
        Qb = AT._windows_q_m(ns, qb, M, N, NP)
        if dt != ttnn.bfloat16:
            Qb = ttnn.typecast(Qb, dt)
        o = ttnn.transformer.scaled_dot_product_attention(Qb, kv_gather(kb, dt), kv_gather(vb, dt), attn_mask=m,
                                                          is_causal=False, scale=s, program_config=pc, compute_kernel_config=ck)
        return ttnn.typecast(relayout(o), ttnn.float32)
    return call

ARMS = {"fp32_loop": arm_fp32(False), "fp32_gather": arm_fp32(True),
        "bf16_sdpa HiFi2": arm_sdpa(ttnn.bfloat16, ttnn.MathFidelity.HiFi2),
        "bfp8_sdpa HiFi2": arm_sdpa(ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        "bfp8_sdpa LoFi": arm_sdpa(ttnn.bfloat8_b, ttnn.MathFidelity.LoFi)}

ref = None
live = {}
for name, call in ARMS.items():
    try:
        o = ttnn.to_torch(call()).float()
        if ref is None:
            ref = o
        rel = float(((o - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
        live[name] = call
        log(ev="check", arm=name, finite=bool(torch.isfinite(o).all()), rel_rms_vs_fp32_loop=rel, shape=list(o.shape))
    except Exception as e:
        log(ev="refused", arm=name, error=str(e).splitlines()[0][:300])

NCALL, REPS = 5, 10
samples = {k: [] for k in live}
clk = {k: [] for k in live}
def clk_window(t0, t1):
    return [int(v) for t, v in (l.split() for l in open(CLKF)) if t0 <= float(t) <= t1]
for rep in range(REPS):
    order = list(live) if rep % 2 == 0 else list(reversed(live))
    for k in order:
        ttnn.synchronize_device(dev); t0 = time.monotonic(); p0 = time.perf_counter()
        for _ in range(NCALL):
            live[k]()
        ttnn.synchronize_device(dev)
        samples[k].append((time.perf_counter() - p0) / NCALL * 1e3); clk[k] += clk_window(t0, time.monotonic())
base = statistics.median(samples["fp32_loop"])
for k, v in samples.items():
    c = sorted(clk[k])
    log(ev="arm", arm=k, ms=statistics.median(v), ms_min=min(v), ms_max=max(v),
        spread_pct=(max(v) - min(v)) / statistics.median(v) * 100, speedup=base / statistics.median(v), reps=REPS,
        aiclk=dict(n=len(c), med=c[len(c) // 2] if c else None, min=c[0] if c else None, max=c[-1] if c else None))
log(ev="end")
