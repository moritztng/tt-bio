"""Shared pieces of the bc4-bet-fusedbuild instruments: one AF2 trimul's inputs at N=224, c=128, its float64
reference (perf/bc4_block/trimul_ref.py, AF2's own module), a 1 Hz AICLK sampler and a traced timer.

The work is AF2's biased trimul forward with its residual, z + mask * update, both directions: 12.74 GFLOP per call
(in-projection 8.22, contraction 2.88, out-projection 1.64), the count fusedtri's bound used.
"""
import json, os, statistics, threading, time

import torch

from perf.bc4_block.trimul_ref import reference

N, C = 224, 128
VALID = 208                     # BC2's 224 grid carries a ragged pad of 16 (fusedtri's baseline)
FLOP = 2 * N * N * C * 640 + 2 * C * N ** 3 + 2 * N * N * C * C


def inputs(seed=0):
    """(z, mask, w): w in af2_reference names (p_in, g_in, norm_in, norm_out, p_out, g_out), bf16-representable."""
    g = torch.Generator().manual_seed(seed)
    w = {"norm_in.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm_in.bias": 0.1 * torch.randn(C, generator=g),
         "norm_out.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm_out.bias": 0.1 * torch.randn(C, generator=g)}
    for k, o in (("g_in", 2 * C), ("p_in", 2 * C), ("g_out", C), ("p_out", C)):
        w[f"{k}.weight"] = torch.randn(o, C, generator=g) / C ** 0.5
        w[f"{k}.bias"] = 0.1 * torch.randn(o, generator=g)
    w = {k: v.bfloat16().double() for k, v in w.items()}
    z = torch.randn(N, N, C, generator=g).bfloat16().double()
    mask = torch.ones(N, N, dtype=torch.float64)
    mask[VALID:, :] = 0
    mask[:, VALID:] = 0
    return z, mask, w


def ref_out(z, mask, w, ending):
    """float64 z + update on the real block, the only part anything downstream reads."""
    return (z + reference(z, mask, w, ending)["out"])[:VALID, :VALID]


def grade(ref, got):
    got = got.double().reshape(N, N, C)[:VALID, :VALID]
    d = got - ref
    return {"rel_l2_vs_f64": round(float(d.norm() / ref.norm()), 6), "max_abs_vs_f64": round(float(d.abs().max()), 5)}


class Clock:
    """AICLK of the leased chip, sampled at 1 Hz in a thread for the whole run."""
    def __init__(self):
        self.card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
        self.samples = []
        threading.Thread(target=self._run, daemon=True).start()

    def now(self):
        return int(open(f"/sys/class/tenstorrent/tenstorrent!{self.card}/tt_aiclk").read())

    def _run(self):
        while True:
            self.samples.append(self.now())
            time.sleep(1)

    def summary(self):
        s = self.samples
        return {"aiclk_median": statistics.median(s), "aiclk_min": min(s), "aiclk_max": max(s), "aiclk_n": len(s)} \
            if s else {}


def traced(dev, f, reps=21, burst=20):
    """Per-call ms of `f` (which returns tensors to free) replayed from a trace of `burst` calls: median, min, max
    over `reps` replays after one warm replay."""
    import ttnn
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(burst):
        for t in f():
            ttnn.deallocate(t)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ms = []
    for _ in range(reps + 1):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / burst)
    ttnn.release_trace(dev, tid)
    ms = ms[1:]
    md = statistics.median(ms)
    return {"ms": round(md, 4), "min_ms": round(min(ms), 4), "max_ms": round(max(ms), 4), "n": len(ms),
            "burst": burst, "tflops": round(FLOP / md / 1e9, 2)}


class Out:
    def __init__(self, path, clock):
        self.f, self.clock = open(path, "a"), clock

    def emit(self, r):
        r.update(aiclk_now=self.clock.now(), load=open("/proc/loadavg").read().split()[:3],
                 t=time.strftime("%FT%TZ", time.gmtime()))
        print(json.dumps(r), flush=True)
        self.f.write(json.dumps(r) + "\n")
        self.f.flush()
