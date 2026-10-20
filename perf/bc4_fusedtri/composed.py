"""Baseline for bc4-bet-fusedtri: main's composed AF2 trimul forward (tt_bio.tenstorrent.TriangleMultiplication,
AF2's biased form, both directions) per call at N=224, c=128, against a float64 reference of the same math.
Whole-op TFLOP/s uses the same 12.74 GFLOP count as phase_bound.py.
"""
import argparse, json, os, statistics, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=21)
ap.add_argument("--burst", type=int, default=10)
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio.af2 import compute_kernel_config

dev = T.get_device(trace="bc2")
CARD = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
CLK = []
threading.Thread(target=lambda: [CLK.append(int(open(f"/sys/class/tenstorrent/tenstorrent!{CARD}/tt_aiclk").read()))
                                 or time.sleep(1) for _ in iter(int, 1)], daemon=True).start()
out = open(A.out, "w")
N, C = A.n, 128
g = torch.Generator().manual_seed(0)
bf = lambda t: t.bfloat16().double()  # noqa: E731
sd = {"norm_in.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm_in.bias": 0.1 * torch.randn(C, generator=g),
      "norm_out.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm_out.bias": 0.1 * torch.randn(C, generator=g)}
for k, o in (("g_in", 2 * C), ("p_in", 2 * C), ("g_out", C), ("p_out", C)):
    sd[f"{k}.weight"] = torch.randn(o, C, generator=g) / C ** 0.5
    sd[f"{k}.bias"] = 0.1 * torch.randn(o, generator=g)
sd = {k: v.bfloat16().float() for k, v in sd.items()}
z = bf(torch.randn(1, N, N, C, generator=g))
mask = torch.ones(1, N, N)
mask[:, N - 16:, :] = 0; mask[:, :, N - 16:] = 0          # a ragged pad, as BC2's 224 grid carries
D = {k: v.double() for k, v in sd.items()}
lin = lambda x, k: x @ D[f"{k}.weight"].T + D[f"{k}.bias"]  # noqa: E731


def ref(ending):
    zn = torch.nn.functional.layer_norm(z, (C,), D["norm_in.weight"], D["norm_in.bias"], 1e-5)
    x = torch.sigmoid(lin(zn, "g_in")) * lin(zn, "p_in") * mask.double()[..., None]
    a, b = x.chunk(2, dim=-1)
    eq = "bkid,bkjd->bijd" if ending else "bikd,bjkd->bijd"
    x = torch.einsum(eq, a, b)
    xn = torch.nn.functional.layer_norm(x, (C,), D["norm_out.weight"], D["norm_out.bias"], 1e-5)
    return lin(xn, "p_out") * torch.sigmoid(lin(zn, "g_out"))


def emit(r):
    r["aiclk_now"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{CARD}/tt_aiclk").read())
    r["load"] = open("/proc/loadavg").read().split()[:3]
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n"); out.flush()


ckc = compute_kernel_config()
Z = ttnn.from_torch(z.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
MK = ttnn.from_torch(mask.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
flop = 2 * N * N * C * 640 + 2 * C * N ** 3 + 2 * N * N * C * C
for ending in (False, True):
    r = dict(arm="composed_main", ending=ending, n=N)
    try:
        m = T.TriangleMultiplication(ending, sd, ckc)
        f = lambda: m(Z, MK)  # noqa: E731
        o = f(); got = ttnn.to_torch(o).double().reshape(1, N, N, C); ttnn.deallocate(o)
        # tt-bio masks only the a half (af2_pair_masks); equal to AF2 on the real block, which is all anyone reads
        R = ref(ending)[:, :N - 16, :N - 16]; got = got[:, :N - 16, :N - 16]; d = got - R
        r.update(rel_l2_vs_f64=round(float(d.norm() / R.norm()), 5), max_abs=round(float(d.abs().max()), 5))
        ttnn.synchronize_device(dev)
        ms = []
        for _ in range(A.reps):
            t0 = time.perf_counter()
            for _ in range(A.burst):
                ttnn.deallocate(f())
            ttnn.synchronize_device(dev)
            ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
        md = statistics.median(ms)
        r.update(ms=round(md, 4), min_ms=round(min(ms), 4), max_ms=round(max(ms), 4), tflops=round(flop / md / 1e9, 2))
        try:  # traced: device time without Python dispatch, the form BC2's traced recomputes run in
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            for _ in range(A.burst):
                ttnn.deallocate(f())
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ms = []
            for _ in range(A.reps + 1):
                t0 = time.perf_counter()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False); ttnn.synchronize_device(dev)
                ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
            ttnn.release_trace(dev, tid); ms = ms[1:]
            md = statistics.median(ms)
            r.update(traced_ms=round(md, 4), traced_min_ms=round(min(ms), 4), traced_max_ms=round(max(ms), 4),
                     traced_tflops=round(flop / md / 1e9, 2))
        except Exception as e:
            r["traced_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    emit(r)
emit(dict(arm="clock", aiclk_median=statistics.median(CLK) if CLK else None, aiclk_min=min(CLK) if CLK else None,
          aiclk_n=len(CLK)))
