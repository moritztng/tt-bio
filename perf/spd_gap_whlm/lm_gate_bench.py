#!/usr/bin/env python3
"""ESMC LM SwiGLU gate, silu(h1) * h2, at the page shapes on Wormhole: what the shipped multiply costs against its DRAM
floor and the cheap alternatives.

The LM gate is ttnn.multiply(h1, h2, input_tensor_a_activations=[SILU]) on DRAM-interleaved [B, 544, hidden] bf16
halves (esmc.SwiGLUFFN, not row-blocked). The 300M census reads ~13 ms of a 121 ms batch for it, ~0.45 ms a call where
three tensor passes at 248 GB/s are ~0.23 ms. Arms are timed as the back-to-back slope of --calls calls between two
synchronizes (each output freed as it is enqueued), round-robin per rep, error vs float64.

    python perf/spd_gap_whlm/lm_gate_bench.py --out OUT.jsonl [--shapes 7:544:2560,7:544:3072,1:544:6912]
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--shapes", default="7:544:2560,7:544:3072,1:544:6912", help="batch:rows:hidden (300M, 600M, 6B)")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=20)
A = ap.parse_args()
LOG = open(A.out, "a")


def log(**kw):
    kw["t_unix"] = time.time(); s = json.dumps(kw, default=str)
    LOG.write(s + "\n"); LOG.flush(); print(s[:600], flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
SILU = ttnn.UnaryOpType.SILU
L1 = ttnn.L1_MEMORY_CONFIG
log(ev="setup", arch=str(dev.arch()), grid=str(dev.compute_with_storage_grid_size()))

for spec in A.shapes.split(","):
    b, n, hid = (int(v) for v in spec.split(":"))
    g = torch.Generator().manual_seed(0)
    # fc1 outputs of a layer-normed input: unit-scale halves
    a_h = torch.randn(b, n, hid, generator=g).to(torch.bfloat16)
    b_h = torch.randn(b, n, hid, generator=g).to(torch.bfloat16)
    ref = torch.nn.functional.silu(a_h.double()) * b_h.double()
    tt = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    h1, h2 = tt(a_h), tt(b_h)
    s1 = ttnn.silu(h1)

    arms = {
        "mul_silu": lambda: ttnn.multiply(h1, h2, input_tensor_a_activations=[SILU]),          # shipped
        "mul_silu_legacy": lambda: ttnn.multiply(h1, h2, input_tensor_a_activations=[SILU], use_legacy=True),
        "mul_silu_bfp8": lambda: ttnn.multiply(h1, h2, input_tensor_a_activations=[SILU], dtype=ttnn.bfloat8_b),
        "mul_silu_l1": lambda: ttnn.multiply(h1, h2, input_tensor_a_activations=[SILU], memory_config=L1),
        "silu_then_mul": lambda: ttnn.multiply(s1, h2),          # the gate if the silu came free from fc1
        "mul_legacy_presilu": lambda: ttnn.multiply(s1, h2, use_legacy=True),
        "silu": lambda: ttnn.silu(h1),
        "copy": lambda: ttnn.clone(h1),                           # one read + one write: the bandwidth yardstick
    }
    live = {}
    for name, fn in arms.items():
        try:
            o = fn(); ttnn.synchronize_device(dev)
            rec = dict(ev="arm_ok", shape=spec, arm=name)
            if name not in ("silu", "copy"):
                d = ttnn.to_torch(o).double().reshape(ref.shape)
                rec.update(finite=bool(torch.isfinite(d).all()),
                           rel_rms=float(((d - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()),
                           max_abs=float((d - ref).abs().max()))
            ttnn.deallocate(o); live[name] = fn; log(**rec)
        except Exception as e:  # noqa: BLE001
            log(ev="arm_refused", shape=spec, arm=name, err=str(e)[:300])
    times = {k: [] for k in live}
    for rep in range(A.reps):
        for name, fn in live.items():
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(A.calls):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            times[name].append((time.perf_counter() - t0) / A.calls * 1e3)
    mb = b * n * hid * 2 / 1e6
    summ = {k: dict(median_ms=round(statistics.median(v), 4), spread_ms=round(max(v) - min(v), 4)) for k, v in times.items()}
    log(ev="summary", shape=spec, tensor_mb=round(mb, 2), floor3_ms_248=round(3 * mb / 248e3, 4), **summ)
    for t in (h1, h2, s1):
        ttnn.deallocate(t)
