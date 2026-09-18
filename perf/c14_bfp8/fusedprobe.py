#!/usr/bin/env python3
"""Does the fused triangle-attention SDPA actually RUN at bfp8, and what does it cost?

KERNEL-PATH, at the op. The census says the two biggest pair-scale buffers in a trunk
PairformerLayer are the 268.4 MB concatenated q/k/v/gate of the two triangle attentions, written by
our fused qkv kernel and read three times by our fused SDPA kernel -- producer and every consumer
ours. `b2x-bfp8-pair-track` deliberately left them bf16 on the assumption the fused kernels would
decline a block-format operand. With the four self-imposed gates widened, this asks the kernel
directly instead of assuming either way, and it reports SERVED vs DECLINED off the kernel's own
counters rather than off the wall clock.
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# The worktree FIRST. Without this the venv's installed tt_bio wins and the probe scores a
# different tree than the one being changed -- which is exactly how this script failed its first
# run: `AttributeError: module 'tt_bio.triatt_sdpa' has no attribute 'fused_pairs'`, from the
# installed package, not from here. Memory `parity-gate-scores-installed-package-not-checkout`.
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio  # noqa: E402
assert Path(tt_bio.__file__).resolve().is_relative_to(REPO), (
    "imported tt_bio from %s, not %s" % (tt_bio.__file__, REPO))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
from tt_bio import triatt_sdpa as TS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "fusedprobe.json"))
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=32)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--mhz", type=int, default=1350)
A = ap.parse_args()

MGD = ensure_p300_mesh_descriptor()
dev = ttnn.open_device(device_id=0)
nodes = clk.nodes_open_by_this_process()
grid = dev.compute_with_storage_grid_size()
cores = grid.x * grid.y
DRAM = ttnn.DRAM_MEMORY_CONFIG
B8, B16 = ttnn.bfloat8_b, ttnn.bfloat16
WIDTH = {B16: 2.0, B8: 1.0625}
S, H, D = A.seq, A.heads, A.head_dim


def t(shape, dt):
    return ttnn.from_torch(torch.randn(*shape, dtype=torch.float32), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=dt, memory_config=DRAM)


res = {"tt_bio": tt_bio.__file__, "seq": S, "heads": H, "head_dim": D, "grid": [grid.y, grid.x], "cores": cores,
       "nodes": nodes, "mgd": MGD, "mhz_requested": A.mhz, "reps": A.reps, "arms": {}}
held = clk.force(A.mhz, nodes)
for _ in range(200):
    if all(clk.aiclk(n) >= A.mhz - 5 for n in held):
        break
    time.sleep(0.05)
sampler = clk.Sampler(held[0])

pairs = TS.fused_pairs(S, H, D, cores, None)
res["pairs"] = [list(p) for p in pairs][:4]
if not pairs:
    sys.exit("no (q_chunk, k_chunk) pair this kernel can serve at seq=%d" % S)
q_chunk, k_chunk = pairs[0][:2] if len(pairs[0]) >= 2 else pairs[0]

for name, dt in (("b16", B16), ("b8", B8), ("b16_again", B16)):
    q, k, v = (t([S, H, S, D], dt) for _ in range(3))
    bias = t([1, H, S, S], dt)
    before = (TS.STATS[0], TS.STATS[1])
    try:
        out = TS.sdpa(q, k, v, bias, 1.0 / (D ** 0.5), q_chunk, k_chunk)
    except Exception as e:                                                   # noqa: BLE001
        res["arms"][name] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}
        for x in (q, k, v, bias):
            ttnn.deallocate(x)
        continue
    served = TS.STATS[0] - before[0]
    if out is None:
        res["arms"][name] = {"served": 0, "declined": TS.STATS[1] - before[1],
                             "rejects": {str(kk): vv for kk, vv in TS.REJECTS.items()}}
        for x in (q, k, v, bias):
            ttnn.deallocate(x)
        continue
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(A.reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = TS.sdpa(q, k, v, bias, 1.0 / (D ** 0.5), q_chunk, k_chunk)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    nbytes = (3 * S * H * S * D + H * S * S + S * H * S * D) * WIDTH[dt]
    res["arms"][name] = {"served": served, "ms_med": statistics.median(ts), "ms_min": min(ts),
                         "ms": ts, "out_dtype": str(out.dtype), "bytes": nbytes,
                         "gbs": nbytes / (statistics.median(ts) * 1e-3) / 1e9,
                         "q_chunk": q_chunk, "k_chunk": k_chunk}
    ttnn.deallocate(out)
    for x in (q, k, v, bias):
        ttnn.deallocate(x)

res["clock"] = sampler.stop()
b16 = res["arms"].get("b16", {}).get("ms_med")
aa = res["arms"].get("b16_again", {}).get("ms_med")
if b16 and aa:
    res["aa_floor_pct"] = abs(aa - b16) / b16 * 100
if b16 and res["arms"].get("b8", {}).get("ms_med"):
    res["b8_vs_b16"] = res["arms"]["b8"]["ms_med"] / b16
    br = res["arms"]["b8"]["bytes"] / res["arms"]["b16"]["bytes"]
    res["byte_ratio"] = br
    res["realization"] = (1 - res["b8_vs_b16"]) / (1 - br)
Path(A.out).write_text(json.dumps(res, indent=2))
print(json.dumps({k: {kk: vv for kk, vv in (v or {}).items() if kk != "ms"}
                  for k, v in res["arms"].items()}, indent=2))
for k in ("aa_floor_pct", "b8_vs_b16", "byte_ratio", "realization", "clock"):
    print(k, res.get(k))
ttnn.close_device(dev)
