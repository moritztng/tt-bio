#!/usr/bin/env python3
"""Two things the decomposition is not allowed to claim without: that the program config which
beats ttnn's own choice computes the SAME answer, and that the gap survives an interleaved A/B
with an A/A twin. A faster config that is wrong is not a finding.

Reference is float64 on the host, not the other device arm, so "same" means both are within the
same distance of the truth rather than within distance of each other.
"""
from __future__ import annotations
import json, statistics as st, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk, torch, ttnn                                                        # noqa: E402
from tt_bio import tenstorrent as T                                            # noqa: E402

MHZ, REPS, NPER = 1350, 21, 8
B, M, K, N = 16, 512, 128, 512
device = T.get_device()
held = clk.force(MHZ, clk.nodes_open_by_this_process())
t0 = time.time()
while time.time() - t0 < 10 and not all(clk.aiclk(n) >= MHZ - 5 for n in held):
    time.sleep(0.01)
assert all(clk.aiclk(n) >= MHZ - 5 for n in held), {n: clk.aiclk(n) for n in held}
g = device.compute_with_storage_grid_size()
kc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
torch.manual_seed(0)
xh = torch.randn(B * M, K) * 0.05
wh = torch.randn(K, N) * 0.05
ref = (xh.double() @ wh.double())
x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                    memory_config=DRAM)
w = ttnn.from_torch(wh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                    memory_config=DRAM)
mt, nt, kt = (B * M) // 32, N // 32, K // 32
gx, gy = 8, 8
per_m, per_n = -(-mt // gy), -(-nt // gx)


def pcfg(obh):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=kt,
        out_subblock_h=min(obh, 4 // per_n), out_subblock_w=per_n,
        out_block_h=obh, out_block_w=per_n, per_core_M=per_m, per_core_N=per_n,
        transpose_mcast=False, fused_activation=None)


def mk(**kw):
    def f():
        return ttnn.linear(x, w, compute_kernel_config=kc, memory_config=DRAM,
                           dtype=ttnn.bfloat16, **kw)
    return f


ARMS = [("ttnn_g8x8", mk(core_grid=ttnn.CoreGrid(x=8, y=8))),
        ("ttnn_g8x8_AA", mk(core_grid=ttnn.CoreGrid(x=8, y=8))),
        ("ttnn_full", mk(core_grid=T.CORE_GRID_MAIN)),
        ("ttnn_g06x10", mk(core_grid=ttnn.CoreGrid(x=6, y=10))),
        ("obh08", mk(program_config=pcfg(8))),
        ("obh08_AA", mk(program_config=pcfg(8))),
        ("obh32", mk(program_config=pcfg(32)))]

out = {"clock_mhz": MHZ, "nodes": held, "grid": [g.x, g.y], "reps": REPS, "nper": NPER,
       "shape": {"rows": B * M, "k": K, "n": N}, "accuracy": {}, "timing": {}}
for nm, fn in ARMS:
    y = fn()
    yh = ttnn.to_torch(y).double()
    ttnn.deallocate(y)
    err = (yh - ref).abs()
    den = ref.abs().mean()
    out["accuracy"][nm] = {
        "max_abs_err": err.max().item(), "mean_abs_err": err.mean().item(),
        "rel_mean_err": (err.mean() / den).item(),
        "pcc": torch.corrcoef(torch.stack([yh.flatten(), ref.flatten()]))[0, 1].item()}

acc = {nm: [] for nm, _ in ARMS}
for r in range(REPS):
    order = ARMS if r % 2 == 0 else list(reversed(ARMS))
    for nm, fn in order:
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        rs = [fn() for _ in range(NPER)]
        ttnn.synchronize_device(device)
        t2 = time.perf_counter()
        for rr in rs:
            ttnn.deallocate(rr)
        acc[nm].append((t2 - t1) * 1e3 / NPER)
for nm, v in acc.items():
    out["timing"][nm] = {"ms_min": min(v), "ms_med": st.median(v), "n": len(v),
                         "spread_pct": 100 * (max(v) - min(v)) / min(v)}
for a_, b_ in (("ttnn_g8x8", "ttnn_g8x8_AA"), ("obh08", "obh08_AA")):
    out["timing"][a_ + "_AA_pct"] = 100 * abs(out["timing"][b_]["ms_min"]
                                              - out["timing"][a_]["ms_min"]) \
        / out["timing"][a_]["ms_min"]
for nm in ("obh08", "obh32", "ttnn_full", "ttnn_g06x10"):
    out["timing"][nm + "_vs_ttnn_g8x8"] = out["timing"]["ttnn_g8x8"]["ms_min"] \
        / out["timing"][nm]["ms_min"]
out["clock"] = clk.Sampler(held[0]).stop() if False else {"checked_after": clk.aiclk(held[0])}
(HERE / "verify.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
