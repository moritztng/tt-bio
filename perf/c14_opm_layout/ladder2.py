#!/usr/bin/env python3
"""Round 2: the composed stack as it is now implemented, against the shipped chain.

Round 1 (ladder_qb1c2.json) found the win needs BOTH the flatten and an unpinned core grid, and
its composed arm pinned the grid, so the composed number was blind to 6.1 ms of the effect. Same
interleaving, same A/A twin, same during-sampled clock; arms rebuilt to match the patched engine.
"""
import json, statistics, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk, torch, ttnn  # noqa: E402

MHZ, REPS = 1350, int(sys.argv[1]) if len(sys.argv) > 1 else 15
S, I, J, C, D, CZ = 64, 512, 512, 32, 32, 128
INSITU = {"projo_batched": 10.2965, "mul_scale_z": 3.0227}
dev = ttnn.open_device(device_id=0)
held = clk.force(MHZ, clk.nodes_open_by_this_process())
for _ in range(200):
    if all(clk.aiclk(n) >= MHZ - 5 for n in held): break
    time.sleep(0.05)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
g = dev.compute_with_storage_grid_size(); CG = ttnn.CoreGrid(y=g.y, x=g.x)
DRAM = ttnn.DRAM_MEMORY_CONFIG
scale = 1.0 / S
torch.manual_seed(0)
Z_EL = I * C * D * J
assert Z_EL * 2 == ((I * C) // 32) * ((D * J) // 32) * 32 * 32 * 2 == 536870912
def tt(x): return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                  memory_config=DRAM)
a0 = tt((torch.randn(S, I, C) / 8).to(torch.bfloat16))
b0 = tt((torch.randn(S, J, D) / 8).to(torch.bfloat16))
W = tt((torch.randn(C * D, CZ) / 32).to(torch.bfloat16))
bias = tt((torch.randn(1, CZ) / 8).to(torch.bfloat16))

def dram_roof():
    x = tt(torch.zeros(1, 1, 16384, 16384, dtype=torch.bfloat16))
    y = tt(torch.zeros(1, 1, 16384, 16384, dtype=torch.bfloat16))
    ttnn.synchronize_device(dev); best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); o = ttnn.add(x, y); ttnn.synchronize_device(dev)
        best = min(best, time.perf_counter() - t0); ttnn.deallocate(o)
    ttnn.deallocate(x); ttnn.deallocate(y)
    return 3 * 536870912 / best / 1e9

def chain(fold_scale, flat, grid):
    a = ttnn.permute(a0, (1, 2, 0))
    if fold_scale:
        a2 = ttnn.multiply_(a, scale); a = a2
    b = ttnn.permute(b0, (2, 1, 0))
    b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT); b = ttnn.reshape(b, (-1, S))
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
    af = ttnn.reshape(a, (I * C, S))
    z = ttnn.matmul(af, b, transpose_b=True, compute_kernel_config=ckc)
    ttnn.deallocate(a); ttnn.deallocate(b)
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    z = ttnn.reshape(z, (I, C * D, J))
    z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
    z = ttnn.permute(z, (0, 2, 1))
    if not fold_scale:
        z = ttnn.multiply_(z, scale)
    if flat:
        z = ttnn.reshape(z, (I * J, C * D))
    kw = {"core_grid": CG} if grid else {}
    out = ttnn.linear(z, W, bias=bias, compute_kernel_config=ckc, **kw)
    ttnn.deallocate(z)
    if flat:
        out = ttnn.reshape(out, (I, J, CZ))
    ttnn.deallocate(out)

Z_FIN = None
def build_zfin():
    global Z_FIN, Z_2D
    a = ttnn.permute(a0, (1, 2, 0)); b = ttnn.permute(b0, (2, 1, 0))
    b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT); b = ttnn.reshape(b, (-1, S))
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
    z = ttnn.matmul(ttnn.reshape(a, (I * C, S)), b, transpose_b=True, compute_kernel_config=ckc)
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT); z = ttnn.reshape(z, (I, C * D, J))
    z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
    Z_FIN = ttnn.permute(z, (0, 2, 1))
    Z_2D = ttnn.reshape(Z_FIN, (I * J, C * D))
build_zfin()
ttnn.synchronize_device(dev)

ARMS = [
    ("full_shipped",    lambda: chain(False, False, True)),
    ("full_shipped_AA", lambda: chain(False, False, True)),
    ("full_scaleonly",  lambda: chain(True, False, True)),
    ("full_new_stack",  lambda: chain(True, True, False)),
    ("projo_batched",   lambda: ttnn.deallocate(ttnn.linear(Z_FIN, W, bias=bias,
                            compute_kernel_config=ckc, core_grid=CG))),
    ("projo_flat_nogrid", lambda: ttnn.deallocate(ttnn.linear(Z_2D, W, bias=bias,
                            compute_kernel_config=ckc))),
    ("mul_scale_z",     lambda: ttnn.deallocate(ttnn.multiply(Z_FIN, scale))),
    ("out_reshape_back", lambda: ttnn.deallocate(ttnn.reshape(
                            ttnn.reshape(Z_2D, (I, J, C * D)), (I * J, C * D)))),
]
for _, fn in ARMS: fn()
ttnn.synchronize_device(dev)
roof = dram_roof(); ttnn.synchronize_device(dev)
s = clk.Sampler(held[0]); time.sleep(0.3)
acc = {n: [] for n, _ in ARMS}
for r in range(REPS):
    for n, fn in (ARMS if r % 2 == 0 else list(reversed(ARMS))):
        ttnn.synchronize_device(dev); t0 = time.perf_counter(); fn()
        ttnn.synchronize_device(dev); acc[n].append((time.perf_counter() - t0) * 1e3)
res = {n: {"ms_min": min(v), "ms_med": statistics.median(v), "n": len(v)} for n, v in acc.items()}
res["_AA_pct"] = abs(res["full_shipped_AA"]["ms_min"] - res["full_shipped"]["ms_min"]) \
                 / res["full_shipped"]["ms_min"] * 100
res["_stack_delta_ms"] = res["full_shipped"]["ms_min"] - res["full_new_stack"]["ms_min"]
res["_stack_ratio"] = res["full_new_stack"]["ms_min"] / res["full_shipped"]["ms_min"]
out = {"arms": res, "insitu": INSITU, "dram_roof_gbs": roof, "clock": s.stop(),
       "grid": [g.y, g.x], "nodes": held, "reps": REPS, "shape": [S, I, J, C, D, CZ]}
ttnn.close_device(dev)
(HERE / "ladder2_qb1c2.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
