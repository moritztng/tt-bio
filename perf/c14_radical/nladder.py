#!/usr/bin/env python3
"""Output-width ladder at fixed K on the pair Transition's executed shape.

C13 fitted the thin-K matmul's cost against K at a FIXED output-tile count, so its 0.0723 ms
intercept cannot distinguish a per-OP fixed cost from a per-OUTPUT-TILE one. That distinction
decides whether merging the SwiGLU's two N=512 in-projections into one N=1024 matmul is worth
anything. This runs the missing ladder.

Refuses to start without perf/c14_radical/prediction.json.
"""
import json, os, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRED = HERE / "prediction.json"
if not PRED.is_file():
    sys.exit("refusing to run: no pre-registered prediction at %s" % PRED)

sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

MHZ = int(os.environ.get("C14R_MHZ", "1350"))
REPS = int(os.environ.get("C14R_REPS", "11"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "nladder.json"

dev = ttnn.open_device(device_id=0)
nodes = clk.nodes_open_by_this_process()
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
grid = dev.compute_with_storage_grid_size()
CORE_GRID = ttnn.CoreGrid(y=grid.y, x=grid.x)
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def t(x, mc=L1, dt=ttnn.bfloat16):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=mc)


def timeit(fn, reps=REPS):
    fn(); ttnn.synchronize_device(dev)          # warm/compile, discarded
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts), statistics.median(ts), ts


res = {"mhz_requested": MHZ, "reps": REPS, "grid": [grid.y, grid.x], "nodes": nodes,
       "prediction": json.loads(PRED.read_text())}
held = clk.force(MHZ, nodes)
res["nodes_forced"] = held
for _ in range(200):
    if all(clk.aiclk(n) >= MHZ - 5 for n in held):
        break
    time.sleep(0.05)
res["aiclk_before"] = {n: clk.aiclk(n) for n in held}
sampler = clk.Sampler(held[0])
time.sleep(0.3)

# ---- instrument control: dense cube, exact FLOPs ----
N = 8192
a = t(torch.randn(N, N, dtype=torch.bfloat16), DRAM)
b = t(torch.randn(N, N, dtype=torch.bfloat16), DRAM)
cube_min, cube_med, _ = timeit(lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                   core_grid=CORE_GRID), reps=5)
flops_shape = 2 * N * N * N
flops_tile = (N // 32) * (N // 32) * (N // 32) * 2 * 32 ** 3
res["control_cube"] = {"flops_shape": flops_shape, "flops_tile": flops_tile,
                       "exact_match": flops_shape == flops_tile == 1099511627776,
                       "ms_min": cube_min, "tflops": flops_shape / (cube_min * 1e-3) / 1e12}
for x in (a, b):
    ttnn.deallocate(x)

# ---- the ladder: in [1,16,512,128] L1, weight [128,N] ----
B, M, K = 16, 512, 128
xn = t(torch.randn(1, B, M, K, dtype=torch.bfloat16), L1)
arms = {}
for width in (512, 1024, 2048):
    w = t(torch.randn(K, width, dtype=torch.bfloat16), DRAM)

    def one(w=w):
        y = ttnn.linear(xn, w, compute_kernel_config=ckc, memory_config=L1,
                        dtype=ttnn.bfloat16, core_grid=CORE_GRID)
        ttnn.deallocate(y)
    mn, md, ts = timeit(one)
    mn2, md2, _ = timeit(one)                    # A/A twin
    arms["N%d" % width] = {"ms_min": mn, "ms_med": md, "aa_ms_min": mn2,
                           "aa_pct": abs(mn2 - mn) / mn * 100,
                           "tflops": 2 * B * M * K * width / (mn * 1e-3) / 1e12}
    ttnn.deallocate(w)

# ---- the shipped pattern vs the merge, on the real SwiGLU widths ----
H = 512
w1 = t(torch.randn(K, H, dtype=torch.bfloat16), DRAM)
w2 = t(torch.randn(K, H, dtype=torch.bfloat16), DRAM)
w12 = t(torch.randn(K, 2 * H, dtype=torch.bfloat16), DRAM)


def split_shipped():
    x1 = ttnn.linear(xn, w1, activation="silu", compute_kernel_config=ckc, memory_config=L1,
                     dtype=ttnn.bfloat16, core_grid=CORE_GRID)
    x2 = ttnn.linear(xn, w2, compute_kernel_config=ckc, memory_config=L1,
                     dtype=ttnn.bfloat16, core_grid=CORE_GRID)
    ttnn.deallocate(ttnn.multiply_(x1, x2))
    ttnn.deallocate(x2)


def merged_noact():
    y = ttnn.linear(xn, w12, compute_kernel_config=ckc, memory_config=L1,
                    dtype=ttnn.bfloat16, core_grid=CORE_GRID)
    ttnn.deallocate(y)


def merged_full():
    y = ttnn.linear(xn, w12, compute_kernel_config=ckc, memory_config=L1,
                    dtype=ttnn.bfloat16, core_grid=CORE_GRID)
    x1 = ttnn.slice(y, [0, 0, 0, 0], [1, B, M, H], memory_config=L1)
    x2 = ttnn.slice(y, [0, 0, 0, H], [1, B, M, 2 * H], memory_config=L1)
    ttnn.deallocate(y)
    x1 = ttnn.silu(x1, memory_config=L1, output_tensor=x1)
    ttnn.deallocate(ttnn.multiply_(x1, x2))
    ttnn.deallocate(x2)


for name, fn in (("split_shipped", split_shipped), ("merged_noact", merged_noact),
                 ("merged_full", merged_full)):
    mn, md, _ = timeit(fn)
    mn2, _, _ = timeit(fn)
    arms[name] = {"ms_min": mn, "ms_med": md, "aa_ms_min": mn2,
                  "aa_pct": abs(mn2 - mn) / mn * 100}

res["arms"] = arms
res["aiclk_after"] = {n: clk.aiclk(n) for n in held}
res["clock"] = sampler.stop()
ttnn.close_device(dev)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("control_cube", "arms", "clock")}, indent=1))
