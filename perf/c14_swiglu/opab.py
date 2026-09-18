#!/usr/bin/env python3
"""Op-level A/B for the fused SwiGLU kernel, at the pair Transition's own executed shape.

Arms, INTERLEAVED rep by rep with the interior order reversed on odd reps, because rounds 1-2 of
c14-radical's ladder ran arms in blocks and disagreed by up to 51 % across sessions:

    prod             production's three ops, silu FUSED into fc1 -- the real baseline
    prod_AA          the same arm again -- the A/A floor this session's numbers are read against
    prod_unfused     radical's `split_unfused`, so this session is comparable to its 0.15175 ms
    prod_unfused_AA  its own A/A twin, because prod vs prod_unfused turned out to be a real gap
    fused            the kernel: silu(x@W1) * (x@W2) in one generic_op
    mm2_generic      TWO UNFUSED minimal_matmuls through generic_op -- the kernel-family control

Run it twice. c14-radical's rounds 1 and 2 disagreed by up to 51 % on one arm ACROSS sessions
even with the arms interleaved inside each, so one session is a reading and two agreeing sessions
are a measurement. `sys.argv[1]` names the session in the output filename.

`mm2_generic` is the control that decides what a win would mean. `fused` differs from `prod` in two
ways at once: it fuses three ops into one AND it runs minimal_matmul kernels instead of ttnn.linear's
multicast matmul. Without this arm a faster `fused` cannot be attributed to the fusion at all. It is
the known-answer case for the timing instrument in the same sense radical's cube was for the counter
instrument: mm2_generic does the same arithmetic as prod's two matmuls with no fusion in it.
"""
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock"))

import clk                                                            # noqa: E402
import torch                                                          # noqa: E402
import ttnn                                                           # noqa: E402

from tt_bio import mm_generic as MG                                   # noqa: E402
from tt_bio import swiglu_fused as SW                                 # noqa: E402

MHZ, REPS = 1350, 21
B, M, K, H = 16, 512, 128, 512
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

dev = ttnn.open_device(device_id=0)
held = clk.force(MHZ, clk.nodes_open_by_this_process())
for _ in range(200):
    if all(clk.aiclk(n) >= MHZ - 5 for n in held):
        break
    time.sleep(0.05)

g = dev.compute_with_storage_grid_size()
CG = ttnn.CoreGrid(y=g.y, x=g.x)
GRID = (g.x, g.y)
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
CKC = MG.ckc_args(ckc)

torch.manual_seed(0)


def to_tt(x, mc=L1):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=mc)


xn = to_tt(torch.randn(1, B, M, K, dtype=torch.bfloat16))
w1 = to_tt(torch.randn(K, H, dtype=torch.bfloat16), DRAM)
w2 = to_tt(torch.randn(K, H, dtype=torch.bfloat16), DRAM)
BLOCK = SW._block(w1)


def lin(w, act=None):
    return ttnn.linear(xn, w, activation=act, compute_kernel_config=ckc, memory_config=L1,
                       dtype=ttnn.bfloat16, core_grid=CG)


def prod():
    a = lin(w1, "silu")
    b = lin(w2)
    ttnn.deallocate(ttnn.multiply_(a, b))
    ttnn.deallocate(b)


def prod_unfused():
    a = lin(w1)
    b = lin(w2)
    a = ttnn.silu(a, memory_config=L1, output_tensor=a)
    ttnn.deallocate(ttnn.multiply_(a, b))
    ttnn.deallocate(b)


def fused():
    ttnn.deallocate(SW.fused_swiglu(xn, w1, w2, CKC, GRID))


def mm2_generic():
    shape = ttnn.Shape([1, B, M, H])
    outs = [ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, L1)
            for _ in range(2)]
    for w, o in zip((w1, w2), outs):
        MG.generic_minimal_matmul(dev, xn, w, [o], (BLOCK, GRID), CKC)
    for o in outs:
        ttnn.deallocate(o)


ARMS = [("prod", prod), ("prod_AA", prod), ("prod_unfused", prod_unfused),
        ("prod_unfused_AA", prod_unfused), ("fused", fused), ("mm2_generic", mm2_generic)]

for _, fn in ARMS:            # JIT build + warmup, outside the measured loop
    fn()
ttnn.synchronize_device(dev)

s = clk.Sampler(held[0])
time.sleep(0.3)
acc = {n: [] for n, _ in ARMS}
for r in range(REPS):
    order = ARMS if r % 2 == 0 else list(reversed(ARMS))
    for n, fn in order:
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        acc[n].append((time.perf_counter() - t0) * 1e3)

res = {n: {"ms_min": min(v), "ms_med": statistics.median(v), "ms_p90": sorted(v)[int(.9 * len(v))],
           "n": len(v)} for n, v in acc.items()}
base = res["prod"]["ms_min"]
out = {
    "arms": res,
    "AA_pct": abs(res["prod_AA"]["ms_min"] - base) / base * 100,
    "AA_pct_unfused": abs(res["prod_unfused_AA"]["ms_min"] - res["prod_unfused"]["ms_min"])
                      / res["prod_unfused"]["ms_min"] * 100,
    "prod_over_prod_unfused": base / res["prod_unfused"]["ms_min"],
    "fused_over_prod": res["fused"]["ms_min"] / base,
    "fused_over_prod_unfused": res["fused"]["ms_min"] / res["prod_unfused"]["ms_min"],
    "fused_over_mm2_generic": res["fused"]["ms_min"] / res["mm2_generic"]["ms_min"],
    "shape": [1, B, M, K], "hidden": H, "grid": list(GRID), "block": list(BLOCK),
    "round": SW.ROUND, "reps": REPS, "interleaved": True,
    "nodes": held, "clock": s.stop(),
}
ttnn.close_device(dev)
tag = sys.argv[1] if len(sys.argv) > 1 else "s1"
(HERE / f"opab_qb1n0_{tag}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
