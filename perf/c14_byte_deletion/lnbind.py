#!/usr/bin/env python3
"""What binds layer_norm_w, the byte class that carries 843.2 of the block s 1,211.1 Mcycles.

c10-fold-census prices the class at 239.0 GB/s against a measured 442.9 GB/s DRAM roof and books
843.2 Mc as above the traffic roof. That figure is only deletable bytes if the op is traffic-bound.
This asks the machine, on the class s own largest executed key, with a same-shape copy as the
known-answer control: the copy moves the identical 2 Z and cannot do anything but move bytes.

Arms interleaved rep by rep, interior order reversed on odd reps, an A/A twin on the shipped arm,
clock forced and sampled DURING by a process-local sampler. Pre-registration in PREREG.json; this
refuses to start without it.
"""
import json, statistics, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk, torch, ttnn  # noqa: E402

PRE = json.loads((HERE / "PREREG.json").read_text())
assert PRE["arms"], "no pre-registration"
MHZ, REPS = 1350, 15
Z = 67108864

dev = ttnn.open_device(device_id=0)
held = clk.force(MHZ, clk.nodes_open_by_this_process())
for _ in range(200):
    if all(clk.aiclk(n) >= MHZ - 5 for n in held):
        break
    time.sleep(0.05)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       math_approx_mode=False, fp32_dest_acc_en=True,
                                       packer_l1_acc=True)
g = dev.compute_with_storage_grid_size()
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
S, C = 512, 128
x = ttnn.from_torch(torch.randn(1, S, S, C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
w = ttnn.from_torch(torch.randn(C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16, memory_config=DRAM)
b = ttnn.from_torch(torch.zeros(C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16, memory_config=DRAM)


def ln(mc=DRAM, affine=True):
    kw = {"weight": w, "bias": b} if affine else {}
    ttnn.deallocate(ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=ckc,
                                    memory_config=mc, **kw))


ARMS = [
    ("ln_shipped", lambda: ln()),
    ("ln_shipped_AA", lambda: ln()),
    ("copy_ctl", lambda: ttnn.deallocate(ttnn.clone(x, memory_config=DRAM))),
    ("ln_noaffine", lambda: ln(affine=False)),
    ("ln_l1_out", lambda: ln(mc=L1)),
    ("mean_ctl", lambda: ttnn.deallocate(ttnn.mean(x, dim=-1, memory_config=DRAM))),
]
ok = []
for n, fn in ARMS:                                    # warm + eligibility: an arm that throws is
    try:                                              # reported as refused, never silently dropped
        fn()
        ok.append((n, fn))
    except Exception as e:
        print(f"REFUSED {n}: {type(e).__name__} {str(e)[:200]}", flush=True)
ttnn.synchronize_device(dev)
s = clk.Sampler(held[0])
time.sleep(0.3)
acc = {n: [] for n, _ in ok}
for r in range(REPS):
    for n, fn in (ok if r % 2 == 0 else list(reversed(ok))):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        acc[n].append((time.perf_counter() - t0) * 1e3)
res = {}
for n, v in acc.items():
    nom = Z if n == "mean_ctl" else 2 * Z              # mean reads 1 Z and writes 0.008 Z
    res[n] = {"ms_min": min(v), "ms_med": statistics.median(v), "n": len(v),
              "GB_s_nominal": nom / (min(v) / 1e3) / 1e9,
              "Mc_at_1350": min(v) / 1e3 * 1350e6 / 1e6}
base = res["ln_shipped"]["ms_min"]
res["AA_pct"] = abs(res["ln_shipped_AA"]["ms_min"] - base) / base * 100
res["ln_over_copy"] = base / res["copy_ctl"]["ms_min"]
res["l1_out_saving_frac_of_copy"] = (base - res["ln_l1_out"]["ms_min"]) / res["copy_ctl"]["ms_min"]
out = {"arms": res, "grid": [g.y, g.x], "nodes": held, "clock": s.stop(), "reps": REPS,
       "interleaved": True, "key": "1x512x512x128 bf16 DRAM, layer_norm_w largest executed key",
       "prereg": PRE["predicted"]}
ttnn.close_device(dev)
(HERE / "lnbind_qb1n0.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
