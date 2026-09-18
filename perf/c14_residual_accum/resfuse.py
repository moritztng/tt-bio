#!/usr/bin/env python3
"""Does ttnn's residual_input_tensor collect c14-radical's 0.30-0.42 s band on the pair tensor?

The band is the claim that all five `z = add_(z, block(z))` residuals in a PairformerLayer can stop
paying for the add. `residual_input_tensor` on layer_norm is the shipped capability that would do it:
`layer_norm(x, residual_input_tensor=r)` computes `layer_norm(x + r)` in one kernel. It returns ONE
tensor -- the normed output -- so the sum is gone, and a residual stream needs the sum.

Three arms separate the mechanism's rate from the rewrite that is actually legal:
  ship_*  what the tree does today: add_ then the consuming norm.
  fused_* the add deleted outright. Fast, and NOT a correct rewrite: it never produces z. This is
          the mechanism's rate and an upper bound, never a lever.
  keep_*  the only rewrite that preserves the stream: fuse the norm AND still do the add_.
Run at both places the update can live, because the five sites take it from L1 today.

copy_ctl is the known-answer control: a same-shape clone can do nothing but move the same 2 Z, and
it must reproduce c14-byte-deletion's 364.84 GB/s on this exact key and node. Arms interleaved rep by
rep with the interior order reversed on odd reps, an A/A twin on the shipped arm, clock forced and
sampled DURING by a process-local sampler. Pre-registration in PREREG.json; refuses to start without.
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
S, C = 512, 128

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


def dev_t(t, mc):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=mc)


zt = torch.randn(1, S, S, C, dtype=torch.bfloat16)
ut = (torch.randn(1, S, S, C) * 0.01).to(torch.bfloat16)   # small: 75 in-place adds must not drift
z = dev_t(zt, DRAM)
w = dev_t(torch.randn(C, dtype=torch.bfloat16), DRAM)
b = dev_t(torch.zeros(C, dtype=torch.bfloat16), DRAM)
u = {"dram": dev_t(ut, DRAM)}
try:
    u["l1"] = dev_t(ut, L1)
except Exception as e:
    print("REFUSED u in L1: %s %s" % (type(e).__name__, str(e)[:160]), flush=True)

KW = dict(weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ckc, memory_config=DRAM)


def ship(where):
    ttnn.add_(z, u[where])
    ttnn.deallocate(ttnn.layer_norm(z, **KW))


def fused(where):
    ttnn.deallocate(ttnn.layer_norm(u[where], residual_input_tensor=z, **KW))


def keep(where):
    ttnn.deallocate(ttnn.layer_norm(u[where], residual_input_tensor=z, **KW))
    ttnn.add_(z, u[where])


ARMS = [("ship_l1", lambda: ship("l1")), ("ship_l1_AA", lambda: ship("l1")),
        ("fused_l1", lambda: fused("l1")), ("keep_l1", lambda: keep("l1")),
        ("copy_ctl", lambda: ttnn.deallocate(ttnn.clone(z, memory_config=DRAM))),
        ("ship_dram", lambda: ship("dram")), ("fused_dram", lambda: fused("dram")),
        ("keep_dram", lambda: keep("dram"))]
ARMS = [(n, f) for n, f in ARMS if "l1" not in n or "l1" in u]

ok = []
for n, fn in ARMS:                     # an arm that throws is reported refused, never dropped
    try:
        fn()
        ok.append((n, fn))
    except Exception as e:
        print("REFUSED %s: %s %s" % (n, type(e).__name__, str(e)[:200]), flush=True)
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
clock = s.stop()

res = {}
for n, v in acc.items():
    res[n] = {"ms_min": min(v), "ms_med": statistics.median(v), "n": len(v),
              "GB_s_on_2Z": 2 * Z / (min(v) / 1e3) / 1e9,
              "Mc_at_1350": min(v) / 1e3 * 1350e6 / 1e6}
ratios = {}
for where in ("l1", "dram"):
    base = res.get("ship_%s" % where)
    if not base:
        continue
    for arm in ("fused", "keep"):
        k = "%s_%s" % (arm, where)
        if k in res:
            ratios["%s_over_ship_%s" % (arm, where)] = res[k]["ms_min"] / base["ms_min"]
if "ship_l1_AA" in res:
    ratios["AA_pct"] = (abs(res["ship_l1_AA"]["ms_min"] - res["ship_l1"]["ms_min"])
                        / res["ship_l1"]["ms_min"] * 100)

# ---- accuracy, on fresh tensors, against a float64 reference and not against each other ----
ttnn.deallocate(z)
zt2 = torch.randn(1, S, S, C, dtype=torch.bfloat16)
ut2 = torch.randn(1, S, S, C, dtype=torch.bfloat16)
z2, u2 = dev_t(zt2, DRAM), dev_t(ut2, DRAM)
wt = torch.randn(C, dtype=torch.bfloat16)
w2, b2 = dev_t(wt, DRAM), dev_t(torch.zeros(C, dtype=torch.bfloat16), DRAM)
K2 = dict(weight=w2, bias=b2, epsilon=1e-5, compute_kernel_config=ckc, memory_config=DRAM)
a_out = ttnn.to_torch(ttnn.layer_norm(ttnn.add(z2, u2), **K2)).to(torch.float64)
b_out = ttnn.to_torch(ttnn.layer_norm(u2, residual_input_tensor=z2, **K2)).to(torch.float64)
ref = torch.nn.functional.layer_norm(
    (zt2.to(torch.float64) + ut2.to(torch.float64)), (C,), eps=1e-5) * wt.to(torch.float64)
acc_out = {
    "shipped_chain_max_abs_err_vs_fp64": float((a_out - ref).abs().max()),
    "fused_max_abs_err_vs_fp64": float((b_out - ref).abs().max()),
    "shipped_vs_fused_bit_exact": bool(torch.equal(a_out, b_out)),
    "shipped_vs_fused_max_abs_diff": float((a_out - b_out).abs().max()),
    "ref_abs_max": float(ref.abs().max()),
}

out = {"arms": res, "ratios": ratios, "accuracy": acc_out, "grid": [g.y, g.x], "nodes": held,
       "clock": clock, "reps": REPS, "interleaved": True,
       "key": "1x512x512x128 bf16, the pair tensor; z DRAM, update in L1 and in DRAM",
       "prereg": PRE["predicted"]}
ttnn.close_device(dev)
(HERE / "resfuse_qb1n0.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("arms", "ratios", "accuracy", "clock", "nodes", "grid")},
                 indent=1))
