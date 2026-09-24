"""What does the discarded recycle tape cost, in seconds and in DRAM?

calls.primal is 0 in every gradient run: recycled_alphafold_outputs stop_gradients every
recycle but the last (af2.py:139-140), so JAX prunes that backward -- but the forward still
went through the TAPED path and banked a tape nobody reads. With design_recycles 1 that is
half the trunk forwards per design step.

This prices it at the real bucket, n=224, both arms in one process so the clock and the
card are the same: untaped forward (tt-bio's ordinary inference path, raw ttnn) against
taped+checkpointed forward, seconds and peak DRAM, AICLK sampled inside each timed region.
"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch, ttnn
import afgrad as A, stack as S

N, REPS = 224, 3
lv = S.Levers(); dm, ref = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")

def dram():
    mv = ttnn.get_memory_view(dev.device, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)

torch.manual_seed(0)
logits = torch.randn(N, 20) * 2.0
m0, z0 = A.embed(ref["bf16"], logits, torch.arange(N))
m0, z0 = m0.detach(), z0.detach()
mask = torch.ones(m0.shape[0], N)
mask_dev = dev.up(mask)

def untaped():
    t0 = time.time()
    mo, zo = dev.stack(dev.up(m0.float()), dev.up(z0.float()), 0, 48, ckpt=False,
                       msa_mask=mask_dev)
    dev.sync()
    t1 = time.time(); peak = dram()
    del mo, zo
    return t1 - t0, peak, (t0, t1)

def taped():
    t0 = time.time()
    ml, zl = dev.leaf(m0.float()), dev.leaf(z0.float())
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, 0, 48, ckpt=True, msa_mask=mask_dev)
    dev.sync()
    t1 = time.time(); peak = dram()
    dev.ag.release_pins()
    del mo, zo, ml, zl
    return t1 - t0, peak, (t0, t1)

out = {"n": N, "reps": REPS, "k_evo": 48, "base_dram_bytes": dram()}
clock = S.Clock()
for name, fn in (("untaped", untaped), ("taped_checkpointed", taped)):
    fn()                                              # warm
    secs, peaks, spans = [], [], []
    for _ in range(REPS):
        s_, p_, sp = fn()
        secs.append(s_); peaks.append(p_); spans.append(sp)
    out[name] = {"seconds": S.dist(secs), "peak_dram_bytes": int(max(peaks)),
                 "peak_dram_gb": round(max(peaks) / 2**30, 3), "spans": spans}
    print(name, round(S.dist(secs)["median"], 3), "s, peak",
          round(max(peaks) / 2**30, 3), "GB", flush=True)
clock.stop()
out["aiclk"] = clock.window([sp for k in ("untaped", "taped_checkpointed") for sp in out[k]["spans"]])
u, t = out["untaped"]["seconds"]["median"], out["taped_checkpointed"]["seconds"]["median"]
out["taped_over_untaped_seconds"] = round(t / u, 3)
out["extra_seconds_per_discarded_recycle"] = round(t - u, 3)
out["extra_dram_gb"] = round((out["taped_checkpointed"]["peak_dram_bytes"]
                              - out["untaped"]["peak_dram_bytes"]) / 2**30, 3)
out["stamp"] = A.stamp(3)
(HERE / "tape_cost.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("taped_over_untaped_seconds",
                                      "extra_seconds_per_discarded_recycle",
                                      "extra_dram_gb", "aiclk")}, indent=1, default=str))
