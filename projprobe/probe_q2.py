#!/usr/bin/env python3
"""Q2: give ttnn.grid_sample its best shot before using it as the baseline.

Q1 measured 16.9 ns per output point, flat in channel count and image size -- the signature of a
per-output-point cost, not a data cost. tt-metal #27904 says the fix already in flight is
"batching support that groups the grid's final dimension to increase page size", i.e. exactly the
4-byte-page read this would produce. So re-measure with the shipped precomputed-grid path before
quoting any speedup against it.
"""
import json, time, inspect
import torch, ttnn

R = {}
dev = ttnn.open_device(device_id=0)
def sync(): ttnn.synchronize_device(dev)
R["sig_grid_sample"] = str(inspect.signature(ttnn.grid_sample)) if hasattr(ttnn.grid_sample, "__signature__") else "n/a"
print(ttnn.grid_sample.__doc__[:2000] if ttnn.grid_sample.__doc__ else "no doc", flush=True)
print("PREP DOC:", (ttnn.prepare_grid_sample_grid.__doc__ or "")[:1200], flush=True)

def run(Hin, Win, C, Ho, Wo, precomp, reps=5):
    x = ttnn.from_torch(torch.randn(1, Hin, Win, C), dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    gt = torch.rand(1, Ho, Wo, 2) * 1.8 - 0.9
    if precomp:
        g_host = ttnn.from_torch(gt, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        gp = ttnn.prepare_grid_sample_grid(g_host, [1, Hin, Win, C], padding_mode="zeros", output_dtype=ttnn.bfloat16)
        g = ttnn.to_device(gp, dev)
        kw = dict(use_precomputed_grid=True)
    else:
        g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        kw = {}
    o = ttnn.grid_sample(x, g, **kw); sync(); ttnn.deallocate(o)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); o = ttnn.grid_sample(x, g, **kw); sync()
        best = min(best, time.perf_counter() - t0); ttnn.deallocate(o)
    ttnn.deallocate(x); ttnn.deallocate(g)
    n = Ho * Wo
    return {"ms": best*1e3, "points": n, "ns_per_point": best*1e9/n, "precomputed": precomp}

for pc in (False, True):
    for (Hin, Win, C, Ho, Wo) in [(512,512,32,256,256), (512,512,32,512,512)]:
        k = f"gs_pc{int(pc)}_{Hin}_C{C}_{Ho}x{Wo}"
        try: R[k] = run(Hin, Win, C, Ho, Wo, pc)
        except Exception as e: R[k] = {"error": f"{type(e).__name__}: {e}"}
        print(k, R[k], flush=True)

ttnn.close_device(dev)
open("/tmp/probe_q2.json","w").write(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
