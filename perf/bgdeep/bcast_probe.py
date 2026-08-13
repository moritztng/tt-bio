"""Can a [1,S,S,1] keep-mask broadcast-multiply a [1,S,S,128] pair tensor on device, bit-exactly?

That is the replacement for `_apply_template_host` when the template module is a proven no-op: its
only remaining effect is re-zeroing the padded rows/cols, which a mask multiply does in one program
instead of an 85 MB round trip plus a full host TemplateModule forward.
"""
import sys, time, json
sys.path.insert(0, "/home/ttuser/.coworker/wt/boltzgen-optimize-on-fixture")
import torch, ttnn

dev = ttnn.open_device(device_id=0)
L, P, C = 514, 576, 128
torch.manual_seed(0)
zt = torch.randn(1, P, P, C, dtype=torch.bfloat16)
keep = torch.zeros(1, P, P, 1, dtype=torch.bfloat16)
keep[:, :L, :L, :] = 1.0
to_d = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
z, k = to_d(zt), to_d(keep)
res = {}
want = zt.clone()
want[:, L:, :, :] = 0
want[:, :, L:, :] = 0

try:
    out = ttnn.multiply(z, k)
    got = ttnn.to_torch(out).to(torch.bfloat16)
    res["broadcast_ok"] = True
    res["broadcast_equal"] = bool(torch.equal(got, want))
    res["broadcast_maxdiff"] = float((got.float() - want.float()).abs().max())
    N = 30
    for _ in range(5):
        ttnn.deallocate(ttnn.multiply(z, k))
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(N):
        ttnn.deallocate(ttnn.multiply(z, k))
    ttnn.synchronize_device(dev)
    res["broadcast_us"] = round((time.perf_counter() - t0) * 1e6 / N, 2)
    ttnn.deallocate(out)
except Exception as exc:
    res["broadcast_ok"] = False
    res["broadcast_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

# fallback: a full-width mask, 85 MB, definitely no broadcast involved
kf = to_d(keep.expand(1, P, P, C).contiguous())
out2 = ttnn.multiply(z, kf)
got2 = ttnn.to_torch(out2).to(torch.bfloat16)
res["full_mask_equal"] = bool(torch.equal(got2, want))
N = 30
for _ in range(5):
    ttnn.deallocate(ttnn.multiply(z, kf))
ttnn.synchronize_device(dev)
t0 = time.perf_counter()
for _ in range(N):
    ttnn.deallocate(ttnn.multiply(z, kf))
ttnn.synchronize_device(dev)
res["full_mask_us"] = round((time.perf_counter() - t0) * 1e6 / N, 2)
print("RESULT " + json.dumps(res), flush=True)
ttnn.close_device(dev)
