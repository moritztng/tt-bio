"""E (D4): cost of the residual's named candidates -- reallocate, deallocate, allocate --
at the 298 aa pair-track shape, synthetically (the model's own buffers are not touched)."""
import json, time, sys
import torch, ttnn
dev = ttnn.open_device(device_id=0)
def eager(fn, it):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(it): fn()
    ttnn.synchronize_device(dev)
    return round((time.perf_counter()-t0)/it*1e6, 3)
res = []
# 298 aa padded N=320, c_z=256 pair track; and the c_z=128 half-width variant
for (N, C, tag) in [(320, 256, "pair z 320x320x256"), (320, 128, "pair 320x320x128"), (320, 32, "small 320x320x32")]:
    mb = N*N*C*2/1e6
    r = {"shape": f"{N}x{N}x{C}", "tag": tag, "MB": round(mb, 2)}
    x = ttnn.from_torch(torch.randn(1, N, N, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # reallocate: ttnn moves the buffer to defragment. Time it in isolation.
    def realloc():
        global x
        x = ttnn.reallocate(x)
    try:
        r["reallocate_us"] = eager(realloc, 30)
        r["reallocate_GBs"] = round(2*N*N*C*2/(r["reallocate_us"]*1e-6)/1e9, 1)
    except Exception as e:
        r["reallocate_err"] = str(e)[:90]
    # deallocate + allocate of the same DRAM buffer
    def dealloc_alloc():
        global x
        ttnn.deallocate(x)
        x = ttnn.from_torch(torch.zeros(1, N, N, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # deallocate alone, measured against an op that reallocates a fresh buffer each time
    y = ttnn.from_torch(torch.randn(1, N, N, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    def clone_keep():
        z = ttnn.clone(y, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(z)
    r["clone_plus_dealloc_us"] = eager(clone_keep, 20)
    def dealloc_only():
        z = ttnn.clone(y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t0 = time.perf_counter(); ttnn.deallocate(z); return time.perf_counter()-t0
    ttnn.synchronize_device(dev)
    ts = [dealloc_only() for _ in range(20)]
    ttnn.synchronize_device(dev)
    r["deallocate_us_host"] = round(sum(sorted(ts)[2:-2])/16*1e6, 3)
    res.append(r); print(json.dumps(r), flush=True)
    ttnn.deallocate(x); ttnn.deallocate(y)
json.dump(res, open(sys.argv[1], "w"), indent=2)
ttnn.close_device(dev); print("DONE", flush=True)
