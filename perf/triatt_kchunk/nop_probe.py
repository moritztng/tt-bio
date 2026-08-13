"""Does ttnn.to_memory_config dispatch a program when the config already matches?

E1c counted 90 `ttnn.to_memory_config` wrappers per diffusion step. 48 of them are AdaLN's trailing
move to DRAM on a tensor that is already DRAM-interleaved. If that call is a no-op the census
overcounts dispatched programs by 3.3 %; if it dispatches, removing it is a lever worth
48 x 23.1 us = 1.11 ms/step on its own.

Measured against two references on the same tensor: a genuine DRAM->L1 move (definitely dispatches)
and a single-tile ttnn.add (the t_d dispatch-floor probe).
"""
import sys, time, json
sys.path.insert(0, "/home/ttuser/.coworker/wt/boltzgen-optimize-on-fixture")
import torch, ttnn

dev = ttnn.open_device(device_id=0)
S, D = 576, 768                     # the token-level AdaLN activation shape
t = torch.randn(1, S, D, dtype=torch.bfloat16)
x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
one = ttnn.from_torch(torch.randn(1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                      device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
N = 200


def timeit(fn, keep=False):
    for _ in range(20):
        o = fn()
        if not keep and o is not x:
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = []
    for _ in range(N):
        o = fn()
        if not keep and o is not x:
            outs.append(o)
    ttnn.synchronize_device(dev)
    us = (time.perf_counter() - t0) * 1e6 / N
    for o in outs:
        try:
            ttnn.deallocate(o)
        except Exception:
            pass
    return us


res = {}
res["to_memory_config_DRAM_to_DRAM_us"] = round(timeit(
    lambda: ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)), 3)
res["identity_returned"] = bool(
    ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG) is x)
res["to_memory_config_DRAM_to_L1_us"] = round(timeit(
    lambda: ttnn.to_memory_config(x, memory_config=ttnn.L1_MEMORY_CONFIG)), 3)
res["single_tile_add_us_t_d"] = round(timeit(lambda: ttnn.add(one, one)), 3)
print("RESULT " + json.dumps(res))
ttnn.close_device(dev)
