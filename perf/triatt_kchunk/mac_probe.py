"""Can AdaLN's epilogue lose a program with no kernel?

AdaLN ends with, at the token-level shape [1, 576, 768] bf16 DRAM:

    a = multiply_(a, s_scale, activation=SIGMOID)     1 program
    a = add_(a, s_bias)                               1 program

`ttnn.mac(a, b, c)` computes a*b + c. If it is a single dispatched program, and the sigmoid moves
into the s_scale linear's fused activation (the pattern this codebase already uses for `s_o`), the
pair collapses to one. 60 AdaLNs per diffusion step => 60 programs/step.

Composite ternary ops in ttnn sometimes lower to several eltwise programs, which would make this a
LOSS. That is the whole question, so it is timed rather than assumed.
"""
import sys, time, json
sys.path.insert(0, "/home/ttuser/.coworker/wt/boltzgen-optimize-on-fixture")
import torch, ttnn

dev = ttnn.open_device(device_id=0)
S, D = 576, 768
torch.manual_seed(0)
mk = lambda: ttnn.from_torch(torch.randn(1, S, D, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
a0, sc, sb = mk(), mk(), mk()
N = 100


def timeit(fn):
    for _ in range(15):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(N)]
    ttnn.synchronize_device(dev)
    us = (time.perf_counter() - t0) * 1e6 / N
    for o in outs:
        ttnn.deallocate(o)
    return us


def arm_pair():
    x = ttnn.multiply(a0, sc, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    y = ttnn.add(x, sb)
    ttnn.deallocate(x)
    return y


def arm_mac_presig():
    return ttnn.mac(a0, sig, sb)


res = {}
res["pair_multiply_sigmoid_then_add_us"] = round(timeit(arm_pair), 3)
sig = ttnn.sigmoid(sc)                      # stands in for the linear's fused activation
res["mac_us"] = round(timeit(arm_mac_presig), 3)
res["sigmoid_alone_us"] = round(timeit(lambda: ttnn.sigmoid(sc)), 3)
res["single_add_us"] = round(timeit(lambda: ttnn.add(a0, sb)), 3)

ref = ttnn.to_torch(arm_pair()).float()
got = ttnn.to_torch(arm_mac_presig()).float()
res["maxdiff_mac_vs_pair"] = float((got - ref).abs().max())
res["equal"] = bool(torch.equal(got, ref))
print("RESULT " + json.dumps(res))
ttnn.close_device(dev)
