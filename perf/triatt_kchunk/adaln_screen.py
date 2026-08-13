"""L5b screen: what a fused token-level AdaLN could actually return.

E1c gave 23.1 us as the token transformer's MEAN cost per dispatched program. The mac probe showed
AdaLN's eltwise ops measure 46-48 us at [1, 576, 768] -- 2.4x the 19.179 us dispatch floor -- so the
mean understates the ops L5b would remove. This times each of the token-level AdaLN's seven programs
at the production shape and the whole chain, so the prize is measured, not derived from an average.

It also settles whether `to_memory_config` dispatches when the config already matches: 48 of the
step's 90 such wrappers are AdaLN's trailing DRAM->DRAM move. It does not -- it aliases the input
buffer, which an earlier run proved by throwing "Buffer is not allocated" on the next call after the
result was deallocated. Timed here without deallocating, and the aliasing checked by address.

Everything bf16, TILE, DRAM interleaved, exactly as the fold issues it.
"""
import sys, time, json
sys.path.insert(0, "/home/ttuser/.coworker/wt/boltzgen-optimize-on-fixture")
import torch, ttnn

dev = ttnn.open_device(device_id=0)
S, D = 576, 768
torch.manual_seed(0)


def mk(*shp):
    return ttnn.from_torch(torch.randn(*shp, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)


a0, s0 = mk(1, S, D), mk(1, S, D)
w_norm = mk(D)
W1, b1, W2 = mk(D, D), mk(D), mk(D, D)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                             fp32_dest_acc_en=False, packer_l1_acc=True)
N = 50
res = {}


def timeit(fn, free=True, n=N):
    """Median-free mean over n calls; outputs freed inside the loop so every arm pays the same."""
    for _ in range(10):
        o = fn()
        if free and o is not None:
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        o = fn()
        if free and o is not None:
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) * 1e6 / n


def rec(key, val):
    res[key] = val
    print(f"  {key} = {val}", flush=True)


sc = ttnn.linear(s0, W1, bias=b1, compute_kernel_config=ckc)
sb = ttnn.linear(s0, W2, compute_kernel_config=ckc)

rec("layer_norm_a_no_weight", round(timeit(
    lambda: ttnn.layer_norm(a0, epsilon=1e-5, compute_kernel_config=ckc)), 3))
rec("layer_norm_s_weighted", round(timeit(
    lambda: ttnn.layer_norm(s0, weight=w_norm, epsilon=1e-5, compute_kernel_config=ckc)), 3))
rec("linear_768x768_bias", round(timeit(
    lambda: ttnn.linear(s0, W1, bias=b1, compute_kernel_config=ckc)), 3))
rec("linear_768x768_nobias", round(timeit(
    lambda: ttnn.linear(s0, W2, compute_kernel_config=ckc)), 3))
rec("multiply_sigmoid", round(timeit(
    lambda: ttnn.multiply(a0, sc, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])), 3))
rec("add", round(timeit(lambda: ttnn.add(a0, sb)), 3))
rec("to_memory_config_DRAM_to_DRAM", round(timeit(
    lambda: ttnn.to_memory_config(a0, memory_config=ttnn.DRAM_MEMORY_CONFIG), free=False), 3))

_t = ttnn.to_memory_config(a0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
rec("tmc_returns_identity", bool(_t is a0))
try:
    rec("tmc_same_buffer_address", bool(_t.buffer_address() == a0.buffer_address()))
except Exception as exc:
    rec("tmc_same_buffer_address", f"unavailable: {type(exc).__name__}")


def chain():
    an = ttnn.layer_norm(a0, epsilon=1e-5, compute_kernel_config=ckc)
    sn = ttnn.layer_norm(s0, weight=w_norm, epsilon=1e-5, compute_kernel_config=ckc)
    x1 = ttnn.linear(sn, W1, bias=b1, compute_kernel_config=ckc)
    x2 = ttnn.linear(sn, W2, compute_kernel_config=ckc)
    ttnn.deallocate(sn)
    an = ttnn.multiply_(an, x1, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(x1)
    an = ttnn.add_(an, x2)
    ttnn.deallocate(x2)
    return ttnn.to_memory_config(an, memory_config=ttnn.DRAM_MEMORY_CONFIG)


rec("whole_adaln_chain", round(timeit(chain), 3))

b = 2  # bf16 bytes
# A perfectly fused AdaLN reads a, s, both weight matrices and the norm weight once, writes once.
rec("fused_bytes", (2 * S * D + 2 * D * D + 2 * D + S * D) * b)
# The chain re-reads and re-writes [S, D] intermediates between every stage.
rec("chain_bytes", (2 * S * D + 2 * S * D + 2 * (S * D + D * D + S * D) + 4 * S * D + 4 * S * D) * b)
print("RESULT " + json.dumps(res), flush=True)
ttnn.close_device(dev)
