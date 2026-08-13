"""L5c screen: does the token-level AdaLN chain get faster with its intermediates in L1?

The AdaLN screen found the two elementwise ops are 43.8 % of the chain at 49.1 / 46.4 us, moving
2.65 MB at an implied 54 GB/s -- neither dispatch-bound (2.6x t_d) nor bandwidth-bound. The tensors
are [1, 576, 768] bf16 = 884736 B, which is 24x smaller than the trimul chunk whose L1 residency
threw the CB clash (L2), and across a 110-core grid it is ~8 KB/core. So L1 residency is a different
regime here and is worth measuring rather than inheriting L2's NO-GO.

`_adaln_memory_config(atom_level, large_seq_len)` already returns L1 for the ATOM level and `None`
(i.e. DRAM) for the token level. This times the whole chain both ways, plus each elementwise op on
its own, and checks the result is unchanged.

PREDICTION, WRITTEN BEFORE THE RUN: if the elementwise ops are occupancy/latency-bound rather than
bandwidth-bound, L1 residency buys little and the chain stays near 218 us -- NO-GO. If they are
paying DRAM round trips, they fall toward the ~25 us the layer_norms take and the chain lands at
140-160 us, i.e. 60-78 us x 48 AdaLNs = 2.9-3.7 ms/step = 3.0-3.9 % end-to-end with no kernel.
Kill gate: under 20 us saved on the chain, stop.
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
w_norm, W1, b1, W2 = mk(D), mk(D, D), mk(D), mk(D, D)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                             fp32_dest_acc_en=False, packer_l1_acc=True)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
N = 50
res = {}


def timeit(fn, n=N):
    for _ in range(10):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) * 1e6 / n


def rec(k, v):
    res[k] = v
    print(f"  {k} = {v}", flush=True)


a_l1 = ttnn.to_memory_config(a0, memory_config=L1)
sc_d = ttnn.linear(s0, W1, bias=b1, compute_kernel_config=ckc)
sc_l1 = ttnn.to_memory_config(sc_d, memory_config=L1)

rec("multiply_sigmoid_DRAM", round(timeit(
    lambda: ttnn.multiply(a0, sc_d, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])), 3))
rec("multiply_sigmoid_L1_out", round(timeit(
    lambda: ttnn.multiply(a0, sc_d, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                          memory_config=L1)), 3))
rec("multiply_sigmoid_L1_all", round(timeit(
    lambda: ttnn.multiply(a_l1, sc_l1, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                          memory_config=L1)), 3))
rec("layer_norm_a_L1_out", round(timeit(
    lambda: ttnn.layer_norm(a0, epsilon=1e-5, compute_kernel_config=ckc, memory_config=L1)), 3))
rec("linear_L1_out", round(timeit(
    lambda: ttnn.linear(s0, W1, bias=b1, compute_kernel_config=ckc, memory_config=L1)), 3))


def chain(mc):
    """mc=None reproduces the shipped token-level path; mc=L1 keeps every intermediate in L1."""
    kw = {} if mc is None else {"memory_config": mc}
    an = ttnn.layer_norm(a0, epsilon=1e-5, compute_kernel_config=ckc, **kw)
    sn = ttnn.layer_norm(s0, weight=w_norm, epsilon=1e-5, compute_kernel_config=ckc, **kw)
    x1 = ttnn.linear(sn, W1, bias=b1, compute_kernel_config=ckc, **kw)
    x2 = ttnn.linear(sn, W2, compute_kernel_config=ckc, **kw)
    ttnn.deallocate(sn)
    an = ttnn.multiply_(an, x1, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(x1)
    an = ttnn.add_(an, x2)
    ttnn.deallocate(x2)
    return ttnn.to_memory_config(an, memory_config=DRAM)


rec("chain_shipped_DRAM", round(timeit(lambda: chain(None)), 3))
try:
    rec("chain_L1", round(timeit(lambda: chain(L1)), 3))
    ref = ttnn.to_torch(chain(None)).float()
    got = ttnn.to_torch(chain(L1)).float()
    rec("equal_L1_vs_DRAM", bool(torch.equal(got, ref)))
    rec("maxdiff_L1_vs_DRAM", float((got - ref).abs().max()))
except Exception as exc:
    rec("chain_L1", f"THREW: {type(exc).__name__}: {str(exc)[:200]}")
print("RESULT " + json.dumps(res), flush=True)
ttnn.close_device(dev)
