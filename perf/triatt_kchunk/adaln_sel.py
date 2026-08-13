"""L5c-selective: L1 only where it wins, after the all-L1 chain measured a 2.1x LOSS.

The all-L1 screen split the ops cleanly:

    layer_norm  ->L1   12.910 us  against 24.926 DRAM     1.93x FASTER
    linear      ->L1  217.558 us  against  33.209 DRAM     6.5x SLOWER, and not torch.equal
    multiply    ->L1   55.878 us  against  56.691 DRAM     no change
    whole chain  L1   469.359 us  against 223.956 DRAM     2.1x SLOWER

So the loss is entirely the two matmuls: an L1 output config makes ttnn pick a different matmul
program config, which is both far slower and numerically different (maxdiff 2.5).

This times the selective chain -- layer_norms to L1, linears and the elementwise ops left exactly as
shipped -- against the shipped chain, and checks torch.equal. A memory config cannot change a value
unless it changes the program config, which is what the matmul did; layer_norm has no such config,
so bit-exactness is expected and verified rather than assumed.

PRE-COMMITTED KILL GATE, unchanged from the L1 screen: under 20 us saved on the chain, stop.
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


def chain(ln_mc):
    """ln_mc=None is the shipped path; ln_mc=L1 puts ONLY the two layer_norm outputs in L1."""
    lk = {} if ln_mc is None else {"memory_config": ln_mc}
    an = ttnn.layer_norm(a0, epsilon=1e-5, compute_kernel_config=ckc, **lk)
    sn = ttnn.layer_norm(s0, weight=w_norm, epsilon=1e-5, compute_kernel_config=ckc, **lk)
    x1 = ttnn.linear(sn, W1, bias=b1, compute_kernel_config=ckc)
    x2 = ttnn.linear(sn, W2, compute_kernel_config=ckc)
    ttnn.deallocate(sn)
    an = ttnn.multiply_(an, x1, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(x1)
    an = ttnn.add_(an, x2)
    ttnn.deallocate(x2)
    return ttnn.to_memory_config(an, memory_config=DRAM)


# A/A first: the same arm twice, so the chain's own run-to-run floor is known before the delta.
rec("chain_shipped_A", round(timeit(lambda: chain(None)), 3))
rec("chain_shipped_B", round(timeit(lambda: chain(None)), 3))
rec("chain_ln_L1", round(timeit(lambda: chain(L1)), 3))
rec("chain_shipped_C", round(timeit(lambda: chain(None)), 3))

ref = ttnn.to_torch(chain(None)).float()
got = ttnn.to_torch(chain(L1)).float()
rec("equal", bool(torch.equal(got, ref)))
rec("maxdiff", float((got - ref).abs().max()))

# and each layer_norm on its own, both ways, to attribute the delta
rec("ln_a_DRAM", round(timeit(
    lambda: ttnn.layer_norm(a0, epsilon=1e-5, compute_kernel_config=ckc)), 3))
rec("ln_a_L1", round(timeit(
    lambda: ttnn.layer_norm(a0, epsilon=1e-5, compute_kernel_config=ckc, memory_config=L1)), 3))
rec("ln_s_DRAM", round(timeit(
    lambda: ttnn.layer_norm(s0, weight=w_norm, epsilon=1e-5, compute_kernel_config=ckc)), 3))
rec("ln_s_L1", round(timeit(
    lambda: ttnn.layer_norm(s0, weight=w_norm, epsilon=1e-5, compute_kernel_config=ckc,
                            memory_config=L1)), 3))
print("RESULT " + json.dumps(res), flush=True)
ttnn.close_device(dev)
