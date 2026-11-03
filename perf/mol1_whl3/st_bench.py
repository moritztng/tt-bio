"""Device micro-bench of Wormhole's single-track Transition variants on trunk block 43's captured input (8fg6 tap.pt,
rows tiled to N tokens with the captured per-token spread): per-call time (warm, synchronized) and small-channel token
rel err vs float64 on the same bf16 weights. TT_VISIBLE_DEVICES=N python st_bench.py TAP.pt LNW.pt N OUT.json"""
import json, sys, time
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
import tt_bio.tenstorrent as T
torch.manual_seed(0)
tap = torch.load(sys.argv[1], weights_only=False); lnw = torch.load(sys.argv[2], weights_only=False); N = int(sys.argv[3])
x0 = tap["in"][0].double().reshape(-1, 384)
idx = torch.arange(N) % x0.shape[0]
x = x0[idx] + (x0 - x0.mean(0)).std(0) * 0.3 * torch.randn(N, 384, dtype=torch.float64)   # more tokens, same spread
bf = lambda t: t.to(torch.bfloat16).double()
lw, lb = bf(lnw["w"].double()), bf(lnw["b"].double())
w1, w2 = bf(torch.randn(384, 1536, dtype=torch.float64) / 384 ** .5 * 8), bf(torch.randn(384, 1536, dtype=torch.float64) / 384 ** .5 * 8)
w3 = bf(torch.randn(1536, 384, dtype=torch.float64) / 1536 ** .5)
xb = bf(x)
xn = torch.nn.functional.layer_norm(xb, (384,), lw, lb, 1e-5)
ref = (torch.nn.functional.silu(xn @ w1) * (xn @ w2)) @ w3
small = ref.abs().amax(0) < ref.abs().amax(0).median()
tok = lambda t: t - t.mean(0)
def err(o): return float((tok(o[:, small]) - tok(ref[:, small])).norm() / tok(ref[:, small]).norm())
dev = T.get_device()
up = lambda t, d=ttnn.bfloat16: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=d)
X = up(x.reshape(1, N, 384)); LW, LB = up(lw), up(lb); W1, W2, W3 = up(w1), up(w2), up(w3)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)
L1 = ttnn.L1_MEMORY_CONFIG
# folded variant weights: LN weight into fc1|fc2, LN bias as a constant row
W12f = up(torch.cat([lw[:, None] * w1, lw[:, None] * w2], 1)); B12 = up((lb @ torch.cat([w1, w2], 1))[None], ttnn.float32)

def base(x):
    n = ttnn.layer_norm(x, weight=LW, bias=LB, epsilon=1e-5, compute_kernel_config=ckc, memory_config=L1)
    a = ttnn.linear(n, W1, activation="silu", compute_kernel_config=ckc, memory_config=L1, core_grid=T.CORE_GRID_MAIN)
    b = ttnn.linear(n, W2, compute_kernel_config=ckc, memory_config=L1, core_grid=T.CORE_GRID_MAIN)
    h = ttnn.multiply_(a, b)
    return ttnn.linear(h, W3, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)

def cur(x):
    return T.fp32_single_transition(x, LW, LB, W1, W2, W3, ckc, ttnn.bfloat16)

def xhat(x):
    x32 = ttnn.typecast(x, ttnn.float32, memory_config=L1)
    xc = ttnn.subtract(x32, ttnn.mean(x32, dim=-1, keepdim=True, compute_kernel_config=ckc, memory_config=L1), memory_config=L1)
    inv = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(xc, xc, memory_config=L1), dim=-1, keepdim=True,
                                        compute_kernel_config=ckc, memory_config=L1), 1e-5, memory_config=L1), memory_config=L1)
    return ttnn.multiply_(xc, inv)

def clin(v, w, bias, dtype, mc):
    c = ttnn.slice(v, [0, 0, 0], [v.shape[0], 1, v.shape[2]], memory_config=L1)
    d = ttnn.typecast(ttnn.subtract(v, c, memory_config=L1), ttnn.bfloat16, memory_config=L1)
    row = ttnn.linear(ttnn.typecast(c, ttnn.bfloat16, memory_config=L1), w, bias=bias, compute_kernel_config=ckc,
                      dtype=ttnn.float32, memory_config=L1)
    y = ttnn.linear(d, w, compute_kernel_config=ckc, dtype=ttnn.float32, core_grid=T.CORE_GRID_MAIN, memory_config=L1)
    return ttnn.add(y, row, dtype=dtype, memory_config=mc)

def fold(x):   # LN weight/bias folded, fc1|fc2 as one matmul
    ab = clin(xhat(x), W12f, B12, ttnn.float32, L1)
    a = ttnn.slice(ab, [0, 0, 0], [1, N, 1536], memory_config=L1); b = ttnn.slice(ab, [0, 0, 1536], [1, N, 3072], memory_config=L1)
    h = ttnn.multiply_(ttnn.silu(a, output_tensor=a), b)
    return clin(h, W3, None, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)

def fold_bias(x):   # as fold, the token-0 row added as the main matmul's bias
    def clinb(v, w, bias, dtype, mc):
        c = ttnn.slice(v, [0, 0, 0], [v.shape[0], 1, v.shape[2]], memory_config=L1)
        d = ttnn.typecast(ttnn.subtract(v, c, memory_config=L1), ttnn.bfloat16, memory_config=L1)
        row = ttnn.linear(ttnn.typecast(c, ttnn.bfloat16, memory_config=L1), w, bias=bias, compute_kernel_config=ckc,
                          dtype=ttnn.float32, memory_config=L1)
        return ttnn.linear(d, w, bias=ttnn.reshape(row, (1, row.shape[-1])), compute_kernel_config=ckc, dtype=dtype,
                           core_grid=T.CORE_GRID_MAIN, memory_config=mc)
    ab = clinb(xhat(x), W12f, B12, ttnn.float32, L1)
    a = ttnn.slice(ab, [0, 0, 0], [1, N, 1536], memory_config=L1); b = ttnn.slice(ab, [0, 0, 1536], [1, N, 3072], memory_config=L1)
    h = ttnn.multiply_(ttnn.silu(a, output_tensor=a), b)
    return clinb(h, W3, None, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)

res = {"N": N}
for name, f in (("base", base), ("cur", cur), ("fold", fold), ("fold_bias", fold_bias)):
    try:
        o = f(X); e = err(ttnn.to_torch(o).double().reshape(-1, 384)[:N])
        for _ in range(5): f(X)
        ttnn.synchronize_device(dev); t = time.perf_counter()
        for _ in range(50): f(X)
        ttnn.synchronize_device(dev); ms = (time.perf_counter() - t) / 50 * 1e3
        res[name] = {"err": e, "ms": ms}
    except Exception as ex:
        res[name] = {"error": repr(ex)[:300]}
    print(name, res[name], flush=True)
json.dump(res, open(sys.argv[4], "w"), indent=1)
