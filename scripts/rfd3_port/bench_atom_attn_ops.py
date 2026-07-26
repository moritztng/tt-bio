"""Per-op cost attribution for the RFD3 atom-attention inner loop.

p5 shipped a sparse-QK / dense-reduction hybrid; p6 and p7 both assumed the
residual cost is the intentionally retained dense fp32 softmax and dense AV.
Nobody measured the split. This replays the exact op sequence of
GatedCrossAttention's sparse_qk branch (tt_bio/rfd3.py) at production shape and
times every op individually, so the residual 22.25x gap gets attributed to a
real op rather than an assumed one.

Run: TT_VISIBLE_DEVICES=<n> TT_BIO_LEASE_HOLDER=... python3 bench_atom_attn_ops.py [L]
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("TT_BIO_ROOT", "/home/ttuser/tt-bio-dev"))
import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 3359
N_HEAD, HEAD_DIM, N_KEYS, C_PAIR = 4, 32, 128, 16
C_MODEL = N_HEAD * HEAD_DIM
N_BLOCKS = 6  # 3 atom encoder + 3 atom decoder blocks per denoiser step

dev = get_device()
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=True,
) if False else ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4)
dt = ttnn.bfloat16

print(f"arch={dev.arch()} L={L} heads={N_HEAD} hd={HEAD_DIM} keys={N_KEYS}", flush=True)

# ---- build the same host-side neighbour structure the model produces -------
g = torch.Generator().manual_seed(0)
# a sequence band plus scattered kNN fill, matching _create_attention_indices
idx = torch.empty(1, L, N_KEYS, dtype=torch.long)
band = min(70, N_KEYS)
for i in range(L):
    lo = max(0, min(L - band, i - band // 2))
    b = torch.arange(lo, lo + band)
    rest = torch.randperm(L, generator=g)[: N_KEYS - band]
    idx[0, i] = torch.sort(torch.cat([b, rest]))[0]

attn_idx = idx.unsqueeze(1).expand(1, N_HEAD, L, N_KEYS).to(torch.int32).contiguous()
kv_idx = idx.to(torch.int32).reshape(1, -1)


def tt(x, dtype=dt, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, layout=layout, device=dev, dtype=dtype)


p_sparse = tt(torch.randn(1, L, N_KEYS, C_PAIR))
b_w = tt(torch.randn(C_PAIR, N_HEAD) * 0.05)
qq = tt(torch.randn(1, N_HEAD, L, HEAD_DIM))
vv = tt(torch.randn(1, N_HEAD, L, HEAD_DIM))
kk_src = tt(torch.randn(L, C_MODEL))
kv_idx_dev = ttnn.from_torch(kv_idx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
attn_idx_dev = ttnn.from_torch(attn_idx, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.uint32)

RESULTS = []


def timed(name, fn, reps=3):
    fn()  # warmup / compile
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1000.0 / reps
    RESULTS.append((name, ms))
    print(f"  {name:<34s} {ms:9.3f} ms", flush=True)
    return out


print("\n--- sparse-QK stage (p5 fix) ---", flush=True)

dense_scores = ttnn.full((1, N_HEAD, L, L), -1e4 * HEAD_DIM ** 0.5, dtype=dt,
                         layout=ttnn.TILE_LAYOUT, device=dev)
dense_bias = ttnn.full((1, N_HEAD, L, L), 0.0, dtype=dt,
                       layout=ttnn.TILE_LAYOUT, device=dev)
timed("ttnn.full dense LxL (x2, /stack)",
      lambda: ttnn.full((1, N_HEAD, L, L), 0.0, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev))


def gather_k():
    k = ttnn.to_layout(kk_src, ttnn.ROW_MAJOR_LAYOUT)
    k = ttnn.embedding(kv_idx_dev, k, layout=ttnn.ROW_MAJOR_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.reshape(k, (1, L, N_KEYS, N_HEAD, HEAD_DIM))
    return ttnn.to_layout(ttnn.permute(k, (0, 3, 1, 2, 4)), ttnn.TILE_LAYOUT)


kk = timed("K gather (embedding+permute+tile)", gather_k)


def pair_bias_fn():
    pb = ttnn.linear(p_sparse, b_w, compute_kernel_config=ckc, dtype=dt)
    return ttnn.permute(pb, (0, 3, 1, 2))


pair_bias = timed("pair_bias linear+permute (Lx128)", pair_bias_fn)

kkT = ttnn.permute(kk, (0, 1, 2, 4, 3))
qq5 = ttnn.unsqueeze(qq, 3)


def qk_fn():
    s = ttnn.matmul(qq5, kkT, compute_kernel_config=ckc)
    return ttnn.squeeze(s, 3)


scores_sp = timed("sparse QK batched matmul (1x32@32x128)", qk_fn)

scores = timed("ttnn.scatter scores -> dense LxL",
               lambda: ttnn.scatter(dense_scores, 3, attn_idx_dev, scores_sp))
bias = timed("ttnn.scatter bias   -> dense LxL",
             lambda: ttnn.scatter(dense_bias, 3, attn_idx_dev, pair_bias))

print("\n--- retained dense reductions (p6/p7 assumed bottleneck) ---", flush=True)

scores_f = timed("typecast scores bf16->fp32",
                 lambda: ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config()))
scores_f2 = timed("multiply fp32 by 1/sqrt(hd)",
                  lambda: ttnn.multiply(scores_f, HEAD_DIM ** -0.5))
bias_f = timed("typecast bias bf16->fp32",
               lambda: ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config()))
scores_f3 = timed("add scores+bias fp32", lambda: ttnn.add(scores_f2, bias_f))
attention = timed("ttnn.softmax fp32 dense LxL",
                  lambda: ttnn.softmax(scores_f3, dim=-1))
attn_bf = timed("typecast attention fp32->bf16",
                lambda: ttnn.typecast(attention, dt, memory_config=attention.memory_config()))
timed("dense AV matmul LxL @ Lx32",
      lambda: ttnn.matmul(attn_bf, vv, compute_kernel_config=ckc, dtype=dt))

total = sum(ms for _, ms in RESULTS)
per_block = total - RESULTS[0][1]  # ttnn.full is per-stack, not per-block
sparse_stage = sum(ms for n, ms in RESULTS if n.startswith(("K gather", "pair_bias", "sparse QK", "ttnn.scatter")))
dense_stage = sum(ms for n, ms in RESULTS if n.startswith(("typecast", "multiply", "add ", "ttnn.softmax", "dense AV")))

print(f"\n{'=' * 62}")
print(f"per-block attn total            {per_block:9.3f} ms")
print(f"  sparse-QK stage               {sparse_stage:9.3f} ms  ({100*sparse_stage/per_block:5.1f}%)")
print(f"    of which ttnn.scatter       {sum(ms for n, ms in RESULTS if n.startswith('ttnn.scatter')):9.3f} ms")
print(f"  dense softmax/AV stage        {dense_stage:9.3f} ms  ({100*dense_stage/per_block:5.1f}%)")
print(f"x{N_BLOCKS} blocks/step               {per_block * N_BLOCKS:9.3f} ms")
print(f"{'=' * 62}", flush=True)
