"""Is dense QK cheaper than p5's sparse batched QK, and bit-exact against it?

p5 replaced the dense atom QK matmul with a per-row gathered [1,32]@[32,128]
batched matmul. That conflated two costs: the dense *pair-bias projection* (which
really is expensive, it needs a dense [L,L,16] tensor) and the dense *QK matmul*
(which is one large efficient matmul). This measures both formulations and checks
value equality at the neighbour positions.

QK reduces over head_dim=32 -- a single tile deep -- so the dot-product tree is
independent of the M/N tiling. Dense and gathered QK should be bit-identical.
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

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4)
dt = ttnn.bfloat16
print(f"arch={dev.arch()} L={L}", flush=True)

g = torch.Generator().manual_seed(0)
idx = torch.empty(1, L, N_KEYS, dtype=torch.long)
band = min(70, N_KEYS)
for i in range(L):
    lo = max(0, min(L - band, i - band // 2))
    rest = torch.randperm(L, generator=g)[: N_KEYS - band]
    idx[0, i] = torch.sort(torch.cat([torch.arange(lo, lo + band), rest]))[0]
attn_idx = idx.unsqueeze(1).expand(1, N_HEAD, L, N_KEYS).to(torch.int32).contiguous()
kv_idx = idx.to(torch.int32).reshape(1, -1)

q_h = torch.randn(1, N_HEAD, L, HEAD_DIM, generator=g)
k_h = torch.randn(1, N_HEAD, L, HEAD_DIM, generator=g)


def tt(x, dtype=dt, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, layout=layout, device=dev, dtype=dtype)


qq, kkh = tt(q_h), tt(k_h)
# K in the [L, n_head*head_dim] layout the model gathers from
kk_src = tt(k_h.permute(0, 2, 1, 3).reshape(L, C_MODEL))
kv_idx_dev = ttnn.from_torch(kv_idx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
attn_idx_dev = ttnn.from_torch(attn_idx, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.uint32)


def timed(name, fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(dev)
    print(f"  {name:<40s} {(time.perf_counter()-t0)*1000/reps:9.3f} ms", flush=True)
    return out


print("\n--- formulation A: p5 sparse gathered QK ---", flush=True)


def sparse_qk():
    k = ttnn.to_layout(kk_src, ttnn.ROW_MAJOR_LAYOUT)
    k = ttnn.embedding(kv_idx_dev, k, layout=ttnn.ROW_MAJOR_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.reshape(k, (1, L, N_KEYS, N_HEAD, HEAD_DIM))
    k = ttnn.to_layout(ttnn.permute(k, (0, 3, 1, 2, 4)), ttnn.TILE_LAYOUT)
    s = ttnn.matmul(ttnn.unsqueeze(qq, 3), ttnn.permute(k, (0, 1, 2, 4, 3)),
                    compute_kernel_config=ckc)
    return ttnn.squeeze(s, 3)


sp = timed("gather K + batched 1x32@32x128 QK", sparse_qk)
dense_scores = ttnn.full((1, N_HEAD, L, L), -1e4 * HEAD_DIM ** 0.5, dtype=dt,
                         layout=ttnn.TILE_LAYOUT, device=dev)
sp_scat = timed("  + ttnn.scatter into dense LxL",
                lambda: ttnn.scatter(dense_scores, 3, attn_idx_dev, sp))

print("\n--- formulation B: dense QK matmul (reference path) ---", flush=True)
de = timed("dense QK  [L,32] @ [32,L]",
           lambda: ttnn.matmul(qq, ttnn.permute(kkh, (0, 1, 3, 2)), compute_kernel_config=ckc))

print("\n--- bit-exactness of scores at neighbour positions ---", flush=True)
sp_t = ttnn.to_torch(sp).float()          # [1,H,L,K]
de_t = ttnn.to_torch(de).float()          # [1,H,L,L]
gathered = torch.gather(de_t, 3, idx.unsqueeze(1).expand(1, N_HEAD, L, N_KEYS))
diff = (gathered - sp_t).abs()
print(f"  maxabs(dense_gathered - sparse) = {diff.max().item():.10g}")
print(f"  n_mismatch                      = {(diff > 0).sum().item()} / {diff.numel()}")
