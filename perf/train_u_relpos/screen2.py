"""SCREEN part 2: the ttnn gather rate at 262,144 indices, and a cheaper expand.

ttnn.embedding refuses fp32 weights (embedding_device_operation.cpp:36 asserts BFLOAT16), so the
rate below is the bf16 one -- the FASTEST thing rung 2 could possibly be built on.
"""
import statistics, sys, time
import torch, ttnn
sys.path.insert(0, "/home/ttuser/.coworker/wt/train-u-relpos-ondevice")
from tt_bio.tenstorrent import get_device
from tt_bio.abodybuilder3_reference import single_and_pair_features, REL_POS_DIM

B, N, C_Z, EMBED = 4, 256, 132, 128
M = B * N * N


def timeit(fn, n=5, warm=2):
    dev = get_device()
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev); del r
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = fn(); ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0); del r
    return statistics.median(ts), min(ts)


def main():
    dev = get_device()
    w = torch.randn(C_Z, EMBED)
    # ---- the gather, at the only dtype the op accepts
    idx = torch.randint(0, C_Z, (1, M), dtype=torch.int32)
    idx_d = ttnn.from_torch(idx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    w_bf = ttnn.from_torch(w.contiguous().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16)
    g = lambda: ttnn.embedding(idx_d, w_bf, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    e = g(); ttnn.synchronize_device(dev)
    print("embedding out:", tuple(e.shape), e.dtype); del e
    t_g, t_g_min = timeit(g, n=5)
    print(f"GATHER embedding {M} idx x {EMBED} wide bf16: median {t_g*1e3:.2f} ms "
          f"min {t_g_min*1e3:.2f} ms = {M/t_g/1e6:.2f} Mindex/s")
    # smaller sizes, to see whether it is per-element or fixed-cost limited
    for m in (8192, 65536, 262144):
        i2 = ttnn.from_torch(torch.randint(0, C_Z, (1, m), dtype=torch.int32),
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
        t, _ = timeit(lambda: ttnn.embedding(i2, w_bf, layout=ttnn.TILE_LAYOUT,
                                             dtype=ttnn.bfloat16), n=5)
        print(f"  ladder m={m:7d}: {t*1e3:7.3f} ms  {m/t/1e6:6.2f} Mindex/s")

    # ---- a cheaper expand: fuse the abs into the subtract's packer
    g2 = torch.Generator().manual_seed(0)
    aatype = torch.randint(0, 21, (B, N), generator=g2)
    is_heavy = torch.zeros(B, N, dtype=torch.int64)
    residue_index = torch.zeros(B, N, dtype=torch.int64)
    for b in range(B):
        nl, nh = 110, 120
        is_heavy[b, nl:nl + nh] = 1
        residue_index[b, :nl] = torch.arange(nl)
        residue_index[b, nl:] = torch.arange(N - nl)
    _, pair_ref = single_and_pair_features(aatype, is_heavy, residue_index)

    ri = ttnn.from_torch(residue_index.float().reshape(B, 1, N, 1), layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.float32)
    ri_t = ttnn.from_torch(residue_index.float().reshape(B, 1, 1, N), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.float32)
    hv = ttnn.from_torch(is_heavy.float().reshape(B, 1, N, 1), layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.float32)
    hv_t = ttnn.from_torch(is_heavy.float().reshape(B, 1, 1, N), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.float32)
    ar = ttnn.from_torch(torch.arange(C_Z, dtype=torch.float32).reshape(1, 1, 1, C_Z),
                         layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    ABS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]

    def keys():
        rel = ttnn.clip(ttnn.subtract(ri_t, ri), -float(REL_POS_DIM), float(REL_POS_DIM))
        k_rel = ttnn.add(rel, float(REL_POS_DIM + 3))
        prod = ttnn.multiply(hv, hv_t)
        chain = ttnn.add(ttnn.multiply(prod, 2.0),
                         ttnn.multiply(ttnn.add(hv, -1.0), ttnn.add(hv_t, -1.0)))
        return ttnn.reshape(k_rel, [B, N, N, 1]), ttnn.reshape(chain, [B, N, N, 1])

    def expand_fused():
        k_rel, chain = keys()
        a = ttnn.subtract(ar, k_rel, activations=ABS)      # |ar - k|
        b_ = ttnn.subtract(ar, chain, activations=ABS)
        return ttnn.relu(ttnn.rsub(ttnn.add(a, b_), 2.0))

    out = expand_fused()
    got = ttnn.to_torch(out).reshape(B, N, N, C_Z)
    print("fused-expand exact:", torch.equal(got, pair_ref),
          "mismatched:", int((got != pair_ref).sum()))
    t_f, t_f_min = timeit(expand_fused, n=5)
    print(f"EXPAND fused: median {t_f*1e3:.2f} ms  min {t_f_min*1e3:.2f} ms")


if __name__ == "__main__":
    main()
