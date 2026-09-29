"""SCREEN + rung-1 probe: build the 132-channel pair one-hot on the card, check it against the
untouched host reference, and measure the three rates the two rungs are decided on.

Run pinned: TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=... python3 screen.py
"""
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/home/ttuser/.coworker/wt/train-u-relpos-ondevice")

from tt_bio.tenstorrent import get_device
from tt_bio.abodybuilder3_reference import single_and_pair_features, REL_POS_DIM

B, N = 4, 256
C_Z, EMBED = 132, 128


def timeit(fn, n=5, warm=2):
    dev = get_device()
    for _ in range(warm):
        r = fn()
        ttnn.synchronize_device(dev)
        del r
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        del r
    return statistics.median(ts), min(ts)


def host_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    aatype = torch.randint(0, 21, (B, N), generator=g)
    is_heavy = torch.zeros(B, N, dtype=torch.int64)
    residue_index = torch.zeros(B, N, dtype=torch.int64)
    for b in range(B):
        nl = int(torch.randint(100, 120, (1,), generator=g))
        nh = int(torch.randint(110, 130, (1,), generator=g))
        is_heavy[b, nl:nl + nh] = 1
        residue_index[b, :nl] = torch.arange(nl)
        residue_index[b, nl:nl + nh] = torch.arange(nh)
        # padded tail continues past the last real residue, as SabdabFvs.host does
        residue_index[b, nl + nh:] = torch.arange(nh, nh + N - nl - nh)
    return aatype, is_heavy, residue_index


def main():
    dev = get_device()
    aatype, is_heavy, residue_index = host_inputs()
    single_ref, pair_ref = single_and_pair_features(aatype, is_heavy, residue_index)
    print(f"reference pair {tuple(pair_ref.shape)} {pair_ref.dtype} "
          f"{pair_ref.numel() * 4 / 2**20:.1f} MiB  sum={pair_ref.sum().item():.0f}")

    # ---- the upload this row is trying to delete
    t_up, t_up_min = timeit(lambda: ttnn.from_torch(pair_ref.contiguous().float(),
                                                    layout=ttnn.TILE_LAYOUT, device=dev,
                                                    dtype=ttnn.float32), n=5)
    mb = pair_ref.numel() * 4 / 2**20
    print(f"UPLOAD  (B,N,N,132) fp32: median {t_up*1e3:.1f} ms  min {t_up_min*1e3:.1f} ms  "
          f"= {mb/t_up:.0f} MiB/s")

    # ---- rung 1: expand on the card from the 8 KB index
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
    print("index bytes uploaded:", (residue_index.numel() + is_heavy.numel()) * 8, "B (int64 host)")

    def expand():
        rel = ttnn.subtract(ri_t, ri)                      # [B,1,N,N]  ri[j] - ri[i]
        rel = ttnn.clip(rel, -float(REL_POS_DIM), float(REL_POS_DIM))
        k_rel = ttnn.add(rel, float(REL_POS_DIM + 3))      # 3 .. 131
        prod = ttnn.multiply(hv, hv_t)
        chain = ttnn.add(ttnn.multiply(prod, 2.0),
                         ttnn.multiply(ttnn.add(hv, -1.0), ttnn.add(hv_t, -1.0)))
        k_rel = ttnn.reshape(k_rel, [B, N, N, 1])
        chain = ttnn.reshape(chain, [B, N, N, 1])
        oh_rel = ttnn.relu(ttnn.add(ttnn.abs(ttnn.subtract(ar, k_rel)), -1.0) * -1.0)
        oh_ch = ttnn.relu(ttnn.add(ttnn.abs(ttnn.subtract(ar, chain)), -1.0) * -1.0)
        return ttnn.add(oh_rel, oh_ch)

    dpair = expand()
    got = ttnn.to_torch(dpair).reshape(B, N, N, C_Z)
    exact = torch.equal(got, pair_ref)
    print(f"EXACT pair equality: {exact}   maxabs={(got - pair_ref).abs().max().item():.6g}  "
          f"mismatched={(got != pair_ref).sum().item()}")
    t_ex, t_ex_min = timeit(expand, n=5)
    print(f"EXPAND on device: median {t_ex*1e3:.1f} ms  min {t_ex_min*1e3:.1f} ms")

    # ---- SCREEN: the gather rate at 262,144 indices, and the matmul it would replace
    M = B * N * N
    w = torch.randn(C_Z, EMBED)
    w_d = ttnn.from_torch(w.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    flat = ttnn.reshape(dpair, [1, 1, M, C_Z])
    t_mm, t_mm_min = timeit(lambda: ttnn.matmul(flat, w_d), n=5)
    flops = 2 * M * C_Z * EMBED
    print(f"MATMUL ({M},{C_Z})x({C_Z},{EMBED}) fp32: median {t_mm*1e3:.1f} ms  "
          f"min {t_mm_min*1e3:.1f} ms = {flops/t_mm/1e9:.1f} GFLOP/s")

    idx = torch.randint(0, C_Z, (1, M), dtype=torch.int32)
    ok = True
    try:
        idx_d = ttnn.from_torch(idx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
        w_rm = ttnn.from_torch(w.contiguous(), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                               dtype=ttnn.float32)
        e = ttnn.embedding(idx_d, w_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        ttnn.synchronize_device(dev)
        print("embedding out:", e.shape, e.dtype)
        del e
        t_g, t_g_min = timeit(lambda: ttnn.embedding(idx_d, w_rm, layout=ttnn.TILE_LAYOUT,
                                                     dtype=ttnn.float32), n=5)
        print(f"GATHER ttnn.embedding {M} indices x {EMBED} wide fp32: median {t_g*1e3:.1f} ms  "
              f"min {t_g_min*1e3:.1f} ms = {M/t_g/1e6:.2f} Mindex/s")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("GATHER FAILED:", type(exc).__name__, str(exc)[:400])
    print("screen ok:", ok)


if __name__ == "__main__":
    main()
