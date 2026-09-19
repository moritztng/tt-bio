"""Cheapest exact expand: one broadcast equality per one-hot."""
import statistics, sys, time
import torch, ttnn
sys.path.insert(0, "/home/ttuser/.coworker/wt/train-u-relpos-ondevice")
from tt_bio.tenstorrent import get_device
from tt_bio.abodybuilder3_reference import single_and_pair_features, REL_POS_DIM

B, N, C_Z = 4, 256, 132


def timeit(fn, n=7, warm=2):
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
    g2 = torch.Generator().manual_seed(0)
    aatype = torch.randint(0, 21, (B, N), generator=g2)
    is_heavy = torch.zeros(B, N, dtype=torch.int64)
    residue_index = torch.zeros(B, N, dtype=torch.int64)
    for b in range(B):
        nl, nh = 100 + 4 * b, 118 - 3 * b
        is_heavy[b, nl:nl + nh] = 1
        residue_index[b, :nl] = torch.arange(nl)
        residue_index[b, nl:nl + nh] = torch.arange(nh)
        residue_index[b, nl + nh:] = torch.arange(nh, nh + N - nl - nh)
    _, pair_ref = single_and_pair_features(aatype, is_heavy, residue_index)

    up = lambda t, shape: ttnn.from_torch(t.float().reshape(*shape), layout=ttnn.TILE_LAYOUT,
                                          device=dev, dtype=ttnn.float32)
    ri, ri_t = up(residue_index, (B, 1, N, 1)), up(residue_index, (B, 1, 1, N))
    hv, hv_t = up(is_heavy, (B, 1, N, 1)), up(is_heavy, (B, 1, 1, N))
    ar = up(torch.arange(C_Z), (1, 1, 1, C_Z))

    def keys():
        rel = ttnn.clip(ttnn.subtract(ri_t, ri), -float(REL_POS_DIM), float(REL_POS_DIM))
        k_rel = ttnn.add(rel, float(REL_POS_DIM + 3))
        chain = ttnn.add(ttnn.multiply(ttnn.multiply(hv, hv_t), 2.0),
                         ttnn.multiply(ttnn.add(hv, -1.0), ttnn.add(hv_t, -1.0)))
        return ttnn.reshape(k_rel, [B, N, N, 1]), ttnn.reshape(chain, [B, N, N, 1])

    variants = {}

    def v_eq():
        k_rel, chain = keys()
        return ttnn.add(ttnn.eq(ar, k_rel), ttnn.eq(ar, chain))
    variants["eq"] = v_eq

    ABS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]

    def v_absfused():
        k_rel, chain = keys()
        a = ttnn.subtract(ar, k_rel, activations=ABS)
        b_ = ttnn.subtract(ar, chain, activations=ABS)
        return ttnn.add(ttnn.relu(ttnn.rsub(a, 1.0)), ttnn.relu(ttnn.rsub(b_, 1.0)))
    variants["abs-fused"] = v_absfused

    def v_orig():
        k_rel, chain = keys()
        oh1 = ttnn.relu(ttnn.multiply(ttnn.add(ttnn.abs(ttnn.subtract(ar, k_rel)), -1.0), -1.0))
        oh2 = ttnn.relu(ttnn.multiply(ttnn.add(ttnn.abs(ttnn.subtract(ar, chain)), -1.0), -1.0))
        return ttnn.add(oh1, oh2)
    variants["orig"] = v_orig

    for name, fn in variants.items():
        try:
            out = fn()
            got = ttnn.to_torch(out).reshape(B, N, N, C_Z)
            ok = torch.equal(got, pair_ref)
            t, tmin = timeit(fn)
            print(f"{name:10s} exact={ok} mismatched={int((got != pair_ref).sum()):8d} "
                  f"dtype={got.dtype} median={t*1e3:7.2f} ms min={tmin*1e3:7.2f} ms")
            del out
        except Exception as exc:  # noqa: BLE001
            print(f"{name:10s} FAILED {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
