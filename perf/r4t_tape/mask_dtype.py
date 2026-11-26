"""A bf16 pair stream times an fp32 [b, n, n, 1] mask (net6): raw ttnn.multiply vs `trunk._masked_copy`.

The raw arm is the failing control: in the fp32 trunk it wrote wrong values into ~0.4 % of the real entries,
different ones each call. `_masked_copy` casts the mask to its own dtype first and must be exact every call.
Exit 0 only if the fixed path is exact on every call; prints how many raw calls were wrong.
"""
import argparse, os, sys
sys.path.insert(0, os.getcwd())
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--real", type=int, default=48)
    ap.add_argument("--c", type=int, default=384)
    ap.add_argument("--calls", type=int, default=12)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio.rfd4 import trunk as TR
    dev = tt.get_device()
    g = torch.Generator().manual_seed(1)
    m = torch.zeros(1, a.n)
    m[:, :a.real] = 1
    pair = m[:, :, None] * m[:, None, :]
    up = lambda t, d: ttnn.from_torch(t.contiguous().float(), dtype=d, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                      device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mask = TR.trunk_masks(m, dev, dtype=ttnn.float32, residual_dtype=ttnn.float32)[3]
    z = up(torch.randn(1, a.n, a.n, a.c, generator=g) * pair[..., None], ttnn.bfloat16)
    want = ttnn.to_torch(z).float() * pair[..., None]
    bad = {"raw": 0, "fixed": 0}
    for i in range(a.calls):
        for arm, f in (("raw", lambda: ttnn.multiply(z, mask)), ("fixed", lambda: TR._masked_copy(z, mask, ttnn.bfloat16))):
            o = f()
            err = (ttnn.to_torch(o).float() - want).abs()
            nbad = int((err > 0).sum())
            bad[arm] += nbad > 0
            print(f"call {i} {arm}: dtype {o.dtype} wrong entries {nbad} max err {float(err.max()):.3e}", flush=True)
            ttnn.deallocate(o)
        churn = [up(torch.randn(1, a.n, a.n, a.c, generator=g), d) for d in (ttnn.float32, ttnn.bfloat16)]
        churn.append(ttnn.multiply(churn[0], mask))
        for t in churn:
            ttnn.deallocate(t)
    print(f"RESULT raw wrong in {bad['raw']}/{a.calls} calls, fixed wrong in {bad['fixed']}/{a.calls}")
    return 0 if bad["fixed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
