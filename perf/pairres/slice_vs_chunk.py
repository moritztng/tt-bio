"""Does slicing x[:, s:e] one block at a time compute what ttnn.chunk did?

71518905d replaced `ttnn.chunk` + `ttnn.concat` in ESMFold2's tiled TransitionLayer with lazy
slices assembled by `_acc_concat`, freeing each slice after use. At 1024 tokens on Wormhole,
where no refusal fires, that one commit moved the esmfold2 fold (main is deterministic run to
run). This checks, on the shapes that path sees, whether the blocks themselves differ, whether a
freed slice corrupts its parent, and whether a row-local body gives different bytes either way.
"""
import sys

import torch
import ttnn

from tt_bio import tenstorrent as T


def body(x, w, g, b):
    xn = ttnn.layer_norm(x, weight=g, bias=b, epsilon=1e-5)
    y = ttnn.linear(xn, w)
    ttnn.deallocate(xn)
    return y


def main():
    dev = T.get_device()
    torch.manual_seed(0)
    bad = 0
    for shape, chunk in [((1, 1024, 1024, 256), 64), ((1, 1024, 384), 256),
                         ((5, 1024, 384), 256), ((1, 1040, 1040, 256), 64)]:
        L, c = shape[1], shape[-1]
        xt = torch.randn(shape).bfloat16()
        x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        w = ttnn.from_torch(torch.randn(c, c).bfloat16() / c ** 0.5, layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        g = ttnn.from_torch(torch.rand(1, c).bfloat16() + 0.5, layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.randn(1, c).bfloat16(), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        n = -(-L // chunk)
        parts = ttnn.chunk(x, n, dim=1)
        sizes_old = [int(p.shape[1]) for p in parts]
        sizes_new = [min(s + chunk, L) - s for s in range(0, L, chunk)]
        print(f"{shape} chunk {chunk}: ttnn.chunk sizes {sizes_old[:3]}..x{len(sizes_old)} "
              f"| slice sizes {sizes_new[:3]}..x{len(sizes_new)}")
        # the blocks themselves, against the host bytes
        off = 0
        for p in parts:
            h = ttnn.to_torch(p)
            ref = xt[:, off:off + h.shape[1]]
            if not torch.equal(h, ref):
                print(f"  ttnn.chunk block at {off} != host bytes"); bad += 1
            off += h.shape[1]
        for s in range(0, L, chunk):
            p = x[:, s:min(s + chunk, L)]
            if not torch.equal(ttnn.to_torch(p), xt[:, s:min(s + chunk, L)]):
                print(f"  slice at {s} != host bytes"); bad += 1
            ttnn.deallocate(p)
        if not torch.equal(ttnn.to_torch(x), xt):
            print("  x CHANGED after freeing its slices"); bad += 1
        old = ttnn.to_torch(ttnn.concat([body(p, w, g, b) for p in parts], dim=1))
        for p in parts:
            ttnn.deallocate(p)
        blocks = []
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            p = x[:, s:e]
            blocks.append(body(p, w, g, b))
            if (s, e) != (0, L):
                ttnn.deallocate(p)
        new = ttnn.to_torch(T._acc_concat(blocks, 1, False))
        whole = ttnn.to_torch(body(x, w, g, b))
        f64 = torch.nn.functional.layer_norm(xt.double(), (c,), ttnn.to_torch(g).double()[0],
                                             ttnn.to_torch(b).double()[0], 1e-5) \
            @ ttnn.to_torch(w).double()
        for name, a in (("old(chunk)", old), ("new(slice)", new), ("whole", whole)):
            err = (a.double() - f64).abs().max().item()
            print(f"  {name}: vs old identical {torch.equal(a, old)} | max|err| vs f64 {err:.4g}")
        if not torch.equal(old, new):
            d = (old.double() - new.double()).abs()
            print(f"  old vs new: {int((d > 0).sum())} of {d.numel()} elements differ, max {d.max().item():.4g}")
            bad += 1
        ttnn.deallocate(x)
    print("RESULT", "DIFFER" if bad else "IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
