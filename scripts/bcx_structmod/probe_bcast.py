"""Does ttnn broadcast these operands the way numpy does, in VALUE and not just in shape?

The layout probe only showed that none of these calls raises. A silently wrong broadcast on
`[1,1,n,1] x [1,h,n,p]` is exactly the shape of bug that leaves layer 0 of the structure module
correct (identity frame) and every later layer wrong.
"""
import numpy as np
import torch
import ttnn

N, H, P = 64, 12, 4


def up(dev, a):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a, np.float32)),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)


def cmp(name, got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    print(f"{'OK  ' if rel < 1e-5 else 'WRONG'} {name:44s} rel={rel:.3e} "
          f"shape={got.shape} vs {want.shape}")


def main():
    rng = np.random.default_rng(0)
    dev = ttnn.open_device(device_id=0)
    try:
        s = rng.standard_normal((1, 1, N, 1))
        p = rng.standard_normal((1, H, N, P))
        cmp("[1,1,n,1] * [1,h,n,p]", ttnn.to_torch(ttnn.multiply(up(dev, s), up(dev, p))), s * p)
        cmp("[1,h,n,p] * [1,1,n,1]", ttnn.to_torch(ttnn.multiply(up(dev, p), up(dev, s))), p * s)
        cmp("[1,h,n,p] + [1,1,n,1]", ttnn.to_torch(ttnn.add(up(dev, p), up(dev, s))), p + s)
        cmp("[1,h,n,p] - [1,1,n,1]", ttnn.to_torch(ttnn.subtract(up(dev, p), up(dev, s))), p - s)

        hw = rng.standard_normal((1, H, 1, 1))
        d2 = rng.standard_normal((1, H, N, N))
        cmp("[1,h,n,n] * [1,h,1,1]", ttnn.to_torch(ttnn.multiply(up(dev, d2), up(dev, hw))),
            d2 * hw)

        qq = rng.standard_normal((1, H, N, 1))
        cmp("[1,h,n,1] + transpose -> [1,h,1,n]",
            ttnn.to_torch(ttnn.add(up(dev, qq), ttnn.transpose(up(dev, qq), -2, -1))),
            qq + qq.transpose(0, 1, 3, 2))

        b = rng.standard_normal((1, H, 1, P))
        cmp("[1,h,n,p] + [1,h,1,p]", ttnn.to_torch(ttnn.add(up(dev, p), up(dev, b))), p + b)

        act = rng.standard_normal((1, 1, N, 384))
        w = rng.standard_normal((1, H, 384, 16))
        cmp("matmul [1,1,n,384] x [1,h,384,16]",
            ttnn.to_torch(ttnn.matmul(up(dev, act), up(dev, w))), act @ w)

        x = rng.standard_normal((1, H, N, P))
        cmp("concat 3 on dim=-1", ttnn.to_torch(
            ttnn.concat([up(dev, x), up(dev, x + 1), up(dev, x + 2)], dim=-1)),
            np.concatenate([x, x + 1, x + 2], axis=-1))

        y = rng.standard_normal((1, H, N, 384))
        cmp("sum dim=1 keepdim", ttnn.to_torch(ttnn.sum(up(dev, y), dim=1, keepdim=True)),
            y.sum(axis=1, keepdims=True))
        cmp("sum dim=-1 keepdim", ttnn.to_torch(ttnn.sum(up(dev, x), dim=-1, keepdim=True)),
            x.sum(axis=-1, keepdims=True))

        a2d = rng.standard_normal((1, N, N, H))
        cmp("permute [1,n,n,h] -> [1,h,n,n]",
            ttnn.to_torch(ttnn.permute(up(dev, a2d), (0, 3, 1, 2))),
            a2d.transpose(0, 3, 1, 2))
        at = rng.standard_normal((1, H, N, N))
        pair = rng.standard_normal((1, N, N, 128))
        cmp("permute+matmul pair attention",
            ttnn.to_torch(ttnn.matmul(ttnn.permute(up(dev, at), (0, 2, 1, 3)), up(dev, pair))),
            np.einsum("bhij,bijc->bihc", at, pair))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
