"""Which of the structure module s five awkward shapes ttnn will actually take.

Every question here is one the forward has to answer before a line of it is worth writing:
sub-tile widths (4, 8, 12, 16) and a 12-wide head axis are the whole module, and guessing
wrong costs a rewrite rather than a bug.
"""
import os, sys, traceback
import torch, ttnn

N = 128
H = 12

def check(name, fn):
    try:
        out = fn()
        print(f"OK   {name:52s} -> {tuple(out.shape) if hasattr(out, chr(115)+chr(104)+chr(97)+chr(112)+chr(101)) else out}")
        return out
    except Exception as e:
        print(f"FAIL {name:52s} -> {type(e).__name__}: {str(e)[:150]}")
        return None

def up(dev, *shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

def main():
    dev = ttnn.open_device(device_id=0)
    try:
        act = up(dev, 1, 1, N, 384)
        wq = up(dev, 1, H, 384, 16)
        q = check("matmul [1,1,n,384] x [1,h,384,16]", lambda: ttnn.matmul(act, wq))
        if q is None:
            actr = check("repeat act -> [1,h,n,384]", lambda: ttnn.repeat(act, ttnn.Shape([1, H, 1, 1])))
            if actr is not None:
                q = check("matmul [1,h,n,384] x [1,h,384,16]", lambda: ttnn.matmul(actr, wq))
        if q is None:
            q = up(dev, 1, H, N, 16)
        k = up(dev, 1, H, N, 16)
        check("matmul q x k^T -> [1,h,n,n]",
              lambda: ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2))))
        scal = up(dev, 1, 1, N, 1)
        pts = up(dev, 1, H, N, 4)
        check("multiply [1,1,n,1] * [1,h,n,4]", lambda: ttnn.multiply(scal, pts))
        check("multiply [1,h,n,4] * [1,1,n,1]", lambda: ttnn.multiply(pts, scal))
        scal32 = up(dev, 1, 1, N, 32)
        check("multiply [1,h,n,4] * [1,1,n,32] (no bcast on W)",
              lambda: ttnn.multiply(pts, ttnn.slice(scal32, [0, 0, 0, 0], [1, 1, N, 4])))
        attn = up(dev, 1, H, N, N)
        ap = check("permute [1,h,n,n] -> [1,n,h,n]", lambda: ttnn.permute(attn, (0, 2, 1, 3)))
        pair = up(dev, 1, N, N, 128)
        if ap is not None:
            check("matmul [1,n,h,n] x [1,n,n,128]", lambda: ttnn.matmul(ap, pair))
        a2d = up(dev, 1, N, N, H)
        check("permute [1,n,n,h] -> [1,h,n,n]", lambda: ttnn.permute(a2d, (0, 3, 1, 2)))
        wide = up(dev, 1, H, N, 384)
        check("sum dim=1 of [1,h,n,384]", lambda: ttnn.sum(wide, dim=1, keepdim=True))
        check("sum dim=-1 keepdim of [1,h,n,4]", lambda: ttnn.sum(pts, dim=-1, keepdim=True))
        check("sqrt", lambda: ttnn.sqrt(ttnn.abs(pts)))
        check("rsqrt", lambda: ttnn.rsqrt(ttnn.add(ttnn.abs(pts), 1.0)))
        check("slice [1,h,n,12] -> [1,h,n,4] at 4",
              lambda: ttnn.slice(up(dev, 1, H, N, 12), [0, 0, 0, 4], [1, H, N, 8]))
        check("concat 3x [1,h,n,4] on dim=-1",
              lambda: ttnn.concat([pts, pts, pts], dim=-1))
        r1 = up(dev, 1, 1, N, 1)
        check("matmul [1,1,n,384] x [1,1,384,1] (rigid column)",
              lambda: ttnn.matmul(act, up(dev, 1, 1, 384, 1)))
        check("subtract [1,h,n,1] - [1,h,1,n]",
              lambda: ttnn.subtract(up(dev, 1, H, N, 1), up(dev, 1, H, 1, N)))
        check("add [1,h,n,n] + [1,1,n,n]",
              lambda: ttnn.add(attn, up(dev, 1, 1, N, N)))
        check("add [1,h,n,16] + [1,h,1,16]", lambda: ttnn.add(up(dev, 1, H, N, 16), up(dev, 1, H, 1, 16)))
        check("softmax dim=-1 [1,h,n,n]", lambda: ttnn.softmax(attn, dim=-1))
    finally:
        ttnn.close_device(dev)
    return 0

if __name__ == "__main__":
    sys.exit(main())
