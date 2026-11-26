"""Can the card do a genuinely fp32 matmul? fp32 operand split into bf16 hi + bf16 lo (+ lo2), products summed in an
fp32 dest. Also ttnn.sum over a cancelling fp32 axis vs the same sum as ones @ (hi, lo). Float64 references."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio.rfd4 import trunk as TR
    dev = tt.get_device()
    ckc = TR.compute_kernel_config()
    g = torch.Generator().manual_seed(0)
    up = lambda t: ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa
    down = lambda t: ttnn.to_torch(t).double()  # noqa

    def split(t, k):
        parts, r = [], t.float()
        for _ in range(k):
            h = r.to(torch.bfloat16).float()
            parts.append(h)
            r = r - h
        return parts

    x = torch.randn(4096, 384, generator=g, dtype=torch.float64)
    w = torch.randn(384, 256, generator=g, dtype=torch.float64) / 384 ** 0.5
    xr, wr = x.float().double(), w.float().double()
    want = xr @ wr
    print("fp32 plain", rel(down(ttnn.matmul(up(x), up(w), compute_kernel_config=ckc)), want), flush=True)
    xb, wb = x.to(torch.bfloat16).double(), w.to(torch.bfloat16).double()
    print("bf16-valued fp32 operands (exact products?)", rel(down(ttnn.matmul(up(xb), up(wb), compute_kernel_config=ckc)), xb @ wb), flush=True)
    for k in (2, 3):
        xs, ws = split(x, k), split(w, k)
        acc = None
        for i in range(k):
            for j in range(k - i):
                p = ttnn.matmul(up(xs[i]), up(ws[j]), compute_kernel_config=ckc)
                acc = p if acc is None else ttnn.add(acc, p)
        print(f"split{k} on device", rel(down(acc), want), flush=True)
    # dW-shaped: K = 4096 rows, cancelling
    gco = torch.randn(4096, 256, generator=g, dtype=torch.float64)
    gco = gco - gco.mean(0, keepdim=True) * 0.999   # columns nearly sum to zero: bias-grad cancellation
    gr = gco.float().double()
    print("bias sum ttnn.sum", rel(down(ttnn.sum(up(gco), dim=0, keepdim=True)).reshape(-1), gr.sum(0)), flush=True)
    ones = torch.ones(32, 4096)
    hs = split(gco, 3)
    acc = None
    for h in hs:
        p = ttnn.matmul(up(ones), up(h), compute_kernel_config=ckc)
        acc = p if acc is None else ttnn.add(acc, p)
    print("bias sum ones@split3", rel(down(acc)[0], gr.sum(0)), flush=True)
    print("bias sum ones@fp32", rel(down(ttnn.matmul(up(ones), up(gco), compute_kernel_config=ckc))[0], gr.sum(0)), flush=True)
    xT = up(xr.t().contiguous())
    print("dW fp32 plain", rel(down(ttnn.matmul(xT, up(gco), compute_kernel_config=ckc)), xr.t() @ gr), flush=True)


if __name__ == "__main__":
    main()
