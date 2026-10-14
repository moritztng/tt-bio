"""ws_resid parity against float64, per (ws_l1acc, resid, weight dtype) arm, at one pair row block.

rel = rms(out - ref) / rms(t), where ref = z + t in float64 and t is the transition. Usage (one chip):
    TT_VISIBLE_DEVICES=<chip> python wsr_parity.py [c_z] [rows] [z_scale] [n_tokens]
"""
import sys
import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import pair_swiglu_ws as W

c, rows, zs = (int(sys.argv[1]) if len(sys.argv) > 1 else 256, int(sys.argv[2]) if len(sys.argv) > 2 else 64,
               float(sys.argv[3]) if len(sys.argv) > 3 else 8.0)
n = int(sys.argv[4]) if len(sys.argv) > 4 else 512
h = 4 * c
dev = T.get_device()
g = torch.Generator().manual_seed(0)
xn = torch.randn(1, rows, n, c, generator=g).bfloat16()
z = (zs * torch.randn(1, rows, n, c, generator=g)).bfloat16()
w1, w2 = [(torch.randn(c, h, generator=g) / c ** 0.5).bfloat16() for _ in range(2)]
w3 = (torch.randn(h, c, generator=g) / h ** 0.5).bfloat16()
d = lambda t: t.double()  # noqa: E731
t_ref = (torch.nn.functional.silu(d(xn) @ d(w1)) * (d(xn) @ d(w2))) @ d(w3)
ref = d(z) + t_ref
# Any bf16 pair output carries this much: ref itself rounded to bf16 once (the shipped add_ rounds t and z + t).
floor = ((d(ref.bfloat16()) - ref).pow(2).mean().sqrt() / t_ref.pow(2).mean().sqrt()).item()
print(f"c {c} rows {rows} n {n} z_scale {zs}: one-rounding floor rel_rms {floor:.5f}", flush=True)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
for wdt in (ttnn.bfloat16, ttnn.bfloat8_b):
    parts = W.weights(w1.float(), w2.float(), w3.float(), dev, wdt)
    fid = ttnn.MathFidelity.HiFi2 if wdt == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi4
    for l1acc in (False, True):
        for resid in (False, True):
            xt = up(xn)
            if not all(W.eligible(xt, a, b, l1acc=l1acc, resid=up(z) if resid else None) for a, b in parts):
                print(f"w={wdt} parts={len(parts)} l1acc={l1acc} resid={resid}: not eligible", flush=True)
                continue
            if resid:
                zt = up(z)
                for a, b in parts:
                    W.pair_swiglu(xt, a, b, fid, l1acc=l1acc, resid=zt)
                out = d(ttnn.to_torch(zt))
            else:
                acc = None
                for a, b in parts:
                    o = d(ttnn.to_torch(W.pair_swiglu(xt, a, b, fid, l1acc=l1acc)))
                    acc = o if acc is None else acc + o
                out = d((d(z) + d(acc.bfloat16())).bfloat16())  # the shipped add_: bf16 t, bf16 z + t
            assert torch.isfinite(out).all(), "non-finite output"
            e = out - ref
            rel = (e.pow(2).mean().sqrt() / t_ref.pow(2).mean().sqrt()).item()
            print(f"w={wdt} parts={len(parts)} l1acc={l1acc} resid={resid}: rel_rms {rel:.5f} "
                  f"max|e| {e.abs().max().item():.4f} n>|0.5| {(e.abs() > 0.5).sum().item()}", flush=True)
