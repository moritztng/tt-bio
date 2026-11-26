"""Host: PairUpdate / TrunkBlock float64 reference with tf32 rounding at matmul inputs (what the card's fp32 matmul
does: SrcA/SrcB hold 10 mantissa bits), vs plain float64. Asks whether D49(2)'s <= 7e-4 PU forward is reachable with
fp32-in-tf32-unpack matmuls at all. Diagnostic only."""
import sys
from pathlib import Path
import torch
from torch.overrides import TorchFunctionMode

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import rfd4_reference as REF

MM = {torch.matmul, torch.Tensor.__matmul__, torch.einsum, torch.bmm, torch.nn.functional.linear}


def tf32(t, trunc):
    if not t.is_floating_point():
        return t
    i = t.float().view(torch.int32)
    if not trunc:
        i = i + 0x1000 + ((i >> 13) & 1) - 1   # round half to even at bit 13
    return (i & ~0x1FFF).view(torch.float32).to(t.dtype)


class Round(TorchFunctionMode):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in MM:
            args = tuple(self.f(a) if torch.is_tensor(a) else a for a in args)
        return func(*args, **kwargs)


def rel(a, b, m):
    return float(((a - b) * m).norm() / (b * m).norm())


n, real = 64, 48
x, z, c, mask = REF.block_inputs(1, n, 768, 384, 256, lengths=[real], seed=1)
x, z, c, mask = (t.double() for t in (x, z, c, mask))
pm = mask[:, :, None] * mask[:, None, :]
x, z = x * mask[..., None], z * pm[..., None]
M = pm[..., None]
w = {k: v.double() for k, v in REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu").items()}
f32 = lambda t: t.float().double()  # noqa: E731
wf = {k: f32(v) for k, v in w.items()}
xf, zf = f32(x), f32(z)
with torch.no_grad():
    ref = {"whole": REF.pair_update(wf, "pu", xf, zf, mask),
           "tri_out": REF.tri_mult(wf, "pu.tri_mult_out", zf, pm, True) * M,
           "tri_in": REF.tri_mult(wf, "pu.tri_mult_in", zf, pm, False) * M,
           "transition": REF.pair_transition(wf, "pu.transition_out", zf, pm)}
    for name, f in (("tf32-trunc", lambda t: tf32(t, True)), ("tf32-rne", lambda t: tf32(t, False)),
                    ("bf16", REF.bf16)):
        with Round(f):
            got = {"whole": REF.pair_update(wf, "pu", xf, zf, mask),
                   "tri_out": REF.tri_mult(wf, "pu.tri_mult_out", zf, pm, True) * M,
                   "tri_in": REF.tri_mult(wf, "pu.tri_mult_in", zf, pm, False) * M,
                   "transition": REF.pair_transition(wf, "pu.transition_out", zf, pm)}
        print(name, " ".join(f"{k} {rel(got[k], ref[k], M):.3e}" for k in ref), flush=True)
    bw = {k: f32(v.double()) for k, v in REF.block_weights(768, 384, 256, n_heads=12).items()}
    cf = f32(c)
    r0 = REF.trunk_block(bw, xf, zf, cf, mask, n_heads=12)
    for name, f in (("tf32-trunc", lambda t: tf32(t, True)), ("tf32-rne", lambda t: tf32(t, False))):
        with Round(f):
            r1 = REF.trunk_block(bw, xf, zf, cf, mask, n_heads=12)
        print("trunk_block", name, f"{rel(r1, r0, mask[..., None]):.3e}", flush=True)
