#!/usr/bin/env python3
"""Every op in `tt_bio/abodybuilder3_ops.py`, forward and backward, against torch in float64.

A third column checks the claim the hook slot exists for: the same call with no tape installed,
on raw ttnn tensors, against the taped forward. It is 0.0 for every op because a taped op computes
its value by calling the shipped function, so "the training forward is the served forward" is a
property of the structure and not of two implementations agreeing.

An op layer is the one place in a port where a wrong sign is invisible: the forward stays right,
the loss still falls, and the model trains to something worse for no visible reason. So each op is
scored twice -- its value against torch, and its gradient against torch's own autograd under a
fixed random cotangent, which is the vector-Jacobian product the tape is supposed to compute.

The float64 side is the reference for both. The bars are the card's, not torch's: eltwise is
fp32-exact here and a matmul keeps ~11 mantissa bits, so a matmul-backed gradient is scored at
1e-2 relative and an eltwise one at 1e-5 (`scripts/abb3_port/precision_probe.py`).

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/op_gradcheck.py
"""
from __future__ import annotations

import torch
import ttnn

from tt_bio import abodybuilder3_ops as ops
from tt_bio.tenstorrent import get_device
from tt_bio.train import abodybuilder3_grad as grad

DT = torch.float64


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    scale = max(ref.abs().max().item(), 1e-30)
    return (got.double() - ref.double()).abs().max().item() / scale


def _to_dev(dev, x: torch.Tensor):
    return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)


CASES = []


def case(name: str, fwd_bar: float, bwd_bar: float | None = None):
    """`fwd_bar` and `bwd_bar` are separate because they are set by different hardware facts.

    An eltwise forward is fp32-exact at ~1e-7, but its backward may contain a reduction -- every
    broadcast gradient does -- and a reduction on this card rounds like a matmul at ~1e-3. A single
    bar per op would have to be the looser of the two and would stop checking the exact half.
    """
    def wrap(fn):
        CASES.append((name, fwd_bar, fwd_bar if bwd_bar is None else bwd_bar, fn))
        return fn
    return wrap


# Each case returns (torch_fn, shapes) where torch_fn takes torch tensors and the device side is
# built by the same call on ops.* -- one definition per op, so the two sides cannot drift.
@case("linear", 1e-2)
def _linear():
    def t(x, w, b):
        return x @ w + b
    def d(x, w, b):
        return ops.linear(x, w, b)
    return t, d, [(2, 64, 96), (96, 128), (128,)]


@case("matmul", 1e-2)
def _matmul():
    return (lambda a, b: a @ b), (lambda a, b: ops.matmul(a, b)), [(2, 3, 64, 32), (2, 3, 32, 64)]


@case("matmul transpose_b", 1e-2)
def _matmul_tb():
    return ((lambda a, b: a @ b.transpose(-1, -2)),
            (lambda a, b: ops.matmul(a, b, transpose_b=True)), [(2, 3, 64, 32), (2, 3, 64, 32)])


@case("matmul transpose_a", 1e-2)
def _matmul_ta():
    return ((lambda a, b: a.transpose(-1, -2) @ b),
            (lambda a, b: ops.matmul(a, b, transpose_a=True)), [(2, 3, 32, 64), (2, 3, 32, 64)])


@case("add broadcast", 1e-5, 1e-2)
def _add():
    return (lambda a, b: a + b), (lambda a, b: ops.add(a, b)), [(2, 4, 32, 64), (2, 1, 32, 1)]


@case("sub broadcast", 1e-5, 1e-2)
def _sub():
    return (lambda a, b: a - b), (lambda a, b: ops.sub(a, b)), [(2, 4, 32, 64), (2, 1, 32, 1)]


@case("mul broadcast", 1e-5, 1e-2)
def _mul():
    return (lambda a, b: a * b), (lambda a, b: ops.mul(a, b)), [(2, 4, 32, 64), (1, 4, 1, 1)]


@case("scale", 1e-5)
def _scale():
    return (lambda x: x * 0.375), (lambda x: ops.scale(x, 0.375)), [(2, 32, 64)]


@case("shift", 1e-5)
def _shift():
    return (lambda x: x + 1.25), (lambda x: ops.shift(x, 1.25)), [(2, 32, 64)]


@case("sum_last", 1e-3, 1e-5)
def _sum_last():
    return (lambda x: x.sum(-1, keepdim=True)), (lambda x: ops.sum_last(x)), [(2, 4, 32, 96)]


@case("sqrt_plus", 1e-5)
def _sqrt():
    return ((lambda x: (x.square() + 1e-7).sqrt()),
            (lambda x: ops.sqrt_plus(ops.mul(x, x), 1e-7)), [(2, 32, 64)])


@case("softplus", 1e-2, 1e-5)
def _softplus():
    return (torch.nn.functional.softplus), (lambda x: ops.softplus(x)), [(1, 32, 32)]


@case("relu", 1e-5)
def _relu():
    return (torch.relu), (lambda x: ops.relu(x)), [(2, 32, 64)]


# The reduction the point term sums its points with, over an axis a free reshape has grouped.
@case("sum_dim leading axis", 1e-3, 1e-5)
def _sum_dim():
    return ((lambda x: x.sum(1, keepdim=True)), (lambda x: ops.sum_dim(x, 1)), [(8, 4, 32, 64)])


@case("softmax", 1e-2, 1e-2)
def _softmax():
    return ((lambda x: torch.softmax(x, dim=-1)), (lambda x: ops.softmax(x)), [(2, 4, 32, 64)])


@case("layer_norm", 1e-2, 1e-2)
def _layer_norm():
    def t(x, g, b):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), g, b, 1e-5)
    return t, (lambda x, g, b: ops.layer_norm(x, g, b)), [(2, 32, 128), (128,), (128,)]


@case("reshape", 1e-5)
def _reshape():
    return ((lambda x: x.reshape(2, 32, 4, 32)), (lambda x: ops.reshape(x, [2, 32, 4, 32])),
            [(2, 32, 128)])


@case("permute", 1e-5)
def _permute():
    return ((lambda x: x.permute(0, 2, 1, 3)), (lambda x: ops.permute(x, [0, 2, 1, 3])),
            [(2, 32, 4, 32)])


# Not a mistake in the bar: a permute that moves the last two axes is the tile transpose and it
# rounds, at 4.95e-04 against 5.3e-08 for a permute of the leading axes. Nothing on the port's
# precision-critical path calls it -- the logits take k^T through `matmul(transpose_b=True)`, which
# rounds at 1.25e-03 regardless -- and it is scored here so that stays a decision and not an
# accident.
@case("transpose_last", 1e-3, 1e-3)
def _transpose():
    return ((lambda x: x.transpose(-2, -1)), (lambda x: ops.transpose_last(x)), [(2, 4, 32, 64)])


@case("slice_dim head axis", 1e-5)
def _slice():
    return ((lambda x: x[:, 4:8]), (lambda x: ops.slice_dim(x, 1, 4, 8)), [(2, 12, 32, 64)])


@case("concat last", 1e-5)
def _concat():
    return ((lambda a, b: torch.cat([a, b], dim=-1)), (lambda a, b: ops.concat([a, b], dim=-1)),
            [(2, 32, 64), (2, 32, 32)])


# The two-sided broadcast Alg. 22's point term is built from: [*, N, 1] against [*, 1, N] gives
# every pair of residues in one program.
@case("sub two-sided bcast", 1e-5, 1e-2)
def _pairwise():
    return ((lambda q, k: q - k), (lambda q, k: ops.sub(q, k)),
            [(2, 4, 64, 1), (2, 4, 1, 64)])


def run(dev) -> int:
    bad = 0
    print(f"{'op':<24} {'forward':>10} {'grads':>10} {'grad-off':>10}")
    for name, fwd_bar, bwd_bar, builder in CASES:
        t_fn, d_fn, shapes = builder()
        torch.manual_seed(len(name))
        refs = [torch.randn(*s, dtype=DT) for s in shapes]
        for r in refs:
            r.requires_grad_(True)
        out_ref = t_fn(*refs)
        cot = torch.randn(*out_ref.shape, dtype=DT)
        out_ref.backward(cot)

        raw = [_to_dev(dev, r.detach()) for r in refs]
        grad.uninstall()
        shipped = ttnn.to_torch(d_fn(*raw))          # no tape installed: the production op
        grad.install()
        args = [grad.param(_to_dev(dev, r.detach())) for r in refs]
        out = d_fn(*args)
        taped = ttnn.to_torch(out.value)
        off_err = (taped - shipped).abs().max().item()
        f_err = _rel(taped, out_ref.detach())
        grad.backward([out], [_to_dev(dev, cot)])
        g_err = 0.0
        for r, a in zip(refs, args):
            if a.grad is None:
                g_err = float("inf")
                continue
            g_err = max(g_err, _rel(ttnn.to_torch(a.grad), r.grad))
        # grad-off against grad-on is bit-exact or the hook is re-implementing a forward.
        ok = f_err <= fwd_bar and g_err <= bwd_bar and off_err == 0.0
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:<19} {f_err:>10.2e} {g_err:>10.2e}"
              f" {off_err:>10.2e}  (bars {fwd_bar:.0e} / {bwd_bar:.0e} / 0)")
    return bad


def main() -> int:
    dev = get_device()
    try:
        bad = run(dev)
    finally:
        grad.uninstall()
        ttnn.close_device(dev)
    print(f"\n{'PASS' if bad == 0 else f'FAIL: {bad} ops'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
