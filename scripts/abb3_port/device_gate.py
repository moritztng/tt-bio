#!/usr/bin/env python3
"""The device IPA against `tt_bio/abodybuilder3_reference.py` in float64.

The reference is scored against upstream and against `af2_reference` by
`scripts/abb3_port/reference_gate.py` and pinned by `tests/test_abodybuilder3_reference.py`, so a
number here is against something proven rather than against another approximation.

**Scored on unmasked residues only, and that is not a convenience.** A fully masked query residue
has `inf * (mask_ij - 1) = -1e7` on every logit in its row, so its softmax is decided entirely by
the terms that survive being added to -1e7. In float64 the logits' own O(1) terms survive and the
row is some non-uniform distribution; in float32 the relative spacing at 1e7 is 0.6, so they are
rounded away and the row is uniform. Both outputs are meaningless -- the loss masks those residues,
and upstream's own float32 run loses them the same way -- but the difference between two kinds of
meaningless dominates a max-abs over the whole tensor. Unmasked rows are unaffected: a masked
COLUMN still reaches `exp(-1e7 - max) = 0` in either precision, so it contributes nothing.

Two arms, and they answer different questions:

* **grad off** is the served forward. Its bar is set by the card, not by taste: a matmul keeps ~11
  mantissa bits, so the logits carry ~1e-3 absolute, and a softmax turns a logit error into a
  comparable RELATIVE error on each weight -- measured 9.6e-3 on the attention itself. The block
  output lands at 6.8e-3 and the bar is 2e-2, above which the cause is a layout or ordering bug
  rather than precision. For scale: upstream trained this model with
  `torch.set_float32_matmul_precision("medium")` (`stages/train.py:20`), i.e. bf16 matmul math at
  ~4e-3, so its own attention weights moved further than ours do.
* **grad on** must be bit-identical to grad off, because a taped op computes its value by calling
  the shipped one. Any difference at all means something re-implemented a forward.
* **the gradient** is scored against torch autograd on the float64 reference under a fixed random
  cotangent -- the whole block, not op by op, because the ops are already covered by
  `scripts/abb3_port/op_gradcheck.py` and what a block adds is the chance to wire two correct ops
  together wrongly. `head_weights` is the one parameter kept on the host, so its gradient is scored
  through `DeviceIPA.head_weights_grad`, which maps the uploaded constant's gradient back through
  the softplus it was built with.

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/device_gate.py [--tokens 64] [--batch 2]
"""
from __future__ import annotations

import argparse

import torch
import ttnn

from tt_bio import abodybuilder3_ops as ops
from tt_bio.abodybuilder3 import DeviceIPA, frame_from_quaternion, to_device_fp32
from tt_bio.abodybuilder3_reference import ABB3Config, InvariantPointAttention
from tt_bio.af2_reference import QuatAffine
from tt_bio.tenstorrent import get_device
from tt_bio.train import abodybuilder3_grad as grad

DT = torch.float64


def build(cfg: ABB3Config, seed: int):
    torch.manual_seed(seed)
    ref = InvariantPointAttention(cfg).to(DT)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.3)
    return ref


def inputs(cfg: ABB3Config, batch: int, n_tok: int, seed: int):
    g = torch.Generator().manual_seed(seed + 1)
    s = torch.randn(batch, n_tok, cfg.embed_dim, generator=g, dtype=DT) * 0.5
    z = torch.randn(batch, n_tok, n_tok, cfg.embed_dim, generator=g, dtype=DT) * 0.5
    quat = torch.randn(batch, n_tok, 4, generator=g, dtype=DT)
    trans = torch.randn(batch, n_tok, 3, generator=g, dtype=DT) * 8.0
    mask = torch.ones(batch, n_tok, dtype=DT)
    mask[-1, -3:] = 0.0
    return s, z, quat, trans, mask


def grads(ref, cfg, args, s, z, quat, trans, mask, want) -> int:
    """The block's backward against torch autograd in float64, under a fixed random cotangent."""
    # Zero at masked residues, because that is what a loss does and the alternative measures
    # nothing: a padded residue's attention row is decided by whatever survives being added to
    # -1e7, which is an O(1) term in float64 and rounding noise in float32, so a cotangent that
    # puts gradient there is asking two implementations to agree on meaningless output. The masked
    # residues still act as KEYS for unmasked queries and their attention weight there underflows
    # to zero in both precisions, so nothing is being skipped.
    cot = torch.randn(*want.shape, generator=torch.Generator().manual_seed(args.seed + 7), dtype=DT)
    cot = cot * mask.reshape(*mask.shape, 1)
    s_ref = s.clone().requires_grad_(True)
    z_ref = z.clone().requires_grad_(True)
    for p in ref.parameters():
        p.grad = None
    ref(s_ref, z_ref, QuatAffine(quat, trans), mask).backward(cot)

    grad.install()
    module = DeviceIPA({k: v.detach() for k, v in ref.state_dict().items()}, cfg,
                       to_device=lambda t: grad.param(to_device_fp32(t)))
    frame = frame_from_quaternion(quat, trans)
    sd, zd = grad.param(to_device_fp32(s)), grad.param(to_device_fp32(z))
    square = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1)
    out = module(sd, zd, frame, to_device_fp32(cfg.inf * (square - 1.0)))
    grad.backward([out], [to_device_fp32(cot)])

    # Weight gradients are compared after undoing the host-side padding and transpose the device
    # layout applies, so a wrong pad shows up as a wrong gradient rather than being hidden by it.
    h, c = cfg.no_heads_ipa, cfg.c_ipa
    checks = [
        ("d/ds", ttnn.to_torch(sd.grad), s_ref.grad),
        ("d/dz", ttnn.to_torch(zd.grad), z_ref.grad),
        ("d/d linear_q.weight",
         ttnn.to_torch(module.w_q.grad).t().reshape(h, 32, -1)[:, :c].reshape(h * c, -1),
         ref.linear_q.weight.grad),
        ("d/d linear_b[0].weight", ttnn.to_torch(module.w_b[0].grad).t(),
         ref.linear_b.weight.grad[:1]),
        ("d/d linear_out.bias", ttnn.to_torch(module.b_out.grad), ref.linear_out.bias.grad),
        ("d/d head_weights", module.head_weights_grad(ttnn.to_torch(module.head_weight.grad)),
         ref.head_weights.grad),
    ]
    bad = 0
    for name, got, ref_g in checks:
        got = got.double().reshape(ref_g.shape)
        sc = max(ref_g.abs().max().item(), 1e-30)
        err = (got - ref_g.double()).abs().max().item() / sc
        ok = err < 5e-2
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<24} rel {err:.2e}  (bar 5e-2)")
    grad.uninstall()
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = ABB3Config()
    ref = build(cfg, args.seed)
    s, z, quat, trans, mask = inputs(cfg, args.batch, args.tokens, args.seed)
    with torch.no_grad():
        want = ref(s, z, QuatAffine(quat, trans), mask)

    dev = get_device()
    try:
        weights = {k: v.detach() for k, v in ref.state_dict().items()}
        square = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1)
        sq_dev = to_device_fp32(cfg.inf * (square - 1.0))
        rows = []
        for taped in (False, True):
            grad.install() if taped else grad.uninstall()
            module = DeviceIPA(weights, cfg, to_device=to_device_fp32)
            frame = frame_from_quaternion(quat, trans)
            sd = to_device_fp32(s)
            zd = to_device_fp32(z)
            if taped:
                sd, zd = grad.param(sd), grad.param(zd)
            out = module(sd, zd, frame, sq_dev)
            got = ttnn.to_torch(out.value if taped else out)
            rows.append(("grad on" if taped else "grad off", got))
        keep = mask.reshape(args.batch, args.tokens, 1).expand_as(want) > 0
        scale = want[keep].abs().max().item()
        for name, got in rows:
            err = (got.double() - want)[keep].abs().max().item()
            print(f"  {name:<9} max abs {err:.3e}  rel {err / scale:.2e}  (|ref|max {scale:.3f},"
                  f" {int(keep[..., 0].sum())} of {args.batch * args.tokens} residues unmasked)")
        off, on = rows[0][1], rows[1][1]
        delta = (on - off).abs().max().item()
        print(f"  grad on vs grad off: {delta:.3e}  (must be 0, everywhere, masked included)")
        err = (rows[0][1].double() - want)[keep].abs().max().item() / scale
        bad = err >= 2e-2 or delta != 0.0
        bad += grads(ref, cfg, args, s, z, quat, trans, mask, want)
        print(f"\n{'PASS' if not bad else 'FAIL'}")
        return 0 if not bad else 1
    finally:
        grad.uninstall()
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
