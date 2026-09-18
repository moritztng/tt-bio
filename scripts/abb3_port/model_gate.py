#!/usr/bin/env python3
"""The whole device model against `tt_bio/abodybuilder3_reference.py` in float64.

`scripts/abb3_port/device_gate.py` scores one IPA block; this scores the 8-block loop, which is
where the parts that only exist between blocks live: the quaternion composition and its
renormalisation, the pairwise distance feature map that the moving frames rebuild every block, and
the angle resnet's sin/cos normalisation.

The reference's output format is the target, so the device outputs are reassembled into it on the
host -- the quaternion and translation into `frames`, the sin/cos tile blocks into `[*, 7, 2]` --
and that reassembly is the same code the training loop will run, since the geometry tail
(torsion angles to frames, then atom14) stays in torch on the host.

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/model_gate.py [--tokens 64] [--batch 2] [--blocks 8]
"""
from __future__ import annotations

import argparse

import torch
import ttnn

from tt_bio.abodybuilder3 import TILE, DeviceABB3, to_device_fp32
from tt_bio.abodybuilder3_reference import (ABB3Config, ABB3StructureModule,
                                            frames_to_atom14_positions,
                                            single_and_pair_features, torsion_angles_to_frames)
from tt_bio.af2_data import RESTYPE_ATOM14_MASK
from tt_bio.af2_reference import QuatAffine
from tt_bio.tenstorrent import get_device
from tt_bio.train import abodybuilder3_grad as grad

DT = torch.float64


def host_outputs(out: dict, n_angles: int, *, taped: bool) -> dict:
    """Reassemble the device outputs into the reference's format, on the host."""
    def pull(t):
        return ttnn.to_torch(t.value if taped else t).double()

    blocks = len(out["states"])
    frames, angles, unnorm, states = [], [], [], []
    for i in range(blocks):
        quat = torch.stack([pull(q) for q in out["quat"][i]], dim=-1)
        trans = torch.stack([pull(t) for t in out["trans"][i]], dim=-1)
        frames.append(torch.cat([quat, trans], dim=-1))
        angles.append(torch.stack([pull(out["sin"][i])[..., :n_angles],
                                   pull(out["cos"][i])[..., :n_angles]], dim=-1))
        unnorm.append(torch.stack([pull(out["unnorm_sin"][i])[..., :n_angles],
                                   pull(out["unnorm_cos"][i])[..., :n_angles]], dim=-1))
        states.append(pull(out["states"][i]))
    got = {"frames": torch.stack(frames), "angles": torch.stack(angles),
           "unnormalized_angles": torch.stack(unnorm), "states": torch.stack(states),
           "single": pull(out["single"])}
    if "plddt" in out:
        got["plddt"] = pull(out["plddt"])
    return got


def positions(out: dict, aatype: torch.Tensor, block: int = -1) -> torch.Tensor:
    """Atom14 positions from a block's frames and angles, through the host geometry tail.

    This is the tail the training loop runs: `torsion_angles_to_frames` then
    `frames_and_literature_positions_to_atom14_pos`, in torch on the host, already scored at
    <= 8.7e-14 against upstream in float64 by `reference_gate.py`. Running it here on the DEVICE
    outputs is what turns a relative error on a quaternion into the only unit the accuracy bar is
    written in: Angstrom on an atom.
    """
    frames = out["frames"][block]
    affine = QuatAffine(frames[..., :4], frames[..., 4:])
    rot, trans = torsion_angles_to_frames(aatype, affine.rotation, affine.translation,
                                          out["angles"][block])
    return frames_to_atom14_positions(aatype, rot, trans)


def grads(ref, cfg, args, single, pair, aatype, mask, sq_dev, bias_dev) -> int:
    """The 8-block loop's backward against torch autograd in float64.

    Seeded on the LAST block's states, quaternion, translation and both angle blocks at once, which
    is what `abodybuilder3_grad.backward(roots, seeds)` exists for: the geometry losses are computed
    on the host from several device outputs, so what comes back is a vector-Jacobian seed per
    output. Seeding only `states` would leave the quaternion chain and the angle resnet untested,
    and they are the two pieces that only exist between blocks.

    Every seed is zeroed at masked residues, for the reason `device_gate.py` records: a padded
    residue's attention row is decided by what survives being added to -1e7, which is an O(1) term
    in float64 and rounding noise in float32.
    """
    gen = torch.Generator().manual_seed(args.seed + 11)
    n = cfg.no_angles
    live = mask.unsqueeze(-1)
    cots = {
        "states": torch.randn(args.batch, args.tokens, cfg.embed_dim, generator=gen, dtype=DT) * live,
        "quat": torch.randn(args.batch, args.tokens, 4, generator=gen, dtype=DT) * live,
        "trans": torch.randn(args.batch, args.tokens, 3, generator=gen, dtype=DT) * live,
        "sin": torch.randn(args.batch, args.tokens, n, generator=gen, dtype=DT) * live,
        "cos": torch.randn(args.batch, args.tokens, n, generator=gen, dtype=DT) * live,
    }

    s_ref = single.clone().requires_grad_(True)
    z_ref = pair.clone().requires_grad_(True)
    for p in ref.parameters():
        p.grad = None
    out = ref(s_ref, z_ref, aatype, mask)
    loss = ((out["states"][-1] * cots["states"]).sum()
            + (out["frames"][-1][..., :4] * cots["quat"]).sum()
            + (out["frames"][-1][..., 4:] * cots["trans"]).sum()
            + (out["angles"][-1][..., 0] * cots["sin"]).sum()
            + (out["angles"][-1][..., 1] * cots["cos"]).sum())
    loss.backward()

    grad.install()
    model = DeviceABB3(ref.state_dict(), cfg,
                       to_device=lambda x: grad.param(to_device_fp32(x)))
    sd = grad.param(to_device_fp32(single))
    zd = grad.param(to_device_fp32(pair))
    dev_out = model(sd, zd, sq_dev, bias_dev)

    pad = torch.zeros(args.batch, args.tokens, TILE - n, dtype=DT)
    roots = [dev_out["states"][-1], *dev_out["quat"][-1], *dev_out["trans"][-1],
             dev_out["sin"][-1], dev_out["cos"][-1]]
    seeds = [to_device_fp32(cots["states"])]
    seeds += [to_device_fp32(cots["quat"][..., i]) for i in range(4)]
    seeds += [to_device_fp32(cots["trans"][..., i]) for i in range(3)]
    seeds += [to_device_fp32(torch.cat([cots["sin"], pad], dim=-1)),
              to_device_fp32(torch.cat([cots["cos"], pad], dim=-1))]
    grad.backward(roots, seeds)

    h, c = cfg.no_heads_ipa, cfg.c_ipa
    checks = [
        ("d/d single", ttnn.to_torch(sd.grad), s_ref.grad),
        ("d/d pair", ttnn.to_torch(zd.grad), z_ref.grad),
        ("d/d linear_in_node.w", ttnn.to_torch(model.w_node.grad).t(),
         ref.linear_in_node.weight.grad),
        ("d/d linear_in_edge.w", ttnn.to_torch(model.w_edge.grad).t()[:cfg.embed_dim - 1],
         ref.linear_in_edge.weight.grad),
        ("d/d ipa[0] linear_q.w",
         ttnn.to_torch(model.ipa[0].w_q.grad).t().reshape(h, TILE, -1)[:, :c].reshape(h * c, -1),
         ref.ipa_layers[0].linear_q.weight.grad),
        ("d/d bb_update[7].w[0]", ttnn.to_torch(model.bb_update[-1].w[0].grad).t(),
         ref.bb_update_layers[-1].linear.weight.grad[:1]),
        ("d/d angle[7] linear_out.w",
         ttnn.to_torch(model.angles[-1].w_out.grad).t()[:n],
         ref.angle_resnet_layers[-1].linear_out.weight.grad[0::2]),
    ]
    # Scored on the two statistics an optimiser is sensitive to, not on max-abs. A gradient is a
    # direction: what matters is the angle between ours and the reference's and the relative size of
    # the difference vector. Max-abs over 10^5 entries is the harshest possible summary of a
    # quantity that is then averaged over a 64-sample batch, and it is printed as a diagnostic.
    #
    # The bars come from the float32 control printed beside them, not from taste. That control is
    # torch against torch, same weights, float32 against float64, and it separates two things a
    # single number cannot: whether a deviation is the backward's own arithmetic or the forward
    # having moved. It says the loss surface here is smooth -- float32 torch moves the FORWARD by
    # 4.4e-02 and still agrees with float64 on the gradient DIRECTION to cos 0.999998 -- so what is
    # left in our number is this card's matmul precision compounded over 8 blocks, at 1.25e-03 per
    # element against float32's 1.2e-07. For scale in the other direction: upstream trained this
    # model with `set_float32_matmul_precision("medium")` (`stages/train.py:20`), i.e. bf16 matmul
    # math at 3.9e-03 per element, 3.1x coarser than ours, so upstream's own training ran on a
    # noisier gradient than this one.
    # The bar depends on the weight scale, and that is a measurement rather than a concession.
    # At sigma 0.3 the block's internal sums cancel hard, and a reduction on this card rounds at
    # ~1e-3 of the TERMS rather than of the result, so the token-axis reductions in the backward
    # lose most of their significant digits: measured cos 0.79 on the angle resnet's output weight
    # at 256 tokens. At sigma 0.05 the same tensors come back at cos 0.9998 and the atom positions
    # at 0.004 A rmsd. A trained checkpoint is narrower still, so the wide arm is kept as a stress
    # row -- it has to stay a descent direction, and that is all it has to do.
    tight = args.sigma <= 0.1
    cos_bar, l2_bar = (0.999, 5e-2) if tight else (0.75, 2.0)
    bad = 0
    print(f"  {'gradient':<26} {'cos':>9} {'L2 rel':>9} {'max rel':>9}"
          f"   sigma {args.sigma}, bars {'tight' if tight else 'stress'}")
    for name, got, ref_g in checks:
        got = got.double().reshape(ref_g.shape).flatten()
        want_g = ref_g.double().flatten()
        cos = torch.dot(got, want_g).item() / max(got.norm().item() * want_g.norm().item(), 1e-30)
        l2 = (got - want_g).norm().item() / max(want_g.norm().item(), 1e-30)
        mx = (got - want_g).abs().max().item() / max(want_g.abs().max().item(), 1e-30)
        ok = cos > cos_bar and l2 < l2_bar
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<21} {cos:>9.6f} {l2:>9.2e} {mx:>9.2e}"
              f"  (bars {cos_bar} / {l2_bar:.0e})")
    grad.uninstall()
    print(f"  {'float32 control (torch vs torch, same weights)':<26}")
    for name, cos, l2 in fp32_control(cfg, args, aatype, mask):
        print(f"       {name:<21} {cos:>9.6f} {l2:>9.2e}")
    return bad


def fp32_control(cfg, args, aatype, mask):
    """The same gradient in torch float32 against torch float64, same weights, on the host.

    Drawing the weights inside each dtype's run is the trap here and it was hit on the way to this
    function: `normal_` consumes the generator differently at float32 and float64, so the two runs
    got different models and the control read cos 0.004 -- which looks like catastrophic chaos and
    is actually two different models. Draw once in float64, cast.
    """
    torch.manual_seed(args.seed)
    base = ABB3StructureModule(cfg).to(DT)
    with torch.no_grad():
        for p in base.parameters():
            p.normal_(0.0, args.sigma)
    state = {k: v.clone() for k, v in base.state_dict().items()}

    def run(dtype):
        ref = ABB3StructureModule(cfg).to(dtype)
        ref.load_state_dict({k: v.to(dtype) for k, v in state.items()})
        g = torch.Generator().manual_seed(args.seed + 1)
        is_heavy = (torch.arange(args.tokens) < args.tokens // 2).expand(
            args.batch, args.tokens).clone()
        ri = torch.arange(args.tokens).expand(args.batch, args.tokens).clone()
        single, pair = single_and_pair_features(aatype, is_heavy, ri, dtype=dtype)
        s = single.clone().requires_grad_(True)
        z = pair.clone().requires_grad_(True)
        out = ref(s, z, aatype, mask.to(dtype))
        gen = torch.Generator().manual_seed(args.seed + 11)
        live = mask.double().unsqueeze(-1)
        n = cfg.no_angles
        c = {k: (torch.randn(args.batch, args.tokens, w, generator=gen, dtype=DT) * live).to(dtype)
             for k, w in (("states", cfg.embed_dim), ("quat", 4), ("trans", 3), ("sin", n),
                          ("cos", n))}
        ((out["states"][-1] * c["states"]).sum() + (out["frames"][-1][..., :4] * c["quat"]).sum()
         + (out["frames"][-1][..., 4:] * c["trans"]).sum()
         + (out["angles"][-1][..., 0] * c["sin"]).sum()
         + (out["angles"][-1][..., 1] * c["cos"]).sum()).backward()
        return {"d/d single": s.grad, "d/d pair": z.grad,
                "d/d ipa[0] linear_q.w": ref.ipa_layers[0].linear_q.weight.grad}, \
               out["states"][-1].double()

    f64, s64 = run(DT)
    f32, s32 = run(torch.float32)
    rows = [("forward states", float("nan"),
             (s32 - s64).abs().max().item() / s64.abs().max().item())]
    for k in f64:
        a, b = f32[k].double().flatten(), f64[k].double().flatten()
        rows.append((k, torch.dot(a, b).item() / (a.norm().item() * b.norm().item()),
                     (a - b).norm().item() / b.norm().item()))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    #: Width of the random parameters. 0.3 is deliberately far wider than a trained checkpoint so
    #: that no path is left near zero, which is right for a forward check and punishing for a
    #: gradient one: wide weights make the block's internal sums cancel harder, and a reduction on
    #: this card rounds at ~1e-3 of the terms rather than of the result. Runs at both.
    ap.add_argument("--sigma", type=float, default=0.3)
    args = ap.parse_args()
    cfg = ABB3Config(use_plddt=True, no_blocks=args.blocks)

    torch.manual_seed(args.seed)
    ref = ABB3StructureModule(cfg).to(DT)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, args.sigma)
    g = torch.Generator().manual_seed(args.seed + 1)
    aatype = torch.randint(0, 21, (args.batch, args.tokens), generator=g)
    is_heavy = (torch.arange(args.tokens) < args.tokens // 2).expand(args.batch, args.tokens).clone()
    residue_index = torch.arange(args.tokens).expand(args.batch, args.tokens).clone()
    single, pair = single_and_pair_features(aatype, is_heavy, residue_index, dtype=DT)
    mask = torch.ones(args.batch, args.tokens, dtype=DT)
    mask[-1, -3:] = 0.0
    with torch.no_grad():
        want = ref(single, pair, aatype, mask)

    dev = get_device()
    try:
        square = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1)
        sq_dev = to_device_fp32(square)
        bias_dev = to_device_fp32(cfg.inf * (square - 1.0))
        rows = {}
        for taped in (False, True):
            grad.install() if taped else grad.uninstall()
            model = DeviceABB3(ref.state_dict(), cfg, to_device=to_device_fp32)
            sd = to_device_fp32(single)
            zd = to_device_fp32(pair)
            if taped:
                sd, zd = grad.param(sd), grad.param(zd)
            rows["on" if taped else "off"] = host_outputs(
                model(sd, zd, sq_dev, bias_dev), cfg.no_angles, taped=taped)

        bad = 0
        print(f"  sigma {args.sigma}, {args.blocks} blocks, batch {args.batch} x {args.tokens} tokens")
        print(f"  {'output':<22} {'rel err':>10} {'on vs off':>11}")
        # Which outputs carry a leading block axis, stated rather than inferred: with batch 2 and
        # 2 blocks the shapes are indistinguishable, and guessing would silently compare the wrong
        # residues against the mask.
        stacked = {"frames", "angles", "unnormalized_angles", "states"}
        for key in ("frames", "angles", "unnormalized_angles", "states", "single", "plddt"):
            ref_t, off, on = want[key].double(), rows["off"][key], rows["on"][key]
            # Masked residues are dropped for the same reason as in device_gate.py: their attention
            # row is decided by what survives being added to -1e7, which is an O(1) term in float64
            # and rounding noise in float32. The loss masks them.
            sel = mask
            if key in stacked:
                sel = sel.unsqueeze(0).expand(ref_t.shape[0], -1, -1)
            while sel.dim() < ref_t.dim():
                sel = sel.unsqueeze(-1)
            sel = sel.expand_as(ref_t) > 0
            scale = max(ref_t[sel].abs().max().item(), 1e-30)
            err = (off - ref_t)[sel].abs().max().item() / scale
            delta = (on - off).abs().max().item()
            note = ""
            if key == "angles":
                # `angles` is `s / |s|`, so its error is the error in `s` divided by `|s|` and an
                # angle whose unnormalised vector is near zero has no well-defined direction in the
                # REFERENCE either. With N(0, 0.3) weights about 1 % of angles land more than 10x
                # below the median magnitude, and those entries -- not the port -- set the
                # unconditional max. Scored on the well-conditioned ones, with both numbers shown.
                norm = want["unnormalized_angles"].double().square().sum(-1).sqrt()
                well = norm > 0.1 * norm[sel[..., 0]].median()
                well = well.unsqueeze(-1).expand_as(ref_t) & sel
                conditioned = (off - ref_t)[well].abs().max().item() / scale
                note = (f"  [unconditional {err:.2e} on {int(sel[..., 0].sum())} angles;"
                        f" {int(well[..., 0].sum())} well-conditioned]")
                err = conditioned
            ok = err < 5e-2 and delta == 0.0
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {key:<17} {err:>10.2e} {delta:>11.2e}"
                  f"  (bars 5e-2 / 0){note}")
        # The number the accuracy bar is written in. Compared on atoms that exist, of residues
        # that are not padding.
        atom_exists = torch.as_tensor(RESTYPE_ATOM14_MASK, dtype=DT)[aatype] * mask.unsqueeze(-1)
        live = (atom_exists > 0).unsqueeze(-1).expand(-1, -1, -1, 3)
        pos_ref = want["positions"][-1].double()
        for name, out in (("grad off", rows["off"]), ("grad on", rows["on"])):
            pos = positions(out, aatype)
            delta = (pos - pos_ref)[live]
            per_atom = (pos - pos_ref).square().sum(-1).sqrt()[atom_exists > 0]
            print(f"  {name:<9} atom14 positions: max {delta.abs().max().item():.3f} A,"
                  f" rmsd {per_atom.square().mean().sqrt().item():.3f} A,"
                  f" median {per_atom.median().item():.4f} A"
                  f" over {int((atom_exists > 0).sum())} atoms")
        bad += grads(ref, cfg, args, single, pair, aatype, mask, sq_dev, bias_dev)
        print(f"\n{'PASS' if not bad else 'FAIL'}")
        return 0 if not bad else 1
    finally:
        grad.uninstall()
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
