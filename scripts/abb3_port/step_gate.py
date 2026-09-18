#!/usr/bin/env python3
"""The COMPLETE ABodyBuilder3 training step, timed and attributed, on one card.

Everything the reproduction's schedule is committed against, in one number: the 8-block forward on
the card, the per-block outputs down to the host through the geometry tail, the stage-1 losses in
torch, the host gradients seeded back onto the device tape, and one RAdam step per
`--accumulate` micro-batches. Every earlier step figure this row reported excluded the last four.

The loss TARGETS are synthetic, and that is a scope statement rather than a shortcut: their
`structures/*.pt` carry every target (`dataloader.py:120-140`) and are not staged on this host, and
a loss's cost is set by its shapes rather than its values -- every clamp, mask and `minimum` in this
set is shape-fixed. What synthetic targets cannot tell you is whether the loss VALUE is right, and
that is what `scripts/abb3_port/loss_gate.py` is for: bit-exact against upstream's own `loss.py`.

Three properties are asserted rather than assumed. Two are what the optimizer design rests on: the
padded weight channels receive exactly zero gradient, and they are still exactly zero after a step.
If either fails, updating the device-layout parameters in place is wrong and the step needs a host
master with a gradient-mapping table.

The third is that EVERY parameter receives a gradient. A parameter that never does is frozen for
the whole run, and nothing else in this row would notice: the loss falls, the gates pass, the
structures look plausible, and one block or one projection simply never learns. It is the cheapest
possible check and the most expensive omission.

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/step_gate.py [--steps 10] [--micro 8] [--accumulate 8] [--tokens 256]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_time import ClockSampler, report_host  # noqa: E402  the same instruments as the device-only step

from tt_bio.abodybuilder3 import to_device_fp32  # noqa: E402
from tt_bio.abodybuilder3_reference import (ABB3Config, ABB3StructureModule,  # noqa: E402
                                            single_and_pair_features)
from tt_bio.af2_data import RESTYPE_ATOM14_MASK  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.train.abodybuilder3_step import TrainStep  # noqa: E402


def synthetic_micro_batch(cfg: ABB3Config, micro: int, n_tok: int, seed: int, device) -> dict:
    """One micro-batch with every field the stage-1 loss set reads, at their real shapes."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.0):
        return torch.randn(*shape, generator=g) * scale

    def rigid_4x4(*shape):
        from tt_bio.af2_reference import quat_to_rot
        q = rand(*shape, 4)
        out = torch.zeros(*shape, 4, 4)
        out[..., :3, :3] = quat_to_rot(q / q.norm(dim=-1, keepdim=True))
        out[..., :3, 3] = rand(*shape, 3, scale=8.0)
        out[..., 3, 3] = 1.0
        return out

    aatype = torch.randint(0, 21, (micro, n_tok), generator=g)
    is_heavy = (torch.arange(n_tok) < n_tok // 2).expand(micro, n_tok).clone()
    residue_index = torch.arange(n_tok).expand(micro, n_tok).clone()
    single, pair = single_and_pair_features(aatype, is_heavy, residue_index)
    seq_mask = torch.ones(micro, n_tok)
    atom_exists = torch.as_tensor(RESTYPE_ATOM14_MASK)[aatype] * seq_mask.unsqueeze(-1)
    square = (seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)).unsqueeze(1)
    chi = rand(micro, n_tok, 4, 2)
    return {
        "device": device,
        "single_d": to_device_fp32(single),
        "pair_d": to_device_fp32(pair),
        "square_d": to_device_fp32(square),
        "bias_d": to_device_fp32(cfg.inf * (square - 1.0)),
        "aatype": aatype,
        "targets": {
            "aatype": aatype,
            "seq_mask": seq_mask,
            "chi_mask": (torch.rand(micro, n_tok, 4, generator=g) > 0.3).float(),
            "chi_angles_sin_cos": chi / chi.norm(dim=-1, keepdim=True),
            "backbone_rigid_tensor": rigid_4x4(micro, n_tok),
            "backbone_rigid_mask": seq_mask.clone(),
            "rigidgroups_gt_frames": rigid_4x4(micro, n_tok, 8),
            "rigidgroups_alt_gt_frames": rigid_4x4(micro, n_tok, 8),
            "rigidgroups_gt_exists": (torch.rand(micro, n_tok, 8, generator=g) > 0.25).float(),
            "atom14_gt_positions": rand(micro, n_tok, 14, 3, scale=8.0),
            "atom14_alt_gt_positions": rand(micro, n_tok, 14, 3, scale=8.0),
            "atom14_gt_exists": atom_exists,
            "atom14_alt_gt_exists": atom_exists.clone(),
            "atom14_atom_is_ambiguous": (torch.rand(micro, n_tok, 14, generator=g) > 0.9).float(),
            "atom14_atom_exists": atom_exists,
            "cdr_mask": ((torch.arange(n_tok) >= n_tok // 3)
                         & (torch.arange(n_tok) < 2 * n_tok // 3)).expand(micro, n_tok).clone(),
            "resolution": torch.full((micro,), 2.0),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accumulate", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--host-fape", action="store_true",
                    help="keep the sidechain FAPE on the host, the arm the device one is scored "
                         "against")
    args = ap.parse_args()
    batch = args.micro * args.accumulate

    report_host("before opening the device")
    cfg = ABB3Config(use_plddt=False, no_blocks=args.blocks)
    torch.manual_seed(0)
    ref = ABB3StructureModule(cfg)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)

    dev = get_device()
    sampler = ClockSampler()
    try:
        step = TrainStep(ref.state_dict(), cfg, accumulate=args.accumulate,
                         device_sidechain=not args.host_fape)
        print(f"  sidechain FAPE on {'the host' if args.host_fape else 'the card'}")
        print(f"  taped device parameters: {len(step.params)}")
        micro = [synthetic_micro_batch(cfg, args.micro, args.tokens, 100 + i, dev)
                 for i in range(args.accumulate)]

        padded = _padded_channel_views(step)
        for _ in range(args.warmup):
            step.step(micro)
        _assert_padding_is_still_zero(step, padded)
        _assert_every_parameter_trains(step)
        ttnn.synchronize_device(dev)
        same, other = report_host("device open, before timing")
        sampler.start()

        totals, stages = [], []
        for i in range(args.steps):
            parts, timing = step.step(micro)
            totals.append(timing.total)
            stages.append(timing.as_dict())
            if i == 0 or (i + 1) % 5 == 0:
                print(f"  step {i + 1:>3}  {timing.total:.3f} s  loss {parts['loss']:.4f}"
                      f"  (median so far {statistics.median(totals):.3f} s)")
        sampler.stop()
        _assert_padding_is_still_zero(step, padded)

        med = statistics.median(totals)
        print(f"\nSTEP: median {med:.3f} s over {len(totals)} COMPLETE steps at batch {batch} "
              f"({args.accumulate} x {args.micro}) and {args.tokens} tokens, "
              f"min {min(totals):.3f} s, max {max(totals):.3f} s")
        keys = ("forward", "download", "losses", "host_backward", "device_backward", "optimizer")
        print("BREAKDOWN (median over steps, seconds and share of the step):")
        for key in keys:
            vals = statistics.median([s[key] for s in stages])
            print(f"  {key:<16} {vals:>8.3f} s  {vals / med * 100:>5.1f} %")
        if step.loss_terms:
            total_terms = sum(step.loss_terms.values())
            print(f"LOSS TERMS (cumulative over {args.steps + args.warmup} steps, "
                  f"share of the loss stage):")
            for name, secs in sorted(step.loss_terms.items(), key=lambda kv: -kv[1]):
                print(f"  {name:<16} {secs:>8.2f} s  {secs / total_terms * 100:>5.1f} %")
        print(f"CLOCK: {sampler.summary()}")
        print(f"INCLUDES: device forward and backward, host geometry tail, the stage-1 losses "
              f"(FAPE, supervised chi, final-block) and one RAdam step")
        print(f"TARGETS: synthetic -- their structures/*.pt are not staged on this host; a loss's "
              f"COST is shape-set and its VALUE is checked by loss_gate.py")
        if same or other:
            print(f"CONTENDED: {same} cotenant(s) on our node and {other} elsewhere")
        return 0
    finally:
        sampler.stop()
        ttnn.close_device(dev)


def _padded_channel_views(step) -> list:
    """Every device parameter that carries zero padding, with the slice that must stay zero.

    Found by value rather than by name: any parameter whose current contents are exactly zero on
    some trailing channels of a 32-wide block is a padded one. Names would have to be kept in sync
    with the layout; this cannot drift.
    """
    out = []
    for i, p in enumerate(step.params):
        host = ttnn.to_torch(p.value)
        if host.ndim != 2 or host.shape[-1] % 32 or host.shape[-1] < 32:
            continue
        cols = host.abs().sum(0)
        zero = (cols == 0).nonzero().flatten()
        if len(zero):
            out.append((p, zero.clone(), f"param[{i}] {tuple(host.shape)}"))
    return out


def _assert_every_parameter_trains(step) -> None:
    """Every taped parameter took a gradient, and it was not all zero.

    `p.grad is None` means the parameter is not on the tape's path at all. An all-zero gradient is
    the subtler case -- it happens when a parameter only feeds channels that are masked away -- and
    it is reported separately because the padding masks are supposed to produce exactly that for
    nothing.
    """
    missing = [i for i, p in enumerate(step.params) if p.grad is None]
    assert not missing, (
        f"{len(missing)} of {len(step.params)} parameters took NO gradient (indices "
        f"{missing[:8]}): they are frozen for the whole run and nothing else here would notice")
    dead = []
    for i, p in enumerate(step.params):
        if float(ttnn.to_torch(p.grad).abs().max()) == 0.0:
            dead.append(i)
    assert not dead, (
        f"{len(dead)} of {len(step.params)} parameters took an all-zero gradient (indices "
        f"{dead[:8]}); a masked-away output or a disconnected branch")
    print(f"  every one of {len(step.params)} parameters took a non-zero gradient")


def _assert_padding_is_still_zero(step, padded) -> None:
    """The optimizer updates the device-layout parameters in place, which is only correct if the
    padding cannot move. Checked after a step, and the gradient checked as zero too."""
    assert padded, "no padded parameter found -- the check would pass vacuously"
    offenders = []
    for p, zero, name in padded:
        host = ttnn.to_torch(p.value)
        moved = host[:, zero].abs().max().item()
        g = (ttnn.to_torch(p.grad)[:, zero].abs().max().item() if p.grad is not None else 0.0)
        if moved != 0.0 or g != 0.0:
            offenders.append(f"{name}: value {moved:.3e}, grad {g:.3e}, "
                             f"{len(zero)} padded of {host.shape[-1]} columns")
    assert not offenders, ("the in-place optimizer moved padding that must stay zero:\n  "
                           + "\n  ".join(offenders))


if __name__ == "__main__":
    raise SystemExit(main())
