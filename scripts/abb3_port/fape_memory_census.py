#!/usr/bin/env python3
"""Where the sidechain FAPE's device memory goes, measured from the allocator.

`train-b3-train` measured the device sidechain FAPE OOM on qb2 at micro 4: 29.36 MB wanted against
5.56 MB free. 29.36 MB is exactly one `[4, 512, 3584]` fp32 chunk tensor, so the failing allocation
names the shape but not the total. This reads the total from
`ttnn.get_memory_view(device, BufferType.DRAM)`, which reports bytes per bank; the card is 8 banks of
4.278 GB, so per-bank x 8 is the whole card and a `[4, 512, 3584]` fp32 allocation moves it by
29 360 128 B -- the OOM's own number, which is how this instrument is calibrated.

Three things are reported separately, because they have different fixes:

* CONSTANTS -- what `prepare_sidechain_constants` holds for the whole batch.
* TAPE -- what the forward retains so the backward can run, sampled per chunk so the growth is
  visible rather than inferred from the end state.
* TRANSIENT -- peak minus retained, i.e. what a chunk needs live at once.

The allocator is sampled after EVERY `abodybuilder3_ops` call, not once per chunk. Sampling at the
chunk boundary reads the peak of a checkpointed chunk as 8.38 MB, because a chunk that retains
nothing has already released its intermediates by the time the boundary is reached -- and the
transient is the entire cost of that design, so a boundary reading answers the wrong question.

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/fape_memory_census.py [--micro 4] [--tokens 256] [--frame-chunk 512]
"""
from __future__ import annotations

import argparse
import gc

import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio.train import losses_geometry as L
from tt_bio.train import fape_device as FD

from fape_device_gate import synthetic  # noqa: E402  (same directory)

MB = 1024.0 * 1024.0


SAMPLED = ("sub", "mul", "add", "sub_square", "minimum", "sqrt_plus", "scale", "sum_last",
           "sum_dim", "div", "shift")


def sample_every_op(probe):
    """Mark the allocator after every `abodybuilder3_ops` call the term makes.

    Wraps the dispatching wrapper rather than replacing it, so the grad hook still sees the same
    name and the same shipped callable; `.shipped` is carried over because that attribute is part of
    the op surface.
    """
    import functools
    from tt_bio import abodybuilder3_ops as O
    for name in SAMPLED:
        inner = getattr(O, name)

        def wrapped(*a, _inner=inner, **kw):
            out = _inner(*a, **kw)
            probe.mark()
            return out

        functools.update_wrapper(wrapped, inner)
        if hasattr(inner, "shipped"):
            wrapped.shipped = inner.shipped
        setattr(O, name, wrapped)


class Probe:
    """Allocated DRAM on the whole card, and the high-water mark since the last reset."""

    def __init__(self, device):
        self.device = device
        self.banks = ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
        self.capacity = (ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_per_bank
                         * self.banks)
        self.peak = self.now()

    def now(self) -> int:
        mv = ttnn.get_memory_view(self.device, ttnn.BufferType.DRAM)
        return mv.total_bytes_allocated_per_bank * self.banks

    def mark(self) -> int:
        v = self.now()
        self.peak = max(self.peak, v)
        return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--frame-chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b = {k: (v.float() if v.is_floating_point() else v)
         for k, v in synthetic(args.micro, args.tokens, args.seed, torch.float64).items()}
    cot = torch.randn(1, args.micro, generator=torch.Generator().manual_seed(7))

    dev = get_device()
    try:
        probe = Probe(dev)
        base = probe.mark()
        sample_every_op(probe)
        flat = L.sidechain_inputs(b["sidechain_frames"], b["sidechain_atom_pos"],
                                  b["rigidgroups_gt_frames"], b["rigidgroups_alt_gt_frames"],
                                  b["rigidgroups_gt_exists"], b["renamed_atom14_gt_positions"],
                                  b["renamed_atom14_gt_exists"], b["alt_naming_is_better"],
                                  b["cdr_mask"])
        n_frames = flat["pred_frames"].shape[-3]
        n_points = flat["pred_positions"].shape[-2]
        one = args.micro * args.frame_chunk * n_points * 4
        print(f"  shape      micro {args.micro}, {args.tokens} tokens -> "
              f"{n_frames} frames x {n_points} points, chunk {args.frame_chunk}")
        print(f"  card       {probe.capacity / MB:.0f} MB over {probe.banks} banks, "
              f"{base / MB:.2f} MB allocated at open")
        print(f"  unit       one [{args.micro}, {args.frame_chunk}, {n_points}] fp32 = "
              f"{one / MB:.2f} MB")

        gt_rot, gt_trans = L.rigid_from_tensor_4x4(flat["gt_frames"])
        const = FD.prepare_sidechain_constants(gt_rot, gt_trans, flat["gt_positions"],
                                               flat["frames_mask"], flat["positions_mask"],
                                               flat["frame_region"], flat["atom_region"],
                                               frame_chunk=args.frame_chunk)
        ttnn.synchronize_device(dev)
        after_const = probe.mark()
        print(f"\n  CONSTANTS  {(after_const - base) / MB:8.2f} MB  "
              f"({(after_const - base) / one:5.2f} units, {len(const['chunks'])} chunks)")

        # Warm the program cache, then release everything the warmup retained, so the numbers below
        # are the term's own footprint and not the warmup's still-live tape.
        wf = flat["pred_frames"].detach().clone().requires_grad_(True)
        wp = flat["pred_positions"].detach().clone().requires_grad_(True)
        (FD.sidechain_fape_device(wf, wp, const) * cot).sum().backward()
        ttnn.synchronize_device(dev)
        del wf, wp
        gc.collect()
        ttnn.synchronize_device(dev)
        warm_base = probe.now()
        probe.peak = warm_base
        print(f"  warm base  {(warm_base - base) / MB:8.2f} MB retained after warmup + collect")

        pf = flat["pred_frames"].detach().clone().requires_grad_(True)
        pp = flat["pred_positions"].detach().clone().requires_grad_(True)
        trace = []
        real_chunks = const["chunks"]

        class Traced(list):
            """A chunk list that samples the allocator as the forward walks it."""

            def __iter__(self):
                for item in list.__iter__(self):
                    trace.append(probe.mark())
                    yield item
                trace.append(probe.mark())

        const["chunks"] = Traced(real_chunks)
        out = FD.sidechain_fape_device(pf, pp, const)
        ttnn.synchronize_device(dev)
        const["chunks"] = real_chunks
        after_fwd = probe.mark()
        print(f"\n  TAPE       {(after_fwd - warm_base) / MB:8.2f} MB  "
              f"({(after_fwd - warm_base) / one:5.2f} units) retained by the forward")
        for i, v in enumerate(trace):
            tag = f"before chunk {i}" if i < len(real_chunks) else "after last chunk"
            print(f"      {tag:<20} {(v - warm_base) / MB:8.2f} MB  "
                  f"({(v - warm_base) / one:5.2f} units)")
        print(f"  TRANSIENT  {(probe.peak - after_fwd) / MB:8.2f} MB peak above the retained tape, "
              f"sampled after every op")

        peak_fwd = probe.peak
        (out * cot).sum().backward()
        ttnn.synchronize_device(dev)
        after_bwd = probe.mark()
        print(f"\n  BACKWARD   peak {(probe.peak - warm_base) / MB:8.2f} MB above warm base "
              f"({(probe.peak - warm_base) / one:5.2f} units), "
              f"{(after_bwd - warm_base) / MB:.2f} MB still held after it")
        print(f"  PEAK       {(max(peak_fwd, probe.peak) - warm_base) / MB:8.2f} MB is what the "
              f"term needs free on the card")
        del pf, pp, out
        gc.collect()
        ttnn.synchronize_device(dev)
        print(f"  released   {(probe.now() - warm_base) / MB:8.2f} MB still held after collect")
        print(f"\n  TOTAL      {(max(peak_fwd, probe.peak) - base) / MB:8.2f} MB constants + "
              f"tape + transient, against {probe.capacity / MB:.0f} MB of card")
        return 0
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
