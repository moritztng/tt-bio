#!/usr/bin/env python3
"""How much DRAM the COMPLETE step actually leaves free, with the model tape resident.

`fape_memory_census.py` measures the FAPE term on its own, which is the right instrument for
deciding what to change but the wrong one for deciding whether a 29.5-day run is safe: the term is
~1.2 GB and the model's tape is ~30 GB, and the margin is what is left after both. `train-b3-train`
had only the OOM's own `free:` field to go on, which is the free space at the moment of failure and
says nothing about the margin of a configuration that succeeds.

This runs the real `TrainStep` on synthetic micro-batches -- `step_gate.py`'s own, imported rather
than re-derived -- and reports the high-water allocated DRAM over a warm step, sampled after every
`abodybuilder3_ops` call, against the card's 32 640 MB.

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/fape_margin_probe.py [--micro 4] [--accumulate 16] [--tokens 256]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fape_memory_census import MB, Probe, sample_every_op  # noqa: E402
from step_gate import synthetic_micro_batch  # noqa: E402

from tt_bio.abodybuilder3_reference import ABB3Config, ABB3StructureModule  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.train.abodybuilder3_step import TrainStep  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--accumulate", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--host-fape", action="store_true")
    args = ap.parse_args()

    cfg = ABB3Config(use_plddt=False, no_blocks=args.blocks)
    torch.manual_seed(0)
    ref = ABB3StructureModule(cfg)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)

    dev = get_device()
    try:
        probe = Probe(dev)
        step = TrainStep(ref.state_dict(), cfg, accumulate=args.accumulate,
                         device_sidechain=not args.host_fape)
        micro = [synthetic_micro_batch(cfg, args.micro, args.tokens, 100 + i, dev)
                 for i in range(args.accumulate)]
        step.step(micro)          # warm the program cache; a cold step allocates differently
        ttnn.synchronize_device(dev)
        sample_every_op(probe)
        probe.peak = probe.now()
        step.step(micro)
        ttnn.synchronize_device(dev)

        peak = probe.peak
        print(f"  config     micro {args.micro} x {args.accumulate} = batch "
              f"{args.micro * args.accumulate}, {args.tokens} tokens, {args.blocks} blocks, "
              f"sidechain FAPE on {'the host' if args.host_fape else 'the card'}")
        print(f"  PEAK       {peak / MB:8.2f} MB allocated at the high-water mark of a warm step")
        print(f"  MARGIN     {(probe.capacity - peak) / MB:8.2f} MB free of "
              f"{probe.capacity / MB:.0f} MB "
              f"({(probe.capacity - peak) / probe.capacity * 100:.2f} % of the card)")
        return 0
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
