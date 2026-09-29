#!/usr/bin/env python3
"""The ABodyBuilder3 training step, shaped so the device profiler can see one of it.

`scripts/abb3_port/step_gate.py` reports STAGE WALL CLOCK: `forward` + `device_backward` came to
17.87 s at micro 2 and PLAN.md §29 called that our device term. It is an upper bound on two counts
and a lower bound on a third:

* both stages carry host dispatch, so wall clock over them is not summed kernel time;
* the `losses` stage runs the sidechain FAPE ON THE CARD (`TrainStep._sidechain`), and
  `host_backward` runs its backward, so device work lives in two stages the split calls host.

This driver exists because a whole step cannot be profiled: at micro 2 it is 32 micro-batches, and
the device profiler's wall is 1000 dispatched programs unless the budget is raised per op. So it
runs the SAME `TrainStep` at a chosen `--accumulate`, and the step is measured at two accumulate
counts so the per-micro-batch device cost comes out of a DIFFERENCE rather than a division -- a
division would assume the linearity the difference measures.

Run bare for wall clock, or under `python -m tracy -r -o OUT --op-support-count N --` for device
kernel time. Nothing here is optimised or changed: it calls `train-b3-train`'s `TrainStep` and
`step_gate.py`'s own synthetic micro-batch builder.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "abb3_port"))
from step_gate import synthetic_micro_batch  # noqa: E402  their builder, unchanged
from step_time import ClockSampler, report_host  # noqa: E402

from tt_bio.abodybuilder3_reference import ABB3Config, ABB3StructureModule  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.train.abodybuilder3_step import TrainStep  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--accumulate", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--host-fape", action="store_true",
                    help="keep the sidechain FAPE on the host. Its device ops live in the `losses` "
                         "stage, which PLAN.md 29 counts as host time, so the difference against "
                         "the default arm is how much device work that split misplaces")
    ap.add_argument("--clock", action="store_true", help="sample AICLK during (off under tracy: "
                                                         "the subprocess perturbs the capture)")
    args = ap.parse_args()

    report_host("before opening the device")
    cfg = ABB3Config(use_plddt=False, no_blocks=args.blocks)
    torch.manual_seed(0)
    ref = ABB3StructureModule(cfg)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)

    dev = get_device()
    sampler = ClockSampler() if args.clock else None
    try:
        step = TrainStep(ref.state_dict(), cfg, accumulate=args.accumulate,
                         device_sidechain=not args.host_fape)
        print(f"  sidechain FAPE on {'the host' if args.host_fape else 'the card'}")
        micro = [synthetic_micro_batch(cfg, args.micro, args.tokens, 100 + i, dev)
                 for i in range(args.accumulate)]
        for _ in range(args.warmup):
            step.step(micro)
        ttnn.synchronize_device(dev)
        same, other = report_host("device open, before timing")
        if sampler:
            sampler.start()

        totals, stages = [], []
        for i in range(args.steps):
            _, timing = step.step(micro)
            totals.append(timing.total)
            stages.append(timing.as_dict())
            print(f"  step {i + 1:>2}  {timing.total:.3f} s", flush=True)
        if sampler:
            sampler.stop()

        keys = ("forward", "download", "losses", "host_backward", "device_backward", "optimizer")
        med = {k: statistics.median([s[k] for s in stages]) for k in keys}
        total = statistics.median(totals)
        print(f"\nCONFIG micro {args.micro} x accumulate {args.accumulate} "
              f"= batch {args.micro * args.accumulate}, {args.tokens} tokens, {args.blocks} "
              f"blocks, sidechain FAPE on {'the host' if args.host_fape else 'the card'}")
        print(f"STEP median {total:.4f} s over {len(totals)} steps "
              f"(min {min(totals):.4f}, max {max(totals):.4f})")
        for k in keys:
            print(f"  {k:<16} {med[k]:>8.4f} s  {med[k] / total * 100:>5.1f} %")
        print(f"  {'device stages':<16} {med['forward'] + med['device_backward']:>8.4f} s")
        if sampler:
            print(f"CLOCK: {sampler.summary()}")
        if same or other:
            print(f"CONTENDED: {same} cotenant(s) on our node and {other} elsewhere")
        else:
            print("CONTENDED: none -- no foreign holder of any node on this host")
        if args.json:
            Path(args.json).write_text(json.dumps({
                "micro": args.micro, "accumulate": args.accumulate, "tokens": args.tokens,
                "host_fape": args.host_fape,
                "blocks": args.blocks, "steps": totals, "stages": stages, "median": total,
                "stage_median": med, "clock": sampler.summary() if sampler else "not sampled",
                "cotenants_same_node": same, "cotenants_elsewhere": other,
                "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=1))
        return 0
    finally:
        if sampler:
            sampler.stop()
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
