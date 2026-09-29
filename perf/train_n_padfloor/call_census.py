#!/usr/bin/env python3
"""Which SOURCE LINE dispatches each device program, counted by operand shape.

`train-k-devicetime`'s census ranks device time by op code and input shape, which is the right
unit for "where does the time go" and the wrong one for "what do I change": two entries carrying
32 % of the step are named `FillPadDeviceOperation` and `BinaryNgDeviceOperation`, and neither
name appears anywhere in the model source. This maps the other way -- it wraps the `ttnn` entry
points the port calls and records, per call, the operand shapes and the deepest `tt_bio` frame
that issued it.

It is a HOST-side counter: no device profiler, no 1000-program wall, and no perturbation of the
number it is used beside. It reports counts and shapes, never a duration.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "abb3_port"))
from step_gate import synthetic_micro_batch  # noqa: E402

from tt_bio.abodybuilder3_reference import ABB3Config, ABB3StructureModule  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.train.abodybuilder3_step import TrainStep  # noqa: E402

WRAPPED = ("add", "subtract", "multiply", "divide", "matmul", "linear", "reshape", "permute",
           "transpose", "slice", "concat", "sum", "mean", "softmax", "layer_norm", "relu",
           "sqrt", "rsqrt", "reciprocal", "neg", "sigmoid", "softplus", "gtz", "lt", "minimum",
           "maximum", "pad", "zeros", "ones_like", "clone", "to_layout", "typecast")

COUNTS: dict = defaultdict(int)
TRACE: list = []
ENABLED = [False]


def _shape(x) -> str:
    try:
        return "x".join(str(int(d)) for d in x.shape)
    except Exception:
        return "-"


def _site() -> str:
    """The deepest frame inside tt_bio that is not this file or the ops shim."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        name = fr.filename
        if "/tt_bio/" not in name:
            continue
        if name.endswith("abodybuilder3_ops.py") or name.endswith("call_census.py"):
            continue
        return f"{Path(name).name}:{fr.lineno} {fr.name}"
    return "?"


def _wrap(fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        if ENABLED[0]:
            shapes = " @ ".join(_shape(a) for a in args[:2] if hasattr(a, "shape"))
            site = _site()
            COUNTS[(fn.__name__, shapes, site)] += 1
            TRACE.append((fn.__name__, shapes, site))
        return fn(*args, **kwargs)
    return call


def install() -> None:
    for name in WRAPPED:
        fn = getattr(ttnn, name, None)
        if fn is not None:
            setattr(ttnn, name, _wrap(fn))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    install()
    cfg = ABB3Config(use_plddt=False, no_blocks=args.blocks)
    torch.manual_seed(0)
    ref = ABB3StructureModule(cfg)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)

    dev = get_device()
    try:
        step = TrainStep(ref.state_dict(), cfg, accumulate=1)
        micro = [synthetic_micro_batch(cfg, args.micro, args.tokens, 100, dev)]
        step.step(micro)                      # warm: program cache and any one-time allocation
        ttnn.synchronize_device(dev)
        COUNTS.clear()
        ENABLED[0] = True
        step.step(micro)
        ENABLED[0] = False
        ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)

    total = sum(COUNTS.values())
    print(f"CALLS: {total} wrapped ttnn calls in one warm micro-{args.micro} step "
          f"({args.tokens} tokens, {args.blocks} blocks, accumulate 1)")
    print(f"{'n':>7}  {'op':<12} {'shapes':<34} site")
    for (op, shapes, site), n in sorted(COUNTS.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"{n:>7}  {op:<12} {shapes[:34]:<34} {site}")
    by_site = defaultdict(int)
    for (op, _s, site), n in COUNTS.items():
        by_site[(op, site)] += n
    print(f"\nBY SITE\n{'n':>7}  {'op':<12} site")
    for (op, site), n in sorted(by_site.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"{n:>7}  {op:<12} {site}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"total": total, "micro": args.micro, "tokens": args.tokens, "blocks": args.blocks,
             "calls": [{"op": o, "shapes": s, "site": si, "n": n}
                       for (o, s, si), n in sorted(COUNTS.items(), key=lambda kv: -kv[1])],
             "trace": TRACE},
            indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
