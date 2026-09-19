#!/usr/bin/env python3
"""Which parameters an optimizer state proves were trained. One file, no card, no model.

RAdam's first moment is an exponential average of the gradients a parameter has been handed, and
nothing else is ever added to it. So `exp_avg` is identically zero after N steps if and only if
every one of those N gradients was exactly zero: a parameter under an all-zero `exp_avg` did not
train, whatever its step counter says, whatever the loss curve did, and whatever `grad_norm`
reported.

That is worth a script because the failure it detects is invisible everywhere else. On 2026-09-19
the `base-loss` leg had taken 1,248 steps on all 436 of its parameters while 356 of them sat in
exactly this state: `tt_bio/train/abodybuilder3_step.py:241` substitutes `torch.zeros_like` for a
missing gradient, so RAdam stepped every parameter, `opt_steps` reached 1,248 on every parameter,
and `grad_norm` averaged the zeros in with the live ones and declined convincingly. The checkpoint
knew. Nothing read it.

The shapes of the live parameters are printed because they carry the diagnosis and not just the
verdict: on that leg every live tensor was 1-D or a `(1, 12, 1, 1)` per-head scale, which says the
gradient reached the bias and scale leaves and no weight matrix at all.

Run:
    python3 scripts/abb3_port/moment_audit.py <checkpoint.safetensors> [--min-live-scalars 0.99]

Exits non-zero when the fraction of scalars under a live moment falls below the threshold, so it
works as a gate on any host that has the file. Reads the checkpoint; never writes it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SPLIT = "|"


def read_metadata(path: Path) -> dict:
    """The safetensors `__metadata__` header, without loading a tensor."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        return json.loads(fh.read(n)).get("__metadata__", {})


def audit(path: Path) -> dict:
    from safetensors.numpy import load_file

    meta = read_metadata(path)
    blob = load_file(str(path))
    n = len([k for k in blob if k.startswith(f"master{SPLIT}")])
    if not n:
        raise SystemExit(f"{path}: no `master{SPLIT}i` groups, so this is not a run checkpoint")
    opt_steps = json.loads(meta.get("opt_steps", "[]"))

    rows = []
    for i in range(n):
        avg = blob[f"exp_avg{SPLIT}{i}"]
        sq = blob[f"exp_avg_sq{SPLIT}{i}"]
        master = blob[f"master{SPLIT}{i}"]
        rows.append({
            "i": i,
            "shape": tuple(int(d) for d in master.shape),
            "scalars": int(master.size),
            "live": bool(np.any(avg != 0.0)),
            # Both moments answer the same question from different arithmetic: exp_avg_sq
            # accumulates g^2. They can only disagree if a moment was written by something other
            # than a gradient, which is worth reporting rather than averaging away.
            "live_sq": bool(np.any(sq != 0.0)),
            # Per-parameter liveness is the verdict; the element-wise count is the honest scalar
            # figure underneath it, because a live tensor can still be mostly zero.
            "nonzero": int(np.count_nonzero(avg)),
            "abs_max": float(np.abs(avg).max()),
            "steps": float(opt_steps[i]) if i < len(opt_steps) else float("nan"),
        })
    return {"path": path, "meta": meta, "rows": rows,
            "global_step": int(meta.get("global_step", -1))}


def report(result: dict, min_live: float, show: int) -> int:
    rows = result["rows"]
    live = [r for r in rows if r["live"]]
    dead = [r for r in rows if not r["live"]]
    scalars = sum(r["scalars"] for r in rows)
    live_scalars = sum(r["scalars"] for r in live)
    frac = live_scalars / scalars if scalars else 0.0
    stepped = [r for r in rows if r["steps"] > 0]

    print(f"CHECKPOINT: {result['path']} at global step {result['global_step']}")
    print(f"PARAMETERS: {len(live)} live, {len(dead)} dead of {len(rows)} "
          f"(dead = exp_avg identically zero, so every gradient it ever got was exactly zero)")
    moved = sum(r["nonzero"] for r in rows)
    print(f"SCALARS:    {live_scalars:,} of {scalars:,} under a live moment ({frac * 100:.3f} %), "
          f"of which {moved:,} ({moved / scalars * 100:.3f} %) carry a non-zero exp_avg "
          f"element-wise")
    print(f"OPT_STEPS:  {len(stepped)} of {len(rows)} parameters stepped by the optimizer, "
          f"min {min((r['steps'] for r in rows), default=0):.0f}, "
          f"max {max((r['steps'] for r in rows), default=0):.0f}")

    disagree = [r for r in rows if r["live"] != r["live_sq"]]
    if disagree:
        print(f"MISMATCH:   {len(disagree)} parameters where exp_avg and exp_avg_sq disagree on "
              f"being live (indices {[r['i'] for r in disagree][:8]})")

    shapes = Counter(r["shape"] for r in live)
    print(f"LIVE SHAPES ({len(shapes)} distinct):")
    for shape, count in sorted(shapes.items(), key=lambda kv: (-kv[1], kv[0])):
        n_scalars = sum(r["scalars"] for r in live if r["shape"] == shape)
        print(f"  {count:>4} x {str(shape):<20} {n_scalars:>10,} scalars")
    if live and max(len(r["shape"]) for r in live) <= 1:
        print("  every live parameter is 1-D: the gradient reached bias-shaped leaves only")

    if dead:
        print(f"DEAD (first {min(show, len(dead))} of {len(dead)}):")
        for r in dead[:show]:
            print(f"  param[{r['i']:>3}] {str(r['shape']):<20} {r['scalars']:>9,} scalars, "
                  f"{r['steps']:.0f} optimizer steps, exp_avg max {r['abs_max']:.3e}")

    if frac < min_live:
        print(f"MOMENT_AUDIT: FAIL -- {frac * 100:.3f} % of scalars under a live moment, below "
              f"the {min_live * 100:.3f} % floor. {len(dead)} parameters are frozen, so the run "
              f"is training {frac * 100:.3f} % of the model")
        return 1
    print(f"MOMENT_AUDIT: PASS -- {frac * 100:.3f} % of scalars under a live moment, at or above "
          f"the {min_live * 100:.3f} % floor")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--min-live-scalars", type=float, default=0.99,
                    help="fail below this fraction of scalars under a non-zero exp_avg "
                         "(default 0.99)")
    ap.add_argument("--show", type=int, default=10, help="how many dead parameters to list")
    args = ap.parse_args()
    if not args.checkpoint.is_file():
        print(f"no such checkpoint: {args.checkpoint}", file=sys.stderr)
        return 2
    return report(audit(args.checkpoint), args.min_live_scalars, args.show)


if __name__ == "__main__":
    raise SystemExit(main())
