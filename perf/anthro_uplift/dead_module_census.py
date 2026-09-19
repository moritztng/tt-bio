#!/usr/bin/env python3
"""Census of checkpoint modules whose weights are numerically dead.

Anthropic's Protenix-v2 kit ships a lever `deadskip` -- "the 41 modules whose checkpoint weights
are ~1e-37 are skipped (outputs exactly 0)" (uplifting-biomolecular-modeling, protenix_v2/CHANGES.md,
`exact` mode). This reproduces that claim against OUR checkpoints, CPU-only, so the count and the
exact module list come from our own weights rather than from their prose.

A leaf module is dead when every float tensor under it has max|w| <= THR. Two chained dead
projections put the unit's output at ~1e-74, which flushes to exactly 0 in both fp32 and bf16,
so the whole unit's residual contribution is zero and the unit can be skipped bit-exactly.

    python3 dead_module_census.py <checkpoint.pt> [...]
"""
import collections
import pickle
import re
import sys

import torch

THR = 1e-30


class _Stub:
    def __init__(self, *a, **k):
        pass


class _U(pickle.Unpickler):
    """Load a checkpoint whose pickled config classes are not importable here."""

    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except Exception:
            return _Stub


class _PM:
    Unpickler = _U
    load = staticmethod(lambda f, **k: _U(f, **k).load())


def _state_dict(obj):
    seen = 0
    while isinstance(obj, dict) and seen < 4:
        keys = list(obj)
        if any(torch.is_tensor(obj[k]) for k in keys):
            return obj
        for probe in ("state_dict", "model"):
            if probe in obj:
                obj = obj[probe]
                break
        else:
            obj = obj[keys[0]]
        seen += 1
    return obj


def census(path):
    sd = _state_dict(torch.load(path, map_location="cpu", pickle_module=_PM, weights_only=False))
    mods = collections.defaultdict(list)
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.numel():
            mods[k.rsplit(".", 1)[0]].append(v)
    peak = {m: max(float(v.abs().max()) for v in ts) for m, ts in mods.items()}
    dead = {m for m, a in peak.items() if a <= THR}

    units = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for m in peak:
        g = re.match(r"(.*blocks\.\d+)\.([^.]+)", m)
        if not g:
            continue
        blk, unit = g.groups()
        units[blk][unit][1] += 1
        units[blk][unit][0] += m in dead

    print(f"== {path}\n   {len(sd)} tensors, {len(mods)} leaf modules, {len(dead)} dead "
          f"(max|w| <= {THR:g})")
    stacks = collections.defaultdict(list)
    for blk, us in units.items():
        stack = re.sub(r"\.blocks\.\d+$", "", blk)
        n = int(blk.rsplit(".", 1)[1])
        stacks[stack].append((n, sorted(u for u, (d, t) in us.items() if d == t)))
    total = 0
    for stack in sorted(stacks):
        rows = sorted(stacks[stack])
        per = collections.Counter(u for _, ds in rows for u in ds)
        if not per:
            continue
        print(f"   {stack}  ({len(rows)} blocks)")
        for u, c in sorted(per.items()):
            total += c
            print(f"      {u:26s} dead in {c}/{len(rows)} blocks: "
                  f"{[n for n, ds in rows if u in ds]}")
    print(f"   dead block-units: {total}")
    return total


if __name__ == "__main__":
    for p in sys.argv[1:]:
        census(p)
