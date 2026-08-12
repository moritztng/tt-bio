#!/usr/bin/env python3
"""Name the device tensors the refiner creates during its first call, and check for mutation.

The refiner holds 26 device tensors before its first call and 146 after, and that first call's
result does not match any later one. The fix currently shipped on this branch is a whole discarded
refiner pass, which costs a full 4-block forward. If the 120 new tensors are only lazily built
weight caches, they can be materialised directly instead, with no compute at all.

The earlier version of this walk keyed its snapshot by `type(obj).__name__ + attr`, which all four
blocks share, so its mutation verdict compared block 0's weight against block 3's and meant nothing.
This keys by the full attribute path, which is stable across the two walks because the structure
only gains entries.

    TT_VISIBLE_DEVICES=26 python3 scripts/probe_refiner_lazy.py predict ... --model opendde-abag
"""
import os
import sys

import torch
import ttnn

MARK = os.environ.get("TT_BIO_LAZY_MARK", "/tmp/refiner_lazy.txt")


def _mark(msg):
    with open(MARK, "a") as fh:
        fh.write(msg + "\n")


def _walk(obj, path="refiner", seen=None, depth=0):
    seen = seen if seen is not None else set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        return
    for k, v in sorted(d.items(), key=lambda kv: str(kv[0])):
        p = f"{path}.{k}"
        if isinstance(v, ttnn.Tensor):
            yield p, v
        elif isinstance(v, (list, tuple)):
            for i, e in enumerate(v):
                yield from _emit(e, f"{p}[{i}]", seen, depth + 1)
        elif isinstance(v, dict):
            for kk in sorted(v.keys(), key=str):
                yield from _emit(v[kk], f"{p}[{kk}]", seen, depth + 1)
        else:
            yield from _walk(v, p, seen, depth + 1)


def _emit(v, p, seen, depth):
    """Yield tensors from any container, not just from objects with a __dict__.

    `_gp_cache` maps (chunk, group) -> LIST of tensors, and a plain `_walk` on that list finds
    nothing because a list has no __dict__. That is why the first run of this probe reported
    new=0: it could not see the one cache that is actually built lazily.
    """
    if isinstance(v, ttnn.Tensor):
        yield p, v
    elif isinstance(v, (list, tuple)):
        for i, e in enumerate(v):
            yield from _emit(e, f"{p}[{i}]", seen, depth + 1)
    elif isinstance(v, dict):
        for kk in sorted(v.keys(), key=str):
            yield from _emit(v[kk], f"{p}[{kk}]", seen, depth + 1)
    else:
        yield from _walk(v, p, seen, depth + 1)


def _snapshot(refiner):
    out = {}
    for nm, t in _walk(refiner):
        try:
            out[nm] = ttnn.to_torch(t).clone()
        except Exception as e:
            out[nm] = f"unreadable: {e}"
    return out


def _install():
    import tt_bio.opendde as od
    orig = od.OpenDDE.expand_and_refine

    def wrapped(self, *a, **kw):
        pre = _snapshot(self.refiner)
        _mark(f"[LAZY] before the first call: {len(pre)} device tensors")
        out = orig(self, *a, **kw)
        post = _snapshot(self.refiner)
        new = sorted(set(post) - set(pre))
        gone = sorted(set(pre) - set(post))
        changed = [k for k in sorted(set(pre) & set(post))
                   if isinstance(pre[k], torch.Tensor) and isinstance(post[k], torch.Tensor)
                   and not torch.equal(pre[k], post[k])]
        _mark(f"[LAZY] after: {len(post)} tensors; new={len(new)} gone={len(gone)} "
              f"changed={len(changed)}")
        for nm in new[:24]:
            _mark(f"[LAZY]   NEW      {nm} {tuple(post[nm].shape) if isinstance(post[nm], torch.Tensor) else post[nm]}")
        for nm in changed[:24]:
            _mark(f"[LAZY]   CHANGED  {nm}")
        return out

    od.OpenDDE.expand_and_refine = wrapped


from tt_bio.main import cli  # noqa: E402

_install()

if __name__ == "__main__":
    sys.exit(cli(standalone_mode=True))
