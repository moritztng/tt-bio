#!/usr/bin/env python3
"""PROTOCOL SS3a, second half: THEIR tensors to the tensors the BUILT MODEL holds on the card.

K22 published the first half from `of3t-equivalence`: 3,275 of their tensors mapped to the flat
dict each device module loads, 0 unmapped in scope, 232 of ours fused from two of theirs. It is
CPU-only, so 1,660 of their tensors sit behind modules that need a card, and on the ones it does
cover it stops one step short: the flat dict is what the remap PRODUCES, and what the model HOLDS
is a further transform of it. `TriangleMultiplication` fuses its four in-projections AND
`linear_g` into one `_gp_gout_cache` of width 640; `TriangleAttention` fuses q, k, v, the gate and
the pair-bias projection into a `qkvgb_weight` of width 544 whose last band is tile padding.
Neither is visible from the flat dict, and both are exactly where a per-parameter comparison has
to cut.

METHOD, and the reason it is not a transcription. Every device weight the walk reaches is read
back and scanned along each axis against THEIR OWN tensors, greedily, taking the longest band
that matches. A match is confirmed elementwise before it is accepted, in either orientation, so a
fingerprint collision yields an unmatched tensor rather than a wrong mapping. Trailing all-zero
space is recorded as padding rather than silently dropped -- a band nobody explains is the thing
this file exists to surface.

Storage that is lossier than bf16 is admitted separately: an exact match is `direct` or
`transposed`, and a match that is only close is `approx` with its relative L2 carried through to
the report, never folded into the exact count.
"""
from __future__ import annotations

import torch

APPROX_BAR = 5e-3


def fingerprint(t: torch.Tensor) -> tuple:
    f = t.reshape(-1).to(torch.float64)
    return (int(f.numel()), round(float(f.sum()), 6), round(float((f * f).sum()), 4))


def _confirm(band: torch.Tensor, cand: torch.Tensor) -> tuple[str, float] | None:
    b = band.to(torch.float32)
    for how, c in (("direct", cand), ("transposed", cand.T if cand.ndim == 2 else None)):
        if c is None or tuple(c.shape) != tuple(b.shape):
            continue
        c = c.contiguous().to(torch.bfloat16).to(torch.float32)
        if torch.equal(b, c):
            return how, 0.0
        d = float((b - c).norm() / (c.norm() + 1e-30))
        if d <= APPROX_BAR:
            return f"approx_{how}", d
    return None


def _index(atoms: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Exact-value index, plus a shape bucket for the lossy-storage fallback.

    The fallback exists because a weight the device stores in something coarser than bf16 never
    reaches the exact index at all. Bucketing it by shape keeps that path O(bucket) instead of
    O(every parameter in the model), which is the difference between minutes and hours once this
    runs over a whole checkpoint rather than one block.
    """
    idx: dict[tuple, list[str]] = {}
    buckets: dict[tuple, list[str]] = {}
    for k, v in atoms.items():
        vb = v.to(torch.bfloat16)
        idx.setdefault((tuple(sorted(v.shape)), fingerprint(vb)), []).append(k)
        buckets.setdefault(tuple(sorted(v.shape)), []).append(k)
    return idx, buckets


def _try_at(band: torch.Tensor, atoms: dict, idx: dict, used: set) -> tuple | None:
    exact, buckets = idx
    shape = tuple(sorted(band.shape))
    for k in exact.get((shape, fingerprint(band)), ()):
        if k in used:
            continue
        c = _confirm(band, atoms[k])
        if c:
            return k, c[0], c[1]
    # fingerprint is exact-value based, so a lossily stored weight never reaches the index.
    for k in buckets.get(shape, ()):
        if k in used:
            continue
        c = _confirm(band, atoms[k])
        if c:
            return k, c[0], c[1]
    return None


def match_tensor(dev: torch.Tensor, atoms: dict, idx: tuple) -> dict:
    """One device tensor against their tensors. Whole first, then a greedy band scan per axis."""
    whole = _try_at(dev, atoms, idx, set())
    if whole:
        return {"kind": "whole", "padding": 0, "unexplained_band": 0,
                "parts": [{"their": whole[0], "axis": 0, "start": 0,
                           "length": int(dev.shape[0]), "layout": whole[1], "rel": whole[2]}]}
    best = None
    for axis in range(dev.ndim):
        n = int(dev.shape[axis])
        lengths = sorted({int(s) for v in atoms.values() for s in v.shape if 0 < s <= n}
                         | ({n} if dev.ndim == 1 else set()), reverse=True)
        parts, used, pos, pad, unexplained = [], set(), 0, 0, 0
        while pos < n:
            hit = None
            for L in lengths:
                if pos + L > n:
                    continue
                band = dev.narrow(axis, pos, L).contiguous()
                got = _try_at(band, atoms, idx, used)
                if got:
                    hit = {"their": got[0], "axis": axis, "start": pos, "length": L,
                           "layout": got[1], "rel": got[2]}
                    break
            if hit is None:
                rest = dev.narrow(axis, pos, n - pos)
                if parts and not rest.any():
                    pad = n - pos                    # tile padding, named rather than dropped
                else:
                    # A band nobody explains does not invalidate the bands that WERE confirmed
                    # elementwise. Discarding the whole tensor was this matcher's own version of
                    # the defect SS3a is about: it silently dropped 190 diffusion tensors whose
                    # q, k and v all matched because a fourth band did not. Keep the parts,
                    # NAME the remainder, and let the report carry it.
                    unexplained = n - pos
                break
            used.add(hit["their"])
            parts.append(hit)
            pos += hit["length"]
        if len(parts) > 1 and (best is None or len(parts) > len(best[0])):
            best = (parts, axis, pad, unexplained)
    if best:
        return {"kind": f"concat_axis{best[1]}", "padding": best[2],
                "unexplained_band": best[3], "parts": best[0]}
    return {"kind": "unmatched", "padding": 0, "unexplained_band": 0, "parts": []}


def device_bijection(device_tensors: dict[str, torch.Tensor],
                     atoms: dict[str, torch.Tensor]) -> dict:
    """`{their key: [placement]}`, plus both unmatched sets, both enumerated."""
    idx = _index(atoms)
    placements: dict[str, list[dict]] = {}
    per_device, unmatched = {}, []
    for path, t in sorted(device_tensors.items()):
        m = match_tensor(t, atoms, idx)
        per_device[path] = m
        if not m["parts"]:
            unmatched.append({"device_path": path, "shape": list(t.shape)})
            continue
        for p in m["parts"]:
            placements.setdefault(p["their"], []).append(
                dict(p, device_path=path, device_shape=list(t.shape)))
    return {"placements": placements, "per_device": per_device,
            "device_unmatched": unmatched,
            "their_unplaced": sorted(set(atoms) - set(placements))}
