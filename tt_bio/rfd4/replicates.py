"""Per-replicate sampling transforms: every design starts from its own condition.

Upstream runs a sampling transform once per replicate, so a batch of designs is not one input
repeated but one input resampled: the VHH recipe draws a different hotspot patch for each design
(`docs/how-to/run-the-model.md`, `examples/antibody_design`). The engine's plain path broadcasts
one featurized example across the design axis, which is right when nothing varies; this builds
the per-replicate batch when something does.

Host-side, and seeded off the design seed so a run reproduces.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

#: A transform takes one example (leading axis 1) and returns one, for replicate `index`.
Transform = Callable[[dict, int, torch.Generator], dict]


def apply_transforms(features: dict, *, num_designs: int, transforms: Sequence[Transform],
                     seed: int) -> dict:
    """One batch of `num_designs` examples, each through every transform in order.

    Without transforms this is the broadcast the engine already does. With them, each replicate
    gets its own generator (`seed + index`), so design 3 of a run is the same whether the run
    asked for 4 designs or 40.
    """
    if not transforms:
        from tt_bio.rfd4.network import _broadcast_batch

        return _broadcast_batch(features, num_designs)

    rows = []
    for index in range(num_designs):
        generator = torch.Generator().manual_seed(seed + index)
        row = features
        for transform in transforms:
            row = transform(row, index, generator)
        rows.append(row)
    return _stack(rows, features)


def _stack(rows: list[dict], original: dict) -> dict:
    """Stack per-replicate examples back into one batch along the design axis.

    Shapes are checked against the example the transforms started from, not against the first
    replicate: a transform that resizes every replicate the same way would otherwise stack
    cleanly and hand the model a different input than it was featurized for.
    """
    out = {}
    for key, value in rows[0].items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            parts = [row[key] for row in rows]
            want = original.get(key, value)
            if any(part.shape != want.shape for part in parts):
                raise ValueError(f"a sampling transform changed the shape of {key}; a transform "
                                 "may resample a condition, not resize the example")
            out[key] = torch.cat(parts, dim=0)
        elif isinstance(value, dict):
            out[key] = _stack([row[key] for row in rows], original.get(key, {}))
        else:
            out[key] = value
    return out


def resample_hotspots(n_hotspots: int, key: str = "condition_hotspot_1_atom") -> Transform:
    """Upstream's VHH case: keep `n_hotspots` of the conditioned hotspot atoms, per design.

    The input marks every hotspot a user is willing to touch; a design is conditioned on a patch
    of them, and a batch covers the surface by drawing a different patch each time. Asking for
    more hotspots than the input marks is an error, not a silent truncation.
    """
    def transform(features: dict, index: int, generator: torch.Generator) -> dict:
        hotspots = features[key]
        flat = hotspots.reshape(-1)
        marked = torch.nonzero(flat, as_tuple=False).flatten()
        if marked.numel() < n_hotspots:
            raise ValueError(f"{key} marks {marked.numel()} atoms, cannot draw {n_hotspots}")
        chosen = marked[torch.randperm(marked.numel(), generator=generator)[:n_hotspots]]
        drawn = torch.zeros_like(flat)
        drawn[chosen] = flat[chosen]
        return dict(features, **{key: drawn.reshape(hotspots.shape)})

    return transform
