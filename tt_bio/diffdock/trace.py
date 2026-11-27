"""Proving a loop is capturable before a card is available to capture it.

`ttnn.begin_trace_capture` records the exact sequence of device operations a block issues,
with the exact buffers they read and write, and replays that recording. A step whose op
sequence or whose tensor shapes depend on the data cannot be replayed: the capture either
refuses, or -- worse -- replays the first step's shapes over every later step's data. The
whole point of the static graphs, the masks, the branch-free quaternion and the coefficient
table is that no such dependence is left.

That property is checkable without a device. :func:`record_ops` runs a callable under a
`TorchFunctionMode` that writes down every torch operation it issues and the shape and dtype
of every tensor that comes back; :func:`assert_shape_stable` then demands that all 19 steps
produced the *same* list. A data-dependent branch shows up as a different op at some index, a
data-dependent width as a different shape, and a dtype promotion as a different dtype. It is a
stronger statement than "the output shape is constant", which is what an eyeball check gives.

What it does not check: that the ttnn implementations of these torch ops are themselves
capture-safe (an op that falls back to the host, or allocates per call, is a device-side
question), and the memory high-water mark, which decides whether the capture fits. Both are
for the device pass. This is the half that can be proved on a CPU, and a failure here is a
failure on the card too.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.overrides import TorchFunctionMode


@dataclass(frozen=True)
class OpRecord:
    """One torch call: what was called, and the shape/dtype of what came back."""
    name: str
    out: tuple

    def __str__(self) -> str:
        return f"{self.name} -> {', '.join(f'{s}{d}' for s, d in self.out) or 'non-tensor'}"


def _describe(out) -> tuple:
    if isinstance(out, torch.Tensor):
        return ((tuple(out.shape), str(out.dtype)),)
    if isinstance(out, (tuple, list)):
        return tuple(x for o in out for x in _describe(o))
    return ()


class _Recorder(TorchFunctionMode):
    def __init__(self):
        self.ops: list[OpRecord] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "__name__", repr(func))
        self.ops.append(OpRecord(name, _describe(out)))
        return out


def record_ops(fn, *args, **kwargs):
    """Run *fn* and return ``(result, [OpRecord, ...])`` for every torch call it made."""
    rec = _Recorder()
    with rec:
        result = fn(*args, **kwargs)
    return result, rec.ops


def first_difference(a: list[OpRecord], b: list[OpRecord]):
    """``(index, a_record, b_record)`` for the first place two op logs differ, else ``None``."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        i = min(len(a), len(b))
        return i, (a[i] if i < len(a) else None), (b[i] if i < len(b) else None)
    return None


def assert_shape_stable(logs: list[list[OpRecord]]) -> int:
    """Every step must issue the same ops on the same shapes. Returns the ops per step.

    The message names the first op that moved and both versions of it, because "the loop is
    not traceable" is useless and "op 418, `index_select`, 1184 edges then 1160" is the bug.
    """
    if not logs:
        raise ValueError("no steps recorded")
    for step, log in enumerate(logs[1:], start=1):
        diff = first_difference(logs[0], log)
        if diff is not None:
            i, a, b = diff
            raise AssertionError(
                f"step {step} is not the same program as step 0: op {i} is\n"
                f"  step 0: {a}\n  step {step}: {b}\n"
                f"({len(logs[0])} ops at step 0, {len(log)} at step {step})")
    return len(logs[0])
