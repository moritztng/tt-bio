#!/usr/bin/env python3
"""Is the fold sensitive to where its tensors land in DRAM?

Three configurations of the AbAg-XM 9j4c work each produced a different 9i3p structure, and the
two mechanisms that could explain it as a value change -- tile-padding content and a dtype
narrowing in `_acc_concat` -- are both refuted by direct measurement. What is left is placement:
the three configurations allocate the same values at different addresses.

If the pipeline is address-sensitive, an allocation that changes nothing else will change the
answer. This holds one dummy buffer alive for the whole run, which shifts every later allocation,
and changes nothing about the model, its inputs, its seed or its arithmetic. Two runs that differ
only in `TT_BIO_ALLOC_SHIFT_BYTES` must produce identical structures; if they do not, the fold
reads memory whose contents depend on the heap, and no configuration's output is privileged.

    TT_BIO_ALLOC_SHIFT_BYTES=67108864 python3 scripts/probe_alloc_shift.py predict ...
"""
import os
import sys

import torch
import ttnn

SHIFT = int(os.environ.get("TT_BIO_ALLOC_SHIFT_BYTES", 0))
# A file, not stderr: the fold runs in a spawned worker whose stdout AND stderr the live-progress
# view owns and drops. The first fixed version of this probe printed its marker to stderr, the
# marker never appeared, and the run looked identical to one where the shim had not fired at all.
MARK = os.environ.get("TT_BIO_ALLOC_SHIFT_MARK", "/tmp/alloc_shift_mark.txt")


def _mark(msg):
    try:
        with open(MARK, "a") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass
    print(msg, file=sys.stderr, flush=True)
_held = []
_orig_from_torch = ttnn.from_torch


def _first_call_shim(*args, **kwargs):
    """Take the shift on the first DEVICE allocation of the run, then get out of the way.

    Stays installed until it actually sees a device. The first version unpatched itself on the
    first call whatever that call was, so a leading host-side `from_torch` consumed it and the
    shift silently never happened -- two runs that were supposed to differ were identical, and
    the only thing that gave it away was a missing log line. Hence the unconditional print.
    """
    dev = kwargs.get("device") or next((a for a in args if hasattr(a, "arch")), None)
    if dev is None:
        return _orig_from_torch(*args, **kwargs)      # not a device alloc; keep looking
    ttnn.from_torch = _orig_from_torch
    if SHIFT > 0:
        rows = max(1, SHIFT // (2 * 1024))
        _held.append(_orig_from_torch(
            torch.zeros(rows, 1024, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16))
        _mark(f"[SHIFT] holding {rows * 1024 * 2} B for the run")
    else:
        _mark("[SHIFT] armed, shift 0 requested")
    return _orig_from_torch(*args, **kwargs)


ttnn.from_torch = _first_call_shim

from tt_bio.main import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli(standalone_mode=True))
