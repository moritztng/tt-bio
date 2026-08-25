#!/usr/bin/env python3
"""p131 -- is the pair Transition's output a function of its chunk height?

p131's fold rungs measured that it is: the R3 3-timestep CIF digest is `6f24e62c705514e1` at
`_PAIR_TRANSITION_L1_BYTES = 138_000_000` (chunk height 64) and `73112401e98737c0` at 70.5/60 MB
(height 63/53), with `RFD3_FC1_SPLIT_SILU` OFF and `RFD3_TUNE_MATMUL=0`, so neither the split nor
calibration is the cause. This localises it to one op with no fold: run `Transition.__call__` on
one live-shaped pair tensor at several chunk heights and compare the outputs to each other and to
the whole-tensor path the `RFD3_PAIR_TRANSITION_L1=0` docstring says it reproduces op for op.

A ULP-scale difference means a rounding-order artefact -- real, but the expected kind. Anything
larger is a bug in the chunking itself.

    p131_chunk_height_exactness.py perf/p131/chunk_height_exactness.json
"""
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.getcwd())
import ttnn                                                              # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402

TOKENS, HIDDENS = 514, (512, 256)
HEIGHTS = (64, 63, 59, 53)


def budget_for(h, w_pad, hidden):
    """The L1 budget whose two-resident cap lands exactly on `h`."""
    if h >= M._PAIR_TRANSITION_H_CHUNK:
        return 138_000_000
    return h * (2 * 2 * w_pad * hidden)


def make(hidden, dev):
    t = M.Transition.__new__(M.Transition)
    t.dtype = ttnn.bfloat16
    t.compute_kernel_config = M._default_compute_kernel_config()
    g = torch.Generator().manual_seed(7)

    def tt(shape):
        return ttnn.from_torch(torch.randn(*shape, generator=g) * 0.1, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev)
    t.norm_w = tt([128])
    t.fc1_w, t.fc2_w, t.fc3_w = tt([128, hidden]), tt([128, hidden]), tt([hidden, 128])
    return t


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "perf/p131/chunk_height_exactness.json"
    dev = M.get_device()
    rec = {"tokens": TOKENS, "heights": list(HEIGHTS), "provisional_on": "pc-card0", "rows": []}
    g = torch.Generator().manual_seed(11)
    x = ttnn.from_torch(torch.randn(1, TOKENS, TOKENS, 128, generator=g) * 0.5,
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w_pad = int(x.padded_shape[2])
    for hidden in HEIGHTS and HIDDENS:
        t = make(hidden, dev)
        # The whole-tensor path, which `RFD3_PAIR_TRANSITION_L1=0` selects.
        whole = t._swiglu(x, None)
        # The card control first: the same call twice, so a difference below has a floor.
        ctrl = M._mm_maxabs(whole, t._swiglu(x, None))
        ref = None
        for h in HEIGHTS:
            M._PAIR_TRANSITION_L1_BYTES = budget_for(h, w_pad, hidden)
            got = t(x)
            got_h = M._pair_transition_chunk_h(w_pad, hidden, TOKENS)
            row = {"hidden": hidden, "asked_h": h, "actual_h": got_h,
                   "l1_bytes": M._PAIR_TRANSITION_L1_BYTES,
                   "vs_whole": M._mm_maxabs(got, whole),
                   "vs_h64": None if ref is None else M._mm_maxabs(got, ref),
                   "card_control_maxabs": ctrl}
            if ref is None:
                ref = got
            else:
                ttnn.deallocate(got)
            rec["rows"].append(row)
            print("hidden=%-4d h=%-3d (asked %d)  vs whole-tensor %.6g   vs h=64 %s   control %.6g"
                  % (hidden, got_h, h, row["vs_whole"],
                     "n/a" if row["vs_h64"] is None else "%.6g" % row["vs_h64"], ctrl), flush=True)
        ttnn.deallocate(ref)
        ttnn.deallocate(whole)
    M._PAIR_TRANSITION_L1_BYTES = 138_000_000
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(rec, indent=2) + "\n")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
