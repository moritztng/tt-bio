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

# `--tokens=T1,T2,...` sweeps the token count at the shipped height instead of sweeping the height
# at 514 tokens. §15.3 localised the divergence to h=64 and guessed the cause is the ragged tail:
# 514 = 8*64 + 2, and a 2-row chunk is [1, 2, w_pad, hidden], which ttnn's matmul heuristic blocks
# differently from a 64-row one. The sweep is what turns that into a count of affected production
# sizes. Keep `w_pad` fixed inside a band or the matmul shape moves too: 513-544 tokens all pad to
# 544 (tails 1-32 at 9 chunks), 545-575 all pad to 576 (tails 33-63 at 9 chunks).


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


def height_sweep(dev, rec, tokens):
    """The original experiment: four chunk heights at one token count."""
    g = torch.Generator().manual_seed(11)
    x = ttnn.from_torch(torch.randn(1, tokens, tokens, 128, generator=g) * 0.5,
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w_pad = int(x.padded_shape[2])
    for hidden in HIDDENS:
        t = make(hidden, dev)
        # The whole-tensor path, which `RFD3_PAIR_TRANSITION_L1=0` selects.
        whole = t._swiglu(x, None)
        # The card control first: the same call twice, so a difference below has a floor.
        ctrl = M._mm_maxabs(whole, t._swiglu(x, None))
        ref = None
        for h in HEIGHTS:
            M._PAIR_TRANSITION_L1_BYTES = budget_for(h, w_pad, hidden)
            got = t(x)
            got_h = M._pair_transition_chunk_h(w_pad, hidden, tokens)
            row = {"tokens": tokens, "hidden": hidden, "asked_h": h, "actual_h": got_h,
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
    ttnn.deallocate(x)


def tail_sweep(dev, rec, token_list):
    """The shipped height at many token counts: which ragged tails diverge from the whole tensor."""
    for tokens in token_list:
        g = torch.Generator().manual_seed(11)
        x = ttnn.from_torch(torch.randn(1, tokens, tokens, 128, generator=g) * 0.5,
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        w_pad = int(x.padded_shape[2])
        for hidden in HIDDENS:
            t = make(hidden, dev)
            whole = t._swiglu(x, None)
            ctrl = M._mm_maxabs(whole, t._swiglu(x, None))
            M._PAIR_TRANSITION_L1_BYTES = 138_000_000
            got = t(x)
            h = M._pair_transition_chunk_h(w_pad, hidden, tokens)
            row = {"tokens": tokens, "hidden": hidden, "actual_h": h, "w_pad": w_pad,
                   "n_chunks": -(-tokens // h), "tail_rows": tokens - (tokens // h) * h or h,
                   "l1_bytes": M._PAIR_TRANSITION_L1_BYTES,
                   "vs_whole": M._mm_maxabs(got, whole), "vs_h64": None,
                   "card_control_maxabs": ctrl}
            rec["rows"].append(row)
            print("tokens=%-4d w_pad=%-4d hidden=%-4d h=%-3d chunks=%-3d tail=%-3d  "
                  "vs whole-tensor %.6g   control %.6g"
                  % (tokens, w_pad, hidden, h, row["n_chunks"], row["tail_rows"],
                     row["vs_whole"], ctrl), flush=True)
            for tt_ in (got, whole):
                ttnn.deallocate(tt_)
        ttnn.deallocate(x)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    tok = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tokens=")), None)
    out = argv[0] if argv else "perf/p131/chunk_height_exactness.json"
    dev = M.get_device()
    rec = {"tokens": TOKENS, "heights": list(HEIGHTS), "provisional_on": "pc-card0", "rows": [],
           "mode": "tail_sweep" if tok else "height_sweep"}
    if tok:
        token_list = [int(v) for v in tok.split(",")]
        rec["token_list"] = token_list
        tail_sweep(dev, rec, token_list)
    else:
        height_sweep(dev, rec, TOKENS)
    M._PAIR_TRANSITION_L1_BYTES = 138_000_000
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(rec, indent=2) + "\n")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
