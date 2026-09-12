#!/usr/bin/env python3
"""Can one program replace the token attention epilogue's four?

`AttentionPairBias.__call__` turns the SDPA output (1, 16, 512, 48-in-64) back into token-major
(1, 1, 512, 768) with four ops:

    o = o[:, :, :, :head_dim]        Slice       11.37 us
    o = ttnn.permute(o, (0,1,3,2))   Transpose   11.18 us
    o = ttnn.reshape(o, ...)         ReshapeView 41.66 us
    o = ttnn.permute(o, (0,2,1))     Transpose    9.07 us

73.28 us and four programs, 24 times a step: 1.759 ms of the step's 41.482 ms WH wall and 96 of
its 1066 programs, all of it pure layout with no arithmetic in it. That is exactly what
`b2z2-sampler-stall-split` priced at 9.76 us of per-program constant apiece.

This screens candidate one-program replacements at the production shape for (a) the right values,
bit for bit, and (b) a time under the chain's. No model code is touched until one passes both.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

N_HEADS, SEQ, HEAD_DIM, PAD_DIM = 16, 512, 48, 64


def timed(ttnn, dev, fn, reps=30, blocks=5):
    for _ in range(5):
        fn()
    ttnn.synchronize_device(dev)
    meds = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        meds.append((time.perf_counter() - t0) / reps)
    return round(1e6 * st.median(meds), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "shape": [1, N_HEADS, SEQ, HEAD_DIM]}}

    # The SDPA output as the model sees it: logical head_dim 48, tile-padded to 64.
    src = torch.randn(1, N_HEADS, SEQ, HEAD_DIM)
    o0 = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    out["logical_shape"] = list(o0.shape)

    def stock(t):
        x = t[:, :, :, :HEAD_DIM]
        x = ttnn.permute(x, (0, 1, 3, 2))
        x = ttnn.reshape(x, (x.shape[0], -1, x.shape[3]))
        return ttnn.permute(x, (0, 2, 1))

    cands = {}

    def cand(name):
        def deco(f):
            cands[name] = f
            return f
        return deco

    @cand("nlp_concat_heads")
    def _c1(t):
        return ttnn.experimental.nlp_concat_heads(t)

    @cand("permute_0213_reshape")
    def _c2(t):
        x = ttnn.permute(t, (0, 2, 1, 3))
        return ttnn.reshape(x, (1, 1, SEQ, N_HEADS * HEAD_DIM))

    @cand("reshape_only")
    def _c3(t):
        return ttnn.reshape(t, (1, 1, SEQ, N_HEADS * HEAD_DIM))

    ref = stock(o0)
    out["stock"] = {"out_shape": list(ref.shape),
                    "us": timed(ttnn, dev, lambda: stock(o0))}
    ref_t = ttnn.to_torch(ref)
    print(f"  stock -> {list(ref.shape)}  {out['stock']['us']:.2f} us", flush=True)

    out["candidates"] = {}
    for name, fn in cands.items():
        rec = {}
        try:
            got = fn(o0)
            rec["out_shape"] = list(got.shape)
            got_t = ttnn.to_torch(got)
            rec["shape_match"] = list(got_t.shape) == list(ref_t.shape)
            if rec["shape_match"]:
                rec["bit_exact"] = bool(torch.equal(got_t, ref_t))
                rec["max_abs"] = float((got_t.float() - ref_t.float()).abs().max())
            rec["us"] = timed(ttnn, dev, lambda: fn(o0))
        except Exception as e:                                            # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["candidates"][name] = rec
        print(f"  {name}: {json.dumps(rec)}", flush=True)

    # negative control: the chain with one head dropped must NOT compare equal
    bad = stock(o0)
    bad_t = ttnn.to_torch(bad).clone()
    bad_t[0, 0, 0] += 1.0
    out["negative_control_equal"] = bool(torch.equal(bad_t, ref_t))
    print(f"  negative control equal (must be false): {out['negative_control_equal']}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
