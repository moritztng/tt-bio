#!/usr/bin/env python3
"""p64c -- time the built L1 subset (`fc2` + the multiply, pinned config) over chunk heights.

p66 killed the wide version: `fc1` re-blocks K and re-rounds whenever its input or output moves
to L1, and no bit-exact program config for its fused silu exists. What survives is `fc2` -- which
`_tuned_linear` pins a config for, so its blocking is independent of L1 pressure -- and the
elementwise multiply. That deletes `b` and `m`: 1975 of the 3951 MB an H=512 call moves.

The two L1 residents are now `b` and `m` rather than `x_norm`/`fc1`/product, so the per-chunk
footprint is a different product (2 x h x W_pad x hidden x 2 B) and the height has to be re-swept
rather than carried over from perf/p64/fc2_l1_heights.json.

This times `Transition.__call__` itself, so what is measured is the shipped code and not a
transcription of it.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-irreducible-traffic -- env TT_VISIBLE_DEVICES=2 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-irreducible-traffic PYTHONPATH=$PWD RFD3_TUNE_MATMUL=1 \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p64c_pinned_l1_heights.py
"""
import json
import os
import pathlib
import sys

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device                              # noqa: E402
from tt_bio.rfd3 import model as M                                    # noqa: E402
import scripts.rfd3_port.p64_pair_transition_l1 as p64                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p64/pinned_l1_heights.json")
I, W_PAD = p64.I, 704
HEIGHTS = {512: [32, 64, 96, 128], 256: [64, 128, 192, 256]}


def main():
    dev = get_device()
    z = ttnn.from_torch(torch.randn(1, I, I, p64.C_Z, generator=torch.Generator().manual_seed(7)),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    rows, best, ship = [], {}, {}
    for hidden in (512, 256):
        mod = p64.mk_transition(hidden, seed=100 + hidden)
        M._PAIR_TRANSITION_L1 = False
        r = p64.row(rows, "A  shipped whole-tensor", hidden, None,
                    p64.timeit(lambda: mod(z), dev), "baseline")
        ship[hidden] = r["ms_median"]
        M._PAIR_TRANSITION_L1 = True
        for h in HEIGHTS[hidden]:
            M._PAIR_TRANSITION_CHUNK_ELEMS = h * W_PAD * hidden
            l1_mb = 2 * h * W_PAD * hidden * 2 / 1e6
            try:
                ms = p64.timeit(lambda: mod(z), dev)
            except Exception as e:
                print("C3 pinned fc2+mul -> L1       H=%3d h=%-4d DID NOT FIT (%.0f MB L1): %s"
                      % (hidden, h, l1_mb, str(e).splitlines()[0][:110]), flush=True)
                rows.append({"arm": "C3 pinned fc2+mul -> L1", "hidden": hidden, "h": h,
                             "ms_median": None, "l1_MB": round(l1_mb, 1), "note": "L1 clash"})
                continue
            rr = p64.row(rows, "C3 pinned fc2+mul -> L1", hidden, h, ms,
                         "%.0f MB live in L1" % l1_mb)
            rr["l1_MB"] = round(l1_mb, 1)
            if hidden not in best or ms[0] < best[hidden][1]:
                best[hidden] = (h, ms[0])
        M._PAIR_TRANSITION_L1 = False
    tot_s = sum(4 * ship[h] for h in (512, 256))
    tot_r = sum(4 * best[h][1] for h in (512, 256) if h in best)
    net = tot_s - tot_r
    print("\nSCREEN  shipped %.1f -> %.1f ms/step   net %+.1f ms/step = %+.2f s/design   h=%s"
          % (tot_s, tot_r, -net, -net * 200 / 1e3,
             {h: best[h][0] for h in (512, 256) if h in best}), flush=True)
    print("GATE    NO-GO if net < 15 ms/step.", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "shipped_ms_call": ship,
        "best": {str(k): {"h": v[0], "ms": round(v[1], 4)} for k, v in best.items()},
        "screen": {"shipped_ms_step": round(tot_s, 2), "route_ms_step": round(tot_r, 2),
                   "net_ms_step": round(net, 2), "net_s_design": round(net * 200 / 1e3, 3)},
        "tokens": I, "n_warm": p64.NWARM, "n_rep": p64.NREP, "host": "qb2", "card": 2,
        "tune_matmul": bool(M._TUNE_MATMUL)}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
