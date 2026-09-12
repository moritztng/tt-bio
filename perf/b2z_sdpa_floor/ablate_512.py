#!/usr/bin/env python3
"""Where the Boltz-2 triangle-attention SDPA's time actually goes, by removing stages from it.

`floor_512.py` put the shipped fused kernel at 34 % of its own measured byte roof AND 33 % of its
own measured matmul roof on whglx card 3 (WH): 6.829 ms against a 2.321 ms floor, a 2.94x deficit
with the same signature as the campaign-wide 2.34x. Neither roof prices the SFPU, and at head_dim
32 this op exponentiates a score matrix 16x larger than any operand either roof counts.

PREDICTED, before the first ablation number:

    The score matrix is S*S per (batch, head) = 1.074e9 elements per call, against 2.68e8 elements
    of q+k+v+out. Every one of them gets a subtract, an exp and an L1-accumulated add; the matmuls
    that produce and consume them are only 256 tile-MACs per (batch, head, k_chunk) against ~500
    tile-passes of eltwise over the same tiles. So removing the exp alone should take 30-55 % off
    the op, and the mask add -- one pass over the same 128 tiles per chunk -- 8-15 %.

    FALSIFIER: if removing the exp moves the op by under 10 %, the SFPU is not the missing term and
    the deficit is somewhere neither this ablation nor either roof can see.

Arms are interleaved in ONE process. Every ablated arm computes WRONG VALUES by construction; only
the times mean anything, which is why nothing here writes a parity number.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import triatt_sdpa as TS

S, H, D = 512, 8, 32
ARMS = (("full", ()), ("no_exp", ("EXP",)), ("no_maskadd", ("MASKADD",)),
        ("no_exp_no_maskadd", ("EXP", "MASKADD")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/b2z_sdpa_floor/ablate_512.json")
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()

    dev = T.get_device()
    res = {"doc": __doc__, "meta": {
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(T.COMPUTE_GRID_MAIN),
        "loadavg": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shape": [S, H, S, D]}}
    print(json.dumps(res["meta"]), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    q, k, v = (dram(torch.randn(S, H, S, D).to(torch.bfloat16)) for _ in range(3))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))
    scale = D ** -0.5

    def run(abl):
        TS._ABLATE = abl
        o = T._tri_att_sdpa_at(q, k, v, bias, scale)
        assert o is not None
        return o

    # compile + warm every arm before any of them is timed, so the first timed rep of arm 0 is
    # not paying a JIT the later arms have already amortised
    for _, abl in ARMS:
        for _ in range(3):
            ttnn.deallocate(run(abl))
    ttnn.synchronize_device(dev)

    acc = {name: [] for name, _ in ARMS}
    for _ in range(args.reps):
        for name, abl in ARMS:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = run(abl)
            ttnn.synchronize_device(dev)
            acc[name].append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
    TS._ABLATE = ()

    med = {n: st.median(v_) for n, v_ in acc.items()}
    full = med["full"]
    res["arms"] = {n: {"ms": med[n], "all_ms": acc[n],
                       "removed_ms": full - med[n], "removed_frac": (full - med[n]) / full}
                   for n, _ in ARMS}
    res["additivity"] = {
        "exp_plus_maskadd_ms": (full - med["no_exp"]) + (full - med["no_maskadd"]),
        "both_together_ms": full - med["no_exp_no_maskadd"]}
    print(json.dumps({n: round(med[n], 4) for n in med}, indent=1), flush=True)
    print(json.dumps(res["additivity"], indent=1), flush=True)

    res["meta"]["loadavg_end"] = os.getloadavg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
