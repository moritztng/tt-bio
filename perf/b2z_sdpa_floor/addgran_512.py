#!/usr/bin/env python3
"""Batch the DST acquire in the SDPA's mask add, and price what that is worth.

`ablate_512.py` measured the mask add at 1.373 ms of a 6.809 ms op (20.2 %) on whglx card 3, while
the SFPU exponential over the same tiles cost 0.276 ms (4.1 %). An eltwise add cannot cost 5x an
exp; what it costs is one `acquire_dst()` per tile -- 14 592 math-to-pack barriers per core -- in
the one block helper the transcription left un-batched.

PREDICTED, before the first number:

    Batching to the DST budget (8 tiles) leaves the same adds in the same order and only removes
    the barrier, so it must be BIT-EXACT, and it should take roughly half to three quarters of the
    add's 1.373 ms: 0.7-1.1 ms, i.e. 1.11-1.19x on the op.

    FALSIFIER: under 0.3 ms means the barrier is not what the add is paying for and the ablation's
    1.373 ms is something else -- most likely the mask CB reads, which the ablation left in place.

Interleaved in one process. The incumbent is `TT_BIO_SDPA_ADD_GRANULARITY=1`, which restores the
per-tile loop verbatim, so this is an A/B against the shipped kernel and not against a rewrite.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T

S, H, D = 512, 8, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/b2z_sdpa_floor/addgran_512.json")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--grans", default="1,2,4,8")
    args = ap.parse_args()
    grans = [g.strip() for g in args.grans.split(",")]

    dev = T.get_device()
    res = {"doc": __doc__, "meta": {
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(T.COMPUTE_GRID_MAIN),
        "loadavg": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shape": [S, H, S, D], "grans": grans}}
    print(json.dumps(res["meta"]), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    q, k, v = (dram(torch.randn(S, H, S, D).to(torch.bfloat16)) for _ in range(3))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))
    scale = D ** -0.5

    def run(g):
        os.environ["TT_BIO_SDPA_ADD_GRANULARITY"] = g
        o = T._tri_att_sdpa_at(q, k, v, bias, scale)
        assert o is not None
        return o

    # parity first, against granularity 1, on the same operands
    ref = ttnn.to_torch(run(grans[0]))
    res["parity_vs_gran1"] = {}
    for g in grans[1:]:
        got = ttnn.to_torch(run(g))
        res["parity_vs_gran1"][g] = {
            "equal": bool(torch.equal(ref, got)),
            "max_abs": float((ref.float() - got.float()).abs().max())}
    print("parity", json.dumps(res["parity_vs_gran1"]), flush=True)

    for g in grans:
        for _ in range(3):
            ttnn.deallocate(run(g))
    ttnn.synchronize_device(dev)

    acc = {g: [] for g in grans}
    for _ in range(args.reps):
        for g in grans:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = run(g)
            ttnn.synchronize_device(dev)
            acc[g].append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
    os.environ.pop("TT_BIO_SDPA_ADD_GRANULARITY", None)

    med = {g: st.median(v_) for g, v_ in acc.items()}
    base = med[grans[0]]
    res["arms"] = {g: {"ms": med[g], "all_ms": acc[g], "over_gran1": base / med[g],
                       "saved_ms": base - med[g]} for g in grans}
    print(json.dumps({g: [round(med[g], 4), round(base / med[g], 4)] for g in grans}, indent=1),
          flush=True)

    res["meta"]["loadavg_end"] = os.getloadavg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
