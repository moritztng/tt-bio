#!/usr/bin/env python3
"""Does the DST-granularity lever generalise off attention? A/B on a structurally unrelated kernel.

`b2z-custom-sdpa` measured, on the Boltz-2 triangle-attention SDPA: the op is at 34.0 % of its byte
roof and 33.4 % of its matmul roof AT ONCE; one 128-tile pass over the score block costs 1.168 ms
(17.8 % of the op), linear in both directions at 71.3 ns a tile against a 64 ns/tile packer floor.
Two sibling rows found the same signature in unrelated kernels -- trimul's matmul-shaped ops at
37-49 % and 15-52 % of their two roofs at once, and 97.28 % of cores busy across a whole
PairformerLayer, which refutes under-parallelisation. The orchestrator's question is whether the
DST treatment is UNIFORM, because a uniform per-core pipeline defect is the campaign and a pile of
per-kernel accidents is not.

`reblock_permute_gated` is the test. It is structurally unrelated to attention -- a gated channel
permute inside TriangleMultiplication -- and it is the DENSEST instance of the defect in the tree:
the SDPA's mask add paid ONE `acquire_dst()` a tile, this pays THREE, because each of sigmoid,
multiply and transpose round-trips through its own bf16 CB to hold a rounding point that the
kernel's bit-exactness against ttnn depends on. At 512 aa it is 3.149 ms of the 16.961 ms
TriangleMultiplication chain (18.6 %).

PREDICTED, before the first number:

    The barrier was NOT what the SDPA's add paid for -- batching it there returned only 1.0267x.
    This kernel has 3x the barrier density per tile, so if the lever is uniform it should return
    MORE than 1.0267x here, but still modest: 1.05-1.15x on the op.

    FALSIFIER, and it is the interesting outcome: at or under 1.01x. The host comment on these CBs
    says "a deeper ring would only hold more of a stream the writer is already the slow end of". If
    the writer is the slow end then removing compute barriers buys nothing, the lever does NOT
    generalise, and the convergence across the three rows is a coincidence of three kernels each
    being slow for its own reason.

Bit-exactness is checked, not assumed, at every granularity: same three stages, same order, same two
bf16 circular buffers, so every rounding point is untouched and `torch.equal` must hold. A parity
failure here is a STOP -- this kernel's `torch.equal` against the two-op ttnn sequence is a standing
claim, not a PCC.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import reblock_permute as RB

N, C = 512, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/b2z_sdpa_floor/gategran_512.json")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--grans", default="1,2,4")
    args = ap.parse_args()
    grans = [int(x) for x in args.grans.split(",")]

    dev = T.get_device()
    res = {"doc": __doc__, "meta": {
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(T.COMPUTE_GRID_MAIN),
        "loadavg": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shape": [1, N, N, 4 * C], "grans": grans}}
    print(json.dumps(res["meta"]), flush=True)

    torch.manual_seed(0)
    h = torch.randn(1, N, N, 4 * C).bfloat16()
    xw = ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def run(g):
        RB.GATE_GRANULARITY = g
        return RB.reblock_permute_gated(xw, 2 * C, 0, C, ttnn.DRAM_MEMORY_CONFIG)

    # parity first, against the incumbent, on the same operand
    ref = ttnn.to_torch(run(1))
    res["parity_vs_gran1"] = {}
    for g in grans:
        if g == 1:
            continue
        got = ttnn.to_torch(run(g))
        eq = bool(torch.equal(ref, got))
        res["parity_vs_gran1"][str(g)] = {
            "equal": eq, "max_abs": float((ref.float() - got.float()).abs().max())}
        print(f"parity gran{g}: equal={eq}", flush=True)
    assert all(v["equal"] for v in res["parity_vs_gran1"].values()), \
        "granularity broke bit-exactness -- STOP, this kernel's torch.equal is a standing claim"

    for g in grans:
        for _ in range(2):
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
    RB.GATE_GRANULARITY = 1

    med = {g: st.median(v_) for g, v_ in acc.items()}
    base = med[grans[0]]
    res["arms"] = {str(g): {"ms": med[g], "all_ms": acc[g], "over_gran1": base / med[g],
                            "saved_ms": base - med[g]} for g in grans}
    best = max(grans, key=lambda g: base / med[g])
    res["verdict"] = {
        "best_gran": best, "best_speedup": base / med[best],
        "generalises": bool(base / med[best] > 1.01),
        "falsifier": "at or under 1.01x means the writer is the slow end and the lever does not "
                     "generalise off attention"}
    print(json.dumps({str(g): [round(med[g], 4), round(base / med[g], 4)] for g in grans}, indent=1),
          flush=True)
    print(json.dumps(res["verdict"], indent=1), flush=True)

    res["meta"]["loadavg_end"] = os.getloadavg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
