#!/usr/bin/env python3
"""Is the daisy chain faster than a multicast where a K loop exists? The gate's own falsifier.

The shipped kernel's comment says the author chose the chain on purpose: *"Critical to performance
for sender to push data to compute before mcasting. This frees sender to start next read earlier"*.
The honest reading is that the chain buys pipelining across K blocks -- a core forwards block k
while it reads k+1 -- and `b2z2-cb-depth-prefetch` measured that every shipped block config has
`K_num_blocks == 1`, so there is no K loop left to pipeline. That is why `mm_generic.mcast_for`
gates on `K_num_blocks == 1` and not on a model name.

The gate is a claim about a regime this tree never enters, so it is worth measuring rather than
asserting. `TT_BIO_MM_MCAST_ANY_K` lifts it, and this sweeps one matmul across K_num_blocks
1, 2, 4 and 8 at a fixed block config, A/B-ing chain against multicast at each. A ratio that falls
towards 1 as the K loop lengthens confirms the gate; a ratio that stays flat says the gate is
narrower than it needs to be, and a ratio below 1 at K_num_blocks > 1 is the reason it exists.

Bit-exactness is checked at every K, against the chain arm's own output.
"""
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

OUT: dict = {"doc": __doc__}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def timed(ttnn, dev, call, reps):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / reps


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "kloop_ab.json")
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--kblocks", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-rep", type=int, default=5)
    a = ap.parse_args()
    OUT_PATH = a.out

    os.environ["TT_BIO_MM_MCAST"] = "0"
    os.environ["TT_BIO_MM_MCAST_ANY_K"] = "1"
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.mm_generic as MG

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    K_BLOCK = 4
    cfg_block = (4, K_BLOCK, 1, 4, 1)
    ckc = MG.ckc_args(T._mm_compute_kernel_config()) if hasattr(T, "_mm_compute_kernel_config") \
        else (ttnn.MathFidelity.HiFi4, False, False, False)
    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(grid),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": open("/proc/loadavg").read().split()[:3],
                  "m": a.m, "n": a.n, "block_cfg": list(cfg_block), "reps": a.reps}
    print(json.dumps(OUT["env"]), flush=True)
    dump()

    torch.manual_seed(0)
    rows = []
    for kb in [int(x) for x in a.kblocks.split(",")]:
        K = 32 * K_BLOCK * kb
        x = ttnn.from_torch(torch.randn(a.m, K).bfloat16(), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(torch.randn(K, a.n).bfloat16(), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def run():
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([a.m, a.n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                ttnn.DRAM_MEMORY_CONFIG)
            MG.generic_minimal_matmul(dev, x, w, out, (cfg_block, grid), ckc)
            return out

        os.environ["TT_BIO_MM_MCAST"] = "0"
        ref = ttnn.to_torch(run())
        os.environ["TT_BIO_MM_MCAST"] = "1"
        got_t = run()
        got = ttnn.to_torch(got_t)
        bit_exact = bool(torch.equal(ref, got))
        built = [(k[-1], e["dims"]["mcast"], e["dims"]["K_blocks"]) for k, e in MG._CACHE.items()]

        def slot(on):
            os.environ["TT_BIO_MM_MCAST"] = "1" if on else "0"
            return timed(ttnn, dev, lambda: ttnn.deallocate(run()), a.reps)

        for _ in range(2):
            slot(False), slot(True)
        chain, mcast = [], []
        for _ in range(a.n_rep):
            c1, m1, m2, c2 = slot(False), slot(True), slot(True), slot(False)
            chain += [c1, c2]
            mcast += [m1, m2]
        row = {"K_num_blocks": kb, "K": K, "bit_exact": bit_exact,
               "chain_ms": round(st.median(chain), 5), "mcast_ms": round(st.median(mcast), 5),
               "ratio_chain_over_mcast": round(st.median(chain) / st.median(mcast), 5),
               "aa_floor": round(st.median(chain[0::2]) / st.median(chain[1::2]), 5),
               "mcast_gate_saw": sorted({(bool(b), int(k)) for _, b, k in built})}
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
        OUT["rows"] = rows
        dump()
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        ttnn.deallocate(got_t)

    OUT["verdict"] = {
        "all_bit_exact": all(r["bit_exact"] for r in rows),
        "ratio_by_k": {r["K_num_blocks"]: r["ratio_chain_over_mcast"] for r in rows}}
    print(json.dumps(OUT["verdict"], indent=1), flush=True)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
